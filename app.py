import logging
import streamlit as st
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.huggingface_hub import HuggingFaceHub

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUGGINGFACE_API_TOKEN = "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"

st.set_page_config(page_title="Chat with your document", page_icon="📚")
st.header("Chat with your document 📚")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# File uploader
with st.sidebar:
    st.title("Upload Your Document")
    file = st.file_uploader("Upload a PDF file to start chatting", type="pdf")

# Process the PDF when uploaded
if file is not None and st.session_state.vector_store is None:
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        if not text.strip():
            st.warning("No text found in the uploaded PDF.")
        else:
            with st.spinner("Processing document..."):
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", ".", "!", "?"],
                    chunk_size=500,
                    chunk_overlap=50,
                    length_function=len
                )
                chunks = text_splitter.split_text(text)
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.vector_store = FAISS.from_texts(
                    chunks, embeddings)
            st.success("Document processed! You can now start chatting.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        st.error(f"An error occurred while processing the PDF: {e}")

# Improved prompt template
prompt_template = """
You are an AI assistant tasked with answering questions about a specific document.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep your answers factual and based solely on the provided context.
Always provide complete sentences and ensure your response ends with a full stop or appropriate punctuation.

Context: {context}

Question: {question}

Helpful Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def ensure_complete_sentence(text):
    """Ensure the text ends with a complete sentence."""
    if not text:
        return text

    # List of sentence-ending punctuation
    end_punctuation = ['.', '!', '?']

    # If the text already ends with sentence-ending punctuation, return it as is
    if text[-1] in end_punctuation:
        return text

    # Find the last occurrence of sentence-ending punctuation
    last_sentence_end = max(text.rfind(p) for p in end_punctuation)

    if last_sentence_end == -1:
        # If no sentence-ending punctuation found, add a period at the end
        return text + "."
    else:
        # Return the text up to and including the last complete sentence
        return text[:last_sentence_end + 1]


def extract_answer(response):
    """Extract only the AI's answer from the full response."""
    helpful_answer_start = response.find("Helpful Answer:")
    if helpful_answer_start != -1:
        return response[helpful_answer_start + len("Helpful Answer:"):].strip()
    return response.strip()


# Chat interface
if st.session_state.vector_store is not None:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    user_question = st.chat_input("Ask a question about your document")

    if user_question:
        # Add user message to chat history
        st.session_state.chat_history.append(
            {"role": "user", "content": user_question})

        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            try:
                # Load HuggingFace model
                llm = HuggingFaceHub(
                    repo_id="HuggingFaceH4/zephyr-7b-beta",
                    task="text-generation",
                    model_kwargs={
                        "temperature": 0.2,
                        "max_new_tokens": 250,
                        "top_p": 0.95,
                        "do_sample": True,
                        "eos_token_id": [13, 198]
                    },
                    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
                )

                # Create RetrievalQA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vector_store.as_retriever(
                        search_kwargs={"k": 3}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": PROMPT}
                )

                # Run the QA chain
                response = qa_chain({"query": user_question})
                extracted_answer = extract_answer(response['result'])
                full_response = ensure_complete_sentence(extracted_answer)

                # Display the full response with simulated typing
                message_placeholder.markdown(full_response)

                # Add assistant response to chat history
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": full_response})

            except Exception as e:
                logger.error(f"An error occurred: {e}")
                st.error(
                    f"An error occurred while generating the response: {e}")

else:
    st.info("Please upload a PDF document to start chatting.")
