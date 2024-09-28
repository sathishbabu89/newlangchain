import logging
import streamlit as st
from PyPDF2 import PdfReader
import pdfplumber
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUGGINGFACE_API_TOKEN = ""

st.set_page_config(page_title="Document Assistant Tool", page_icon="📚")
st.header("Document Assistant Tool 📚")

# Initialize session state for conversation history, vector store, and feedback tracking
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'feedback' not in st.session_state:
    st.session_state.feedback = []  # Track feedback for each question-answer pair

# Sidebar for file upload
with st.sidebar:
    st.title("Upload Your Document")
    file = st.file_uploader("Upload a PDF file to start chatting", type="pdf")

    if file is not None:
        # Display PDF preview in sidebar for all pages
        try:
            with pdfplumber.open(file) as pdf:
                st.subheader("PDF Preview")
                for page_num, page in enumerate(pdf.pages):
                    pdf_image = page.to_image()
                    img = pdf_image.original
                    st.image(img, caption=f"Page {page_num + 1}", use_column_width=True)
        except Exception as e:
            logger.error(f"An error occurred while previewing the PDF: {e}")
            st.warning("Unable to display PDF preview.")

# If a file is uploaded, process the document
if file is not None:
    if st.session_state.vector_store is None:
        try:
            with st.spinner("Processing document..."):
                pdf_reader = PdfReader(file)
                TEXT = ""
                for page in pdf_reader.pages:
                    TEXT += page.extract_text() or ""

                if not TEXT.strip():
                    st.warning("No text found in the uploaded PDF.")
                else:
                    # Process the document and create embeddings
                    text_splitter = RecursiveCharacterTextSplitter(
                        separators="\n",
                        chunk_size=1000,
                        chunk_overlap=100,
                        length_function=len
                    )
                    chunks = text_splitter.split_text(TEXT)

                    # Vector store creation
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
                    st.success("Document processed successfully!")
        except Exception as e:
            logger.error(f"An error occurred while processing the document: {e}")
            st.error(str(e))

    # Display the conversation history
    for idx, message in enumerate(st.session_state.conversation_history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display feedback for previous answers
            if message["role"] == "assistant" and idx < len(st.session_state.feedback):
                feedback = st.session_state.feedback[idx]
                st.markdown(f"**Feedback: {feedback}**")

    # Accept user question input
    user_question = st.chat_input("Type your question here")

    if user_question:
        if len(user_question.strip()) == 0:
            st.warning("Please enter a question.")
        else:
            st.session_state.conversation_history.append(
                {"role": "user", "content": user_question}
            )
            with st.chat_message("user"):
                st.markdown(user_question)

            if st.session_state.vector_store is not None:
                with st.spinner("Thinking..."):
                    try:
                        retriever = st.session_state.vector_store.similarity_search(
                            user_question
                        )
                        if not retriever:
                            st.warning("No relevant information found.")
                        else:
                            llm = HuggingFaceEndpoint(
                                repo_id="mistralai/Mistral-Nemo-Instruct-2407",
                                max_new_tokens=512,
                                top_k=10,
                                top_p=0.95,
                                typical_p=0.95,
                                temperature=0.01,
                                repetition_penalty=1.03,
                                huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
                            )
                            chain = load_qa_chain(llm, chain_type="stuff")
                            response = chain.invoke(
                                {"input_documents": retriever, "question": user_question}
                            )
                            answer = response['output_text'].split("Helpful Answer:")[-1].strip()
                            st.session_state.conversation_history.append(
                                {"role": "assistant", "content": answer}
                            )
                            with st.chat_message("assistant"):
                                st.markdown(answer)

                            # Add interactive Like/Dislike buttons and track feedback
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button('👍 Like', key=f"like_{len(st.session_state.feedback)}"):
                                    st.session_state.feedback.append("Liked")
                            with col2:
                                if st.button('👎 Dislike', key=f"dislike_{len(st.session_state.feedback)}"):
                                    st.session_state.feedback.append("Disliked")

                            # If no feedback is given, default to 'No Feedback'
                            if len(st.session_state.feedback) == len(st.session_state.conversation_history) - 1:
                                st.session_state.feedback.append("No Feedback")
                    except Exception as e:
                        logger.error(f"An error occurred while processing the question: {e}")
                        st.error(str(e))
            else:
                st.warning("Please wait for the document to finish processing before asking questions.")
else:
    st.info("Please upload a PDF document to start chatting.")

# Display total Like and Dislike counts at the bottom
like_count = st.session_state.feedback.count("Liked")
dislike_count = st.session_state.feedback.count("Disliked")
st.markdown(f"**Total Feedback Summary:**")
st.write(f"👍 Likes: {like_count}")
st.write(f"👎 Dislikes: {dislike_count}")
