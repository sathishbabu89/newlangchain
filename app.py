import logging
import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.huggingface_hub import HuggingFaceHub

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUGGINGFACE_API_TOKEN = ""

st.set_page_config(page_title="Chat with Your PDF", page_icon="📚")
st.header("Chat with Your PDF 📚")

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
                    separators="\n",
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text)
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
            st.success("Document processed! You can now start chatting.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        st.error(f"An error occurred while processing the PDF: {e}")

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
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                match = st.session_state.vector_store.similarity_search(user_question)
                if not match:
                    full_response = "I couldn't find any relevant information in the document to answer your question. Could you please rephrase or ask something else?"
                else:
                    llm = HuggingFaceHub(
                        repo_id="tiiuae/falcon-7b-instruct",
                        task="text-generation",
                        model_kwargs={
                            "temperature": 0.5,
                            "max_length": 4096,
                            "top_k": 3
                        },
                        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
                    )
                    chain = load_qa_chain(llm, chain_type="stuff")
                    response = chain.invoke({"input_documents": match, "question": user_question})
                    full_response = response['output_text'].split("Helpful Answer:")[-1].strip()

                # Simulate typing effect
                for chunk in full_response.split():
                    full_response += chunk + " "
                    message_placeholder.markdown(full_response + "▌")
                    # Adjust the sleep duration to control the typing speed
                    # time.sleep(0.05)

                message_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                logger.error(f"An error occurred: {e}")
                st.error(f"An error occurred while generating the response: {e}")

else:
    st.info("Please upload a PDF document to start chatting.")