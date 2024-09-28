import logging
import streamlit as st
from PyPDF2 import PdfReader
import pdfplumber
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define your Hugging Face API token
HUGGINGFACE_API_TOKEN = "your_huggingface_api_token_here"

# Set up the Streamlit page configuration
st.set_page_config(page_title="Document Assistant Tool", page_icon="📚")
st.header("Document Assistant Tool 📚")

# Initialize session state variables
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'pdf_texts' not in st.session_state:
    st.session_state.pdf_texts = []
if 'chunks' not in st.session_state:
    st.session_state.chunks = []

# Sidebar for uploading files
with st.sidebar:
    st.title("Upload Your Document")
    multiple_files = st.file_uploader("Upload PDF files to start chatting", type="pdf", accept_multiple_files=True)

    # Preview the uploaded PDFs in the sidebar
    if multiple_files:
        try:
            for file in multiple_files:
                with pdfplumber.open(file) as pdf:
                    st.subheader(f"PDF Preview: {file.name}")
                    for page_num, page in enumerate(pdf.pages):
                        pdf_image = page.to_image()
                        img = pdf_image.original
                        st.image(img, caption=f"Page {page_num + 1}", use_column_width=True)
        except Exception as e:
            logger.error(f"An error occurred while previewing the PDF: {e}")
            st.warning("Unable to display PDF preview.")

# Process PDF and split text into chunks
if multiple_files and st.session_state.vector_store is None:
    try:
        with st.spinner("Processing documents..."):
            all_texts = []
            for file in multiple_files:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                all_texts.append(text)

            # Combine all text from multiple PDFs
            combined_text = "\n".join(all_texts)

            # Use RecursiveCharacterTextSplitter to split the text into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100, length_function=len
            )
            chunks = text_splitter.split_text(combined_text)
            st.session_state.chunks = chunks  # Save chunks in session state for search later

            # Create embeddings using Hugging Face's Sentence Transformers
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
            st.success("Document processed successfully!")

            # Summarize the document using Hugging Face pipeline
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            summary = summarizer(combined_text[:1000])  # Summarize the first 1000 characters
            st.write("**Document Summary:**", summary[0]['summary_text'])

    except Exception as e:
        logger.error(f"An error occurred while processing the document: {e}")
        st.error(str(e))

# If a PDF has been processed, allow users to ask questions
if st.session_state.vector_store:
    st.write("## Ask Questions about the Document")
    
    # Display conversation history
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Type your question here")

    if user_question:
        # Store the user's question in the conversation history
        st.session_state.conversation_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Perform a similarity search in the vector store using the user's question
        with st.spinner("Searching for relevant information..."):
            try:
                retriever = st.session_state.vector_store.similarity_search(user_question)
                if retriever:
                    # Load the LLM and ask a question
                    llm = HuggingFaceHub(
                        repo_id="mistralai/Mistral-Nemo-Instruct-2407",
                        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
                    )
                    qa_chain = load_qa_chain(llm, chain_type="stuff")
                    answer = qa_chain.run(
                        {"input_documents": retriever, "question": user_question}
                    )

                    # Add the assistant's response to the conversation history
                    st.session_state.conversation_history.append({"role": "assistant", "content": answer})
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                else:
                    st.warning("No relevant information found.")

            except Exception as e:
                logger.error(f"An error occurred while answering the question: {e}")
                st.error(str(e))

# Implement a keyword search functionality
search_term = st.text_input("Search for a keyword in the document:")
if search_term and st.session_state.vector_store:
    chunks = st.session_state.chunks
    search_results = [chunk for chunk in chunks if search_term.lower() in chunk.lower()]
    if search_results:
        st.write(f"Found {len(search_results)} result(s) for '{search_term}':")
        for result in search_results:
            st.write(result)
    else:
        st.write(f"No results found for '{search_term}'.")

