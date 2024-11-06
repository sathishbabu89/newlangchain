import requests
from bs4 import BeautifulSoup
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

st.set_page_config(page_title="Web & Document Assistant Tool", page_icon="🌐")
st.header("Web & Document Assistant Tool 🌐")

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Function to fetch website content using requests
def fetch_website_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"An error occurred while fetching the website: {e}")
        return None

with st.sidebar:
    st.title("Input Source")
    st.subheader("Options:")
    option = st.selectbox("Choose your input type:", ["Upload PDF", "Enter Website URL"])

    if option == "Upload PDF":
        file = st.file_uploader("Upload a PDF file to start chatting", type="pdf")

        if file is not None:
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

    elif option == "Enter Website URL":
        url = st.text_input("Enter a website URL (e.g., https://www.example.com/)")

        if url:
            with st.spinner("Fetching website content..."):
                page_content = fetch_website_content(url)
                if page_content:
                    soup = BeautifulSoup(page_content, "html.parser")
                    website_text = ' '.join([p.get_text() for p in soup.find_all("p")])
                    if not website_text.strip():
                        st.warning("No text found on the webpage.")
                    else:
                        text_splitter = RecursiveCharacterTextSplitter(
                            separators="\n", chunk_size=1000, chunk_overlap=100, length_function=len
                        )
                        chunks = text_splitter.split_text(website_text)
                        embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2"
                        )
                        st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
                        st.success("Website content processed successfully!")
                else:
                    st.error("Error fetching or processing the website content.")

# Document processing for PDFs
if (option == "Upload PDF" and file):
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
                    text_splitter = RecursiveCharacterTextSplitter(
                        separators="\n", chunk_size=1000, chunk_overlap=100, length_function=len
                    )
                    chunks = text_splitter.split_text(TEXT)
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
                    st.success("Document processed successfully!")
        except Exception as e:
            logger.error(f"An error occurred while processing the document: {e}")
            st.error(str(e))

# Chatbot interface for questions
if (option == "Upload PDF" and file) or (option == "Enter Website URL" and url):
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Type your question here")

    if user_question:
        if len(user_question.strip()) == 0:
            st.warning("Please enter a question.")
        else:
            st.session_state.conversation_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            if st.session_state.vector_store is not None:
                with st.spinner("Thinking..."):
                    try:
                        retriever = st.session_state.vector_store.similarity_search(user_question)
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
                                huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
                            )
                            chain = load_qa_chain(llm, chain_type="stuff")
                            response = chain.invoke({"input_documents": retriever, "question": user_question})
                            answer = response['output_text'].split("Helpful Answer:")[-1].strip()
                            st.session_state.conversation_history.append({"role": "assistant", "content": answer})
                            with st.chat_message("assistant"):
                                st.markdown(answer)
                    except Exception as e:
                        logger.error(f"An error occurred while processing the question: {e}")
                        st.error(str(e))
            else:
                st.warning("Please wait for the content to finish processing before asking questions.")
else:
    st.info("Please upload a PDF document or enter a URL to start chatting.")
