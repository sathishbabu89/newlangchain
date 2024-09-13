"""Module providing a function QA generation."""
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

HUGGINGFACE_API_TOKEN = "hf_SkQojHETEBQqnseFmIXIwhEPvMlRICxo"

st.header("Welcome to docs assistant bot 👋")
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader(
        "Upload a PDF file and start asking questions", type="pdf")


if file is not None:
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        if not text.strip():
            st.warning("No text found in the uploaded PDF.")
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                separators="\n",
                chunk_size=1000,
                chunk_overlap=150,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            embeddings = HuggingFaceEmbeddings(
                model_name="cointegrated/rubert-tiny")
            vector_store = FAISS.from_texts(chunks, embeddings)
            user_question = st.text_input("Type your question here")
            if user_question:
                if len(user_question.strip()) == 0:
                    st.warning("Please enter a question.")
                else:
                    match = vector_store.similarity_search(user_question)
                    if not match:
                        st.warning("No relevant information found.")
                    else:
                        llm = HuggingFaceHub(
                            repo_id="tiiuae/falcon-7b-instruct",
                            task="text-generation",
                            model_kwargs={
                                "temperature": 0.5,
                                "max_length": 1000
                            },
                            huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
                        )
                        chain = load_qa_chain(llm, chain_type="stuff")
                        response = chain.invoke(
                            {"input_documents": match, "question": user_question})
                        answer = response['output_text'].split(
                            "Helpful Answer:")[-1].strip()
                        st.write(answer)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        st.error(e)
