"""Document Assistant Tool"""
import logging
import streamlit as st
from langchain import hub
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
API_TOKEN = "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"

st.set_page_config(page_title="Document Assistant Tool", page_icon="📚")
st.header("Document Assistant Tool 📚")

with st.sidebar:
    st.title("Upload Your Document")
    file = st.file_uploader("Upload a PDF file to start chatting", type="pdf")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

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
                    chunk_overlap=100,
                    length_function=len
                )
                splits = text_splitter.split_text(text)
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.vector_store = FAISS.from_texts(
                    splits, embeddings)
            st.success("Document processed! You can now start chatting.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        st.error(f"An error occurred while processing the PDF: {e}")


def format_docs(docs: any):
    """Format documentation"""
    st.write(docs)
    return "\n\n".join(doc.page_content for doc in docs)


if st.session_state.vector_store is not None:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    user_question = st.chat_input("Ask a question about your document")
    if user_question:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                retriever = st.session_state.vector_store.as_retriever(
                    search_kwargs={"k": 1})
                llm = HuggingFaceEndpoint(
                    repo_id="mistralai/Mistral-Nemo-Instruct-2407",
                    task="text-generation",
                    max_new_tokens=1000,
                    temperature=0.5,
                    # top_k=10,
                    # top_p=0.95,
                    # typical_p=0.95,
                    # repetition_penalty=1.03,
                    huggingfacehub_api_token=API_TOKEN
                )
                prompt = hub.pull("rlm/rag-prompt")
                qa_chain = (
                    {"context": retriever | format_docs,
                     "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                response = qa_chain.invoke(user_question)
                message_placeholder.markdown(response)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response})

            except Exception as e:
                logger.error(f"An error occurred: {e}")
                st.error(
                    f"An error occurred while generating the response: {e}")
else:
    st.info("Please upload a PDF document to start chatting.")
