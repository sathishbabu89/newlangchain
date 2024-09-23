"""Document Assistant Tool"""
import os
import logging
import streamlit as st
from langchain import hub
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["LANGCHAIN_API_KEY"] = "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"

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
        with open("temp.pdf", "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load_and_split()
        st.write(docs)
        with st.spinner("Processing document..."):
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", "!", "?"],
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            splits = text_splitter.split_documents(docs)
            st.write(splits)
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2")
            st.session_state.vector_store = Chroma.from_documents(
                documents=splits, embedding=embeddings)
        st.success("Document processed! You can now start chatting.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        st.error(f"An error occurred while processing the PDF: {e}")


def format_docs(docs):
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
                    search_kwargs={"k": 3})
                llm = HuggingFaceEndpoint(
                    repo_id="HuggingFaceH4/zephyr-7b-beta",
                    task="text-generation",
                    max_new_tokens=512,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.03,
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
