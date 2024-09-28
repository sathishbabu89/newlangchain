import logging
import streamlit as st
from PyPDF2 import PdfReader
import pdfplumber
from PIL import Image
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUGGINGFACE_API_TOKEN = ""

st.set_page_config(page_title="Project Status Tracker", page_icon="📊")
st.header("Project Status Tracker 📊")

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'vector_store_1' not in st.session_state:
    st.session_state.vector_store_1 = None
if 'vector_store_2' not in st.session_state:
    st.session_state.vector_store_2 = None

# Sidebar for document upload
with st.sidebar:
    st.title("Upload Project Documents")
    
    # Upload first document
    file1 = st.file_uploader("Upload the first PDF file (Last Year)", type="pdf")
    
    # Upload second document
    file2 = st.file_uploader("Upload the second PDF file (Current Year)", type="pdf")

    # Preview first document
    if file1 is not None:
        try:
            with pdfplumber.open(file1) as pdf:
                st.subheader("PDF Preview - Last Year")
                for page_num, page in enumerate(pdf.pages):
                    pdf_image = page.to_image()
                    img = pdf_image.original
                    st.image(img, caption=f"Page {page_num + 1}", use_column_width=True)
        except Exception as e:
            logger.error(f"An error occurred while previewing the PDF: {e}")
            st.warning("Unable to display PDF preview for last year.")

    # Preview second document
    if file2 is not None:
        try:
            with pdfplumber.open(file2) as pdf:
                st.subheader("PDF Preview - Current Year")
                for page_num, page in enumerate(pdf.pages):
                    pdf_image = page.to_image()
                    img = pdf_image.original
                    st.image(img, caption=f"Page {page_num + 1}", use_column_width=True)
        except Exception as e:
            logger.error(f"An error occurred while previewing the PDF: {e}")
            st.warning("Unable to display PDF preview for current year.")

# Process documents and generate vector stores
if file1 is not None and st.session_state.vector_store_1 is None:
    try:
        with st.spinner("Processing last year's document..."):
            pdf_reader = PdfReader(file1)
            TEXT_1 = ""
            for page in pdf_reader.pages:
                TEXT_1 += page.extract_text() or ""

            if not TEXT_1.strip():
                st.warning("No text found in the first uploaded PDF.")
            else:
                text_splitter = RecursiveCharacterTextSplitter(
                    separators="\n",
                    chunk_size=1000,
                    chunk_overlap=100,
                    length_function=len
                )
                chunks = text_splitter.split_text(TEXT_1)
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.vector_store_1 = FAISS.from_texts(chunks, embeddings)
                st.success("Last year's document processed successfully!")
    except Exception as e:
        logger.error(f"An error occurred while processing the last year's document: {e}")
        st.error(str(e))

if file2 is not None and st.session_state.vector_store_2 is None:
    try:
        with st.spinner("Processing current year's document..."):
            pdf_reader = PdfReader(file2)
            TEXT_2 = ""
            for page in pdf_reader.pages:
                TEXT_2 += page.extract_text() or ""

            if not TEXT_2.strip():
                st.warning("No text found in the second uploaded PDF.")
            else:
                text_splitter = RecursiveCharacterTextSplitter(
                    separators="\n",
                    chunk_size=1000,
                    chunk_overlap=100,
                    length_function=len
                )
                chunks = text_splitter.split_text(TEXT_2)
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.vector_store_2 = FAISS.from_texts(chunks, embeddings)
                st.success("Current year's document processed successfully!")
    except Exception as e:
        logger.error(f"An error occurred while processing the current year's document: {e}")
        st.error(str(e))

# Compare documents
if st.session_state.vector_store_1 is not None and st.session_state.vector_store_2 is not None:
    st.subheader("Document Comparison Results")
    
    # Input criteria for comparison
    comparison_criteria = st.text_input("Enter the criteria for comparison (e.g., resource constraint, budget, infrastructure):")
    
    if comparison_criteria:
        with st.spinner("Comparing documents..."):
            try:
                # Retrieve relevant information from both documents
                retriever_1 = st.session_state.vector_store_1.similarity_search(comparison_criteria)
                retriever_2 = st.session_state.vector_store_2.similarity_search(comparison_criteria)
                
                if not retriever_1 and not retriever_2:
                    st.warning("No relevant information found in either document.")
                else:
                    # Prepare LLM for comparison
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
                    
                    response_1 = chain.invoke({"input_documents": retriever_1, "question": comparison_criteria})
                    answer_1 = response_1['output_text'].split("Helpful Answer:")[-1].strip()

                    response_2 = chain.invoke({"input_documents": retriever_2, "question": comparison_criteria})
                    answer_2 = response_2['output_text'].split("Helpful Answer:")[-1].strip()

                    st.markdown("### Last Year Document:")
                    st.write(answer_1)

                    st.markdown("### Current Year Document:")
                    st.write(answer_2)

                    # Analyze and suggest bottlenecks
                    if answer_1 and answer_2:
                        # Simple comparison logic (could be more sophisticated)
                        if answer_1 == answer_2:
                            st.success("No significant issues found between the two documents.")
                        else:
                            st.warning("Differences detected between last year's and current year's documents, indicating potential bottlenecks.")
            except Exception as e:
                logger.error(f"An error occurred while comparing the documents: {e}")
                st.error(str(e))
else:
    st.info("Please upload both PDF documents to compare their statuses.")
