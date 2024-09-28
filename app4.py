import logging
import streamlit as st
from PyPDF2 import PdfReader
import pdfplumber
from PIL import Image
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from transformers import pipeline
from streamlit_option_menu import option_menu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUGGINGFACE_API_TOKEN = ""

# Set up the page layout and title
st.set_page_config(page_title="Document Assistant Tool", page_icon="📚")
st.header("Document Assistant Tool 📚")

# Initialize session states
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'pdf_texts' not in st.session_state:
    st.session_state.pdf_texts = []

# Sidebar and menu options for Uploading and Chatting
selected = option_menu(
    menu_title=None, options=["Upload Document", "Ask Questions"],
    icons=["upload", "chat"],
    menu_icon="cast", default_index=0, orientation="horizontal"
)

# PDF upload logic
if selected == "Upload Document":
    with st.sidebar:
        st.title("Upload Your Document")
        multiple_files = st.file_uploader("Upload PDF files to start chatting", type="pdf", accept_multiple_files=True)

        # Display PDF preview
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
        with st.spinner("Processing document..."):
            all_texts = []
            for file in multiple_files:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                all_texts.append(text)

            combined_text = "".join(all_texts)
            text_splitter = RecursiveCharacterTextSplitter(
                separators="\n", chunk_size=1000, chunk_overlap=100, length_function=len
            )
            chunks = text_splitter.split_text(combined_text)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
            st.success("Document processed successfully!")

            # Summarize the document
            summarizer = pipeline("summarization")
            doc_summary = summarizer(combined_text[:1000])  # Summarize first 1000 characters
            st.write("**Document Summary:**", doc_summary[0]['summary_text'])

    except Exception as e:
        logger.error(f"An error occurred while processing the document: {e}")
        st.error(str(e))

# Ask Questions and Display Conversation
if selected == "Ask Questions":
    if st.session_state.vector_store:
        # Display conversation history
        for message in st.session_state.conversation_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_question = st.chat_input("Type your question here")

        if user_question:
            st.session_state.conversation_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

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

                        # Add the assistant response to chat history
                        st.session_state.conversation_history.append({"role": "assistant", "content": answer})
                        with st.chat_message("assistant"):
                            st.markdown(answer)

                        # Collect feedback on the response
                        feedback = st.radio("Was this helpful?", options=["👍 Yes", "👎 No"])

                except Exception as e:
                    logger.error(f"An error occurred while processing the question: {e}")
                    st.error(str(e))
    else:
        st.warning("Please upload a PDF document and let it process before asking questions.")

# Search functionality
search_term = st.text_input("Search for a keyword in the document:")
if search_term and st.session_state.vector_store:
    search_results = [chunk for chunk in chunks if search_term.lower() in chunk.lower()]
    if search_results:
        st.write(f"Found {len(search_results)} result(s) for '{search_term}':")
        for result in search_results:
            st.write(result)
    else:
        st.write(f"No results found for '{search_term}'.")
else:
    st.info("Please upload a PDF and search across the document.")
