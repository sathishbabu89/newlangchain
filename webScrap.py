import logging
import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Corrected import
from langchain.huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUGGINGFACE_API_TOKEN = ""  # Add your HuggingFace API token here

st.set_page_config(page_title="Website Content Assistant Tool", page_icon="🌐")
st.header("Website Content Assistant Tool 🌐")

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Initialize WebDriver
def initialize_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run in headless mode (no browser UI)
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')  # Disable GPU acceleration for headless mode
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

with st.sidebar:
    st.title("Enter Website URL")
    url = st.text_input("Enter the URL of the website", "")

    if url:
        try:
            # Initialize the WebDriver
            driver = initialize_driver()

            # Open the URL in the WebDriver
            driver.get(url)

            # Wait for the page to load (you can adjust the time based on the website)
            time.sleep(3)  # Wait 3 seconds for the page to load (you can customize this)

            # Get the page source after JavaScript has rendered the content
            page_source = driver.page_source

            # Close the driver after getting the content
            driver.quit()

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(page_source, 'html.parser')

            # Extract text from the website, removing unnecessary tags like <script>, <style>, etc.
            website_text = ' '.join([p.get_text() for p in soup.find_all('p')])

            # Display a preview of the website content (first 1500 characters)
            st.subheader("Website Content Preview")
            st.write(website_text[:1500] + "...")

        except Exception as e:
            logger.error(f"An error occurred while fetching or parsing the website: {e}")
            st.error("Error fetching or parsing the website content.")

if url and website_text:
    if st.session_state.vector_store is None:
        try:
            with st.spinner("Processing website content..."):
                # Split the text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    separators="\n",
                    chunk_size=1000,
                    chunk_overlap=100,
                    length_function=len
                )
                chunks = text_splitter.split_text(website_text)

                # Create embeddings using HuggingFace
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)

                st.success("Website content processed successfully!")
        except Exception as e:
            logger.error(f"An error occurred while processing the website: {e}")
            st.error("Error processing the website content.")

    # Display conversation history
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input for question
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
                        # Search for relevant chunks from the vector store
                        retriever = st.session_state.vector_store.similarity_search(user_question)
                        if not retriever:
                            st.warning("No relevant information found.")
                        else:
                            # Use a HuggingFace model for the QA chain
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
                            response = chain.invoke(
                                {"input_documents": retriever, "question": user_question}
                            )
                            answer = response['output_text'].split("Helpful Answer:")[-1].strip()

                            # Display the answer
                            st.session_state.conversation_history.append(
                                {"role": "assistant", "content": answer}
                            )
                            with st.chat_message("assistant"):
                                st.markdown(answer)
                    except Exception as e:
                        logger.error(f"An error occurred while processing the question: {e}")
                        st.error(str(e))
            else:
                st.warning("Please wait for the website content to finish processing before asking questions.")
else:
    st.info("Please enter a URL to start chatting.")
