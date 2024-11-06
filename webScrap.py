import logging
import streamlit as st
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceEndpoint

# Force PyTorch to use CPU
import torch
device = torch.device("cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUGGINGFACE_API_TOKEN = ""  # Add your Hugging Face API token

st.set_page_config(page_title="Website Content Summarization Tool", page_icon="🌐")
st.header("Website Content Summarization Tool 🌐")

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

with st.sidebar:
    st.title("Enter Website URL")
    url = st.text_input("Enter the website URL to start extracting content")

    if url:
        try:
            # Initialize Chrome WebDriver with Selenium to scrape content
            chrome_options = Options()
            chrome_options.add_argument("--headless")  # Run headless mode (no browser window)
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            
            # Open the URL
            driver.get(url)
            time.sleep(5)  # Wait for JavaScript to render content (adjust timing as necessary)
            
            # Get all the paragraphs from the page (or any other tags you want)
            paragraphs = driver.find_elements(By.TAG_NAME, 'p')
            text_content = " ".join([para.text for para in paragraphs])

            driver.quit()  # Close the browser

            st.subheader("Website Content Preview")
            st.text(text_content[:2000])  # Preview the first 2000 characters of content

        except Exception as e:
            logger.error(f"An error occurred while fetching the website content: {e}", exc_info=True)
            st.warning("Unable to extract content from the provided URL.")

if url:
    if st.session_state.vector_store is None:
        try:
            with st.spinner("Processing and summarizing website content..."):

                # Function to clean and split the content
                def extract_summary(content):
                    # You can add more complex logic to analyze the content, such as identifying key sections
                    summary = []
                    # Example: A simple summary of the length of content
                    summary.append(f"Content Length: {len(content)} characters")

                    return summary

                # Get basic content summary
                content_summary = extract_summary(text_content)

                if not content_summary:
                    st.warning("No useful content extracted.")
                else:
                    # Use LLM to generate detailed content summary
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50
                    )
                    chunks = text_splitter.split_text(text_content)

                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2", device=device
                    )

                    st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)

                    # Load the LLM for content summarization
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

                    # Combine structural and LLM-based logic analysis
                    st.success("Website content processed successfully!")

                    st.subheader("Website Content Summary")
                    for section in content_summary:
                        st.markdown(f"- {section}")

                    st.subheader("Business Logic or Insights Summary")
                    with st.spinner("Generating business logic summary..."):
                        try:
                            # Concatenate the content with an instruction for summarization
                            prompt = f"Summarize the key insights and business logic from this website content:\n\n{text_content}"
                            response = llm.invoke(prompt)

                            # Assuming the response is a string, display it directly
                            st.markdown(response)

                        except Exception as e:
                            logger.error(f"An error occurred while summarizing the website content: {e}", exc_info=True)
                            st.error("Unable to generate summary.")

        except Exception as e:
            logger.error(f"An error occurred while processing the website content: {e}", exc_info=True)
            st.error(str(e))

else:
    st.info("Please enter a website URL to start extracting content.")
