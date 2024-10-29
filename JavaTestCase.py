import os
import logging
import streamlit as st
import zipfile
import tempfile
import re
from langchain.llms import HuggingFaceEndpoint

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit app configuration
st.set_page_config(page_title="Java Test Class Generator with LLM", page_icon="💻")
st.header("Java Test Class Generator with LLM and Mockito 💻")

# Your Hugging Face API token
HUGGINGFACE_API_TOKEN = ""  # Add your Hugging Face API token

if 'generated_tests' not in st.session_state:
    st.session_state.generated_tests = {}

# Function to create a test class using LLM
def generate_test_class_with_llm(java_code):
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-Nemo-Instruct-2407",  # Replace with your model ID
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
    )

    prompt = f"""
    Generate a JUnit test class using Mockito for the following Java class:
    
    {java_code}
    
    The test class should:
    - Include test cases for all public methods.
    - Identify and mock all dependencies used in the class, including services, repositories, DTOs, and any other classes.
    - Cover edge cases and typical scenarios to ensure robust testing.
    - Follow best practices for unit testing in Java with Mockito.
    """

    response = llm.invoke(prompt)
    return response

# File upload in the sidebar
with st.sidebar:
    st.title("Upload Your Java Class Directory")
    uploaded_zip = st.file_uploader("Upload a ZIP file containing Java classes", type="zip")

    if uploaded_zip is not None:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                logger.info("Creating temporary directory at %s", tmpdir)
                
                # Extract the uploaded ZIP file
                with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                    logger.info("Files extracted to temporary directory.")

                # List extracted files
                extracted_files = []
                for root, _, files in os.walk(tmpdir):
                    for file in files:
                        extracted_files.append(os.path.join(root, file))
                        logger.info(f"Extracted: {os.path.join(root, file)}")  # Log extracted file paths

                # Filter for Java files
                java_files = [file for file in extracted_files if file.endswith(".java")]

                if java_files:
                    st.subheader("Java Classes Found:")
                    for java_file in java_files:
                        st.text(java_file)
                else:
                    st.warning("No Java files found in the uploaded directory.")

                # Only process Java files if they exist
                if java_files:
                    with st.spinner("Generating test classes..."):
                        for java_file in java_files:
                            logger.info(f"Attempting to open file: {java_file}")  # Log file path
                            try:
                                with open(java_file, 'r', encoding='utf-8') as f:
                                    java_content = f.read()

                                # Generate the test class using LLM
                                test_class_code = generate_test_class_with_llm(java_content)
                                class_name_pattern = r'class\s+(\w+)'
                                class_name_match = re.search(class_name_pattern, java_content)

                                if class_name_match:
                                    class_name = class_name_match.group(1)
                                    st.session_state.generated_tests[class_name] = test_class_code

                            except FileNotFoundError:
                                logger.error(f"File not found: {java_file}")
                                st.error(f"File not found: {java_file}")
                            except Exception as e:
                                logger.error(f"An error occurred while processing the file: {e}", exc_info=True)
                                st.error(f"An error occurred while processing the file: {java_file}. Error: {e}")

                        st.success("Test classes generated successfully!")

                        # Display generated test classes
                        st.subheader("Generated Test Classes")
                        for class_name, test_class_code in st.session_state.generated_tests.items():
                            st.code(test_class_code, language='java')

                            # Add download option for the generated test class
                            st.download_button(
                                label=f"Download {class_name}Test.java",
                                data=test_class_code,
                                file_name=f"{class_name}Test.java",  # Name of the file to download
                                mime="text/java"
                            )

        except Exception as e:
            logger.error(f"An error occurred while reading the ZIP file: {e}", exc_info=True)
            st.warning("Unable to process the ZIP file.")

# If no file is uploaded
if uploaded_zip is None:
    st.info("Please upload a ZIP file containing Java classes to start generating test classes.")
