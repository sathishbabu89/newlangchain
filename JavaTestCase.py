import os
import logging
import streamlit as st
import re
from langchain.llms import HuggingFaceEndpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Java Test Class Generator with LLM", page_icon="💻")
st.header("Java Test Class Generator with LLM and Mockito 💻")

HUGGINGFACE_API_TOKEN = ""  # Add your Hugging Face API token

if 'generated_tests' not in st.session_state:
    st.session_state.generated_tests = {}

with st.sidebar:
    st.title("Upload Your Java Class Directory")
    directory = st.file_uploader("Upload a ZIP file containing Java classes", type="zip")

    if directory is not None:
        try:
            import zipfile
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(directory, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)

                java_files = []
                for root, _, files in os.walk(tmpdir):
                    for file in files:
                        if file.endswith(".java"):
                            java_files.append(os.path.join(root, file))

                if java_files:
                    st.subheader("Java Classes Found:")
                    for java_file in java_files:
                        st.text(java_file)
                else:
                    st.warning("No Java files found in the uploaded directory.")

        except Exception as e:
            logger.error(f"An error occurred while reading the ZIP file: {e}", exc_info=True)
            st.warning("Unable to process the ZIP file.")

if directory is not None and java_files:
    if not st.session_state.generated_tests:
        try:
            with st.spinner("Generating test classes..."):

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

                # Iterate through each Java file and generate test classes
                for java_file in java_files:
                    with open(java_file, 'r', encoding='utf-8') as f:
                        java_content = f.read()

                    # Generate the test class using LLM
                    test_class_code = generate_test_class_with_llm(java_content)
                    class_name_pattern = r'class\s+(\w+)'
                    class_name_match = re.search(class_name_pattern, java_content)

                    if class_name_match:
                        class_name = class_name_match.group(1)
                        st.session_state.generated_tests[class_name] = test_class_code

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
            logger.error(f"An error occurred while generating the test classes: {e}", exc_info=True)
            st.error(str(e))

else:
    st.info("Please upload a ZIP file containing Java classes to start generating test classes.")
