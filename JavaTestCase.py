import logging
import streamlit as st
import re
from langchain.llms import HuggingFaceEndpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Java Test Class Generator with LLM", page_icon="💻")
st.header("Java Test Class Generator with LLM and Mockito 💻")

HUGGINGFACE_API_TOKEN = ""  # Add your Hugging Face API token

if 'test_class' not in st.session_state:
    st.session_state.test_class = None

with st.sidebar:
    st.title("Upload Your Java Class")
    file = st.file_uploader("Upload a Java file (.java) to generate a test class", type="java")

    if file is not None:
        try:
            java_content = file.read().decode("utf-8")
            st.subheader("Java Class Preview")
            st.text(java_content[:5000])  # Preview the first 5000 characters of code
        except Exception as e:
            logger.error(f"An error occurred while reading the Java file: {e}", exc_info=True)
            st.warning("Unable to display Java class preview.")

if file is not None:
    if st.session_state.test_class is None:
        try:
            with st.spinner("Generating test class..."):

                # Function to create a test class using LLM
                def generate_test_class_with_llm(java_code):
                    # Load the LLM for generating test class code
                    llm = HuggingFaceEndpoint(
                        repo_id="mistralai/Mistral-Nemo-Instruct-2407",  # Replace with your model ID
                        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
                    )

                    prompt = f"""
                    Generate a JUnit test class using Mockito for the following Java class:
                    
                    {java_code}
                    
                    The test class should include test cases for all public methods and necessary mocks.
                    """

                    response = llm.invoke(prompt)
                    return response

                # Generate the test class using LLM
                test_class_code = generate_test_class_with_llm(java_content)

                st.session_state.test_class = test_class_code

                st.success("Test class generated successfully!")

                st.subheader("Generated Test Class")
                st.code(test_class_code, language='java')

                # Add download option for the generated test class
                class_name_pattern = r'class\s+(\w+)'
                class_name_match = re.search(class_name_pattern, java_content)
                if class_name_match:
                    class_name = class_name_match.group(1)
                    st.download_button(
                        label="Download Test Class",
                        data=test_class_code,
                        file_name=f"{class_name}Test.java",  # Name of the file to download
                        mime="text/java"
                    )

        except Exception as e:
            logger.error(f"An error occurred while generating the test class: {e}", exc_info=True)
            st.error(str(e))

else:
    st.info("Please upload a Java class file to start generating a test class.")
