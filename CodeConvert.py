import logging
import streamlit as st
import re
import torch  # Add the import for PyTorch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceEndpoint

# Force PyTorch to use CPU
device = torch.device("cpu")  # Ensure torch is imported

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUGGINGFACE_API_TOKEN = ""  # Add your Hugging Face API token

st.set_page_config(page_title="C++ to Java Microservices Converter", page_icon="💻")
st.header("C++ to Java Spring Boot Microservices Converter 💻")

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

with st.sidebar:
    st.title("Upload Your C++ Code")
    file = st.file_uploader("Upload a C++ file (.cpp) to start analyzing", type="cpp")

    if file is not None:
        try:
            code_content = file.read().decode("utf-8")
            st.subheader("Code Preview")
            st.text(code_content[:5000])  # Preview the first 5000 characters of code
        except Exception as e:
            logger.error(f"An error occurred while reading the code file: {e}", exc_info=True)
            st.warning("Unable to display code preview.")

if file is not None:
    if st.session_state.vector_store is None:
        try:
            with st.spinner("Processing and summarizing code..."):

                # Function to extract code summaries using regex (structure)
                def extract_summary(code):
                    # Using regex to identify function definitions, classes, etc.
                    function_pattern = r'\w+\s+\w+\s*[^)]*\s*{'
                    functions = re.findall(function_pattern, code)

                    # Example: Identify class declarations
                    class_pattern = r'class\s+\w+\s*{'
                    classes = re.findall(class_pattern, code)

                    summary = []
                    if functions:
                        summary.append("Functions identified in the code:")
                        summary.extend(functions)
                    if classes:
                        summary.append("\nClasses identified in the code:")
                        summary.extend(classes)

                    return summary

                # Get basic structure summary
                code_summary = extract_summary(code_content)

                if not code_summary:
                    st.warning("No functions, classes, or critical logic found in the code.")
                else:
                    # Use LLM to generate detailed business logic summary
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50
                    )
                    chunks = text_splitter.split_text(code_content)

                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2", device=device
                    )

                    st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)

                    # Load the LLM for business logic summarization
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
                    st.success("Code processed successfully!")

                    st.subheader("Code Structure Summary")
                    for section in code_summary:
                        st.markdown(f"- {section}")

                    st.subheader("Business Logic Summary")
                    with st.spinner("Generating business logic summary..."):
                        try:
                            # Concatenate the code with an instruction for summarization
                            prompt = f"Summarize the business logic and key functionality of this C++ code:\n\n{code_content}"
                            response = llm.invoke(prompt)

                            # Assuming the response is a string, display it directly
                            st.markdown(response)

                            # Now proceed to the Java Spring Boot conversion
                            st.subheader("Java Spring Boot Microservice Code")
                            with st.spinner("Converting to Java Spring Boot..."):

                                # Call a function to perform the conversion
                                java_code = convert_cpp_to_java_springboot(code_content)

                                # Display the converted Java code
                                st.code(java_code, language='java')

                        except Exception as e:
                            logger.error(f"An error occurred while summarizing the code: {e}", exc_info=True)
                            st.error("Unable to generate business logic summary.")
        except Exception as e:
            logger.error(f"An error occurred while processing the code: {e}", exc_info=True)
            st.error(str(e))

else:
    st.info("Please upload a C++ code file to start analyzing.")


# Function to convert C++ code to Java Spring Boot microservice
def convert_cpp_to_java_springboot(cpp_code):
    """
    Convert C++ code into a basic Java Spring Boot microservice structure.
    This function focuses on mapping C++ classes and functions to Java classes and REST controllers.
    """
    # Regex patterns for identifying C++ classes and functions
    class_pattern = r'class\s+(\w+)\s*{'
    function_pattern = r'(\w+)\s+(\w+)\s*([^)]*)\s*{'

    # Start building the Java Spring Boot code
    java_code = """
package com.example.microservice;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.*;

@SpringBootApplication
public class Application {
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}
"""

    # Find all classes in the C++ code
    classes = re.findall(class_pattern, cpp_code)

    # Convert each C++ class into a Java class
    for class_name in classes:
        java_code += f"""
@RestController
@RequestMapping("/api/{class_name.lower()}")
public class {class_name}Controller {{
"""

        # Find functions in each class and map them to Java methods
        functions = re.findall(function_pattern, cpp_code)

        for return_type, func_name, params in functions:
            java_code += f"""
    @GetMapping("/{func_name}")
    public {map_type_cpp_to_java(return_type)} {func_name}() {{
        // TODO: Implement the logic for {func_name}
        return null;
    }}
"""
        java_code += "}\n"

    return java_code

# Helper function to map C++ types to Java types
def map_type_cpp_to_java(cpp_type):
    """
    Map basic C++ types to equivalent Java types.
    """
    cpp_to_java = {
        'int': 'int',
        'float': 'float',
        'double': 'double',
        'char': 'String',
        'void': 'void',
        # Add more type mappings as needed
    }
    return cpp_to_java.get(cpp_type, 'Object')  # Default to Object if type is not found
