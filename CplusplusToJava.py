import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import os

# Load the embedding model and code generation model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('microsoft/codebert-base')

@st.cache_resource
def load_codegen_model():
    return pipeline("text2text-generation", model="Salesforce/codet5-base")

# Initialize FAISS Index (if needed for future retrieval-augmented generation)
@st.cache_resource
def initialize_faiss_index(embedding_model):
    return FAISS(embedding_model)

# Read the uploaded C++ file content
def read_cpp_file(uploaded_file):
    return uploaded_file.read().decode("utf-8")

# Step 1: Convert C++ code to plain Java code
def convert_cpp_to_plain_java(cpp_code, codegen_model):
    prompt = f"Convert the following C++ code into equivalent Java code:\n\n{cpp_code}"
    response = codegen_model(prompt)
    return response[0]['generated_text']

# Step 2: Convert the plain Java code to Spring Boot microservice
def convert_java_to_spring_boot(java_code, codegen_model):
    prompt = (
        f"Refactor the following Java code into a Spring Boot microservice. "
        f"Include REST controllers, services, and repository layers:\n\n{java_code}"
    )
    response = codegen_model(prompt)
    return response[0]['generated_text']

# Streamlit App
def main():
    st.title("C++ to Java Spring Boot Microservice Converter")

    # File Upload Section
    st.subheader("Upload your C++ code file:")
    uploaded_file = st.file_uploader("Choose a C++ file", type=["cpp", "h", "hpp"])

    if uploaded_file is not None:
        # Read the uploaded C++ file
        cpp_code = read_cpp_file(uploaded_file)
        
        # Display the uploaded C++ code
        st.subheader("Uploaded C++ Code:")
        st.code(cpp_code, language='cpp')

        # Button to trigger the conversion
        if st.button("Convert to Java Microservice"):
            try:
                codegen_model = load_codegen_model()

                # Step 1: Convert C++ to Java
                java_code = convert_cpp_to_plain_java(cpp_code, codegen_model)
                st.subheader("Step 1: Generated Plain Java Code:")
                st.code(java_code, language='java')

                # Step 2: Convert Java to Spring Boot microservice
                spring_boot_code = convert_java_to_spring_boot(java_code, codegen_model)
                st.subheader("Step 2: Generated Java Spring Boot Microservice Code:")
                st.code(spring_boot_code, language='java')

                # Option to download the generated Spring Boot code
                st.download_button("Download Spring Boot Code", spring_boot_code, file_name="ConvertedSpringBootMicroservice.java")

            except Exception as e:
                st.error(f"Error during conversion: {e}")

if __name__ == "__main__":
    main()
