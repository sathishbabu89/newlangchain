import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import os

# Load the embedding model and code generation model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/codebert-base')

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

# Convert C++ code to Java Spring Boot microservice using the code generation model
def convert_cpp_to_java_microservice(cpp_code, codegen_model):
    prompt = f"Convert this C++ code to a Java Spring Boot microservice:\n{cpp_code}"
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
                embedding_model = load_embedding_model()
                codegen_model = load_codegen_model()

                # Convert C++ to Java Spring Boot microservice code
                java_code = convert_cpp_to_java_microservice(cpp_code, codegen_model)

                # Display the generated Java code
                st.subheader("Generated Java Spring Boot Microservice Code:")
                st.code(java_code, language='java')

                # Option to download the generated Java code as a file
                st.download_button("Download Java Code", java_code, file_name="ConvertedMicroservice.java")

            except Exception as e:
                st.error(f"Error during conversion: {e}")

if __name__ == "__main__":
    main()
