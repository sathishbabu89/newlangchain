import streamlit as st
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Load the tokenizer and code generation model
tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-6B")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('microsoft/codebert-base')

@st.cache_resource
def load_codegen_model():
    return pipeline("text2text-generation", model="facebook/incoder-6B")

# Read the uploaded C++ file content
def read_cpp_file(uploaded_file):
    return uploaded_file.read().decode("utf-8")

# Function to chunk input text
def chunk_input(input_text, chunk_size=300):
    tokens = input_text.split()
    for i in range(0, len(tokens), chunk_size):
        yield ' '.join(tokens[i:i + chunk_size])

# Function to check if Java code is valid
def is_valid_java_code(java_code):
    return "class" in java_code and "void" in java_code and "{" in java_code and "}" in java_code

# Step 1: Convert C++ code to plain Java code
def convert_cpp_to_plain_java(cpp_code, codegen_model):
    all_java_code = []
    for chunk in chunk_input(cpp_code):
        prompt = (
            "You are a programming assistant. "
            "Your task is to convert valid C++ code into equivalent Java code. "
            "Make sure to maintain correct syntax and semantics. "
            "Here is the C++ code:\n\n"
            f"{chunk}\n\n"
            "Please provide the corresponding Java code."
        )
        response = codegen_model(prompt)
        if not response or 'generated_text' not in response[0]:
            return "Error: Model output was not as expected."
        java_chunk = response[0]['generated_text']
        all_java_code.append(java_chunk)

    return "\n".join(all_java_code)

# Step 2: Convert the plain Java code to Spring Boot microservice
def convert_java_to_spring_boot(java_code, codegen_model):
    all_spring_boot_code = []
    for chunk in chunk_input(java_code):
        # Check the token length
        tokenized_chunk = tokenizer(chunk, return_tensors='pt', truncation=True, max_length=512, padding="max_length")
        if tokenized_chunk['input_ids'].size(1) > 512:
            st.error("Chunk exceeds the token limit after tokenization.")
            continue

        prompt = (
            "You are a programming assistant. "
            "Your task is to refactor valid Java code into a Spring Boot microservice. "
            "Make sure it includes proper annotations, controllers, services, and repository layers. "
            "Here is the Java code:\n\n"
            f"{chunk}\n\n"
            "Please provide the corresponding Spring Boot microservice code."
        )
        response = codegen_model(prompt)
        if not response or 'generated_text' not in response[0]:
            return "Error: Model output was not as expected."
        spring_boot_chunk = response[0]['generated_text']
        all_spring_boot_code.append(spring_boot_chunk)

    return "\n".join(all_spring_boot_code)

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
                if not is_valid_java_code(java_code):
                    st.error("Generated Java code is invalid.")
                    return
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
