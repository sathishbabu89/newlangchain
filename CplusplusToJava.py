import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# Load the tokenizer and code generation model
tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-6B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/incoder-6B")

# Add a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use the end-of-sequence token as the padding token

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('microsoft/codebert-base')

# Read the uploaded C++ file content
def read_cpp_file(uploaded_file):
    return uploaded_file.read().decode("utf-8")

# Function to chunk input text
def chunk_input(input_text, chunk_size=300):
    tokens = input_text.split()
    for i in range(0, len(tokens), chunk_size):
        yield ' '.join(tokens[i:i + chunk_size])

# Step 1: Convert C++ code to plain Java code
def convert_cpp_to_plain_java(cpp_code):
    all_java_code = []
    for chunk in chunk_input(cpp_code):
        prompt = (
            "You are a programming assistant. "
            "Convert the following valid C++ code to equivalent Java code. "
            "Maintain correct syntax and avoid redundancy. "
            "Here is the C++ code:\n\n"
            f"{chunk}\n\n"
            "Please provide the corresponding Java code."
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        output_sequences = model.generate(**inputs, max_new_tokens=1000)
        java_chunk = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        all_java_code.append(java_chunk)

    return "\n".join(all_java_code)

# Step 2: Convert the plain Java code to Spring Boot microservice
def convert_java_to_spring_boot(java_code):
    all_spring_boot_code = []
    for chunk in chunk_input(java_code):
        # Check the token length
        tokenized_chunk = tokenizer(chunk, return_tensors='pt', truncation=True, max_length=512, padding="max_length")
        if tokenized_chunk['input_ids'].size(1) > 512:
            st.error("Chunk exceeds the token limit after tokenization.")
            continue

        prompt = (
            "You are a programming assistant. "
            "Refactor the following valid Java code into a Spring Boot microservice. "
            "Ensure it includes proper annotations, controllers, services, and repository layers. "
            "Here is the Java code:\n\n"
            f"{chunk}\n\n"
            "Please provide the corresponding Spring Boot microservice code."
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        output_sequences = model.generate(**inputs, max_new_tokens=1000)
        spring_boot_chunk = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
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
                # Step 1: Convert C++ to Java
                java_code = convert_cpp_to_plain_java(cpp_code)
                if not java_code.strip():
                    st.error("Generated Java code is empty or invalid.")
                    return
                st.subheader("Step 1: Generated Plain Java Code:")
                st.code(java_code, language='java')

                # Step 2: Convert Java to Spring Boot microservice
                spring_boot_code = convert_java_to_spring_boot(java_code)
                st.subheader("Step 2: Generated Java Spring Boot Microservice Code:")
                st.code(spring_boot_code, language='java')

                # Option to download the generated Spring Boot code
                st.download_button("Download Spring Boot Code", spring_boot_code, file_name="ConvertedSpringBootMicroservice.java")

            except Exception as e:
                st.error(f"Error during conversion: {e}")

if __name__ == "__main__":
    main()
