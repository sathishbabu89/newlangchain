import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Load models
incoder_tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B")
incoder_model = AutoModelForCausalLM.from_pretrained("facebook/incoder-1B")

# Check and set the padding token
if incoder_tokenizer.pad_token is None:
    incoder_tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Define a padding token

codebert_model = SentenceTransformer('microsoft/codebert-base')

def read_cpp_file(uploaded_file):
    """Read the uploaded C++ file and return its content as a string."""
    return uploaded_file.read().decode("utf-8")

def convert_cpp_to_java(cpp_code):
    """Convert C++ code to Java code using the Incoder model."""
    prompt = (
        "You are a programming assistant. "
        "Convert the following C++ code to Java code:\n"
        f"{cpp_code}\n"
        "Java code:"
    )
    inputs = incoder_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    output_sequences = incoder_model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'], 
        max_new_tokens=200  # Set the number of tokens to generate
    )
    java_code = incoder_tokenizer.decode(output_sequences[0], skip_special_tokens=True).strip()
    return java_code

def convert_java_to_spring_boot(java_code):
    """Convert Java code to a Spring Boot microservice code."""
    prompt = (
        "You are a programming assistant. "
        "Convert the following Java code into a Spring Boot microservice:\n"
        f"{java_code}\n"
        "Spring Boot code:"
    )
    inputs = incoder_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    output_sequences = incoder_model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'], 
        max_new_tokens=200  # Set the number of tokens to generate
    )
    spring_boot_code = incoder_tokenizer.decode(output_sequences[0], skip_special_tokens=True).strip()
    return spring_boot_code

def main():
    """Main function to run the Streamlit app."""
    st.title("C++ to Java Spring Boot Microservice Converter")

    uploaded_file = st.file_uploader("Choose a C++ file", type=["cpp", "h", "hpp"])
    
    if uploaded_file is not None:
        cpp_code = read_cpp_file(uploaded_file)
        st.subheader("Uploaded C++ Code:")
        st.code(cpp_code, language='cpp')

        # Convert C++ to Java
        if st.button("Convert C++ to Java"):
            try:
                java_code = convert_cpp_to_java(cpp_code)
                st.subheader("Generated Java Code:")
                st.code(java_code, language='java')

                # Use CodeBERT for semantic understanding (optional)
                java_embeddings = codebert_model.encode(java_code)
                st.write(f"Java Code Embeddings: {java_embeddings[:5]}...")  # Display first few embedding values

                # Button to convert Java code to Spring Boot microservice
                if st.button("Convert Java to Spring Boot Microservice"):
                    spring_boot_code = convert_java_to_spring_boot(java_code)
                    st.subheader("Generated Spring Boot Code:")
                    st.code(spring_boot_code, language='java')

                    st.download_button("Download Spring Boot Code", spring_boot_code, file_name="SpringBootMicroservice.java")

            except Exception as e:
                st.error(f"Error during conversion: {e}")

if __name__ == "__main__":
    main()
