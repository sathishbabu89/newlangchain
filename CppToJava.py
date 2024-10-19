import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import re

# Load models
incoder_tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B")
incoder_model = AutoModelForCausalLM.from_pretrained("facebook/incoder-1B")

# Check and set the padding token
if incoder_tokenizer.pad_token is None:
    incoder_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

codebert_model = SentenceTransformer('microsoft/codebert-base')

def read_cpp_file(uploaded_file):
    """Read the uploaded C++ file and return its content as a string."""
    return uploaded_file.read().decode("utf-8")

def handle_complex_patterns(cpp_code):
    """Handle specific complex patterns in C++ code."""
    # Example: Convert CURL requests and JSON parsing
    cpp_code = re.sub(r'#include <curl/curl.h>', 'import java.net.*;\nimport java.io.*;', cpp_code)
    cpp_code = re.sub(r'#include <jsoncpp/json/json.h>', 'import com.google.gson.*;', cpp_code)
    cpp_code = re.sub(r'CURL\s*\*\s*curl;', 'HttpURLConnection connection;', cpp_code)
    cpp_code = re.sub(r'curl_easy_setopt\(.*?\);', '', cpp_code)  # Remove CURL options for simplicity
    return cpp_code

def post_process_java_code(java_code):
    """Post-process the generated Java code for improvements."""
    # Simple replacements or fixes can be added here
    return java_code

def convert_cpp_to_java(cpp_code):
    """Convert C++ code to Java code using the Incoder model."""
    
    # Preprocess complex patterns
    cpp_code = handle_complex_patterns(cpp_code)

    prompt = (
        "You are a programming assistant. "
        "Convert the following C++ code to Java code, handling CURL and JSON parsing:\n"
        f"{cpp_code}\n"
        "Java code:"
    )
    inputs = incoder_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    output_sequences = incoder_model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'], 
        max_new_tokens=500
    )
    
    java_code = incoder_tokenizer.decode(output_sequences[0], skip_special_tokens=True).strip()
    
    # Post-process the Java code
    java_code = post_process_java_code(java_code)
    
    return java_code

def main():
    """Main function to run the Streamlit app."""
    st.title("C++ to Java Converter")

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

            except Exception as e:
                st.error(f"Error during conversion: {e}")

if __name__ == "__main__":
    main()
