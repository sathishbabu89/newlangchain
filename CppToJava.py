import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import re

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

def extract_functions(cpp_code):
    """Extract functions from the C++ code."""
    func_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(.*?\)\s*{.*?}'
    functions = re.findall(func_pattern, cpp_code, re.DOTALL)
    return functions

def convert_cpp_chunk(chunk):
    """Convert a chunk of C++ code to Java code."""
    prompt = (
        "You are a programming assistant. "
        "Convert the following C++ function to Java code:\n"
        f"{chunk}\n"
        "Java code:"
    )
    inputs = incoder_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    output_sequences = incoder_model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'], 
        max_new_tokens=500
    )
    
    java_code = incoder_tokenizer.decode(output_sequences[0], skip_special_tokens=True).strip()
    return java_code.split("Java code:")[-1].strip()

def convert_cpp_to_java(cpp_code):
    """Convert C++ code to Java code by chunking."""
    functions = extract_functions(cpp_code)
    java_code = ""
    
    if not functions:
        st.warning("No valid functions found in the C++ code.")
        return java_code
    
    for function in functions:
        st.write(f"Converting function: {function}")  # Log the function being converted
        java_code += convert_cpp_chunk(function) + "\n\n"  # Convert each function and append
    
    return java_code.strip()

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
                if java_code:  # Only display if conversion was successful
                    st.subheader("Generated Java Code:")
                    st.code(java_code, language='java')

                    # Use CodeBERT for semantic understanding (optional)
                    java_embeddings = codebert_model.encode(java_code)
                    st.write(f"Java Code Embeddings: {java_embeddings[:5]}...")  # Display first few embedding values

            except Exception as e:
                st.error(f"Error during conversion: {e}")

if __name__ == "__main__":
    main()
