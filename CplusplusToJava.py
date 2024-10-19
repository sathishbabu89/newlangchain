import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import re  # Import regular expressions

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
        "Convert the following C++ code to Java code, including any API calls:\n"
        f"{cpp_code}\n"
        "Please ensure that the Java code is functional and mirrors the functionality of the C++ code:\n"
        "Java code:"
    )
    inputs = incoder_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    output_sequences = incoder_model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'], 
        max_new_tokens=500  # Keep this high for detailed output
    )
    
    # Decode the output and filter unwanted parts
    java_code = incoder_tokenizer.decode(output_sequences[0], skip_special_tokens=True).strip()

    # Clean the output by splitting and taking relevant parts
    if "Java code:" in java_code:
        java_code = java_code.split("Java code:")[-1].strip()
    
    # Further clean unwanted patterns
    cleaned_java_code = re.sub(r'<\/?code.*|<\|.*|\bThanks for your answer\b.*', '', java_code, flags=re.DOTALL).strip()
    
    return cleaned_java_code

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
