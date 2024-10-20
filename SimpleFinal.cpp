import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import re
import time

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

def convert_cpp_to_java(cpp_code):
    """Convert C++ code to Java code using the Incoder model."""
    prompt = (
        "You are a programming assistant. "
        "Convert the following C++ code to Java code completely:\n"
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

    if "Java code:" in java_code:
        java_code = java_code.split("Java code:")[-1].strip()
    
    cleaned_java_code = re.sub(r'<\/?code.*|<\|.*|\bThanks for your answer\b.*', '', java_code, flags=re.DOTALL).strip()
    
    return cleaned_java_code

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="C++ to Java Converter", layout="wide")
    st.title("C++ to Java Converter")

    # Add an image or logo
    st.image("your_logo.png", width=100)  # Replace with your logo file path

    st.sidebar.header("Options")
    st.sidebar.info("Upload your C++ code file to convert it to Java.")

    uploaded_file = st.file_uploader("Choose a C++ file", type=["cpp", "h", "hpp"])
    
    if uploaded_file is not None:
        cpp_code = read_cpp_file(uploaded_file)
        st.subheader("Uploaded C++ Code:")
        st.code(cpp_code, language='cpp')

        # Convert C++ to Java
        if st.button("Convert C++ to Java"):
            with st.spinner("Converting..."):
                progress_bar = st.progress(0)  # Initialize a progress bar
                time.sleep(1)  # Simulate waiting time; you can adjust this
                progress_bar.progress(50)  # Update progress

                try:
                    # Perform the actual conversion
                    java_code = convert_cpp_to_java(cpp_code)
                    progress_bar.progress(100)  # Complete the progress

                    st.success("Conversion completed!")
                    st.subheader("Generated Java Code:")
                    st.code(java_code, language='java')

                    # Download button for the generated Java code
                    st.download_button(
                        label="Download Java Code",
                        data=java_code,
                        file_name="converted_code.java",
                        mime="text/java"
                    )

                    # Use CodeBERT for semantic understanding (optional)
                    java_embeddings = codebert_model.encode(java_code)
                    st.write(f"Java Code Embeddings: {java_embeddings[:5]}...")

                except Exception as e:
                    st.error(f"Error during conversion: {e}")

    # Expander for additional information
    with st.expander("Help & Tips", expanded=False):
        st.write("""
        - **Step 1**: Upload your C++ file.
        - **Step 2**: Click on the 'Convert C++ to Java' button.
        - **Step 3**: Download the converted Java code using the provided button.
        """)

if __name__ == "__main__":
    main()
