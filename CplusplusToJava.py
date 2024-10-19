import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import re

# Load models
incoder_tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B")
incoder_model = AutoModelForCausalLM.from_pretrained("facebook/incoder-1B")
codebert_model = SentenceTransformer('microsoft/codebert-base')

# Check and set the padding token
if incoder_tokenizer.pad_token is None:
    incoder_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

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

def convert_java_to_springboot(java_code):
    """Convert Java code to Spring Boot microservices code (placeholder function)."""
    # Placeholder for actual conversion logic
    springboot_code = f"// Spring Boot Service\n@Service\npublic class MyService {{\n    // Converted code from Java to Spring Boot\n{java_code}\n}}"
    return springboot_code

def main():
    """Main function to run the Streamlit app."""
    st.title("Code Conversion Tool")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Upload C++ Code")
        uploaded_file = st.file_uploader("Choose a C++ file", type=["cpp", "h", "hpp"])
        
        if uploaded_file is not None:
            cpp_code = read_cpp_file(uploaded_file)
            st.subheader("Uploaded C++ Code:")
            st.code(cpp_code, language='cpp')

            # Convert C++ to Java
            if st.button("Convert C++ to Java"):
                try:
                    java_code = convert_cpp_to_java(cpp_code)
                    st.session_state.java_code = java_code  # Store Java code in session state
                    st.subheader("Generated Java Code:")
                    st.code(java_code, language='java')
                except Exception as e:
                    st.error(f"Error during conversion: {e}")

    with col2:
        st.header("Convert Java Code to Spring Boot")
        if 'java_code' in st.session_state:
            java_code = st.session_state.java_code
            st.subheader("Java Code Ready for Conversion:")
            st.code(java_code, language='java')

            if st.button("Convert to Spring Boot"):
                try:
                    springboot_code = convert_java_to_springboot(java_code)
                    st.subheader("Generated Spring Boot Code:")
                    st.code(springboot_code, language='java')
                except Exception as e:
                    st.error(f"Error during conversion: {e}")
        else:
            st.warning("Please upload C++ code and convert it to Java first.")

if __name__ == "__main__":
    main()
