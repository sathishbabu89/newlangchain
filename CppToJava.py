import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import re

# Load models
try:
    incoder_tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B")
    incoder_model = AutoModelForCausalLM.from_pretrained("facebook/incoder-1B")
    codebert_model = SentenceTransformer('microsoft/codebert-base')
except Exception as e:
    st.error(f"Error loading models: {e}")

# Check and set the padding token
if incoder_tokenizer.pad_token is None:
    incoder_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def read_cpp_file(uploaded_file):
    return uploaded_file.read().decode("utf-8")

def convert_cpp_to_java(cpp_code):
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
    st.title("C++ to Java Converter")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Choose a C++ file", type=["cpp", "h", "hpp"])
        
        if uploaded_file is not None:
            cpp_code = read_cpp_file(uploaded_file)
            st.subheader("Uploaded C++ Code:")
            st.code(cpp_code, language='cpp')
            
            if st.button("Next"):
                st.session_state.cpp_code = cpp_code
                st.session_state.conversion_done = False

    with col2:
        if 'cpp_code' in st.session_state:
            st.subheader("Convert C++ to Java")
            
            if not st.session_state.get('conversion_done', False):
                if st.button("Convert C++ to Java"):
                    try:
                        java_code = convert_cpp_to_java(st.session_state.cpp_code)
                        st.session_state.java_code = java_code
                        st.session_state.conversion_done = True
                        st.subheader("Generated Java Code:")
                        st.code(java_code, language='java')
                        st.session_state.accepted = False

                    except Exception as e:
                        st.error(f"Error during conversion: {e}")
            else:
                java_code = st.session_state.java_code
                st.subheader("Generated Java Code:")
                st.code(java_code, language='java')

                accept_button = st.button("Accept")
                reject_button = st.button("Reject")
                
                if accept_button:
                    st.success("Conversion accepted!")
                    st.session_state.clear()  # Reset state for a new conversion

                if reject_button:
                    issue = st.text_input("What is the issue with the conversion?")
                    if issue:
                        st.session_state.cpp_code = issue  # Use issue as new C++ input for conversion
                        st.session_state.conversion_done = False  # Reset conversion state
                        st.write("Ready to retry conversion...")

if __name__ == "__main__":
    main()
