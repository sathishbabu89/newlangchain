import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2-1B-P")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-2-1B-P")

def read_cpp_file(uploaded_file):
    return uploaded_file.read().decode("utf-8")

def convert_cpp_to_java(cpp_code):
    prompt = (
        "Convert the following C++ code to Java:\n"
        f"{cpp_code}\n"
        "Java code:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    # Limit the output to prevent excessive responses
    output_sequences = model.generate(**inputs, max_new_tokens=300)
    
    java_code = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return java_code.strip()  # Clean up the response

def main():
    st.title("C++ to Java Converter")

    uploaded_file = st.file_uploader("Choose a C++ file", type=["cpp", "h", "hpp"])
    
    if uploaded_file is not None:
        cpp_code = read_cpp_file(uploaded_file)
        st.subheader("Uploaded C++ Code:")
        st.code(cpp_code, language='cpp')

        if st.button("Convert to Java"):
            try:
                java_code = convert_cpp_to_java(cpp_code)
                if not java_code or "Java code:" in java_code:
                    st.error("Generated Java code is empty or invalid.")
                    return
                st.subheader("Generated Java Code:")
                st.code(java_code, language='java')

            except Exception as e:
                st.error(f"Error during conversion: {e}")

if __name__ == "__main__":
    main()
