import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# Load models
incoder_tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B")
incoder_model = AutoModelForCausalLM.from_pretrained("facebook/incoder-1B")

# Check and set the padding token
if incoder_tokenizer.pad_token is None:
    incoder_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def read_cpp_file(uploaded_file):
    """Read the uploaded C++ file and return its content as a string."""
    return uploaded_file.read().decode("utf-8")

def convert_curl_and_json(cpp_code):
    """Convert CURL and JSON specific constructs from C++ to Java."""
    # Replace C++ CURL code with Java's HttpURLConnection
    java_code = re.sub(
        r'CURL\s*\*\s*curl;\s*.*?curl_easy_setopt\((.*?)\);.*?curl_easy_perform\(curl\);',
        r'HttpURLConnection connection = (HttpURLConnection) new URL("\1").openConnection();\n'
        r'connection.setRequestMethod("POST");\n'
        r'connection.setDoOutput(true);',
        cpp_code, flags=re.DOTALL
    )

    # Replace JSON parsing with Gson
    java_code = re.sub(
        r'Json::Value root;\s*Json::Reader reader;\s*if\s*\(!reader.parse\((.*?)\);\)',
        r'JsonObject jsonObject = JsonParser.parseString("\1").getAsJsonObject();',
        java_code, flags=re.DOTALL
    )

    return java_code

def convert_cpp_to_java(cpp_code):
    """Convert C++ code to Java code."""
    
    # Handle specific patterns first
    java_code = convert_curl_and_json(cpp_code)

    # Generate Java code from the model for simpler constructs
    prompt = (
        "You are a programming assistant. "
        "Convert the following C++ code to Java code, handling basic constructs:\n"
        f"{java_code}\n"
        "Java code:"
    )
    
    inputs = incoder_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    output_sequences = incoder_model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'], 
        max_new_tokens=500
    )
    
    java_code = incoder_tokenizer.decode(output_sequences[0], skip_special_tokens=True).strip()

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

            except Exception as e:
                st.error(f"Error during conversion: {e}")

if __name__ == "__main__":
    main()
