import streamlit as st
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the LLM model and set padding token
llm_tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B")
if llm_tokenizer.pad_token is None:
    llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
llm_model = AutoModelForCausalLM.from_pretrained("facebook/incoder-1B")

def read_cpp_file(uploaded_file):
    """Read the uploaded C++ file and return its content as a string."""
    return uploaded_file.read().decode("utf-8")

def convert_cpp_to_java(cpp_code):
    """Convert C++ code to a high-level Java representation."""
    
    # Initialize imports and track which ones to include
    imports = set()
    
    # Handle includes and translate to Java imports
    if "#include <iostream>" in cpp_code:
        imports.add("import java.io.*;")
    if "#include <string>" in cpp_code:
        imports.add("import java.util.*;")
    if "#include <vector>" in cpp_code:
        imports.add("import java.util.*;")  # Vector equivalent in Java
    if "#include <ctime>" in cpp_code:
        imports.add("import java.util.*;")  # For time handling in Java

    # Start with the original code
    java_code = cpp_code
    
    # Remove C++ headers
    for header in ["#include <iostream>", "#include <string>", "#include <vector>", "#include <ctime>"]:
        java_code = java_code.replace(header, "")

    # Replace main function
    java_code = java_code.replace("int main() {", "public static void main(String[] args) {")
    
    # Replace std:: and specific C++ constructs
    java_code = java_code.replace("std::", "")  # Remove std::
    java_code = java_code.replace("cout", "System.out.println")  # Prepare for print statements
    java_code = java_code.replace("<<", " + ")  # Replace << with string concatenation
    java_code = java_code.replace("endl", "");  # Remove endl since it will be handled by + "\n"
    
    # Convert constructors
    java_code = re.sub(r'(\w+)\s*::(\w+)\s*\((.*?)\)', r'\2(\3) {', java_code)  # Adjust C++ constructors to Java

    # Return the basic Java code and imports
    return java_code, imports

def refine_java_code(java_code):
    """Use LLM to refine the generated Java code."""
    prompt = (
        "Please correct and improve the following Java code:\n"
        f"{java_code}\n"
        "Corrected Java code:"
    )

    inputs = llm_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

    output_sequences = llm_model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=500
    )

    refined_java_code = llm_tokenizer.decode(output_sequences[0], skip_special_tokens=True).strip()

    # Avoid duplicating the "Corrected Java code:" prompt in the output
    if "Corrected Java code:" in refined_java_code:
        refined_java_code = refined_java_code.split("Corrected Java code:")[-1].strip()

    return refined_java_code

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
                java_code, imports = convert_cpp_to_java(cpp_code)
                
                # Add imports to the beginning of the java_code
                if imports:
                    java_code = "\n".join(imports) + "\n" + java_code
                
                st.subheader("Generated Java Code:")
                st.code(java_code, language='java')
                
                # Store the Java code in session state for refinement
                st.session_state.java_code = java_code

            except Exception as e:
                st.error(f"Error during conversion: {e}")

        # Button to refine the Java code
        if "java_code" in st.session_state and st.button("Refine Java Code"):
            try:
                refined_java_code = refine_java_code(st.session_state.java_code)
                st.subheader("Refined Java Code:")
                st.code(refined_java_code, language='java')
            except Exception as e:
                st.error(f"Error during refinement: {e}")

if __name__ == "__main__":
    main()
