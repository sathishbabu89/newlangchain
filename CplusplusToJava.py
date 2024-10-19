import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the CodeGen2 model
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2-1B-P")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-2-1B-P")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def read_cpp_file(uploaded_file):
    return uploaded_file.read().decode("utf-8")

def dynamic_chunk_input(input_text):
    # Split based on functions or classes (this is a simple heuristic)
    return input_text.split('\n\n')  # Splits by double newlines

def convert_cpp_to_java(cpp_code):
    all_java_code = []
    chunks = dynamic_chunk_input(cpp_code)
    
    for chunk in chunks:
        prompt = (
            "You are a programming assistant specialized in converting code. "
            "Convert the following valid C++ code to equivalent Java code. "
            "Please maintain correct syntax and provide clean and efficient Java code.\n\n"
            f"C++ code:\n{chunk}\n\nJava code:"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        output_sequences = model.generate(**inputs, max_new_tokens=500)
        java_chunk = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        all_java_code.append(java_chunk)

    return "\n\n".join(all_java_code)

def convert_java_to_spring_boot(java_code):
    all_spring_boot_code = []
    chunks = dynamic_chunk_input(java_code)

    for chunk in chunks:
        prompt = (
            "You are a programming assistant specialized in Spring Boot. "
            "Refactor the following valid Java code into a Spring Boot microservice. "
            "Ensure proper annotations, controllers, services, and repository layers are included.\n\n"
            f"Java code:\n{chunk}\n\nSpring Boot code:"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        output_sequences = model.generate(**inputs, max_new_tokens=500)
        spring_boot_chunk = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        all_spring_boot_code.append(spring_boot_chunk)

    return "\n\n".join(all_spring_boot_code)

def main():
    st.title("C++ to Java Spring Boot Microservice Converter")

    uploaded_file = st.file_uploader("Choose a C++ file", type=["cpp", "h", "hpp"])
    
    if uploaded_file is not None:
        cpp_code = read_cpp_file(uploaded_file)
        st.subheader("Uploaded C++ Code:")
        st.code(cpp_code, language='cpp')

        if st.button("Convert to Java Microservice"):
            try:
                java_code = convert_cpp_to_java(cpp_code)
                if not java_code.strip():
                    st.error("Generated Java code is empty or invalid.")
                    return
                st.subheader("Generated Java Code:")
                st.code(java_code, language='java')

                spring_boot_code = convert_java_to_spring_boot(java_code)
                st.subheader("Generated Java Spring Boot Microservice Code:")
                st.code(spring_boot_code, language='java')

                # Feedback Loop
                feedback = st.text_area("Provide feedback or adjustments for the generated Spring Boot code:")
                if st.button("Submit Feedback"):
                    st.success("Thank you for your feedback!")
                    # Here, you could log the feedback or take further action

                st.download_button("Download Spring Boot Code", spring_boot_code, file_name="ConvertedSpringBootMicroservice.java")

            except Exception as e:
                st.error(f"Error during conversion: {e}")

if __name__ == "__main__":
    main()
