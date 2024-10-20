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

def convert_java_to_springboot(java_code):
    """Convert Java code to Spring Boot microservices structure."""
    entity = f"""@Entity
public class ExampleEntity {{
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    private String name;
    
    // Getters and Setters
}}"""

    repository = f"""import org.springframework.data.jpa.repository.JpaRepository;
    
public interface ExampleRepository extends JpaRepository<ExampleEntity, Long> {{
}}"""

    service = f"""import org.springframework.stereotype.Service;

@Service
public class ExampleService {{
    private final ExampleRepository repository;

    public ExampleService(ExampleRepository repository) {{
        this.repository = repository;
    }}

    // Service methods
}}"""

    controller = f"""import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/examples")
public class ExampleController {{
    private final ExampleService service;

    public ExampleController(ExampleService service) {{
        this.service = service;
    }}

    // Endpoint methods
}}"""

    return {
        "entity": entity,
        "repository": repository,
        "service": service,
        "controller": controller
    }

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="C++ to Java Converter", layout="wide")
    st.title("C++ to Java Converter")

    # Add an image or logo
    st.image("your_logo.png", width=100)  # Replace with your logo file path

    st.sidebar.header("Options")
    st.sidebar.info("Upload your C++ code file to convert it to Java or Spring Boot.")

    uploaded_file = st.file_uploader("Choose a C++ file", type=["cpp", "h", "hpp"])
    
    if uploaded_file is not None:
        cpp_code = read_cpp_file(uploaded_file)
        st.subheader("Uploaded C++ Code:")
        st.code(cpp_code, language='cpp')

        # Convert C++ to Java
        if st.button("Convert C++ to Java"):
            with st.spinner("Converting..."):
                progress_bar = st.progress(0)
                time.sleep(1)  # Simulate waiting time
                progress_bar.progress(50)

                try:
                    # Perform the actual conversion
                    java_code = convert_cpp_to_java(cpp_code)
                    progress_bar.progress(100)

                    st.success("Conversion to Java completed!")
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

        # Convert Java to Spring Boot Microservices
        if st.button("Convert Java to Spring Boot"):
            with st.spinner("Converting to Spring Boot..."):
                try:
                    springboot_code = convert_java_to_springboot(java_code)

                    st.success("Conversion to Spring Boot completed!")
                    st.subheader("Generated Spring Boot Code:")

                    st.code(springboot_code["entity"], language='java')
                    st.code(springboot_code["repository"], language='java')
                    st.code(springboot_code["service"], language='java')
                    st.code(springboot_code["controller"], language='java')

                    # Download buttons for Spring Boot components
                    st.download_button(
                        label="Download Entity",
                        data=springboot_code["entity"],
                        file_name="ExampleEntity.java",
                        mime="text/java"
                    )
                    st.download_button(
                        label="Download Repository",
                        data=springboot_code["repository"],
                        file_name="ExampleRepository.java",
                        mime="text/java"
                    )
                    st.download_button(
                        label="Download Service",
                        data=springboot_code["service"],
                        file_name="ExampleService.java",
                        mime="text/java"
                    )
                    st.download_button(
                        label="Download Controller",
                        data=springboot_code["controller"],
                        file_name="ExampleController.java",
                        mime="text/java"
                    )

                except Exception as e:
                    st.error(f"Error during Spring Boot conversion: {e}")

    # Expander for additional information
    with st.expander("Help & Tips", expanded=False):
        st.write("""
        - **Step 1**: Upload your C++ file.
        - **Step 2**: Click on the 'Convert C++ to Java' button.
        - **Step 3**: Download the converted Java code using the provided button.
        - **Step 4**: Click on 'Convert Java to Spring Boot' to generate microservices structure.
        """)

if __name__ == "__main__":
    main()
