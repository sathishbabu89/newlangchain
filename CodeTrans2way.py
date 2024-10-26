import logging
import streamlit as st
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceEndpoint

# Set PyTorch to use CPU
device = torch.device("cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUGGINGFACE_API_TOKEN = ""  # Add your Hugging Face API token

# Ensure the API token is provided before invoking the LLM
if not HUGGINGFACE_API_TOKEN:
    st.error("Please set the Hugging Face API token.")

st.set_page_config(page_title="C++ to Java Spring Boot Converter", page_icon="💻")

# Sidebar page selection
page = st.sidebar.selectbox("Choose Page", ["File Upload Converter", "Inline Code Converter"])

# Helper function for code conversion logic
def convert_cpp_to_java_spring_boot(cpp_code, token):
    try:
        progress_bar = st.progress(0)
        progress_stage = 0

        with st.spinner("Processing and converting code..."):
            # Step 1: Splitting text into smaller chunks
            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 1: Splitting code into manageable chunks...")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(cpp_code)

            # Step 2: Creating embeddings only if needed (commented out for performance)
            # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            # vector_store = FAISS.from_texts(chunks, embeddings)

            # Step 3: Initialize LLM
            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 2: Loading the language model...")

            llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-Nemo-Instruct-2407",
                max_new_tokens=1024,
                top_k=10,
                top_p=0.95,
                temperature=0.01,
                repetition_penalty=1.03,
                huggingfacehub_api_token=token
            )

            # Function to send separate prompts for each Spring Boot layer
            def generate_layer(layer_name, cpp_code):
                prompt = f"""
Convert the following C++ code into a separate Java Spring Boot {layer_name} class:
1. Ensure proper Java Spring Boot conventions.
2. Include relevant annotations like @RestController, @Service, @Repository, and configuration files.
3. Ensure each layer follows best practices for maintainability.
Here is the C++ code snippet to convert:
{cpp_code}
"""
                return llm.invoke(prompt)

            # Step 4: Request LLM to generate separate files for each layer
            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 3: Converting C++ to Java Spring Boot components...")

            try:
                # Generate each layer separately
                controller_code = generate_layer("controller", cpp_code)
                service_code = generate_layer("service", cpp_code)
                entity_code = generate_layer("entity", cpp_code)
                repository_code = generate_layer("repository", cpp_code)
                exception_code = generate_layer("exception", cpp_code)
                config_code = generate_layer("configuration (application.yaml and pom.xml)", cpp_code)

                # Display individual results for each file
                st.success("Conversion complete! 🎉")
                st.code(controller_code, language='java')
                st.code(service_code, language='java')
                st.code(entity_code, language='java')
                st.code(repository_code, language='java')
                st.code(exception_code, language='java')
                st.code(config_code, language='yaml')  # Config files (e.g., application.yaml, pom.xml)

                # Download each file
                st.download_button("Download Controller", controller_code, file_name="Controller.java", mime="text/x-java-source")
                st.download_button("Download Service", service_code, file_name="Service.java", mime="text/x-java-source")
                st.download_button("Download Entity", entity_code, file_name="Entity.java", mime="text/x-java-source")
                st.download_button("Download Repository", repository_code, file_name="Repository.java", mime="text/x-java-source")
                st.download_button("Download Exception", exception_code, file_name="Exception.java", mime="text/x-java-source")
                st.download_button("Download Configuration Files", config_code, file_name="application.yaml", mime="application/x-yaml")

            except Exception as e:
                logger.error(f"Error during conversion: {e}")
                st.error("Failed to convert C++ code to Java Spring Boot. Please try again.")

    except Exception as e:
        logger.error(f"Error in processing: {e}")
        st.error("Unexpected error occurred during processing.")

# Page 1: File Upload Converter
if page == "File Upload Converter":
    st.header("C++ to Java Spring Boot Code Converter 💻")

    # Sidebar for file upload and instructions
    with st.sidebar:
        with st.expander("Upload Your C++ Code", expanded=True):
            file = st.file_uploader("Upload a C++ file (.cpp)", type="cpp")
            if file:
                code_content = file.read().decode("utf-8")
                st.subheader("C++ Code Preview")
                st.code(code_content[:5000], language='cpp')

    with st.expander("Tutorials & Tips", expanded=True):
        st.write("""
        ### Tips for Best Results:
        - Format code properly before upload.
        - Break large files into smaller parts if possible.
        - Add meaningful names and comments in C++ code for clarity in conversion.
        """)

    if file and st.button("Convert C++ to Java Spring Boot"):
        convert_cpp_to_java_spring_boot(code_content, HUGGINGFACE_API_TOKEN)

# Page 2: Inline Code Converter
if page == "Inline Code Converter":
    st.header("Inline C++ to Java Spring Boot Converter 💻")

    cpp_code_input = st.text_area("Enter C++ Code to Convert", height=300)
    if cpp_code_input and st.button("Convert to Java"):
        convert_cpp_to_java_spring_boot(cpp_code_input, HUGGINGFACE_API_TOKEN)
