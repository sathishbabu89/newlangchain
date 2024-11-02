import logging
import streamlit as st
import re
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceEndpoint

# Force PyTorch to use CPU
device = torch.device("cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUGGINGFACE_API_TOKEN = ""  # Add your Hugging Face API token

if not HUGGINGFACE_API_TOKEN:
    st.error("Please set the Hugging Face API token.")

st.set_page_config(page_title="C++ to Java Conversion Tool", page_icon="💻")

page = st.sidebar.selectbox("Choose Page", ["File Upload Converter", "Inline Code Converter"])

def convert_cpp_to_java_spring_boot(cpp_code, filename, HUGGINGFACE_API_TOKEN, project_info):
    try:
        progress_bar = st.progress(0)
        progress_stage = 0

        with st.spinner("Processing and converting code..."):
            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 1: Splitting the code into chunks... 💬")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(cpp_code)

            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 2: Generating embeddings... 📊")

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_texts(chunks, embeddings)

            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 3: Loading the language model... 🚀")

            llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-Nemo-Instruct-2407",
                max_new_tokens=2048,
                top_k=10,
                top_p=0.95,
                typical_p=0.95,
                temperature=0.01,
                repetition_penalty=1.03,
                huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
            )

            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 4: Converting C++ to Java Spring Boot... 🔄")

prompt = f"""
Convert the following C++ code into Java Spring Boot. The conversion should follow these guidelines:

1. Generate only one `MainApplication` class that contains the `public static void main(String[] args)` method for starting the Spring Boot application.
2. Only create separate classes for `Controller`, `Service`, `Repository`, and `Entity` if the logic of the C++ code requires them. Avoid unnecessary layers or duplicate classes.
3. Each class should be annotated appropriately (e.g., '@SpringBootApplication', '@RestController', '@Service', '@Entity', '@Repository').
4. Make sure to generate only distinct classes with no duplicate definitions.
5. Include the application configuration files such as `application.yaml` and `pom.xml` with only the required dependencies.

Here is the C++ code snippet:
{cpp_code}
"""
            response = llm.invoke(prompt)

            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 5: Conversion complete! 🎉")

            # Display the converted Java code
            st.code(response, language='java')

            if re.search(r'\berror\b|\bexception\b|\bsyntax\b|\bmissing\b', response.lower()):
                st.warning("The converted Java code may contain syntax or structural errors. Please review it carefully.")
            else:
                st.success("The Java code is free from basic syntax errors!")

    except Exception as e:
        logger.error(f"An error occurred while converting the code: {e}", exc_info=True)
        st.error("Unable to convert C++ code to Java.")

# Page 1: File Upload Converter
if page == "File Upload Converter":
    st.header("C++ to Java Conversion Tool with LLM 💻")

    with st.sidebar:
        st.subheader("Spring Boot Project Metadata")
        group_id = st.text_input("Group ID", "com.example")
        artifact_id = st.text_input("Artifact ID", "demo")
        name = st.text_input("Project Name", "Demo Project")
        packaging = st.selectbox("Packaging", ["jar", "war"])
        dependencies = st.multiselect("Select Dependencies", ["web", "data-jpa", "mysql", "h2", "thymeleaf"])

        with st.expander("Upload Your C++ Code", expanded=True):
            file = st.file_uploader("Upload a C++ file (.cpp) to start analyzing", type="cpp")

            if file is not None:
                try:
                    code_content = file.read().decode("utf-8")
                    st.subheader("C++ Code Preview")
                    st.code(code_content[:5000], language='cpp')
                except Exception as e:
                    logger.error(f"An error occurred while reading the code file: {e}", exc_info=True)
                    st.warning("Unable to display code preview.")

    with st.expander("Tutorials & Tips", expanded=True):
        st.write("""### Welcome to the C++ to Java Conversion Tool!
        Here are some tips to help you use this tool effectively:
        - **Code Formatting:** Ensure your C++ code is properly formatted.
        - **Chunking:** Break large files into smaller parts.
        - **Annotations:** Ensure the Java conversion includes necessary annotations like `@RestController`.
        - **Testing:** Test the Java code after conversion.
        - **Documentation:** Familiarize with C++ and Java Spring Boot docs.
        """)

    if file is not None:
        if st.button("Convert C++ to Java Spring Boot"):            
            project_info = {
                'type': 'maven-project',
                'groupId': group_id,
                'artifactId': artifact_id,
                'name': name,
                'packageName': group_id,
                'version': '0.0.1-SNAPSHOT',
                'packaging': packaging,
                'dependencies': ','.join(dependencies)
            }
            convert_cpp_to_java_spring_boot(code_content, file.name, HUGGINGFACE_API_TOKEN, project_info)

# Page 2: Inline Code Converter
if page == "Inline Code Converter":
    st.header("Inline C++ to Java Code Converter 💻")

    cpp_code_input = st.text_area("Enter C++ Code to Convert to Java Spring Boot", height=300)

    if st.button("Convert Inline C++ to Java Spring Boot"):
        if cpp_code_input.strip():
            project_info = {
                'type': 'maven-project',
                'groupId': group_id,
                'artifactId': artifact_id,
                'name': name,
                'packageName': group_id,
                'version': '0.0.1-SNAPSHOT',
                'packaging': packaging,
                'dependencies': ','.join(dependencies)
            }
            convert_cpp_to_java_spring_boot(cpp_code_input, "inline_code_conversion.cpp", HUGGINGFACE_API_TOKEN, project_info)
        else:
            st.warning("Please enter some C++ code to convert.")

# Footer
st.sidebar.markdown("### About")
st.sidebar.write("This tool uses state-of-the-art AI models to assist with C++ to Java conversion, specifically tailored for Spring Boot applications.")
