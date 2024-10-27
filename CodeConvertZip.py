import logging
import streamlit as st
import re
import torch
import zipfile
import io
from concurrent.futures import ThreadPoolExecutor
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

def convert_cpp_to_java_spring_boot(cpp_code, filename, HUGGINGFACE_API_TOKEN):
    progress_messages = [
        "Starting conversion...",
        "Splitting the code into chunks...",
        "Generating embeddings...",
        "Loading the language model...",
        "Converting C++ to Java Spring Boot..."
    ]
    
    try:
        for i, message in enumerate(progress_messages):
            st.session_state.progress_bar.progress((i + 1) * 20)
            st.session_state.progress_message = message
            st.experimental_rerun()  # Update the UI

            if i == 0:  # Step 1
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = text_splitter.split_text(cpp_code)

            elif i == 1:  # Step 2
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vector_store = FAISS.from_texts(chunks, embeddings)

            elif i == 2:  # Step 3
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

            elif i == 3:  # Step 4
                prompt = f"""
Convert the following C++ code into Java Spring Boot. Generate separate classes only if needed by the logic of the C++ code, avoiding unnecessary layers.
Only generate a separate `Controller`, `Service`, and `Repository` if the C++ code includes logic for handling HTTP requests, database interactions, or business logic. If the code is simple (e.g., "Hello World"), convert it within a single `MainApplication` class.

Here is the C++ code snippet:

{cpp_code}
"""
                response = llm.invoke(prompt)

        # Final step
        st.session_state.progress_bar.progress(100)
        st.session_state.progress_message = "Conversion complete! 🎉"
        st.experimental_rerun()  # Update the UI
        
        components = {}
        lines = response.splitlines()
        current_class = None
        
        for line in lines:
            if line.startswith("public class ") or line.startswith("class "):
                current_class = line.split()[2].strip()
                components[current_class] = []
            if current_class:
                components[current_class].append(line)

        zip_buffer = io.BytesIO()
        zip_filename = filename.rsplit('.', 1)[0] + '.zip'
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for class_name, class_lines in components.items():
                class_code = "\n".join(class_lines)
                zip_file.writestr(f"{class_name}.java", class_code)

        zip_buffer.seek(0)

        return response, components, zip_buffer, zip_filename

    except Exception as e:
        logger.error(f"An error occurred while converting the code: {e}", exc_info=True)
        return None, None, None, None

def run_conversion(cpp_code, filename, token):
    return convert_cpp_to_java_spring_boot(cpp_code, filename, token)

# Page 1: File Upload Converter
if page == "File Upload Converter":
    st.header("C++ to Java Conversion Tool with LLM 💻")

    with st.sidebar:
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
            # Initialize progress bar and message if not already set
            if "progress_bar" not in st.session_state:
                st.session_state.progress_bar = st.progress(0)
            if "progress_message" not in st.session_state:
                st.session_state.progress_message = "Starting conversion..."

            with st.spinner("Processing..."):
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(run_conversion, code_content, file.name, HUGGINGFACE_API_TOKEN)
                    response, components, zip_buffer, zip_filename = future.result()

            st.success(st.session_state.progress_message)  # Show final message

            if response:
                st.code(response, language='java')
                for class_name, class_lines in components.items():
                    class_code = "\n".join(class_lines)
                    st.download_button(
                        label=f"Download {class_name}.java",
                        data=class_code,
                        file_name=f"{class_name}.java",
                        mime="text/x-java-source"
                    )

                st.download_button(
                    label="Download All Classes as Zip",
                    data=zip_buffer,
                    file_name=zip_filename,
                    mime="application/zip"
                )
            else:
                st.error("Conversion failed. Please try again.")

# Page 2: Inline Code Converter
if page == "Inline Code Converter":
    st.header("Inline C++ to Java Code Converter 💻")

    cpp_code_input = st.text_area("Enter C++ Code to Convert to Java Spring Boot", height=300)
    
    if cpp_code_input and st.button("Convert to Java"):
        # Initialize progress bar and message if not already set
        if "progress_bar" not in st.session_state:
            st.session_state.progress_bar = st.progress(0)
        if "progress_message" not in st.session_state:
            st.session_state.progress_message = "Starting conversion..."

        with st.spinner("Processing..."):
            with ThreadPoolExecutor() as executor:
                future = executor.submit(run_conversion, cpp_code_input, "converted_code.zip", HUGGINGFACE_API_TOKEN)
                response, components, zip_buffer, zip_filename = future.result()

        st.success(st.session_state.progress_message)  # Show final message

        if response:
            st.code(response, language='java')
            for class_name, class_lines in components.items():
                class_code = "\n".join(class_lines)
                st.download_button(
                    label=f"Download {class_name}.java",
                    data=class_code,
                    file_name=f"{class_name}.java",
                    mime="text/x-java-source"
                )

            st.download_button(
                label="Download All Classes as Zip",
                data=zip_buffer,
                file_name=zip_filename,
                mime="application/zip"
            )
        else:
            st.error("Conversion failed. Please try again.")
