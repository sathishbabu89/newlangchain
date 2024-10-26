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

# Ensure the API token is provided before invoking the LLM
if not HUGGINGFACE_API_TOKEN:
    st.error("Please set the Hugging Face API token.")

st.set_page_config(page_title="C++ to Java Conversion Tool", page_icon="💻")

# Sidebar for page selection
page = st.sidebar.selectbox("Choose Page", ["File Upload Converter", "Inline Code Converter"])

# Helper function for code conversion logic
def convert_cpp_to_java_spring_boot(cpp_code, HUGGINGFACE_API_TOKEN):
    try:
        # Initialize the progress bar
        progress_bar = st.progress(0)
        progress_stage = 0

        with st.spinner("Processing and converting code..."):

            # Stage 1: Splitting text into chunks (20% progress)
            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 1: Splitting the code into chunks... 💬")

            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = text_splitter.split_text(cpp_code)

            # Stage 2: Creating embeddings (40% progress)
            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 2: Generating embeddings... 📊")

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            vector_store = FAISS.from_texts(chunks, embeddings)

            # Stage 3: LLM initialization (60% progress)
            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 3: Loading the language model... 🚀")

            # Load the LLM for code conversion
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

            # Stage 4: Code conversion (80% progress)
            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 4: Converting C++ to Java Spring Boot... 🔄")

            # Prompt to convert C++ to Java Spring Boot
            prompt = f"""
Convert the following C++ code snippet into equivalent Java Spring Boot code. Ensure that the translated code:

1. Adheres to Java Spring Boot conventions, including proper use of annotations (e.g., @RestController, @Service, @Repository).
2. Follows best practices in Java, such as meaningful class and method names, use of proper access modifiers (e.g., private, public), and exception handling.
3. Includes the necessary imports, such as `org.springframework.web.bind.annotation`, `org.springframework.beans.factory.annotation.Autowired`, and other relevant Spring Boot components.
4. Uses Spring Boot components appropriately, such as:
   - Controllers for handling HTTP requests.
   - Services for business logic.
   - Repositories for database interaction (if applicable).
5. Converts any C++ I/O operations into their Java Spring Boot equivalents (e.g., handling HTTP requests/responses).
6. If the C++ code involves threading, synchronization, or concurrency, implement them using Java’s `ExecutorService`, `CompletableFuture`, or other Java concurrency utilities.
7. Includes comments where necessary to explain complex logic and ensure maintainability.
8. Adds any additional dependencies or configuration required (e.g., instructions for `pom.xml` or `application.properties` for Spring Boot).

Here is the C++ code snippet to convert:

{cpp_code}
"""
            # Call the LLM to convert the code
            response = llm.invoke(prompt)

            # Stage 5: Displaying converted code (100% progress)
            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.success("Step 5: Conversion complete! 🎉")

            # Display the converted Java code
            st.code(response, language='java')

            # Basic error-checking logic
            if re.search(r'\berror\b|\bexception\b|\bsyntax\b|\bmissing\b', response.lower()):
                st.warning("The converted Java code may contain syntax or structural errors. Please review it carefully.")
            else:
                st.success("The Java code is free from basic syntax errors!")

            # Implementing the downloadable file feature
            st.download_button(
                label="Download Java Code",
                data=response,
                file_name="converted_code.java",
                mime="text/x-java-source"
            )
        
    except Exception as e:
        logger.error(f"An error occurred while converting the code: {e}", exc_info=True)
        st.error("Unable to convert C++ code to Java.")

# Page 1: File Upload Converter
if page == "File Upload Converter":
    st.header("C++ to Java Conversion Tool with LLM 💻")

    # Collapsible sidebar for uploading code and tutorials
    with st.sidebar:
        with st.expander("Upload Your C++ Code", expanded=True):  # Expanded by default
            file = st.file_uploader("Upload a C++ file (.cpp) to start analyzing", type="cpp")

            if file is not None:
                try:
                    code_content = file.read().decode("utf-8")
                    st.subheader("C++ Code Preview")
                    st.code(code_content[:5000], language='cpp')  # Display the C++ code with proper syntax highlighting
                except Exception as e:
                    logger.error(f"An error occurred while reading the code file: {e}", exc_info=True)
                    st.warning("Unable to display code preview.")

    # Tutorials or Tips Section
    with st.expander("Tutorials & Tips", expanded=True):
        st.write("""
        ### Welcome to the C++ to Java Conversion Tool!
        
        Here are some tips to help you use this tool effectively:
        - **Code Formatting:** Ensure your C++ code is properly formatted.
        - **Chunking:** Break large files into smaller parts.
        - **Annotations:** Ensure the Java conversion includes necessary annotations like `@RestController`.
        - **Testing:** Test the Java code after conversion.
        - **Documentation:** Familiarize with C++ and Java Spring Boot docs.
        """)

    if file is not None:
        if st.button("Convert C++ to Java Spring Boot"):
            convert_cpp_to_java_spring_boot(code_content, HUGGINGFACE_API_TOKEN)

# Page 2: Inline Code Converter
if page == "Inline Code Converter":
    st.header("Inline C++ to Java Code Converter 💻")

    cpp_code_input = st.text_area("Enter C++ Code to Convert to Java Spring Boot", height=300)
    
    if cpp_code_input and st.button("Convert to Java"):
        convert_cpp_to_java_spring_boot(cpp_code_input, HUGGINGFACE_API_TOKEN)
