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
st.header("C++ to Java Conversion Tool with LLM 💻")

# Initialize vector_store in session_state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

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

        - **Code Formatting:** Ensure your C++ code is properly formatted and follows standard conventions for better conversion results.
        - **Chunking:** If your C++ code is too large, consider breaking it into smaller files for easier processing.
        - **Annotations:** When converting to Java Spring Boot, remember to include necessary annotations such as `@RestController` for controllers.
        - **Testing:** After conversion, test the Java code in your development environment to ensure it works as expected.
        - **Refer to Documentation:** Familiarize yourself with both C++ and Java Spring Boot documentation for best practices in coding.
        
        ### Frequently Asked Questions (FAQs):
        - **What types of C++ code can be converted?**
          This tool can handle various C++ constructs, but complex libraries and system-level code might require manual adjustments.
        - **How can I ensure the converted code is error-free?**
          After conversion, carefully review the Java code, and test it thoroughly. Look for any syntactic or semantic errors.
        """)

# Code conversion logic
if file is not None:
    if st.session_state.vector_store is None:
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
                chunks = text_splitter.split_text(code_content)

                # Stage 2: Creating embeddings (40% progress)
                progress_stage += 20
                progress_bar.progress(progress_stage)
                st.info("Step 2: Generating embeddings... 📊")

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)

                # Stage 3: LLM initialization (60% progress)
                progress_stage += 20
                progress_bar.progress(progress_stage)
                st.info("Step 3: Loading the language model... 🚀")

                # Load the LLM for code conversion
                llm = HuggingFaceEndpoint(
                    repo_id="mistralai/Mistral-Nemo-Instruct-2407",
                    max_new_tokens=2048,  # Increased token limit for longer output
                    top_k=10,
                    top_p=0.95,
                    typical_p=0.95,
                    temperature=0.01,
                    repetition_penalty=1.03,
                    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
                )

                try:
                    # Stage 4: Code conversion (80% progress)
                    progress_stage += 20
                    progress_bar.progress(progress_stage)
                    st.info("Step 4: Converting C++ to Java Spring Boot... 🔄")

                    # Prompt for C++ to Java conversion
                    prompt_conversion = f"""
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

Here is the C++ code snippet to convert:

{code_content}
"""
                    # Call the LLM to convert the code
                    response_conversion = llm.invoke(prompt_conversion)

                    # Stage 5: Displaying converted code (100% progress)
                    progress_stage += 20
                    progress_bar.progress(progress_stage)
                    st.success("Step 5: Conversion complete! 🎉")
                    
                    # Display the converted Java code
                    st.code(response_conversion, language='java')

                    # Basic error-checking logic
                    if re.search(r'\berror\b|\bexception\b|\bsyntax\b|\bmissing\b', response_conversion.lower()):
                        st.warning("The converted Java code may contain syntax or structural errors. Please review it carefully.")
                    else:
                        st.success("The Java code is free from basic syntax errors!")

                    # Implementing the downloadable file feature
                    st.download_button(
                        label="Download Java Code",
                        data=response_conversion,
                        file_name="converted_code.java",
                        mime="text/x-java-source"
                    )

                    # Button for explaining business logic
                    if st.button("Explain Business Logic"):
                        st.info("Generating business logic explanation... Please wait.")
                        
                        # Prompt for business logic explanation
                        prompt_business_logic = f"""
Explain the business logic of the following C++ code. Focus on describing the purpose and functionality of the code, any important methods or classes, and how the code handles input, output, and processing. Also, describe any key algorithms, design patterns, or architectural choices made in the code.

Here is the C++ code:

{code_content}
"""
                        # Call the LLM to explain the business logic
                        response_business_logic = llm.invoke(prompt_business_logic)

                        # Display the explanation
                        st.subheader("Business Logic Explanation")
                        st.write(response_business_logic)

                    # Button for execution instructions
                    if st.button("Execute"):
                        st.info("Providing instructions for running the code...")

                        st.write("""
                        ### Instructions for Running Java Spring Boot Code
                        
                        Here are the steps to run the converted Java Spring Boot code in popular IDEs:

                        #### Spring Tool Suite (STS):
                        1. **Download STS**: Ensure you have [Spring Tool Suite](https://spring.io/tools) installed.
                        2. **Create a New Spring Boot Project**:
                            - Go to `File -> New -> Spring Starter Project`.
                            - Fill in the project details like name, group, and package.
                            - Add the required Spring Boot dependencies (Web, JPA, etc.).
                        3. **Add the Converted Code**: 
                            - Copy the converted code into the appropriate package under `src/main/java`.
                            - Ensure the `@RestController` and other components are in place.
                        4. **Run the Project**: 
                            - Right-click the project and select `Run As -> Spring Boot App`.
                            - The Spring Boot application should start, and you can access it via `http://localhost:8080`.

                        #### IntelliJ IDEA:
                        1. **Download IntelliJ IDEA**: Ensure you have [IntelliJ IDEA](https://www.jetbrains.com/idea/download/) installed.
                        2. **Create a New Spring Boot Project**:
                            - Go to `File -> New -> Project`.
                            - Select `Spring Initializr` and configure the project.
                            - Add the necessary Spring Boot dependencies (Web, JPA, etc.).
                        3. **Add the Converted Code**
