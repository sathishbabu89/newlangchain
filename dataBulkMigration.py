import logging
import streamlit as st
import zipfile
import os
import shutil
import torch
import io
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceEndpoint

# Force PyTorch to use CPU
device = torch.device("cpu")

# Setup logger for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

HUGGINGFACE_API_TOKEN = ""  # Add your Hugging Face API token here

# Ensure the API token is provided before invoking the LLM
if not HUGGINGFACE_API_TOKEN:
    st.error("Please set the Hugging Face API token.")

st.set_page_config(page_title="PL/SQL to Database Conversion Tool", page_icon="💻")

# Sidebar for page selection
page = st.sidebar.selectbox("Choose Page", ["File Upload Converter", "PL/SQL Project Converter"])

# Helper function for database conversion logic
def convert_plsql_to_database(plsql_code, db_type, HUGGINGFACE_API_TOKEN):
    try:
        # Initialize the progress bar
        progress_bar = st.progress(0)
        progress_stage = 0

        with st.spinner("Processing and converting PL/SQL code..."):

            # Stage 1: Splitting text into chunks (20% progress)
            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 1: Splitting the code into chunks... 💬")

            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = text_splitter.split_text(plsql_code)

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
                repo_id="meta-llama/Llama-3.2-11B-Vision-Instruct",  # Using T5 instead of Mistral
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
            st.info(f"Step 4: Converting PL/SQL to {db_type}... 🔄")

            # Set prompt based on the selected database conversion target
            prompt = f"""
Convert the following PL/SQL code into {db_type} SQL format. Ensure the syntax matches {db_type} conventions and update the structure as needed.

Here is the PL/SQL code snippet:

{plsql_code}
"""
            # Call the LLM to convert the code
            response = llm.invoke(prompt)

            # Stage 5: Displaying converted code (100% progress)
            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.success("Step 5: Conversion complete! 🎉")

            # Display the converted SQL code
            st.code(response, language='sql')

            # Basic error-checking logic
            if re.search(r'\berror\b|\bexception\b|\bsyntax\b|\bmissing\b', response.lower()):
                st.warning(f"The converted {db_type} SQL code may contain syntax or structural errors. Please review it carefully.")
            else:
                st.success(f"The {db_type} SQL code is free from basic syntax errors!")

            return response
        
    except Exception as e:
        logger.error(f"An error occurred while converting the code: {e}", exc_info=True)
        st.error("Unable to convert PL/SQL code.")

# Function to process and convert multiple PL/SQL files in a zip
def process_plsql_zip(file, db_type, HUGGINGFACE_API_TOKEN):
    # Create a temporary directory to extract zip contents
    with zipfile.ZipFile(file, 'r') as zip_ref:
        # List all files in the zip
        file_list = zip_ref.namelist()
        logger.info(f"Files in uploaded zip: {file_list}")
        
        zip_ref.extractall("temp_plsql")
    
    converted_files = []
    conversion_summary = []

    # Iterate through extracted files
    extracted_files = os.listdir("temp_plsql")
    logger.info(f"Extracted files: {extracted_files}")
    
    for filename in extracted_files:
        if filename.endswith(".plsql"):
            plsql_file_path = os.path.join("temp_plsql", filename)
            try:
                with open(plsql_file_path, 'r') as f:
                    plsql_code = f.read()
                    logger.debug(f"Read {len(plsql_code)} characters from {filename}.")
                    
                    # If no content, log it
                    if not plsql_code:
                        logger.warning(f"File {filename} is empty!")
                        st.warning(f"File {filename} is empty.")
                        continue
                    
                    st.info(f"Converting {filename}...")

                    # Convert the PL/SQL code to target DB type
                    converted_code = convert_plsql_to_database(plsql_code, db_type, HUGGINGFACE_API_TOKEN)
                    
                    # Ensure conversion returned something
                    if converted_code:
                        converted_files.append((filename.replace(".plsql", f"_{db_type}.sql"), converted_code))
                        conversion_summary.append(f"Converted {filename} to {db_type} SQL format.")
                    else:
                        logger.warning(f"Conversion failed for {filename}. No converted code returned.")
                        st.warning(f"Conversion failed for {filename}.")
            
            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}", exc_info=True)
                st.error(f"Error processing {filename}.")
    
    # Check if we actually converted any files
    if not converted_files:
        logger.warning("No files were converted.")
        st.warning("No files were converted. Please check your PL/SQL code.")
        return None, conversion_summary

    # Create a new zip file with the converted SQL files
    output_zip = io.BytesIO()
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zip_out:
        for new_filename, code in converted_files:
            zip_out.writestr(new_filename, code)

    output_zip.seek(0)  # Go to the beginning of the BytesIO object
    return output_zip, conversion_summary

# Page 1: File Upload Converter (Existing Page)
if page == "File Upload Converter":
    st.header("PL/SQL to Database Conversion Tool 💻")

    # Collapsible sidebar for uploading code and tutorials
    with st.sidebar:
        with st.expander("Upload Your PL/SQL Code", expanded=True):  # Expanded by default
            file = st.file_uploader("Upload a PL/SQL file (.plsql) to start analyzing", type="plsql")

            if file is not None:
                try:
                    plsql_code = file.read().decode("utf-8")
                    st.subheader("PL/SQL Code Preview")
                    st.code(plsql_code[:5000], language='sql')  # Display the PL/SQL code with proper syntax highlighting
                except Exception as e:
                    logger.error(f"An error occurred while reading the code file: {e}", exc_info=True)
                    st.warning("Unable to display code preview.")

    # Tutorials or Tips Section
    with st.expander("Tutorials & Tips", expanded=True):
        st.write(""" 
        ### Welcome to the PL/SQL to Database Conversion Tool!
        
        Here are some tips to help you use this tool effectively:
        - **Code Formatting:** Ensure your PL/SQL code is properly formatted.
        - **Chunking:** Break large files into smaller parts.
        - **Testing:** Test the converted code in the target database system.
        - **Documentation:** Familiarize with PL/SQL and the target database SQL syntax (e.g., PostgreSQL, MySQL).
        """)

    # Database conversion options
    conversion_options = [
        "PL/SQL to PostgreSQL",
        "PL/SQL to Google BigQuery",
        "PL/SQL to MongoDB",
        "PL/SQL to SQLite",
        "PL/SQL to SQL Server (T-SQL)",
        "PL/SQL to MySQL"
    ]
    
    # Dropdown to select conversion target
    selected_option = st.selectbox("Select the target database conversion format", conversion_options)

    # Convert based on selection
    if file is not None and selected_option:
        db_type = selected_option.split(" ")[-1]  # Get the target database type (PostgreSQL, MySQL, etc.)
        if st.button(f"Convert to {selected_option}"):
            convert_plsql_to_database(plsql_code, db_type, HUGGINGFACE_API_TOKEN)

# Page 2: PL/SQL Project Converter (New Page for ZIP Upload)
if page == "PL/SQL Project Converter":
    st.header("PL/SQL Project to Database Conversion Tool 🚀")

    # File upload for ZIP
    uploaded_zip = st.file_uploader("Upload a zip file containing PL/SQL files", type="zip")

    if uploaded_zip is not None:
        conversion_options = [
            "PostgreSQL", "Google BigQuery", "MongoDB", "SQLite", "SQL Server (T-SQL)", "MySQL"
        ]
        db_type = st.selectbox("Select the target database conversion format", conversion_options)
        
        if st.button(f"Convert to {db_type}"):
            with st.spinner(f"Converting PL/SQL project to {db_type}..."):
                # Process the ZIP file
                output_zip, conversion_summary = process_plsql_zip(uploaded_zip, db_type, HUGGINGFACE_API_TOKEN)

                # Provide download link for converted project
                if output_zip:
                    st.success("Conversion complete! 🎉")
                    st.download_button(
                        label="Download Converted Project",
                        data=output_zip,
                        file_name=f"converted_plsql_project_{db_type}.zip",
                        mime="application/zip"
                    )

                    # Show conversion summary
                    st.subheader("Conversion Summary")
                    for summary in conversion_summary:
                        st.write(summary)
                else:
                    st.warning("No files were converted. Please check the log above for details.")