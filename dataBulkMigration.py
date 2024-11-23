import logging
import streamlit as st
import zipfile
import os
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
        st.progress(0)
        
        # Show a single progress bar for the entire conversion
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

    # Iterate through extracted files recursively using os.walk
    extracted_files = []
    for root, dirs, files in os.walk("temp_plsql"):
        for filename in files:
            extracted_files.append(os.path.join(root, filename))
    logger.info(f"Extracted files (recursive): {extracted_files}")
    
    # Show the global progress for all files
    total_files = len(extracted_files)
    progress_bar = st.progress(0)
    progress_stage = 0
    
    # Iterate through all the files found in the directories
    for idx, plsql_file_path in enumerate(extracted_files):
        if plsql_file_path.endswith(".sql"):
            try:
                with open(plsql_file_path, 'r') as f:
                    plsql_code = f.read()
                    logger.debug(f"Read {len(plsql_code)} characters from {plsql_file_path}.")
                    
                    # If no content, log it
                    if not plsql_code:
                        logger.warning(f"File {plsql_file_path} is empty!")
                        st.warning(f"File {plsql_file_path} is empty.")
                        continue
                    
                    # Extract file name relative to the original directory
                    file_name = os.path.relpath(plsql_file_path, "temp_plsql")

                    # Convert the PL/SQL code to target DB type
                    converted_code = convert_plsql_to_database(plsql_code, db_type, HUGGINGFACE_API_TOKEN)
                    
                    # Ensure conversion returned something
                    if converted_code:
                        new_file_name = file_name.replace(".sql", f"_{db_type}.sql")
                        converted_files.append((new_file_name, converted_code))
                        conversion_summary.append(f"Converted {file_name} to {db_type} SQL format.")
                    else:
                        logger.warning(f"Conversion failed for {plsql_file_path}. No converted code returned.")
                        st.warning(f"Conversion failed for {file_name}.")
            
            except Exception as e:
                logger.error(f"Failed to process {plsql_file_path}: {e}", exc_info=True)
                st.error(f"Error processing {plsql_file_path}.")
        
        # Update the progress bar for each file processed
        progress_stage = int((idx + 1) / total_files * 100)
        progress_bar.progress(progress_stage)
    
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

            # Show a global progress bar for the entire conversion process
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
