import logging
import streamlit as st
import re
import torch
import zipfile
import io
import requests
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

st.set_page_config(page_title="SQL Conversion Tool", page_icon="💻")

page = st.sidebar.selectbox("Choose Page", ["File Upload SQL Converter", "Inline SQL Converter"])

# Conversion function for PL/SQL to target database SQL dialect
def convert_plsql_to_target_db(plsql_content, target_db):
    converted_content = plsql_content
    
    if target_db == "PostgreSQL":
        # Convert PL/SQL syntax to PostgreSQL-compatible SQL
        converted_content = converted_content.replace("SYSDATE", "CURRENT_DATE")
        converted_content = converted_content.replace("TO_CHAR", "TO_TIMESTAMP")
        converted_content = converted_content.replace("VARCHAR2", "VARCHAR")

    elif target_db == "MySQL":
        # Convert PL/SQL syntax to MySQL-compatible SQL
        converted_content = converted_content.replace("||", "CONCAT")
        converted_content = converted_content.replace("VARCHAR2", "VARCHAR")
        converted_content = converted_content.replace("NUMBER", "DECIMAL")

    elif target_db == "SQL Server":
        # Convert PL/SQL syntax to SQL Server-compatible SQL
        converted_content = converted_content.replace("SYSDATE", "GETDATE()")
        converted_content = converted_content.replace("VARCHAR2", "VARCHAR")
        converted_content = converted_content.replace("||", "+")
        converted_content = converted_content.replace("TO_CHAR", "CONVERT")

    elif target_db == "SQLite":
        # Convert PL/SQL syntax to SQLite-compatible SQL
        converted_content = converted_content.replace("VARCHAR2", "TEXT")
        converted_content = converted_content.replace("NUMBER", "REAL")

    elif target_db == "MariaDB":
        # Convert PL/SQL syntax to MariaDB-compatible SQL
        converted_content = converted_content.replace("||", "CONCAT")
        converted_content = converted_content.replace("VARCHAR2", "VARCHAR")
    
    return converted_content

# Page 1: File Upload SQL Converter
if page == "File Upload SQL Converter":
    st.header("PL/SQL to Target Database SQL Conversion Tool 💻")

    target_db = st.selectbox("Select Target Database", ["PostgreSQL", "MySQL", "SQL Server", "SQLite", "MariaDB"])
    file = st.file_uploader("Upload a PL/SQL file (.sql)", type="sql")

    if file is not None:
        plsql_content = file.read().decode("utf-8")
        st.subheader("PL/SQL Code Preview")
        st.code(plsql_content[:5000], language='sql')

        if st.button("Convert PL/SQL to Target Database SQL"):
            converted_content = convert_plsql_to_target_db(plsql_content, target_db)
            st.subheader(f"Converted SQL for {target_db}")
            st.code(converted_content, language='sql')

            # Download button for the converted file
            st.download_button(
                label=f"Download Converted SQL for {target_db}",
                data=converted_content,
                file_name=f"converted_{target_db.lower()}.sql",
                mime="text/sql"
            )

# Page 2: Inline SQL Converter
if page == "Inline SQL Converter":
    st.header("Inline PL/SQL to Target Database SQL Converter 💻")

    target_db_inline = st.selectbox("Select Target Database for Inline Conversion", ["PostgreSQL", "MySQL", "SQL Server", "SQLite", "MariaDB"])
    plsql_code_input = st.text_area("Enter PL/SQL Code to Convert", height=300)

    if plsql_code_input and st.button("Convert Inline PL/SQL to Target SQL"):
        converted_inline_content = convert_plsql_to_target_db(plsql_code_input, target_db_inline)
        st.subheader(f"Converted SQL for {target_db_inline}")
        st.code(converted_inline_content, language='sql')

        # Download button for inline converted file
        st.download_button(
            label=f"Download Converted SQL for {target_db_inline}",
            data=converted_inline_content,
            file_name=f"converted_inline_{target_db_inline.lower()}.sql",
            mime="text/sql"
)
