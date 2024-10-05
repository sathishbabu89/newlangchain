import logging
import streamlit as st
import re
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceEndpoint
from langchain.chains import load_qa_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUGGINGFACE_API_TOKEN = ""  # Add your token here

st.set_page_config(page_title="Code Analysis Tool", page_icon="💻")
st.header("Code Analysis Tool 💻")

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

with st.sidebar:
    st.title("Upload Your C++ Code")
    file = st.file_uploader("Upload a C++ file (.cpp) to analyze", type="cpp")

    if file is not None:
        try:
            code_content = file.read().decode("utf-8")
            st.subheader("Code Preview")
            st.text(code_content[:5000])  # Preview first 5000 characters
        except Exception as e:
            logger.error(f"An error occurred while reading the code file: {e}")
            st.warning("Unable to display code preview.")

if file is not None:
    if st.session_state.vector_store is None:
        try:
            with st.spinner("Processing code..."):
                # Function to extract functions, API calls, and business logic hints
                def extract_functions_and_apis(code):
                    # Regular expressions to capture key elements of C++ code
                    function_pattern = r'(\w+\s+\w+\s*\([^)]*\)\s*{)'  # function definitions
                    api_call_pattern = r'\w+\s*\(\s*\"[^\"]+\"'  # API calls (simple regex)
                    loop_pattern = r'(for\s*\(.*\)|while\s*\(.*\))'  # loops
                    
                    functions = re.findall(function_pattern, code)
                    api_calls = re.findall(api_call_pattern, code)
                    loops = re.findall(loop_pattern, code)
                    
                    return {
                        "functions": functions,
                        "api_calls": api_calls,
                        "loops": loops
                    }

                # Extracting functions, API calls, and loops from the code
                extracted_elements = extract_functions_and_apis(code_content)
                
                # Split the code for embedding and summary purposes
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )
                chunks = text_splitter.split_text(code_content)

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
                st.success("Code processed successfully!")

                # Displaying the extracted information in a structured summary
                st.subheader("Summary of Code Elements")
                
                if extracted_elements["functions"]:
                    st.write("### Functions Found:")
                    for func in extracted_elements["functions"]:
                        st.write(f"- {func.strip()}")
                else:
                    st.write("No functions found.")
                    
                if extracted_elements["api_calls"]:
                    st.write("### API Callouts Found:")
                    for api in extracted_elements["api_calls"]:
                        st.write(f"- {api.strip()}")
                else:
                    st.write("No API callouts found.")
                    
                if extracted_elements["loops"]:
                    st.write("### Loops Found:")
                    for loop in extracted_elements["loops"]:
                        st.write(f"- {loop.strip()}")
                else:
                    st.write("No loops found.")
                
                # Using the LLM to generate a high-level summary
                llm = HuggingFaceEndpoint(
                    repo_id="mistralai/Mistral-Nemo-Instruct-2407",
                    max_new_tokens=512,
                    top_k=10,
                    top_p=0.95,
                    typical_p=0.95,
                    temperature=0.01,
                    repetition_penalty=1.03,
                    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
                )
                
                # Creating a chain to summarize the chunks
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.invoke(
                    {"input_documents": st.session_state.vector_store.similarity_search("Summarize the code logic")}
                )
                summary = response['output_text']
                st.write("### LLM-Generated Summary of Code Logic:")
                st.write(summary)
                
        except Exception as e:
            logger.error(f"An error occurred while processing the code: {e}")
            st.error(str(e))
else:
    st.info("Please upload a C++ code file to start analyzing.")
