import logging
import streamlit as st
import re
import lizard  # For code complexity analysis
import plotly.express as px  # For pie chart
import pandas as pd  # For DataFrames
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceEndpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUGGINGFACE_API_TOKEN = ""  # Add your Hugging Face API token

st.set_page_config(page_title="C++ Code Summarization Tool", page_icon="💻")
st.header("C++ Code Summarization Tool with LLM 💻")

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

with st.sidebar:
    st.title("Upload Your C++ Code")
    file = st.file_uploader("Upload a C++ file (.cpp) to start analyzing", type="cpp")

    if file is not None:
        try:
            code_content = file.read().decode("utf-8")
            st.subheader("Code Preview")
            st.text(code_content[:5000])  # Preview the first 5000 characters of code
        except Exception as e:
            logger.error(f"An error occurred while reading the code file: {e}", exc_info=True)
            st.warning("Unable to display code preview.")

if file is not None:
    if st.session_state.vector_store is None:
        try:
            with st.spinner("Processing and summarizing code..."):

                # Function to extract code summaries using regex (structure)
                def extract_summary(code):
                    # Using regex to identify function definitions, classes, etc.
                    function_pattern = r'\w+\s+\w+\s*\([^)]*\)\s*{'
                    functions = re.findall(function_pattern, code)

                    # Example: Identify class declarations
                    class_pattern = r'class\s+\w+\s*{'
                    classes = re.findall(class_pattern, code)

                    summary = []
                    if functions:
                        summary.append("Functions identified in the code:")
                        summary.extend(functions)
                    if classes:
                        summary.append("\nClasses identified in the code:")
                        summary.extend(classes)

                    return summary

                # Perform complexity analysis using lizard
                def get_code_complexity(code):
                    complexity_metrics = lizard.analyze_file.analyze_source_code("uploaded.cpp", code)
                    metrics = {
                        "cyclomatic_complexity": complexity_metrics.average_cyclomatic_complexity,
                        "functions": len(complexity_metrics.function_list),
                        "lines_of_code": complexity_metrics.nloc,
                        "average_nloc": complexity_metrics.average_nloc
                    }
                    return metrics

                # Get basic structure summary
                code_summary = extract_summary(code_content)

                if not code_summary:
                    st.warning("No functions, classes, or critical logic found in the code.")
                else:
                    # Perform code complexity analysis
                    complexity = get_code_complexity(code_content)
                    
                    # Use LLM to generate detailed business logic summary
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50
                    )
                    chunks = text_splitter.split_text(code_content)

                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)

                    # Load the LLM for business logic summarization
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

                    # Combine structural and LLM-based logic analysis
                    st.success("Code processed successfully!")

                    st.subheader("Code Structure Summary")
                    for section in code_summary:
                        st.markdown(f"- {section}")

                    # Code Complexity Visualization using Bar Charts
                    st.subheader("Code Complexity Analysis")

                    complexity_df = pd.DataFrame({
                        'Metric': ['Cyclomatic Complexity', 'Number of Functions', 'Lines of Code'],
                        'Value': [complexity['cyclomatic_complexity'], complexity['functions'], complexity['lines_of_code']]
                    })

                    # Show Bar Chart
                    st.bar_chart(complexity_df.set_index('Metric'))

                    # Show Pie Chart for number of functions vs classes
                    pie_data = pd.DataFrame({
                        'Type': ['Functions', 'Classes'],
                        'Count': [len(re.findall(r'\w+\s+\w+\s*\([^)]*\)\s*{', code_content)), len(re.findall(r'class\s+\w+\s*{', code_content))]
                    })

                    fig = px.pie(pie_data, values='Count', names='Type', title='Functions vs Classes')
                    st.plotly_chart(fig)

                    # Business Logic Summary using LLM
                    st.subheader("Business Logic Summary")
                    with st.spinner("Generating business logic summary..."):
                        try:
                            # Concatenate the code with an instruction for summarization
                            prompt = f"Summarize the business logic and key functionality of this C++ code:\n\n{code_content}"
                            response = llm.invoke(prompt)

                            # Assuming the response is a string, display it directly
                            st.markdown(response)

                        except Exception as e:
                            logger.error(f"An error occurred while summarizing the code: {e}", exc_info=True)
                            st.error("Unable to generate business logic summary.")

        except Exception as e:
            logger.error(f"An error occurred while processing the code: {e}", exc_info=True)
            st.error(str(e))

else:
    st.info("Please upload a C++ code file to start analyzing.")
