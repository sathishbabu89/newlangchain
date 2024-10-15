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
from graphviz import Source

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
HUGGINGFACE_API_TOKEN = ""  # Add your Hugging Face API token

# Set up Streamlit page
st.set_page_config(page_title="C++ Code Summarization Tool", page_icon="💻")
st.header("C++ Code Summarization Tool with LLM 💻")

# Initialize session state for vector store
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Sidebar for file upload
def sidebar_file_upload():
    """Handle file upload in the sidebar."""
    st.title("Upload Your C++ Code")
    return st.file_uploader("Upload a C++ file (.cpp) to start analyzing", type="cpp")

# Function to read the uploaded file
def read_uploaded_file(file):
    """Read the content of the uploaded file.

    Args:
        file: The uploaded file.

    Returns:
        str: Content of the file.
    """
    try:
        code_content = file.read().decode("utf-8")
        st.subheader("Code Preview")
        st.text(code_content[:5000])  # Preview the first 5000 characters of code
        return code_content
    except Exception as e:
        logger.error(f"An error occurred while reading the code file: {e}", exc_info=True)
        st.warning("Unable to display code preview.")
        return None

# Function to generate the sequence diagram
def generate_sequence_diagram():
    """Generate a simple sequence diagram."""
    diagram = """
    sequenceDiagram
        participant User
        participant StreamlitApp
        participant LLM
        participant CodeAnalyzer
        
        User->>StreamlitApp: Upload C++ Code
        StreamlitApp->>CodeAnalyzer: Analyze Code
        CodeAnalyzer->>LLM: Generate Business Logic Summary
        LLM-->>CodeAnalyzer: Return Summary
        CodeAnalyzer-->>StreamlitApp: Provide Summary & Metrics
        StreamlitApp-->>User: Display Summary & Visualizations
    """
    
    # Render the sequence diagram using Graphviz
    src = Source(diagram, format='png')
    return src

# Function to extract code summaries using regex
def extract_summary(code):
    """Extracts function and class definitions from the given C++ code.

    Args:
        code (str): The C++ code to analyze.

    Returns:
        list: A summary list containing identified functions and classes.
    """
    function_pattern = r'\w+\s+\w+\s*\([^)]*\)\s*{'
    functions = re.findall(function_pattern, code)

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

# Function to perform code complexity analysis
def get_code_complexity(code):
    """Perform complexity analysis using lizard.

    Args:
        code (str): The C++ code to analyze.

    Returns:
        dict: Complexity metrics including cyclomatic complexity, number of functions, and lines of code.
    """
    complexity_metrics = lizard.analyze_file.analyze_source_code("uploaded.cpp", code)
    metrics = {
        "cyclomatic_complexity": complexity_metrics.average_cyclomatic_complexity,
        "functions": len(complexity_metrics.function_list),
        "lines_of_code": complexity_metrics.nloc,
        "average_nloc": complexity_metrics.average_nloc
    }
    return metrics

# Main application logic
def main():
    file = sidebar_file_upload()

    if file is not None:
        code_content = read_uploaded_file(file)
        
        if code_content:
            with st.spinner("Processing and summarizing code..."):
                # Get basic structure summary
                code_summary = extract_summary(code_content)

                if not code_summary:
                    st.warning("No functions, classes, or critical logic found in the code.")
                else:
                    # Perform code complexity analysis
                    complexity = get_code_complexity(code_content)
                    
                    # Prepare embeddings and LLM
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
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

                    # Display code structure summary
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
                            prompt = f"Summarize the business logic and key functionality of this C++ code:\n\n{code_content}"
                            response = llm.invoke(prompt)
                            st.markdown(response)

                        except Exception as e:
                            logger.error(f"An error occurred while summarizing the code: {e}", exc_info=True)
                            st.error("Unable to generate business logic summary.")
                    
                    # Generate and display the sequence diagram
                    st.subheader("Sequence Diagram")
                    diagram_src = generate_sequence_diagram()
                    st.graphviz_chart(diagram_src)

    else:
        st.info("Please upload a C++ code file to start analyzing.")

# Run the main function
if __name__ == "__main__":
    main()
