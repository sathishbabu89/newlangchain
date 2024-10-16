import hashlib  # For hashing code blocks to detect duplicates
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

# Dependency analysis function
def analyze_dependencies(code):
    """
    Extract dependencies (libraries) from the code.
    Looks for #include directives and identifies external dependencies.
    """
    include_pattern = r'#include\s*[<"]([^">]+)[">]'
    dependencies = re.findall(include_pattern, code)

    # Standard C++ libraries (you can expand this list)
    std_libs = ['iostream', 'vector', 'map', 'cmath', 'string', 'algorithm', 'iomanip', 'fstream', 'sstream', 'queue', 'stack']
    
    # Check if the dependency is standard or external
    external_dependencies = [dep for dep in dependencies if dep.split('/')[0] not in std_libs]

    return {
        'all_dependencies': dependencies,
        'external_dependencies': external_dependencies
    }

def calculate_maintainability_index(cyclomatic_complexity, lines_of_code):
    """
    Calculate the maintainability index based on cyclomatic complexity and lines of code.
    Returns a value between 0 and 100, where 100 represents the most maintainable code.
    """
    if lines_of_code == 0:
        return 0
    avg_line_length = lines_of_code / max(1, lines_of_code)  # Avoid division by zero
    maintainability_index = 171 - 5.2 * avg_line_length - 0.3 * cyclomatic_complexity

    # Debug prints
    logger.info(f"Lines of Code: {lines_of_code}")
    logger.info(f"Cyclomatic Complexity: {cyclomatic_complexity}")
    logger.info(f"Average Line Length: {avg_line_length}")
    logger.info(f"Calculated Maintainability Index (before clamping): {maintainability_index}")

    return max(0, min(100, maintainability_index))  # Clamping the value between 0 and 100

def get_code_complexity(code):
    """
    Use lizard to analyze code complexity and return relevant metrics.
    """
    complexity_metrics = lizard.analyze_file.analyze_source_code("uploaded.cpp", code)
    metrics = {
        "cyclomatic_complexity": complexity_metrics.average_cyclomatic_complexity,
        "functions": len(complexity_metrics.function_list),
        "lines_of_code": complexity_metrics.nloc,
        "average_nloc": complexity_metrics.average_nloc,
        "maintainability_index": calculate_maintainability_index(
            complexity_metrics.average_cyclomatic_complexity,
            complexity_metrics.nloc
        )
    }
    return metrics

# Duplicate code detection function
def detect_duplicate_code(code):
    """
    Detect duplicate code blocks (functions) by hashing them and checking for duplicates.
    Returns a list of duplicate functions.
    """
    function_pattern = r'(\w+\s+\w+\s*\([^)]*\)\s*{(?:[^{}]*|{(?:[^{}]*|{[^{}]*})*})*})'
    functions = re.findall(function_pattern, code, re.DOTALL)

    function_hashes = {}
    duplicates = []

    for func in functions:
        # Hash the function code
        func_hash = hashlib.sha256(func.encode('utf-8')).hexdigest()

        if func_hash in function_hashes:
            duplicates.append(func)
        else:
            function_hashes[func_hash] = func

    return duplicates

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

                # Get basic structure summary
                code_summary = extract_summary(code_content)

                # Perform Dependency Analysis
                dependencies = analyze_dependencies(code_content)
                st.subheader("Code Dependency Analysis")

                # Display all dependencies
                st.markdown(f"**All Dependencies:** {', '.join(dependencies['all_dependencies'])}")
                
                # Highlight external dependencies
                if dependencies['external_dependencies']:
                    st.warning(f"External Dependencies Found: {', '.join(dependencies['external_dependencies'])}")
                else:
                    st.success("No external dependencies detected.")

                if not code_summary:
                    st.warning("No functions, classes, or critical logic found in the code.")
                else:
                    # Perform code complexity analysis
                    complexity = get_code_complexity(code_content)

                    # Perform Duplicate Code Detection
                    duplicates = detect_duplicate_code(code_content)

                    if duplicates:
                        st.subheader("Duplicate Code Detected")
                        for i, dup in enumerate(duplicates, 1):
                            st.error(f"Duplicate #{i}:\n{dup[:300]}...")  # Showing first 300 characters for readability
                    else:
                        st.success("No duplicate code detected.")

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
                        'Metric': ['Cyclomatic Complexity', 'Number of Functions', 'Lines of Code', 'Maintainability Index'],
                        'Value': [complexity['cyclomatic_complexity'], complexity['functions'], complexity['lines_of_code'], complexity['maintainability_index']]
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

# Test function to validate the Maintainability Index calculation
def test_maintainability_index():
    test_cases = [
        {"lines_of_code": 10, "cyclomatic_complexity": 1},
        {"lines_of_code": 20, "cyclomatic_complexity": 5},
        {"lines_of_code": 100, "cyclomatic_complexity": 15},
        {"lines_of_code": 0, "cyclomatic_complexity": 0},  # Edge case
    ]
    
    for case in test_cases:
        mi = calculate_maintainability_index(case["cyclomatic_complexity"], case["lines_of_code"])
        logger.info(f"Test Case - Lines of Code: {case['lines_of_code']}, Cyclomatic Complexity: {case['cyclomatic_complexity']}, MI: {mi}")

# Call the test function (You can run this separately or as needed)
test_maintainability_index()
