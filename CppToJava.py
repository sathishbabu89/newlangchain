import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# Load models
incoder_tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B")
incoder_model = AutoModelForCausalLM.from_pretrained("facebook/incoder-1B")

# Check and set the padding token
if incoder_tokenizer.pad_token is None:
    incoder_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def read_cpp_file(uploaded_file):
    """Read the uploaded C++ file and return its content as a string."""
    return uploaded_file.read().decode("utf-8")

def convert_curl_to_java(cpp_code):
    """Convert CURL requests from C++ to Java HttpURLConnection."""
    # Convert the function signature
    cpp_code = re.sub(r'std::string makeRequest\((.*?)\)', 
                      r'public static String makeRequest(String url, String method, String data) throws IOException', 
                      cpp_code)

    # Replace CURL setup with HttpURLConnection
    cpp_code = re.sub(r'CURL\s*\*\s*curl;', 'HttpURLConnection connection;', cpp_code)
    cpp_code = re.sub(r'curl_easy_init\(\);', 'connection = (HttpURLConnection) new URL(url).openConnection();', cpp_code)
    cpp_code = re.sub(r'curl_easy_setopt\(curl, CURLOPT_URL, url.c_str\(\);\)', 'connection.setRequestMethod(method);', cpp_code)
    
    # Handling request headers
    cpp_code = re.sub(r'curl_easy_setopt\(curl, CURLOPT_HTTPHEADER, &headers\);', 
                      'connection.setRequestProperty("Content-Type", "application/json");\n'
                      'connection.setRequestProperty("Accept", "application/json");', cpp_code)
    
    # Handle data
    cpp_code = re.sub(r'if\s*\(!data.empty\(\)\)\s*{(.*?)}', 
                      'if (data != null && !data.isEmpty()) {\n'
                      '    connection.setDoOutput(true);\n'
                      '    try (OutputStream os = connection.getOutputStream()) {\n'
                      '        os.write(data.getBytes());\n'
                      '    }\n'
                      '}', cpp_code, flags=re.DOTALL)

    # Replace response handling
    cpp_code = re.sub(r'curl_easy_setopt\(curl, CURLOPT_WRITEDATA, &response\);', 
                      'BufferedReader in = new BufferedReader(new InputStreamReader(connection.getInputStream()));\n'
                      'StringBuilder response = new StringBuilder();\n'
                      'String inputLine;\n'
                      'while ((inputLine = in.readLine()) != null) {\n'
                      '    response.append(inputLine);\n'
                      '}\n'
                      'in.close();\n'
                      'return response.toString();', cpp_code)

    return cpp_code

def convert_json_to_java(cpp_code):
    """Convert JSON parsing from C++ to Java using Gson without field mapping."""
    cpp_code = re.sub(r'Json::Value root;\s*Json::Reader reader;', 
                      'JsonObject jsonObject;', cpp_code)
    cpp_code = re.sub(r'if\s*\(!reader.parse\((.*?)\);\)', 
                      'jsonObject = JsonParser.parseString("\1").getAsJsonObject();', cpp_code)
    
    # Keep user-defined object parsing flexible
    cpp_code = re.sub(r'User user;', 
                      'User user = new User();', cpp_code)

    # General structure for parsing JSON
    cpp_code = re.sub(r'// Handle parsing error', 
                      '// Handle parsing error if needed', cpp_code)

    return cpp_code

def convert_cpp_to_java(cpp_code):
    """Convert C++ code to Java code."""
    cpp_code = convert_curl_to_java(cpp_code)
    cpp_code = convert_json_to_java(cpp_code)

    # Clean up any unnecessary C++ specific constructs
    cpp_code = re.sub(r'#include.*\n', '', cpp_code)  # Remove includes
    cpp_code = re.sub(r'std::string', 'String', cpp_code)
    cpp_code = re.sub(r'std::cout', 'System.out.println', cpp_code)

    return cpp_code

def main():
    """Main function to run the Streamlit app."""
    st.title("C++ to Java Converter")

    uploaded_file = st.file_uploader("Choose a C++ file", type=["cpp", "h", "hpp"])
    
    if uploaded_file is not None:
        cpp_code = read_cpp_file(uploaded_file)
        st.subheader("Uploaded C++ Code:")
        st.code(cpp_code, language='cpp')

        # Convert C++ to Java
        if st.button("Convert C++ to Java"):
            try:
                java_code = convert_cpp_to_java(cpp_code)
                st.subheader("Generated Java Code:")
                st.code(java_code, language='java')

            except Exception as e:
                st.error(f"Error during conversion: {e}")

if __name__ == "__main__":
    main()
