# Commands

1. pip install pipenv
2. pipenv shell
3. pipenv --venv
4. pipenv install langchain
5. pipenv install langchain-community
6. pipenv install langchain-core
7. pipenv install streamlit pypdf2 faiss-cpu langchain langchain-community langchain-core pdfplumber
8. pipenv graph
9. pip freeze > requirements.txt
10. pipenv install -r requirements.txt

# Added following features in app4.py file

1. Multiple PDF Uploads: You can now upload and process multiple PDF files.
2. Document Summarization: After processing, the tool provides a brief summary of the document.
3. Improved Question-Answer Flow: The tool allows conversation with a chat-like interface and collects user feedback on answers.
4. Search Feature: Users can search specific keywords within the uploaded documents.
