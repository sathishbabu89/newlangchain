import logging
import gradio as gr
from PyPDF2 import PdfReader
import pdfplumber
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUGGINGFACE_API_TOKEN = ""

# Initialize session state equivalent
conversation_history = []
vector_store = None

def process_pdf(file):
    global vector_store
    try:
        # Extract text from the PDF file
        pdf_reader = PdfReader(file)
        TEXT = ""
        for page in pdf_reader.pages:
            TEXT += page.extract_text() or ""

        if not TEXT.strip():
            return "No text found in the uploaded PDF."

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
        chunks = text_splitter.split_text(TEXT)

        # Use HuggingFaceEmbeddings to embed text and create a vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(chunks, embeddings)
        return "Document processed successfully!"
    except Exception as e:
        logger.error(f"An error occurred while processing the document: {e}")
        return str(e)

def chat_with_ai(user_question, vector_store):
    try:
        # Retrieve relevant documents from vector store
        retriever = vector_store.similarity_search(user_question)
        if not retriever:
            return "No relevant information found."

        # Initialize the model for QA
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
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.invoke(
            {"input_documents": retriever, "question": user_question})
        answer = response['output_text'].split("Helpful Answer:")[-1].strip()
        return answer
    except Exception as e:
        logger.error(f"An error occurred while processing the question: {e}")
        return str(e)

def handle_pdf_upload(file):
    return process_pdf(file)

def handle_user_query(user_question):
    global vector_store
    if not user_question.strip():
        return "Please enter a valid question."
    
    if vector_store is None:
        return "Please upload and process a document before asking questions."
    
    return chat_with_ai(user_question, vector_store)

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("""
    <style>
        /* Custom Dark Theme */
        .gradio-container { background-color: #121212; color: #ffffff; font-family: 'Helvetica Neue', sans-serif; }
        .gradio-markdown { font-size: 1.5rem; font-weight: bold; text-align: center; }
        
        /* Chat Message Styling */
        .gr-chat-message-user { 
            background-color: #2a9d8f; 
            color: #ffffff; 
            border-radius: 12px; 
            padding: 12px; 
            margin-bottom: 8px;
            max-width: 75%;
            word-wrap: break-word;
        }
        
        .gr-chat-message-assistant { 
            background-color: #264653; 
            color: #ffffff; 
            border-radius: 12px; 
            padding: 12px; 
            margin-bottom: 8px;
            max-width: 75%;
            word-wrap: break-word;
        }

        /* File Upload Section */
        .gr-file { 
            background-color: #264653; 
            color: #ffffff; 
            border-radius: 8px; 
            padding: 15px; 
        }

        /* Input Fields */
        .gr-textbox { 
            border-radius: 12px; 
            padding: 15px; 
            background-color: #2a9d8f; 
            color: white; 
        }

        .gr-button { 
            border-radius: 12px; 
            padding: 12px; 
            background-color: #2a9d8f; 
            color: white;
            border: none;
        }
    </style>
    """, unsafe_allow_html=True)
    
    gr.Markdown("<h1 style='color:white;'>Document Assistant Tool 📚</h1>")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            output_pdf = gr.Textbox(label="PDF Processing Output", lines=2)
            file_input.change(fn=handle_pdf_upload, inputs=file_input, outputs=output_pdf)

        with gr.Column():
            conversation_output = gr.Chatbot(label="Conversation History", elem_id="chatbox")
            user_input = gr.Textbox(placeholder="Type your question here", label="Your Question", elem_id="user_input")
            user_input.submit(fn=handle_user_query, inputs=user_input, outputs=conversation_output)

    demo.launch(share=True)
