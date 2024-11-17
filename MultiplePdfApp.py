import logging
import gradio as gr
from PyPDF2 import PdfReader
import pdfplumber
from PIL import Image
import io
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings

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
            return "No text found in the uploaded PDF.", None

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
        chunks = text_splitter.split_text(TEXT)

        # Use HuggingFaceEmbeddings to embed text and create a vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(chunks, embeddings)

        # Generate a preview image from the first page of the PDF
        with pdfplumber.open(file) as pdf:
            first_page = pdf.pages[0]
            preview_image = first_page.to_image()
            img_byte_arr = io.BytesIO()
            preview_image.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            return "Document processed successfully!", img_byte_arr.getvalue()

    except Exception as e:
        logger.error(f"An error occurred while processing the document: {e}")
        return str(e), None

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
    # Clear previous state
    global vector_store
    vector_store = None

    status, preview = process_pdf(file)
    
    return status, preview

def handle_user_query(user_question):
    global vector_store
    if not user_question.strip():
        return "Please enter a valid question."
    
    if vector_store is None:
        return "Please upload and process a document before asking questions."

    answer = chat_with_ai(user_question, vector_store)

    conversation_history.append(('user', user_question))
    conversation_history.append(('assistant', answer))
    
    return conversation_history

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("""
    <style>
        /* Background and Layout */
        .gradio-container { 
            background: linear-gradient(145deg, #2e3440, #3b4252); 
            border-radius: 16px; 
            padding: 20px;
            color: #e5e9f0;
            font-family: 'Arial', sans-serif;
        }

        /* Header */
        .gr-markdown h1 {
            font-size: 2rem;
            color: #88c0d0;
            font-weight: bold;
            text-align: center;
        }

        /* File Upload Styling */
        .gr-file {
            background-color: #434c5e; 
            color: #e5e9f0; 
            border-radius: 12px;
            padding: 20px;
            transition: 0.3s ease;
        }
        .gr-file:hover {
            background-color: #4c566a;
        }

        /* Textboxes */
        .gr-textbox {
            background-color: #3b4252;
            color: #e5e9f0;
            border-radius: 12px;
            padding: 15px;
            border: none;
        }
        .gr-textbox:focus {
            border-color: #88c0d0;
            outline: none;
        }

        /* Buttons */
        .gr-button {
            background-color: #88c0d0;
            color: white;
            border-radius: 12px;
            padding: 12px 18px;
            border: none;
            cursor: pointer;
            transition: 0.3s ease;
        }
        .gr-button:hover {
            background-color: #81a1c1;
        }

        /* Chat Message Styling */
        .gr-chat-message-user { 
            background-color: #88c0d0; 
            color: #2e3440; 
            border-radius: 15px; 
            padding: 12px; 
            margin-bottom: 8px;
            max-width: 75%;
            word-wrap: break-word;
        }
        .gr-chat-message-assistant { 
            background-color: #4c566a; 
            color: #e5e9f0; 
            border-radius: 15px; 
            padding: 12px; 
            margin-bottom: 8px;
            max-width: 75%;
            word-wrap: break-word;
        }

        /* Chat Container Styling */
        .gr-chatbot {
            max-height: 500px;
            overflow-y: scroll;
            padding: 15px;
            border-radius: 16px;
            background-color: #2e3440;
        }

        /* Input Field */
        .gr-textbox {
            background-color: #434c5e;
            color: #e5e9f0;
            border-radius: 12px;
            padding: 15px;
            border: none;
        }
    </style>
    """)

    gr.Markdown("<h1>Document Assistant Tool 📚</h1>")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])  # Single file upload
            output_pdf = gr.Textbox(label="PDF Processing Output", lines=2)
            pdf_preview = gr.Image(label="PDF Preview", show_label=False)
            file_input.change(fn=handle_pdf_upload, inputs=file_input, outputs=[output_pdf, pdf_preview])

        with gr.Column():
            conversation_output = gr.Chatbot(label="Conversation History", elem_id="chatbox")
            user_input = gr.Textbox(placeholder="Type your question here", label="Your Question", elem_id="user_input")
            user_input.submit(fn=handle_user_query, inputs=user_input, outputs=conversation_output)

    demo.launch(share=True)
