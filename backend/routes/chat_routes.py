from flask import Blueprint, request, jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

chat_bp = Blueprint("chat", __name__)

# Global variables for RAG
vectorstore = None
embeddings = None

def init_rag():
    global vectorstore, embeddings
    try:
        kb_path = "/Users/apple/Desktop/Enigma_24/rag chatbot/KB.pdf"
        index_path = "/Users/apple/Desktop/Enigma_24/backend/faiss_index_backend"
        
        # Use HuggingFace embeddings (free, no API limits)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        if os.path.exists(index_path):
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        elif os.path.exists(kb_path):
            loader = PyPDFLoader(kb_path)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(data)
            vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
            vectorstore.save_local(index_path)
        else:
            print("Warning: Knowledge base (KB.pdf) not found.")
            return
        
        print("✅ FAISS Vectorstore Initialized successfully.")
    except Exception as e:
        print(f"❌ RAG Initialization error: {e}")

# Initialize RAG on startup
init_rag()

@chat_bp.route("/chat", methods=["POST"])
def chat():
    global vectorstore
    try:
        data = request.json
        message = data.get("message", "")
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        # Configure Gemini
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel('gemini-flash-latest')
        
        context = ""
        # Get context from FAISS if available
        if vectorstore:
            docs = vectorstore.similarity_search(message, k=3)
            context = "\n".join([doc.page_content for doc in docs])
        
        system_prompt = (
            "You are TradeLens AI, a helpful stock market assistant. "
            "Use the provided context to answer the user's question. "
            "If the answer isn't in the context, use your general knowledge but mention it's outside the direct TradeLens docs. "
            "Keep responses concise (3-4 sentences). Be professional."
        )
        
        full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nUser Question: {message}"
        
        response = model.generate_content(full_prompt)
        answer = response.text

        return jsonify({
            "response": answer,
            "status": "success"
        })
        
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({
            "error": str(e),
            "response": "Sorry, I'm having trouble responding right now. Please try again.",
            "status": "error"
        }), 500
