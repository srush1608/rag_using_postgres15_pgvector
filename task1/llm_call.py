# llm_call.py

import faiss
import numpy as np
from db import get_session  # Import the get_session function
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import requests
from sqlalchemy import text  # Import text for raw SQL

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize model and FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384
index = faiss.IndexFlatL2(dimension)

def get_relevant_documents(query):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=1)
    return I[0]

def query_llm_with_retrieval(query):
    relevant_idx = get_relevant_documents(query)
    
    # Open a session using get_session() directly
    db_session = get_session()
    try:
        # Use text() to explicitly mark the query as a raw SQL expression
        result = db_session.execute(text("SELECT text FROM embeddings WHERE id = :id"), {'id': relevant_idx[0]}).fetchone()
        retrieved_text = result[0] if result else "No relevant document found."
    finally:
        db_session.close()  # Ensure the session is closed after use

    prompt = f"{retrieved_text}\n\nUser Query: {query}\n\nAnswer based on the text above:"
    
    # Send to Groq API
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    data = {
        "prompt": prompt
    }
    
    response = requests.post("https://api.groq.com/v1/completions", headers=headers, json=data)
    return response.json().get("choices")[0].get("text")
