# main.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from db import store_document
import llm_call

# Initialize model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Knowledge base (you can expand this list as needed)
documents = [
    "The Eiffel Tower is located in Paris, France.",
    "The Great Wall of China stretches across northern China.",
    "The Statue of Liberty is in New York, USA.",
    "The Colosseum is an ancient amphitheater in Rome, Italy.",
    "Machu Picchu is an Incan citadel in Peru."
]

# Generate embeddings and store document text and embeddings as JSON in DB
dimension = 384  # Embedding dimension for 'all-MiniLM-L6-v2'
index = faiss.IndexFlatL2(dimension)

# Store document embeddings
for doc in documents:
    embedding = model.encode([doc])[0]
    store_document(doc, embedding)  # Store document text and JSON-encoded embedding in PostgreSQL
    index.add(np.array([embedding]).astype('float32'))  # Store embeddings in FAISS

# Ask for user query
user_query = input("Please enter your query: ")

# Get generated response based on the user query
response = llm_call.query_llm_with_retrieval(user_query)

print(f"Generated Response: {response}")
