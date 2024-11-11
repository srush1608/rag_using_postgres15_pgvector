from embedding_generator import EmbeddingGenerator
from database import DatabaseManager
import os
from PyPDF2 import PdfReader

DATA_FILE_PATH = 'documents.pdf'
is_pdf = True

def extract_text_from_pdf(file_path):
    """Function to extract text from a PDF file."""
    text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text.replace("\x00", "")  # Remove NUL characters if any
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def load_text_samples(file_path, is_pdf=False):
    texts = []
    if is_pdf:
        texts.append(extract_text_from_pdf(file_path))
    else:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file '{file_path}' not found.")
        with open(file_path, 'r') as file:
            texts = list(dict.fromkeys(filter(None, file.read().splitlines())))
    return texts

def generate_augmented_response(query, retrieved_items, max_context_length=1000):
    context = " ".join(text for text, _ in retrieved_items[:3])  # Only take top 3 documents
    if len(context) > max_context_length:
        context = context[:max_context_length]  # Truncate context if too long
    
    # Simulate a response generation (replace with actual model call)
    response = f"Response based on context: {context} for query: {query}"
    return response

def main():
    db_manager = DatabaseManager()
    embedding_gen = EmbeddingGenerator()

    # Load .pdf document data
    texts = load_text_samples(DATA_FILE_PATH, is_pdf=is_pdf)

    for idx, text in enumerate(texts):
        embedding = embedding_gen.generate_embedding(text)
        if embedding is not None:
            db_manager.add_embedding_to_db(embedding, text_id=str(idx), text_content=text)

    print("Embeddings added to the database.")
    
    # Loop to continuously prompt user for queries
    while True:
        query_text = input("Enter your query (or type 'stop' to finish): ")
        if query_text.lower() == 'stop':
            print("Stopping the query input...")
            break

        query_embedding = embedding_gen.generate_embedding(query_text)

        # Search for similar items in the database
        similar_items = db_manager.search_similar_vectors(query_embedding, top_k=3)

        # Generate the augmented response
        response = generate_augmented_response(query_text, similar_items)

        # Print the generated response
        print("Response:", response)

        # Save the query and response to the database
        db_manager.save_query_response(query_text, response)

    db_manager.close()

if __name__ == "__main__":
    main()