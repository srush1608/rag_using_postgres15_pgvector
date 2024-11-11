# db.py

import os
import json
import psycopg2
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# Initialize the database engine and session
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
metadata = MetaData()

# Function to create a new session
def get_session():
    return SessionLocal()

def store_document(text, embedding):
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
    )
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            text TEXT,
            embedding JSONB
        );
    """)
    conn.commit()

    # Convert numpy array to a list of Python floats
    embedding_as_list = [float(val) for val in embedding]
    
    # Insert document text and embedding as JSON
    cur.execute("INSERT INTO embeddings (text, embedding) VALUES (%s, %s)", (text, json.dumps(embedding_as_list)))
    conn.commit()
    cur.close()
    conn.close()
