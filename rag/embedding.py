import os
import chromadb
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------
# Load Embedding Model
# --------------------------------------------------------------
embed_model = SentenceTransformer("BAAI/bge-small-en")
MAX_TOKENS = 50  # BGE models have a 512-token limit

# --------------------------------------------------------------
# Read the Markdown (.md) File
# --------------------------------------------------------------
file_path = "C:/Users/ACER/Desktop/rag/output.md"

with open(file_path, "r", encoding="utf-8") as f:
    document_text = f.read()

# --------------------------------------------------------------
# Apply Chunking
# --------------------------------------------------------------
def chunk_text(text, max_tokens=MAX_TOKENS):
    sentences = sent_tokenize(text)
    chunks, current_chunk, current_length = [], [], 0

    for sentence in sentences:
        sentence_length = len(sentence.split())  # Approximate token count

        if current_length + sentence_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_length = [sentence], sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

chunks = chunk_text(document_text)

# --------------------------------------------------------------
# Create ChromaDB Database and Collection
# --------------------------------------------------------------
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="doc_chunks")

# --------------------------------------------------------------
# Embed and Store Chunks in ChromaDB
# --------------------------------------------------------------
for idx, chunk in enumerate(chunks):
    embedding = embed_model.encode(chunk).tolist()
    collection.add(
        ids=[str(idx)],
        embeddings=[embedding],
        documents=[chunk],  # ✅ Store the actual text!
        metadatas=[{"filename": os.path.basename(file_path), "chunk_index": idx}]
    )

print(f"✅ Successfully stored {len(chunks)} chunks in ChromaDB!")





