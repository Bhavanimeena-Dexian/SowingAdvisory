import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model for text similarity search
embed_model = SentenceTransformer("BAAI/bge-small-en")

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="doc_chunks")

def search_query(query, top_k=5):
    """
    Searches for relevant text chunks in ChromaDB using the query.
    
    Args:
        query (str): The user's search query.
        top_k (int): Number of results to return.

    Returns:
        list: List of retrieved document chunks.
    """
    query_embedding = embed_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    retrieved_chunks = []
    for i in range(top_k):
        chunk = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        retrieved_chunks.append({
            "document": chunk,
            "filename": metadata["filename"],
            "chunk_index": metadata["chunk_index"]
        })

    return retrieved_chunks
