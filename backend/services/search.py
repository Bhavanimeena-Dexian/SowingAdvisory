import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
embed_model = SentenceTransformer("BAAI/bge-small-en")

# Ensure ChromaDB uses the correct database
chroma_client = chromadb.PersistentClient(path="backend/COMBINE/chroma_db")
collection = chroma_client.get_or_create_collection(name="doc_chunks")  # ✅ Ensure it exists

def search_query(query, top_k=5):
    """
    Searches for relevant text chunks in ChromaDB using the query.
    """
    if collection.count() == 0:
        print("⚠️ No chunks found in the collection!")
        return []

    query_embedding = embed_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    retrieved_chunks = []
    for i in range(len(results["documents"][0])):
        chunk = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        retrieved_chunks.append({
            "document": chunk,
            "filename": metadata["filename"],
            "chunk_index": metadata["chunk_index"]
        })

    return retrieved_chunks
