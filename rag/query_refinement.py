from embedding import *
def query_chromadb(query_text, top_k=3):
    query_embedding = embed_model.encode(query_text).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

   
    if not results.get("documents") or not results["documents"][0]:
        print(" No relevant chunks found.")
        return None

    
    retrieved_chunks = results["documents"][0]
    metadata = results.get("metadatas", [[]])[0]  

    print(f"\n🔍 **Query:** {query_text}\n")
    retrieved_data = []
    for i, (chunk, meta) in enumerate(zip(retrieved_chunks, metadata)):
        print(f"🔹 **Result {i+1}:**")
        print(chunk)
        print(f"Metadata: {meta}\n")
        retrieved_data.append({"text": chunk, "metadata": meta})

    return retrieved_data  
