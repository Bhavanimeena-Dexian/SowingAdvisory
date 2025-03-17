import os
from openai import AzureOpenAI 
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI Credentials
endpoint = os.getenv("ENDPOINT_URL", "https://dines-m7lp6cc0-japaneast.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "cutn2-gpt4o")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY" ,  "3fodGsX48w3Ln4oFdVjDZxN56MrnzM352n7wYvQTXEjVSoV0PdMfJQQJ99BBACi0881XJ3w3AAAAACOGSoaO")

# Initialize Azure OpenAI Client
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2024-05-01-preview",
)

# Initialize embedding model and ChromaDB
embed_model = SentenceTransformer("BAAI/bge-small-en")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="doc_chunks")

# Function to search for relevant chunks
def search_query(query, top_k=5):
    query_embedding = embed_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    retrieved_chunks = []
    for i in range(top_k):
        chunk = results['documents'][0][i]
        metadata = results['metadatas'][0][i]
        retrieved_chunks.append({"document": chunk, "filename": metadata['filename'], "chunk_index": metadata['chunk_index']})

    return retrieved_chunks

# Function to format the prompt
def format_context(query_text, retrieved_chunks):
    extracted_chunks = [chunk if isinstance(chunk, str) else chunk.get("document", "") for chunk in retrieved_chunks]

    extracted_chunks = [chunk for chunk in extracted_chunks if chunk]  

    if not extracted_chunks:
        return "No relevant information found."

    context = "\n\n".join(extracted_chunks)

    prompt = f"""
    You are a helpful and knowledgeable assistant. Answer the user's question **only** using the provided retrieved information. 

    If the answer **is not found** in the retrieved information, respond with "I don't know." Do not make up an answer. 

    **Relevant Information from Retrieved Documents:**  
    {context}  

    **User Query:**  
    {query_text}  

    **Answer:**
    """
    return prompt

# Function to generate GPT-4o response
def generate_gpt4o_response(query_text, retrieved_chunks):
    prompt = format_context(query_text, retrieved_chunks)

    chat_prompt = [
        {"role": "system", "content": "You are an AI assistant that helps people find information."},
        {"role": "user", "content": prompt},
    ]

    completion = client.chat.completions.create(
        model=deployment,
        messages=chat_prompt,
        max_tokens=800,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False
    )

    return completion.choices[0].message.content

# Example usage
if __name__ == "__main__":
    query = "how can i get the better yield of rice in tamilnadu in monsoon season " \
    "?"
    retrieved_chunks = search_query(query)

    if retrieved_chunks:
        answer = generate_gpt4o_response(query, retrieved_chunks)
        print("\n💡 GPT-4o Response:\n", answer)
    else:
        print(" No relevant documents found.")
