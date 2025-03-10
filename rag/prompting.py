def format_context(query_text, retrieved_chunks):
    if not retrieved_chunks:
        context = "No relevant information found."
    else:
        context = "\n\n".join(retrieved_chunks)

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
