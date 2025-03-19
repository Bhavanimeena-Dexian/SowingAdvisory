from openai import AzureOpenAI
from utils.env_loader import ENDPOINT_URL, DEPLOYMENT_NAME, AZURE_OPENAI_API_KEY
from services.search import search_query

# Initialize Azure OpenAI Client
client = AzureOpenAI(
    azure_endpoint=ENDPOINT_URL,
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-05-01-preview",
)

def generate_gpt_response(query_text):
    """
    Generates a response using GPT-4o based on retrieved document context.

    Args:
        query_text (str): The user's question.

    Returns:
        str: GPT-4o generated response.
    """
    retrieved_chunks = search_query(query_text)

    # Format prompt with retrieved context
    if not retrieved_chunks:
        return "I don't know."

    context = "\n\n".join([chunk["document"] for chunk in retrieved_chunks])

    prompt = f"""
    You are a knowledgeable AI assistant. Answer the user's query **only** using the provided retrieved information. 

    If the answer **is not found** in the retrieved information, respond with "I don't know." 

    **Relevant Information from Retrieved Documents:**  
    {context}  

    **User Query:**  
    {query_text}  

    **Answer:**
    """

    # Send query to GPT-4o
    chat_prompt = [
        {"role": "system", "content": "You are an AI assistant that helps find information."},
        {"role": "user", "content": prompt},
    ]

    completion = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=chat_prompt,
        max_tokens=800,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stream=False
    )

    return completion.choices[0].message.content
