import os
from openai import AzureOpenAI 
from prompting  import *
from query_refinement import *

# Azure OpenAI Credentials  
endpoint = os.getenv("ENDPOINT_URL", "https://dines-m7lp6cc0-japaneast.openai.azure.com/")  
deployment = os.getenv("DEPLOYMENT_NAME", "cutn2-gpt4o")  
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", )  

# Initialize Azure OpenAI Client  
client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key,  
    api_version="2024-05-01-preview",
)

def generate_gpt4o_response(query_text, retrieved_chunks):
    """ Uses retrieved chunks as context and queries GPT-4o for response. """
    

    extracted_chunks = [chunk if isinstance(chunk, str) else chunk.get("text", "") for chunk in retrieved_chunks]

    extracted_chunks = [chunk for chunk in extracted_chunks if chunk]  

    if not extracted_chunks:
        print("❌ No valid text chunks found in retrieval.")
        return None

    prompt = format_context(query_text, extracted_chunks)  

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



query = "recommand the variety of millets during kharif season in Tamilnadu?"


    
retrieved_chunks = query_chromadb(query) 

if retrieved_chunks:
    answer = generate_gpt4o_response(query, retrieved_chunks)
    print("\n💡 GPT-4o Response:\n", answer)
else:
    print(" No relevant documents found.")





