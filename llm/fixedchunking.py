import json
import os
from sentence_transformers import SentenceTransformer
from unstructured.chunking.basic import chunk_elements
from unstructured.documents.elements import Text

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def load_cleaned_data(folder_path):
    text_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                data = json.load(file)
                for entry in data.get("text_data", []):
                    if "text" in entry:
                        text_data.append(entry["text"].strip())
    return text_data

def chunk_text(text_data, max_chars=1024, overlap=150):
    elements = [Text(text) for text in text_data if text] 
    return chunk_elements(elements, max_characters=max_chars, overlap=overlap)



cleaned_data_folder = "C:/Users/ACER/Desktop/PROPER/clean text"
chunked_output_folder = "C:/Users/ACER/Desktop/PROPER/chunked text"

if not os.path.exists(chunked_output_folder):
    os.makedirs(chunked_output_folder)

for filename in os.listdir(cleaned_data_folder):
    if filename.endswith(".json"):
        input_file = os.path.join(cleaned_data_folder, filename)
        output_file = os.path.join(chunked_output_folder, filename.replace(".json", "_chunked.json"))
        
        text_data = load_cleaned_data(cleaned_data_folder)
        chunks = chunk_text(text_data, max_chars=1024, overlap=150)
        
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump([{"text": chunk.text} for chunk in chunks], file, indent=4)
        
        print(f" Chunked data saved to: {output_file}")




