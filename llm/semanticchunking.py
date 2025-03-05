import json
import os
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
nltk.download("punkt")
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

def semantic_chunking(text_data, num_clusters=10):
    sentences = [sent for text in text_data for sent in sent_tokenize(text)]
    embeddings = model.encode(sentences)
    clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='euclidean', linkage='ward')
    cluster_labels = clustering.fit_predict(embeddings)
    clustered_chunks = {}
    for i, label in enumerate(cluster_labels):
        if label not in clustered_chunks:
            clustered_chunks[label] = []
        clustered_chunks[label].append(sentences[i])
    chunks = [{"text": " ".join(clustered_chunks[label])} for label in clustered_chunks]
    return chunks
cleaned_data_folder = "C:/Users/ACER/Desktop/PROPER/clean text"
chunked_output_folder = "C:/Users/ACER/Desktop/PROPER/chunked2 text"

if not os.path.exists(chunked_output_folder):
    os.makedirs(chunked_output_folder)

for filename in os.listdir(cleaned_data_folder):
    if filename.endswith(".json"):
        input_file = os.path.join(cleaned_data_folder, filename)
        output_file = os.path.join(chunked_output_folder, filename.replace(".json", "_chunked.json"))
        
        text_data = load_cleaned_data(cleaned_data_folder)
        chunks = semantic_chunking(text_data, num_clusters=10)  
        
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(chunks, file, indent=4)
        
        print("Semantic chunks saved to: {output_file}")
