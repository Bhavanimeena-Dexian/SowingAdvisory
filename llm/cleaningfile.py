import json
import os
from unstructured.cleaners.core import clean

def clean_text(text):
    return clean(
        text,
        bullets=True,
        extra_whitespace=True,
        dashes=True,
        trailing_punctuation=True,
        lowercase=True
    )

def filter_metadata(entry):
    #remove unnecessary metadata
    return {"text": entry["text"]} if "text" in entry else None

def remove_duplicates(data):
   
    seen = set()
    unique_data = []
    for entry in data:
        text = entry.get("text", "").strip()
        if text and text not in seen:
            seen.add(text)
            unique_data.append(entry)
    return unique_data

def merge_fragments(data):
    
    merged_data = []
    temp_text = ""
    for entry in data:
        text = entry.get("text", "").strip()
        if text:
            if temp_text:
                temp_text += " " + text
            else:
                temp_text = text
        else:
            if temp_text:
                merged_data.append({"text": temp_text})
                temp_text = ""
    if temp_text:
        merged_data.append({"text": temp_text})
    return merged_data

def extract_tables(data):
    tables = [entry for entry in data if entry.get("type") == "Table"]
    return tables

def process_json_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            with open(input_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            
            data = [filter_metadata(entry) for entry in data if filter_metadata(entry)]
            data = remove_duplicates(data)
            data = merge_fragments(data)
            tables = extract_tables(data)
            
            cleaned_data = {"text_data": data, "tables": tables}
            
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(cleaned_data, file, indent=4)
            
            print(f"Processed and saved: {output_path}")


input_folder = "C:/Users/ACER/Desktop/PROPER/extracted text"  
output_folder = "C:/Users/ACER/Desktop/PROPER/clean text"
process_json_folder(input_folder, output_folder)
