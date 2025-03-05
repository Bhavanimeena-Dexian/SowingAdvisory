from unstructured.partition.pdf import partition_pdf

pdf_path = "C:/Users/ACER/Desktop/PROPER/AGRICULTURE.pdf"


elements = partition_pdf(
    filename=pdf_path,
    strategy="hi_res",  
    extract_images_in_pdf=True,  
    extract_image_block_types=["Image", "Table"],  
    extract_image_block_output_dir="path/to/save/images"  
)
for element in elements:
    print(element.to_dict())  

    import json

output_path = "C:/Users/ACER/Desktop/PROPER/extracted_data.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump([element.to_dict() for element in elements], f, ensure_ascii=False, indent=4)

print(f"Extracted data saved to {output_path}")

# used same code for all other pdf's