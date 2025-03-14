from docling.document_converter import DocumentConverter

pdf_path = "C:/Users/ACER/Desktop/clone/Sowing-Advisory/rag/KharifAgroAdvisoryForFarmers (2).pdf"  

converter = DocumentConverter()

# Convert the local PDF file
result = converter.convert(pdf_path)

document = result.document
markdown_output = document.export_to_markdown()

# Save the extracted Markdown content to a file
output_file = "C:/Users/ACER/Desktop/clone/Sowing-Advisory/rag/output.md"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(markdown_output)

print(f"Markdown file saved as {output_file}")
