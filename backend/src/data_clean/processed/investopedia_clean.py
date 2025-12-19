import re
import json
import os
from glob import glob

# --- 核心清洗和切块函数（沿用并微调） ---

def clean_and_chunk_investopedia_text(text: str, file_name: str) -> list:
    """
    clean Investopedia text and chunk into RAG chunks
    
    Args:
        text: raw text。
        file_name: raw file name (to be used as metadata).
        
    Returns:
        list of RAG chunks
    """
    
    # 1. metadata
    title_match = re.search(r"Title:\s*(.+?)\n", text)
    title = title_match.group(1).strip() if title_match else file_name.replace('.txt', '')
    
    # remove Title line (Investopedia)
    cleaned_text = re.sub(r"Title:\s*.+?\n", "", text, 1)

    # 2. remove noise and clean up the text
    # remove images and links (Investopedia)
    cleaned_text = re.sub(r"Investopedia\s*/\s*.+", "", cleaned_text)
    # remove redundant lines or introductory text at the end
    cleaned_text = re.sub(r"\n\s*Understanding Form 10-K\s*$", "", cleaned_text) 
    
    # standardize formatting errors and unnecessary line breaks
    cleaned_text = cleaned_text.replace("with theU.S.", "with the U.S.")
    cleaned_text = re.sub(r"(\r\n|\r|\n){3,}", "\n\n", cleaned_text).strip()
    
    # 3. Semantic-based chunking
    
    # try to split the text into Investopedia sections
    # match "Title: X" or "X?" as the semantic boundaries
    sections = re.split(r"(What Is Form .+?\?|Key Takeaways|Understanding .+?\?|.+?:\s*\n)", cleaned_text)
    
    structured_chunks = []
    current_section_title = "Introduction"
    
    for part in sections:
        part = part.strip()
        if not part:
            continue
            
        # Identify new paragraph titles
        if re.match(r"(What Is Form .+?\?|Key Takeaways|Understanding .+?\?|.+?:)", part):
            current_section_title = part.strip(':').strip()
            continue

        # Treat each paragraph (separated by double newline characters) as a chunk
        for paragraph in part.split('\n\n'):
            paragraph = paragraph.strip()
            if paragraph:
                # construct RAG chunk
                chunk = {
                    "document_title": title,
                    "section_title": current_section_title,
                    "text": paragraph,
                    "source_file": file_name,
                    "source_path": os.path.join(INPUT_DIR, file_name)
                }
                structured_chunks.append(chunk)

    return structured_chunks

# --- main process ---

def process_files(input_dir: str, output_dir: str):
    """
    Traverse all .txt files in the input directory, clean and chunk the content, 
    and write the results to the output directory.
    """
    print(f"--- Starting to process Investopedia files ---")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ensure the output directory exists: {output_dir}")

    # Find all .txt files in the input directory
    file_paths = glob(os.path.join(input_dir, "*.txt"))
    
    if not file_paths:
        print(f"Warning: No .txt files found in directory {input_dir}.")
        return

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        print(f"Processing file: {file_name}")
        
        try:
            # 1. Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            # 2. Clean and chunk the content
            rag_chunks = clean_and_chunk_investopedia_text(raw_text, file_name)
            
            # 3. Write to the output file
            output_file_name = file_name.replace('.txt', '_processed.json')
            output_file_path = os.path.join(output_dir, output_file_name)
            
            # Write processed data in JSON format
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(rag_chunks, f, indent=4, ensure_ascii=False)
                
            print(f"Successfully wrote {len(rag_chunks)} chunks to: {output_file_path}")
            
        except Exception as e:
            print(f"Error: An error occurred while processing file {file_name}: {e}")
            
    print(f"--- Finished processing Investopedia files ---")

# --- main ---
if __name__ == "__main__":
    # --- config paths ---
    INPUT_DIR = 'backend/data/raw/investopedia/'
    OUTPUT_DIR = 'backend/data/processed/investopedia/'
    
    
    process_files(INPUT_DIR, OUTPUT_DIR)
    print("\n--- all files processed ---")