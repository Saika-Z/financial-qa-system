import re
import json
import os
from glob import glob

# --- 核心清洗和切块函数（沿用并微调） ---

def clean_and_chunk_investopedia_text(text: str, file_name: str) -> list:
    """
    清洗 Investopedia 文本，提取元数据，并进行基于段落的切块。
    
    Args:
        text: 原始文本内容。
        file_name: 原始文件的名称（用于元数据）。
        
    Returns:
        结构化的 RAG chunks 列表。
    """
    
    # 1. 元数据提取 (Title)
    title_match = re.search(r"Title:\s*(.+?)\n", text)
    title = title_match.group(1).strip() if title_match else file_name.replace('.txt', '')
    
    # 移除 Title 行，准备进行后续处理
    cleaned_text = re.sub(r"Title:\s*.+?\n", "", text, 1)

    # 2. 清除噪音和非内容元素
    # 清除图片引用/归属行 (针对 Investopedia 的通用清洗)
    cleaned_text = re.sub(r"Investopedia\s*/\s*.+", "", cleaned_text)
    # 清除末尾的冗余标题或引导语
    cleaned_text = re.sub(r"\n\s*Understanding Form 10-K\s*$", "", cleaned_text) 
    
    # 标准化排版错误和多余换行
    cleaned_text = cleaned_text.replace("with theU.S.", "with the U.S.")
    cleaned_text = re.sub(r"(\r\n|\r|\n){3,}", "\n\n", cleaned_text).strip()
    
    # 3. 基于语义的切块 (Chunking)
    
    # 尝试按常见的 Investopedia 结构标题进行分割
    # 匹配 "Title: X" 或 "X?" 形式的标题，作为切块的语义边界
    sections = re.split(r"(What Is Form .+?\?|Key Takeaways|Understanding .+?\?|.+?:\s*\n)", cleaned_text)
    
    structured_chunks = []
    current_section_title = "Introduction"
    
    for part in sections:
        part = part.strip()
        if not part:
            continue
            
        # 识别新的段落标题
        if re.match(r"(What Is Form .+?\?|Key Takeaways|Understanding .+?\?|.+?:)", part):
            current_section_title = part.strip(':').strip()
            continue

        # 将每个段落（以双换行符分割）作为一个 chunk
        for paragraph in part.split('\n\n'):
            paragraph = paragraph.strip()
            if paragraph:
                # 构造 RAG 友好的数据结构
                chunk = {
                    "document_title": title,
                    "section_title": current_section_title,
                    "text": paragraph,
                    "source_file": file_name,
                    "source_path": os.path.join(INPUT_DIR, file_name)
                }
                structured_chunks.append(chunk)

    return structured_chunks

# --- 主流程函数 ---

def process_files(input_dir: str, output_dir: str):
    """
    遍历输入目录下的所有 .txt 文件，进行清洗和切块，并将结果写入输出目录。
    """
    print(f"--- 开始处理 Investopedia 文件 ---")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"确保输出目录存在: {output_dir}")

    # 查找输入目录下所有的 .txt 文件
    file_paths = glob(os.path.join(input_dir, "*.txt"))
    
    if not file_paths:
        print(f"警告：在目录 {input_dir} 中未找到任何 .txt 文件。")
        return

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        print(f"正在处理文件: {file_name}")
        
        try:
            # 1. 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            # 2. 清洗和切块
            rag_chunks = clean_and_chunk_investopedia_text(raw_text, file_name)
            
            # 3. 写入输出文件
            output_file_name = file_name.replace('.txt', '_processed.json')
            output_file_path = os.path.join(output_dir, output_file_name)
            
            # 以 JSON 格式写入处理后的数据
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(rag_chunks, f, indent=4, ensure_ascii=False)
                
            print(f"成功写入 {len(rag_chunks)} 个数据块到: {output_file_path}")
            
        except Exception as e:
            print(f"错误：处理文件 {file_name} 时发生错误: {e}")

# --- 运行主函数 ---
if __name__ == "__main__":
    # --- 配置路径 ---
    INPUT_DIR = 'backend/data/raw/investopedia/'
    OUTPUT_DIR = 'backend/data/processed/investopedia/'
    
    
    process_files(INPUT_DIR, OUTPUT_DIR)
    print("\n--- 所有文件处理完成 ---")