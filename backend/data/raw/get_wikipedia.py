import wikipediaapi
import re
import os

def fetch_and_clean_wiki_content(concept, user_agent):
    wiki = wikipediaapi.Wikipedia('en', user_agent=user_agent)
    page = wiki.page(concept.replace(' ', '_'))
    
    if not page.exists():
        return None
    
    # 获取原始文本
    raw_text = page.text
    
    # 清理：移除括号中的内容（通常是引用、注释或不必要的上下文）
    cleaned_text = re.sub(r'\([^()]*?\)', '', raw_text)
    
    # 进一步清理：移除多余的空行和特殊标记
    cleaned_text = '\n'.join(line.strip() for line in cleaned_text.splitlines() if line.strip())
    
    return cleaned_text


if __name__ == "__main__":
    # 示例概念列表
    financial_concepts = ["Free cash flow", "Price-to-earnings ratio", "Balance sheet", "Amortization"]
    user_agent = 'FinanceQASystem/1.0 (your.email@example.com)'

    # 将所有知识库内容存放在一个总文件夹中
    rag_data_dir = "backend/data/raw/wikipedia"
    if not os.path.exists(rag_data_dir):
        os.makedirs(rag_data_dir)

    print("--- 开始下载 Wikipedia 金融概念 ---")
    for concept in financial_concepts:
        content = fetch_and_clean_wiki_content(concept,user_agent)
        if content:
            # 将每个概念保存为一个文本文件
            file_name = f"{concept.replace(' ', '_')}.txt"
            file_path = os.path.join(rag_data_dir, file_name)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 成功保存概念：{concept}")
        else:
            print(f"❌ 未找到页面：{concept}")

    print("--- Wikipedia 数据准备完成 ---")