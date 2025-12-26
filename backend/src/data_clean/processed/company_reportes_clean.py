from bs4 import BeautifulSoup
import re
import os

def clean_with_beautifulsoup(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 1. 提取 <TEXT> 标签（兼容大小写）
        text_match = re.search(r'<(TEXT|text)>(.*?)</\1>', content, re.DOTALL)
        html_content = text_match.group(2) if text_match else content

        soup = BeautifulSoup(html_content, 'html.parser')

        # 2. 【优化】去除真正的噪声标签（不含正文内容的）
        for element in soup(['title', 'meta', 'head', 'script', 'style', 'header', 'footer']):
            element.decompose()

        # 3. 【核心修改】处理 XBRL 标签
        # 不要 decompose，而是使用 unwrap()。这会去掉标签但保留里面的文字。
        # 例如 <ix:nonNumeric>None.</ix:nonNumeric> 变成 None.
        for xbrl_tag in soup.find_all(re.compile(r'^(ix|xbrl|dei|us-gaap|srt|country|aapl|xbrli|xlink):.*', re.IGNORECASE)):
            xbrl_tag.unwrap() 

        # 4. 表格处理
        # 很多财报数据在表格里，直接 decompose 会丢失数据。
        # 建议先将表格转为简单的文本，或者如果确实不要表格数据，再 decompose。
        # 这里我们保留文字，只去掉结构
        for table in soup.find_all('table'):
            # 如果你确实不想看复杂的表格数据，可以用空格替换它，但不要删掉里面的字
            table.insert_after(soup.new_string(" [Table Data] "))
            table.unwrap()

        # 5. 获取纯文本
        # 使用 separator=' ' 确保 <div> 之间有空格，防止单词粘在一起
        raw_text = soup.get_text(separator=' ')

        # 6. 强力文本清洗
        # a. 处理 HTML 转义字符 (如 &#160;)
        text = raw_text.replace('\xa0', ' ').replace('&#160;', ' ')
        
        # b. 去除特殊符号（保留标点）
        text = re.sub(r'[☐☒\t]', ' ', text)

        # c. 处理 Item 标题的换行问题，确保 Item 9C. [Title] 格式连贯
        # 这有助于向量模型识别“标题-内容”关系
        text = re.sub(r'(Item\s+\d+[A-Z]?\.?)\s+', r'\n\1 ', text, flags=re.IGNORECASE)

        # d. 合并过多的空格和换行
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # 7. 【重要】不再强制截断 MD&A
        # 如果你想保留整份报告用于 RAG，直接返回全文
        return text.strip()

    except Exception as e:
        print(f"❌ 解析错误: {e}")
        return None

# def clean_edgar_html_v3(file_path):
#     # ... 之前的 BeautifulSoup 处理 ...
#     text = soup.get_text(separator=' ')
    
#     # 找到正文开始的标志性词汇
#     start_keywords = ["FORM 10-K", "UNITED STATES SECURITIES AND EXCHANGE COMMISSION"]
#     start_index = -1
#     for kw in start_keywords:
#         idx = text.find(kw)
#         if idx != -1:
#             start_index = idx
#             break
            
#     if start_index != -1:
#         # 只保留从 FORM 10-K 开始的内容
#         text = text[start_index:]
    
#     # 过滤掉包含过多 http 或 xbrl 关键字的行
#     lines = text.split('\n')
#     cleaned_lines = [line for line in lines if "http://" not in line and "xbrli:" not in line]
#     text = '\n'.join(cleaned_lines)
    
#     return text.strip()

def clean_edgar_full_pipeline(file_path):
    # --- 第一步：BeautifulSoup 基础清洗 (你现有的逻辑) ---
    # 记得把之前的 .decompose() 改为 .unwrap() 以免丢掉 "Not applicable"
    raw_cleaned_text = clean_with_beautifulsoup(file_path) 

    if not raw_cleaned_text:
        return None

    # --- 第二步：针对性过滤 (v3 建议) ---
    # 1. 找到真正的正文起点，切掉开头的元数据
    # SEC 文件的正文通常从 "UNITED STATES" 或 "FORM 10-K" 开始
    start_match = re.search(r'(UNITED\s+STATES\s+SECURITIES|FORM\s+10-K)', raw_cleaned_text, re.IGNORECASE)
    if start_match:
        raw_cleaned_text = raw_cleaned_text[start_match.start():]

    # 2. 移除残留的 XBRL 专用术语和无效链接
    # 这些通常是清洗后剩下的 http 链接或 xbrli 标识
    patterns_to_remove = [
        r'http[s]?://\S+',               # 链接
        r'\b\w+:\S+',                    # 类似 us-gaap:xxx 或 xbrli:xxx 的内容
        r'\b000\d{7}\b',                 # CIK 编号
        r'\b[A-Z0-9]{10,}\b',            # 长的机器编码
        r'(false|true|FY|P1Y|iso4217)',  # 常见的 XBRL 状态词
    ]
    
    for pattern in patterns_to_remove:
        raw_cleaned_text = re.sub(pattern, '', raw_cleaned_text)

    # 3. 最后的美化：合并空格，保持段落
    final_text = re.sub(r'\s{2,}', ' ', raw_cleaned_text)
    
    return final_text.strip()


if __name__ == "__main__":
    

    file_path = 'backend/data/raw/company_reports/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/full-submission.txt'
    
    
    cleaned_text = clean_edgar_full_pipeline(file_path)

    save_path = 'backend/data/processed/company_reports'
    filename = 'cleaned_edgar_AAPL_10K.txt'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if cleaned_text:
        print("✅ Cleaned text content:")
        try:
            with open(os.path.join(save_path, filename), 'w', encoding='utf-8') as out_file:
                out_file.write(cleaned_text)
            print(f"The cleaned text has been saved to...: {os.path.join(save_path, filename)}")
        except Exception as e:
            print(f"Failed to write to file.: {e}")
        print(cleaned_text[:500] + "...")
    else:
        print("❌ Text cleaning failed.。")