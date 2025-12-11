import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, urlunparse
import time
import os
import re



def clean_and_normalize_url(full_link):
    """
    清理 Google 追踪参数，只保留原始文章的 URL。
    """
    # 查找是否有 Google 追踪参数 (&ved=...)，如果有，只取问号前的部分
    if '&ved=' in full_link:
        # 尝试使用 urllib.parse 来清理，这是更健壮的方法
        parsed = urlparse(full_link)
        # 移除 'ved', 'usg' 等追踪参数
        query_params = parse_qs(parsed.query)
        
        # 保留所有非追踪参数
        clean_query = '&'.join(f'{k}={v[0]}' for k, v in query_params.items() 
                               if not k in ['ved', 'usg', 'sa', 'cad'])
        
        # 重构 URL
        parsed = parsed._replace(query=clean_query)
        # 如果没有查询参数了，确保不留下问号
        clean_url = urlunparse(parsed)
        return clean_url.split('&ved=')[0].split('?')[0] # 最终保障
    
    return full_link.split('?')[0] # 简单清理问号后的所有参数

def scrape_article_content(url):
    """
    抓取单个 URL 的文章主体内容。
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # 检查 HTTP 错误
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 通用内容定位：新闻网站的正文通常在 <article>, <div> 或包含特定 class 的标签内
        # 尝试查找最有可能包含正文的标签
        
        # 优先使用 <article> 标签
        article = soup.find('article') 
        
        # 如果没有 <article>，尝试查找主内容 div
        if not article:
             article = soup.find('div', {'class': re.compile(r'content|body|main|article', re.I)}) 
        
        if not article:
            # 最后的尝试：获取所有段落，但这可能包含广告
            paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True)]
            return '\n\n'.join(paragraphs) if paragraphs else None

        # 提取文章主体内的所有段落文本
        article_text_parts = []
        for element in article.find_all('p'):
            text = element.get_text(strip=True)
            if text and len(text) > 30: # 过滤掉过短的段落（可能是图注或广告）
                article_text_parts.append(text)

        return '\n\n'.join(article_text_parts)

    except requests.exceptions.RequestException as e:
        print(f"   ❌ 请求失败或被阻止: {e}")
        return None

def process_news_data(csv_path, output_dir, sleep_secondes):
    """
    主处理函数：读取 CSV，循环爬取正文。
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_path}")
        return

    processed_count = 0
    for index, row in df.iterrows():
        full_link = row['link']
        
        # 1. 清理链接
        clean_url = clean_and_normalize_url(full_link)
        
        # 使用标题作为文件名（清理特殊字符）
        safe_title = re.sub(r'[^\w\-]', '_', row['title'][:50])
        file_path = os.path.join(output_dir, f"{safe_title}_{index}.txt")

        # 检查是否已爬取
        if os.path.exists(file_path):
            print(f"[{index+1}/{len(df)}] 跳过: {row['title']} (已存在)")
            continue
            
        print(f"[{index+1}/{len(df)}] 正在抓取: {row['title']}")
        
        # 2. 爬取正文
        article_content = scrape_article_content(clean_url)
        
        if article_content:
            # 3. 组织并保存内容
            # 在顶部添加关键元数据，方便 RAG 分块时提取
            metadata_header = (
                f"TITLE: {row['title']}\n"
                f"SOURCE: {row['media']} ({row['datetime']})\n"
                f"URL: {clean_url}\n\n"
            )
            
            full_content = metadata_header + article_content
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(full_content)
                
            print(f"   ✅ 成功保存内容，长度: {len(article_content)} 字符。")
            processed_count += 1
        else:
            print(f"   ❌ 无法提取正文。链接: {clean_url}")

        # 4. 延迟请求，避免被封锁
        time.sleep(sleep_secondes)
        
    print(f"\n--- 任务完成。成功处理 {processed_count} 篇文章。 ---")





if __name__ == "__main__":
    # 1. 配置
    NEWS_CSV_PATH = 'backend/data/raw/company_history_news/AAPL_news_12-11-2023_to_12-10-2025.csv' # 假设您的文件名为此
    NEWS_OUTPUT_DIR = 'backend/data/processed/company_history_news'
    DELAY_SECONDS = 20 # 设置请求间隔，避免被新闻网站封锁 IP

    if not os.path.exists(NEWS_OUTPUT_DIR):
        os.makedirs(NEWS_OUTPUT_DIR)

    # --- 执行主函数 ---
    process_news_data(NEWS_CSV_PATH,NEWS_OUTPUT_DIR,DELAY_SECONDS)