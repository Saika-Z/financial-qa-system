'''
Author: Zhao
Date: 2025-12-25 20:53:17
LastEditors: 
LastEditTime: 2025-12-25 20:53:23
FilePath: get_investopedia_concurrent.py
Description: 

'''
import os
import time
import random
import re
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

# --- 配置区 ---
KEYWORDS_FILE = "keywords.txt"
SAVE_DIR = "backend/data/raw/investopedia"
HEADLESS = True  # Docker 中设为 True

def scrape_with_browser(url):
    with sync_playwright() as p:
        # 启动浏览器
        browser = p.chromium.launch(headless=HEADLESS)
        # 模拟真实的 User-Agent 和窗口大小
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        
        try:
            print(f"正在访问: {url}")
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            
            # 模拟人的行为：随机滚动，让反爬虫认为你是真人
            page.mouse.wheel(0, random.randint(500, 1000))
            time.sleep(random.uniform(2, 4)) 

            # 获取页面内容
            html = page.content()
            return html
        except Exception as e:
            print(f"❌ 抓取失败 {url}: {e}")
            return None
        finally:
            browser.close()

def parse_and_save(html, keyword):
    if not html: return
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # 你的原逻辑：寻找正文
    article_body = soup.find('div', {'id': 'article-body_1-0'}) or \
                   soup.find('div', {'class': 'article-content'}) or \
                   soup.find('div', {'class': 'comp-body-content'})
    
    if not article_body:
        print(f"⚠️ 无法解析正文: {keyword}")
        return

    # 提取并清理文本
    text_parts = [el.get_text(strip=True) for el in article_body.find_all(['p', 'h2', 'h3', 'li'])]
    full_text = f"Keyword: {keyword}\n\n" + '\n\n'.join(text_parts)
    
    # 保存文件（增量逻辑）
    safe_name = re.sub(r'[-\s]+', '_', keyword.lower()).strip()
    file_path = os.path.join(SAVE_DIR, f"{safe_name}.txt")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    print(f"✅ 已保存: {file_path}")

def main():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    
    # 从文件读取关键词
    with open(KEYWORDS_FILE, 'r') as f:
        keywords = [line.strip() for line in f if line.strip()]

    for kw in keywords:
        # 增量检查：如果文件已存在，跳过
        safe_name = re.sub(r'[-\s]+', '_', kw.lower()).strip()
        if os.path.exists(os.path.join(SAVE_DIR, f"{safe_name}.txt")):
            print(f"⏩ 跳过已存在的词条: {kw}")
            continue

        url = get_url_from_keyword(kw)
        html = scrape_with_browser(url)
        parse_and_save(html, kw)
        
        # 随机休息，防止被封
        wait = random.uniform(5, 15)
        print(f"休息 {wait:.1f} 秒...")
        time.sleep(wait)

if __name__ == "__main__":
    main()