import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, urlunparse
import time
import os
import re



def clean_and_normalize_url(full_link):
    """
    clean Google tracking parameters only keep orgin article
    """
    # check if has Google tracking parameters (&ved=...), if so, only keep before the question mark
    if '&ved=' in full_link:
        # try to use urllib.parse, this is more robust
        parsed = urlparse(full_link)
        # remove 'ved' and 'usg' parameters
        query_params = parse_qs(parsed.query)
        
        # keep all tracking parameters
        clean_query = '&'.join(f'{k}={v[0]}' for k, v in query_params.items() 
                               if not k in ['ved', 'usg', 'sa', 'cad'])
        
        # replace the query
        parsed = parsed._replace(query=clean_query)
        # if there are no parameters left, remove the question mark
        clean_url = urlunparse(parsed)
        # final check
        return clean_url.split('&ved=')[0].split('?')[0]
    
    # simply remove the question mark 
    return full_link.split('?')[0] 

def scrape_article_content(url):
    """
    get single article content
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() 

        soup = BeautifulSoup(response.text, 'html.parser')
        

        # the content is usually in <article>, <div> or <div class="content">
        # try to find the tag that contains the content
        
        # first try <article>
        article = soup.find('article') 
        
        # if no <article> tag, try <div> tag
        if not article:
             article = soup.find('div', {'class': re.compile(r'content|body|main|article', re.I)}) 
        
        if not article:
            # final resort: find all <p> tags but may get a lot of noise eg. ads or comments
            paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True)]
            return '\n\n'.join(paragraphs) if paragraphs else None

        # extract text from <p> tags
        article_text_parts = []
        for element in article.find_all('p'):
            text = element.get_text(strip=True)
            if text and len(text) > 30: # filter out short paragraphs
                article_text_parts.append(text)

        return '\n\n'.join(article_text_parts)

    except requests.exceptions.RequestException as e:
        print(f"   ❌  requsets failed or blocked: {e}")
        return None

def process_news_data(csv_path, output_dir, sleep_secondes):
    """
    the main function: read csv file and extract content
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"   ❌  error: file not found: {csv_path}")
        return

    processed_count = 0
    for index, row in df.iterrows():
        full_link = row['link']
        
        # 1. remove tracking parameters
        clean_url = clean_and_normalize_url(full_link)
        
        # use title as file name
        safe_title = re.sub(r'[^\w\-]', '_', row['title'][:50])
        file_path = os.path.join(output_dir, f"{safe_title}_{index}.txt")

        # check if file already exists
        if os.path.exists(file_path):
            print(f"[{index+1}/{len(df)}] skip: {row['title']} (already exists)")
            continue
        
        print(f"[{index+1}/{len(df)}] extracting: {row['title']}")
        
        # 2. get article content
        article_content = scrape_article_content(clean_url)
        
        if article_content:
            # 3. save article content
            # add metadata header
            metadata_header = (
                f"TITLE: {row['title']}\n"
                f"SOURCE: {row['media']} ({row['datetime']})\n"
                f"URL: {clean_url}\n\n"
            )
            
            full_content = metadata_header + article_content
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(full_content)
                
            print(f"  ✅ success save content, length: {len(article_content)} characters.")
            processed_count += 1
        else:
            print(f"   ❌ cannot extract content from: {clean_url}")

        # 4. delay request to avoid being blocked
        time.sleep(sleep_secondes)
        
    print(f"\n--- ✅ task complete, processed {processed_count} files ---")





if __name__ == "__main__":
    # assume you have a CSV file named "AAPL_news_12-11-2023_to_12-10-2025.csv"
    NEWS_CSV_PATH = 'backend/data/raw/company_history_news/AAPL_news_12-11-2023_to_12-10-2025.csv' 
    NEWS_OUTPUT_DIR = 'backend/data/processed/company_history_news'
    DELAY_SECONDS = 20 # set the delay in seconds to avoid being blocked

    if not os.path.exists(NEWS_OUTPUT_DIR):
        os.makedirs(NEWS_OUTPUT_DIR)


    process_news_data(NEWS_CSV_PATH,NEWS_OUTPUT_DIR,DELAY_SECONDS)