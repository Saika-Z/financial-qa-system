from bs4 import BeautifulSoup
import re
import os

def clean_edgar_html(file_path):
    """
    Process SEC file, extracting and cleaning the HTML/XBRL content within the <TEXT> tags.

    file_path (str): Path to the HTML file to be cleaned.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # --- 1. find the <TEXT> tag content ---
        # tags in SEC usually are uppercase
        text_match = re.search(r'<TEXT>(.*?)</TEXT>', content, re.DOTALL)
        if not text_match:
            print("❌ 未找到<TEXT>标签内容")
            return None
        
        html_content = text_match.group(1)

        # --- 2. BeautifulSoup parsing ---
        soup = BeautifulSoup(html_content, 'html.parser')

        # --- 3. Remove noisy tags ---
        # remove <script> and <style> tags
        for element in soup(['title', 'meta', 'head', 'script', 'style']):
            element.decompose()
        

        # Remove XBRL related tags, but keep their text content
        for xbrl_tag in soup.find_all(re.compile(r'^(ix|xbrl|dei|us-gaap|srt|country|aapl|xbrli|xlink):.*', re.IGNORECASE)):
            if xbrl_tag:
                xbrl_tag.decompose()

        # Remove tags with style display:none or ix tags entirely
        for xbrl_tag in soup.find_all(True):
            if xbrl_tag and xbrl_tag.get('style', '') == 'display:none' or 'ix' in xbrl_tag.name:
                xbrl_tag.decompose()

        # Remove tables to avoid noisy data
        for table in soup.find_all('table'):
            table.decompose()
        
        # --- 4. get key text ---
        cleaned_content = re.sub(r'http[s]?://\S+', '', str(soup))
        raw_text = BeautifulSoup(cleaned_content, 'html.parser').get_text('\n\n')

        # --- 5. Clean text ---
        # Remove multiple spaces, newlines, and tabs
        text = ' '.join(raw_text.split())

        # Remove page markers like "Page X of Y" or "F-X"
        text = re.sub(r'UNITED STATES SECURITIES AND EXCHANGE COMMISSION.*?None', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'\s{2,}', ' ', text).strip()


        # Extract Management's Discussion and Analysis section
        #mda_start_pattern = re.compile(r'ITEM\s+7[.\s-:]*MANAGEMENT\'S\s+DISCUSSION\s+AND\s+ANALYSIS', re.IGNORECASE)
        #mda_end_pattern = re.compile(r'ITEM\s+8[.\s-:]*FINANCIAL\s+STATEMENTS', re.IGNORECASE)
        # [.\s-:] 修改为 [-. \s:] 或 [\s.:-] 或 [.\s\-:]
        # mda_start_pattern = re.compile(r'ITEM\s*7[.\s:-]*MANAGEMENT\'S\s*DISCUSSION\s*AND\s*ANALYSIS', re.IGNORECASE)
        # mda_end_pattern = re.compile(r'ITEM\s*8[.\s:-]*FINANCIAL\s*STATEMENTS', re.IGNORECASE)
        
        # mda_match = mda_start_pattern.search(text)
        # if mda_match:
        #     mda_start = mda_match.end()
        #     mda_end_match = mda_end_pattern.search(text, mda_start)
            
        #     if mda_end_match:
        #         mda_text = text[mda_start:mda_end_match.start()]
        #         print("✅ 成功粗略提取 MD&A 部分 (V2)。")
        #         return mda_text.strip()
        mda_start_pattern = re.compile(r'ITEM\s*7\W*MANAGEMENT\'S\s+DISCUSSION\s+AND\s+ANALYSIS', re.IGNORECASE)
        mda_end_pattern = re.compile(r'ITEM\s*8\W*FINANCIAL\s+STATEMENTS', re.IGNORECASE)
        mda_match = mda_start_pattern.search(text)
        text = re.sub(r'[☐☒]', '', text)
        if mda_match:
            mda_start = mda_match.end()
            mda_end_match = mda_end_pattern.search(text, mda_start)
            
            if mda_end_match:
                mda_text = text[mda_start:mda_end_match.start()]
                print("✅ 成功粗略提取 MD&A 部分 (V2)。")
                return mda_text.strip()
        return text.strip()

    except Exception as e:
        print(f"❌ 解析 EDGAR 文件时发生错误: {e}")
        return None
    
def test_clean_demo(test_file_path):
    # Example usage
    document_content = """
    <SEC-DOCUMENT>0000320193-25-000079.txt : 20251031
    <SEC-HEADER>... (省略元数据) ...</SEC-HEADER>
    <DOCUMENT>
    <TYPE>10-K
    <SEQUENCE>1
    <FILENAME>aapl-20250927.htm
    <DESCRIPTION>10-K
    <TEXT>
    <XBRL>
    <?xml version='1.0' encoding='ASCII'?>
    ... (大量 XBRL/HTML 标签) ...
    <html xmlns="http://www.w3.org/1999/xhtml" ...><head>...</head><body><div style="display:none">...</div>
    <p>This is the start of the report narrative.</p>
    <h2>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</h2>
    <p>The Company achieved record revenues in 2025 in the Services segment. This segment grew by 15% year-over-year.</p>
    <p>Our cash position remains strong, allowing for continued share buybacks.</p>
    <h2>ITEM 8. FINANCIAL STATEMENTS</h2>
    <p>The detailed financial tables follow below.</p>
    </body></html>
    </TEXT>
    </DOCUMENT>
    </SEC-DOCUMENT>
    """
    with open(test_file_path, 'w', encoding='utf-8') as test_file:
        test_file.write(document_content) 

    
if __name__ == "__main__":
    
    
    #file_path = 'backend/data/processed/edgar_test.txt'
    #test_clean_demo(file_path)

    file_path = 'backend/data/raw/company_reports/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/full-submission.txt'
    
    
    cleaned_text = clean_edgar_html(file_path)

    save_path = 'backend/data/processed/company_reports'
    filename = 'cleaned_edgar_AAPL_10K.txt'
    #filename = '112.txt'


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if cleaned_text:
        print("✅ 清理后的文本内容:")
        try:
            with open(os.path.join(save_path, filename), 'w', encoding='utf-8') as out_file:
                out_file.write(cleaned_text)
            print(f"清理后的文本已保存到: {os.path.join(save_path, filename)}")
        except Exception as e:
            print(f"写入文件失败: {e}")
        print(cleaned_text[:500] + "...")
    else:
        print("❌ 文本清理失败。")