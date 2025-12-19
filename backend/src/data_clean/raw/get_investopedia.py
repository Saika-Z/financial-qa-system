import requests
from bs4 import BeautifulSoup
import re
import os
import time 

def scrape_investopedia_article(url, proxy_config=None):
    # using a complex user agent
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7'
    }
    
    try:
        print(f"trying request URL: {url}")
        # add request timeout
        response = requests.get(url, headers=headers, proxies=proxy_config, timeout=15)
        
        # this line will check if it is 400 error or 500 error
        response.raise_for_status() 
        print(f"status code: {response.status_code}") # return 200 means success

        # --- key part: check for Cloudflare or other anti-spider ---
        if "Just a moment..." in response.text or "Cloudflare" in response.text:
            print("⚠️ warning: Cloudflare or other anti-spider detected. Skipping this URL.")
            return None
        # --------------------------------------------------

        soup = BeautifulSoup(response.text, 'html.parser')

        
        # try to find the article body  
        article_body = soup.find('div', {'id': 'article-body_1-0'})
        
        if not article_body:
            article_body = soup.find('div', {'class': 'article-content'})
            
        if not article_body:
            article_body = soup.find('div', {'class': 'comp-body-content'})
            
        if not article_body:
            print("❌ error: article body not found on Investopedia. The structure may have changed.")
            return None
   
        
        article_text_parts = []
        for element in article_body.find_all(['p', 'h2', 'h3', 'li']):
            text = element.get_text(strip=True)
            if text:
                article_text_parts.append(text)

        full_text = '\n\n'.join(article_text_parts)
        cleaned_text = re.sub(r'\[\d+\]', '', full_text)  
        
        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else "N/A"
        
        return title, f"Title: {title}\n\n{cleaned_text}"

    except requests.exceptions.RequestException as e:
        # 
        print(f"❌ request failed: {e}")
        return None, None
    
if __name__ == "__main__":
    # --- config key points ---
    TARGET_URLS = [
        # P/E Ratio (already done)
        "https://www.investopedia.com/terms/p/price-earningsratio.asp", 
        # Unlevered Free Cash Flow
        "https://www.investopedia.com/terms/u/unlevered-free-cash-flow-ufcf.asp",
        # Gross Margin
        "https://www.investopedia.com/terms/g/grossmargin.asp",
        # Net Income
        "https://www.investopedia.com/terms/n/netincome.asp",
        # Capital Expenditure
        "https://www.investopedia.com/terms/c/capitalexpenditure.asp",
        # EBIT
        "https://www.investopedia.com/terms/e/ebit.asp",
        # Current Ratio
        "https://www.investopedia.com/terms/c/currentratio.asp",
        # Debt-to-Equity Ratio
        "https://www.investopedia.com/terms/d/debtequityratio.asp",
        # Return on Equity (ROE)
        "https://www.investopedia.com/terms/r/returnonequity.asp",
        # Working Capital
        "https://www.investopedia.com/terms/w/workingcapital.asp",
        # Initial Public Offering (IPO)
        "https://www.investopedia.com/terms/i/ipo.asp",
        # Dividend Yield
        "https://www.investopedia.com/terms/d/dividendyield.asp",
        # Market Capitalization
        "https://www.investopedia.com/terms/m/marketcapitalization.asp",
        # Free Cash Flow (FCF)
        "https://www.investopedia.com/terms/f/freecashflow.asp",
        # Earnings Before Interest, Taxes, Depreciation, and Amortization (EBITDA)
        "https://www.investopedia.com/terms/e/ebitda.asp",
        # Beta
        "https://www.investopedia.com/terms/b/beta.asp",
        # Price-to-Book Ratio (P/B Ratio)
        "https://www.investopedia.com/terms/p/price-to-bookratio.asp",
        # Asset Turnover Ratio
        "https://www.investopedia.com/terms/a/assetturnover.asp",
        # Cash Conversion Cycle (CCC)
        "https://www.investopedia.com/terms/c/cashconversioncycle.asp",
        # Inventory Turnover
        "https://www.investopedia.com/terms/i/inventoryturnover.asp",
        # Debt Service Coverage Ratio (DSCR)
        "https://www.investopedia.com/terms/d/dscr.asp",
        # Quick Ratio
        "https://www.investopedia.com/terms/q/quickratio.asp",
        # Earnings Per Share (EPS)
        "https://www.investopedia.com/terms/e/eps.asp",
        # Discounted Cash Flow (DCF)
        "https://www.investopedia.com/terms/d/dcf.asp",
        # Comparable Company Analysis (CCA)
        "https://www.investopedia.com/terms/c/comparable-company-analysis-cca.asp",
        # GAAP
        "https://www.investopedia.com/terms/g/gaap.asp",
        # IFRS
        "https://www.investopedia.com/terms/i/ifrs.asp",
        # Accrual Accounting
        "https://www.investopedia.com/terms/a/accrualaccounting.asp",
        # Revenue Recognition
        "https://www.investopedia.com/terms/r/revenuerecognition.asp",
        # Inflation
        "https://www.investopedia.com/terms/i/inflation.asp",
        # Federal Funds Rate
        "https://www.investopedia.com/terms/f/federalfundsrate.asp",
        # Quantitative Easing (QE)
        "https://www.investopedia.com/terms/q/quantitative-easing.asp",
        # Quantitative Easing 2 (QE2)
        "https://www.investopedia.com/terms/q/quantitative-easing-2-qe2.asp",
        # Quantitative Analysis
        "https://www.investopedia.com/terms/q/quantitativeanalysis.asp",
        # Hedge Fund
        "https://www.investopedia.com/terms/h/hedgefund.asp",
        # Short Selling
        "https://www.investopedia.com/terms/s/shortselling.asp",
        # Volatility
        "https://www.investopedia.com/terms/v/volatility.asp",
        # Credit Default Swap (CDS)
        "https://www.investopedia.com/terms/c/creditdefaultswap.asp",
        # Income Statement
        "https://www.investopedia.com/terms/i/incomestatement.asp",
        # Balance Sheet
        "https://www.investopedia.com/terms/b/balancesheet.asp",
        # Cash Flow Statement
        "https://www.investopedia.com/terms/c/cashflowstatement.asp",
        # 10-K Report
        "https://www.investopedia.com/terms/1/10-k.asp",
        # 10-Q Report
        "https://www.investopedia.com/terms/1/10q.asp"
        # ... continue adding URLs ...
    ]

    # set the save directory
    SAVE_DIR = "backend/data/raw/investopedia"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    # ---------------------------

    # set proxies, if you don't have a proxy, keep it empty
    proxies = {}
    # ---

    print("\n" + "="* 50)
    print(f"🚀  extracting {len(TARGET_URLS)} articles from Investopedia")=


    for url in TARGET_URLS:
        print("\n" + "="* 50)
        print(f"sloving URL: {url}")
        # --- execute  ---
        article_title, article_content = scrape_investopedia_article(url, proxy_config=proxies)

        if article_content:
            # 1. make sure the directory exists
            os.makedirs(SAVE_DIR, exist_ok=True)

            # 2. accoding to article title generate a file name
            # clean the title and generate a safe filename
            safe_title = re.sub(r'[^\w\s-]', '', article_title).strip()
            safe_title = re.sub(r'[-\s]+', '_', safe_title).lower()
            filename = f"{safe_title}.txt"

            # 3. format the file path
            file_path = os.path.join(SAVE_DIR, filename)
            # 4. save the article content
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(article_content)
                
                print("\n🎉 successfully extracted and saved!")
                print(f"article saved to: {file_path}")
                # print the first 200 characters
                print("--- article content preview ---")
                print(article_content[:200] + "...")
                print("----------------")
                
            except Exception as e:
                print(f"failed to save article: {e}")
        else:
            print("\n cannot find article content for this url. please check the url.")
        
        # avoid too many requests
        sleep_time = 5  # delay in seconds
        print(f"waiting {sleep_time} seconds before next request...")
        time.sleep(sleep_time)
    print("\n✅ all articles extracted and saved!")


    