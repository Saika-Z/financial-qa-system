import requests
from bs4 import BeautifulSoup
import re
import os
import time # 导入 time 模块用于设置延时，防止请求过于频繁

def scrape_investopedia_article(url, proxy_config=None):
    # 使用一个更复杂、更难被识别的 User-Agent
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7'
    }
    
    try:
        print(f"尝试请求 URL: {url}")
        # 增加超时时间，确保网络没有问题
        response = requests.get(url, headers=headers, proxies=proxy_config, timeout=15)
        
        # 这一行是关键！它会检查是否是 4xx 或 5xx 错误
        response.raise_for_status() 
        print(f"状态码: {response.status_code}") # 成功应返回 200

        # --- 核心调试点：检查返回的内容是否是文章内容 ---
        if "Just a moment..." in response.text or "Cloudflare" in response.text:
            print("⚠️ 警告：检测到 Cloudflare 或其他反爬机制。请求可能被阻碍。")
            return None
        # --------------------------------------------------

        soup = BeautifulSoup(response.text, 'html.parser')

        # ... (跳转到步骤 2 的内容提取部分) ...
        # ...
        
        # 尝试定位文章的主体内容（Investopedia 结构）
        article_body = soup.find('div', {'id': 'article-body_1-0'})
        
        if not article_body:
            # 尝试查找最新的通用主体类名
            article_body = soup.find('div', {'class': 'article-content'})
            
        if not article_body:
            # 尝试查找您代码中原来的备用类名
            article_body = soup.find('div', {'class': 'comp-body-content'})
            
        if not article_body:
            print("❌ 错误：未能找到 Investopedia 文章主体内容标签。网站结构可能已改变。")
            return None
        
        # ... (后续内容提取和清理逻辑不变) ...
        
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
        # 捕获 404, 500, Timeout 等所有请求错误
        print(f"❌ 请求 Investopedia 失败: {e}")
        return None, None
    
if __name__ == "__main__":
    # --- 配置：关键修改部分 ---
    TARGET_URLS = [
        # P/E Ratio (您已验证成功的链接)
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
        # ... 您可以继续添加更多 Investopedia 文章链接 ...
    ]

    # 定义保存路径
    SAVE_DIR = "backend/data/raw/investopedia"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    # ---------------------------

    # 示例代理配置（请替换为您的实际代理信息，如果不需要代理，则设置为 None）
    proxies = {} # 如果您没有代理配置，保持为空字典或设置为 None
    # ---
    print(f"🚀 开始抓取 {len(TARGET_URLS)} 篇文章...")

    for url in TARGET_URLS:
        print("\n" + "="* 50)
        print(f"正在处理 URL: {url}")
        # --- 执行代码 ---
        article_title, article_content = scrape_investopedia_article(url, proxy_config=proxies)

        if article_content:
            # 1. 确保目标目录存在
            os.makedirs(SAVE_DIR, exist_ok=True)

            # 2. 根据文章标题生成文件名
            # 清理标题，将非字母数字字符替换为下划线，用于生成文件名
            safe_title = re.sub(r'[^\w\s-]', '', article_title).strip()
            safe_title = re.sub(r'[-\s]+', '_', safe_title).lower()
            filename = f"{safe_title}.txt"

            # 3. 构造完整的文件路径
            file_path = os.path.join(SAVE_DIR, filename)
            # 4. 写入文件
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(article_content)
                
                print("\n🎉 成功抓取文章内容！")
                print(f"文章已保存到: {file_path}")
                # 打印部分内容进行确认
                print("--- 文章预览 ---")
                print(article_content[:200] + "...")
                print("----------------")
                
            except Exception as e:
                print(f"写入文件失败: {e}")
        else:
            print("\n未能成功抓取文章内容，请检查 URL 和目标网站的结构。")
        
        # 为了防止请求过于频繁，添加延时
        sleep_time = 5  # 设置延时秒数
        print(f"等待 {sleep_time} 秒后继续下一个请求...")
        time.sleep(sleep_time)
    print("\n✅ 所有文章抓取任务完成！")


    