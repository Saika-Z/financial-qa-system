'''
 # Author: Wenqing Zhao
 # Date: 2025-12-10 10:33:51
 # LastEditTime: 2025-12-10 10:52:08
 # Description: 
 # FilePath: /financial-qa-system/backend/data/raw/getCompanyNews.py
'''
from GoogleNews import GoogleNews
import pandas as pd
import os
from datetime import datetime, timedelta

def fetch_and_save_news(company, start_date, end_date, output_dir):
    """
    使用 GoogleNews 库获取指定公司在时间范围内的新闻，并保存为 CSV。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 初始化 GoogleNews 对象
    googlenews = GoogleNews()
    googlenews.setlang('en')  # 设置语言为英文
    googlenews.setperiod('7d') # 设置搜索周期 (虽然设置了日期范围，但这个参数有助于避免被认为是机器人)
    googlenews.set_time_range(start_date, end_date)
    
    # 2. 执行搜索
    # 搜索查询：公司名称 + "news" (可以优化为更具体的查询)
    search_query = f'"{company}" news stock' 
    print(f"🔄 正在搜索查询: {search_query}")
    print(f"📅 时间范围: {start_date} 到 {end_date}")

    googlenews.search(search_query)

    # 3. 抓取所有页面的结果
    results = []
    # GoogleNews 默认只抓取第一页，需要循环抓取更多页面
    for page in range(1, 6): # 尝试抓取前 5 页 (每页约 10-15 条新闻)
        # 注意：googlenews.get_page(page) 有时可能触发反爬机制，谨慎使用
        try:
            googlenews.get_page(page)
            page_results = googlenews.results()
            if not page_results:
                print(f"✅ 第 {page} 页没有更多结果，停止搜索。")
                break
            results.extend(page_results)
            print(f"    - 成功抓取第 {page} 页，当前总计 {len(results)} 条新闻。")
        except Exception as e:
            print(f"❌ 抓取第 {page} 页时发生错误或触发反爬机制: {e}")
            break
            
    # 4. 转换并保存数据
    if not results:
        print("❌ 未获取到任何新闻结果。")
        return

    df = pd.DataFrame(results)
    
    # 清理并筛选关键列
    # 关键列：['title', 'date', 'media', 'desc', 'link']
    # 注意：'desc' 是新闻摘要，'link' 是新闻原文链接，'text' 列可能包含摘要或空值
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'published'})
    
    # 确保只保留 RAG 训练有用的列
    columns_to_keep = ['title', 'published', 'media', 'desc', 'link', 'datetime']
    df_clean = df[[col for col in columns_to_keep if col in df.columns]]
    
    # 生成文件名
    file_name = f"{TICKER}_news_{START_DATE.replace('/', '-')}_to_{END_DATE.replace('/', '-')}.csv"
    file_path = os.path.join(output_dir, file_name)

    df_clean.to_csv(file_path, index=False, encoding='utf-8')
    print(f"\n🎉 成功将 {len(df_clean)} 条新闻数据保存到：{file_path}")


if __name__ == "__main__":
    # --- 配置 ---
    TICKER = "AAPL"
    COMPANY_NAME = "Apple"  # 用于搜索的公司名称，比股票代码更有效
    DOWNLOAD_DIR = "backend/data/raw/company_history_news"
    START_DATE = (datetime.now() - timedelta(days=365*2)).strftime("%m/%d/%Y")  # 过去两年
    END_DATE = datetime.now().strftime("%m/%d/%Y")
    # ----------
    # 执行函数
    fetch_and_save_news(COMPANY_NAME, START_DATE, END_DATE, DOWNLOAD_DIR)