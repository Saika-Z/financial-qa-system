
from GoogleNews import GoogleNews
import pandas as pd
import os
from datetime import datetime, timedelta

def fetch_and_save_news(company, start_date, end_date, output_dir):
    """
    Use GoogleNews to fetch news about a company and save it to a CSV file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. initialize GoogleNews
    googlenews = GoogleNews()
    googlenews.setlang('en')  # set language
    googlenews.setperiod('7d') # set check period, this parameter to avoid being blocked
    googlenews.set_time_range(start_date, end_date)
    
    # 2. execute search
    # search: company + "news"
    search_query = f'"{company}" news stock' 
    print(f"🔄  searching: {search_query}")
    print(f"📅 time range: {start_date} to {end_date}")

    googlenews.search(search_query)

    # 3. collect results
    results = []
    # GoogleNews default return 1 page, need to loop
    for page in range(1, 6): # try up to 5 pages
        # Note: GoogleNews has anti-crawler mechanism
        try:
            googlenews.get_page(page)
            page_results = googlenews.results()
            if not page_results:
                print(f"✅ page {page} has no results, stop searching.")
                break
            results.extend(page_results)
            print(f"    - successfully fetched page {page}, total results: {len(results)}")
        except Exception as e:
            print(f"❌ feching page {page} occurs exception or anti-crawler mechanism triggered: {e}")
            break
            
    # 4. save results
    if not results:
        print(f"❌ cannot find any results for {search_query}")
        return

    df = pd.DataFrame(results)
    
    # clean key columns
    # ['title', 'date', 'media', 'desc', 'link']
    # Note: 'date' is not always available
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'published'})
    
    # only keep for RAG training columns
    columns_to_keep = ['title', 'published', 'media', 'desc', 'link', 'datetime']
    df_clean = df[[col for col in columns_to_keep if col in df.columns]]
    
    # file name and path
    file_name = f"{TICKER}_news_{START_DATE.replace('/', '-')}_to_{END_DATE.replace('/', '-')}.csv"
    file_path = os.path.join(output_dir, file_name)

    df_clean.to_csv(file_path, index=False, encoding='utf-8')
    print(f"\n🎉  successfully saved {len(df_clean)} news to {file_path}")


if __name__ == "__main__":
    # --- config ---
    TICKER = "AAPL"
    COMPANY_NAME = "Apple"  # company name more effective than stock ticker
    DOWNLOAD_DIR = "backend/data/raw/company_history_news"
    START_DATE = (datetime.now() - timedelta(days=365*2)).strftime("%m/%d/%Y")  # last 2 years
    END_DATE = datetime.now().strftime("%m/%d/%Y")
    # ----------
    # execute
    fetch_and_save_news(COMPANY_NAME, START_DATE, END_DATE, DOWNLOAD_DIR)