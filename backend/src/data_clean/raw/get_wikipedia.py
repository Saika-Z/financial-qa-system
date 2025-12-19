import wikipediaapi
import re
import os

def fetch_and_clean_wiki_content(concept, user_agent):
    wiki = wikipediaapi.Wikipedia('en', user_agent=user_agent)
    page = wiki.page(concept.replace(' ', '_'))
    
    if not page.exists():
        return None
    
    # get original text
    raw_text = page.text
    
    # clean: remove contents in brackets 
    cleaned_text = re.sub(r'\([^()]*?\)', '', raw_text)
    
    # next: remove empty lines and leading/trailing spaces
    cleaned_text = '\n'.join(line.strip() for line in cleaned_text.splitlines() if line.strip())
    
    return cleaned_text


if __name__ == "__main__":
    # example
    financial_concepts = ["Free cash flow", "Price-to-earnings ratio", "Balance sheet", "Amortization"]
    user_agent = 'FinanceQASystem/1.0 (your.email@example.com)'

    # save all data in one folder
    rag_data_dir = "backend/data/raw/wikipedia"
    if not os.path.exists(rag_data_dir):
        os.makedirs(rag_data_dir)

    print("--- start to fetch Wikipedia ---")
    for concept in financial_concepts:
        content = fetch_and_clean_wiki_content(concept,user_agent)
        if content:
            # save each concept to a file
            file_name = f"{concept.replace(' ', '_')}.txt"
            file_path = os.path.join(rag_data_dir, file_name)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ successfully fetched and saved: {file_path}")
        else:
            print(f"❌ cannot find content for: {concept}")

    print("--- Wikipedia data fetch complete ---")