'''
Author: Zhao
Date: 2025-12-29 21:26:10
LastEditors: Please set LastEditors
LastEditTime: 2025-12-30 17:51:22
FilePath: random_gerated_data.py
Description: 

'''
import random
import json

def  generate_random_data(file_path):
# 1. 准备关键词库
    stocks = ["苹果", "Apple", "特斯拉", "TSLA", "英伟达", "NVDA", "微软", "MSFT"]
    indicators = ["价格", "股价", "走势", "行情", "现在的点位", "报多少"]

    # 2. 准备句式模板库 (改变语序和语气)
    templates = [
        "请问{stock}现在的{indicator}是多少？",
        "{stock}的{indicator}帮我查一下",
        "我想看下{stock}的{indicator}",
        "{stock}现在的{indicator}怎么样？",
        "帮我查询{stock}目前的{indicator}",
        "告诉我{stock}的{indicator}"
    ]

    # 3. 交叉生成
    augmented_data = []
    for s in stocks:
        for i in indicators:
            for t in templates:
                sentence = t.format(stock=s, indicator=i)
                augmented_data.append({"Text": sentence, "Label": "FINANCE"}) # 0: finance

    print(f"生成的样本数: {len(augmented_data)}") # 8*6*6 = 288 条

    # 4. 保存为json文件
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(augmented_data, json_file, ensure_ascii=False, indent=4)
    # df = pd.DataFrame(augmented_data)
    # df.to_csv(file_path, index=False)

    print(f" 数据已保存到 {file_path}")

if __name__ == "__main__":

    file_path = "data/raw/intention/random_generated_data.json"    
    generate_random_data(file_path)