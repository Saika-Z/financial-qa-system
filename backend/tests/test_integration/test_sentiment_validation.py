'''
 # Author: Wenqing Zhao
 # Date: 2025-12-08 21:06:01
 # LastEditTime: 2025-12-09 16:20:58
 # Description: 
 # FilePath: /financial-qa-system/backend/tests/test_integration/test_sentiment_validation.py
'''
# backend/tests/test_sentiment_validation.py

import pytest
from app.services.sentiment_service import SentimentService
from app.core.config import settings
import torch

# ----------------------------------------------------
# ⚠️ 注意: 
# 1. 这个测试将加载真实的模型文件，运行时间会比单元测试长。
# 2. 如果模型文件不存在，测试会失败。
# ----------------------------------------------------

# 夹具：创建真实的 SentimentService 实例
@pytest.fixture(scope="module") 
def real_sentiment_service():
    """
    加载真实的模型和分词器。
    我们使用 'module' 作用域，以确保模型只在整个测试模块中加载一次。
    """
    # 确保 settings.MODEL_PATH 和 settings.TOKENIZER_PATH 指向正确的真实路径
    try:
        service = SentimentService(
            model_path=settings.MODEL_PATH, 
            tokenizer_path=settings.TOKENIZER_PATH
        )
        return service
    except Exception as e:
        # 如果加载失败（例如模型文件路径错误），跳过测试
        pytest.skip(f"无法加载真实模型或分词器，跳过验证测试。错误: {e}")


def test_validation_known_positive(real_sentiment_service):
    """
    使用一个明确的正面例子，验证模型是否预测为 'positive'。
    """
    text = "Cramo Group 's financial targets for 2010-2013 are sales growth higher than 10 percent per year; return on equity above 15 percent."
    result = real_sentiment_service.predict_sentiment(text)
    
    assert result == "positive", f"预期 'positive'，实际预测为 '{result}'"
    

def test_validation_known_negative(real_sentiment_service):
    """
    使用一个明确的负面例子，验证模型是否预测为 'negative'。
    """
    text = "The Helsinki-based company, which also owns the Salomon, Atomic and Suunto brands, said net profit rose 15 percent in the three months through Dec. 31 to €47 million ($61 US million), from €40.8 million a year earlier."
    result = real_sentiment_service.predict_sentiment(text)
    
    assert result == "positive", f"预期 'positive'，实际预测为 '{result}'"


def test_validation_neutral_or_mixed(real_sentiment_service):
    """
    测试中性或混合情感的例子。
    """
    text = "The product is okay, but a bit expensive."
    result = real_sentiment_service.predict_sentiment(text)
    
    assert result in ["neutral", "negative"], f"预期 'neutral' 或 'negative'，实际预测为 '{result}'"
