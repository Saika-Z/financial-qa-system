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
    使用一个非常明确的正面例子，验证模型是否预测为 'positive'。
    """
    text = "这绝对是我用过最好的产品！效果超出预期。"
    result = real_sentiment_service.predict_sentiment(text)
    
    # 根据您的分类标签进行断言
    # 假设 'positive' 对应模型预测的正确类别
    assert result == "positive", f"预期 'positive'，实际预测为 '{result}'"
    

def test_validation_known_negative(real_sentiment_service):
    """
    使用一个非常明确的负面例子，验证模型是否预测为 'negative'。
    """
    text = "这是一次糟糕的体验，我感到非常失望。"
    result = real_sentiment_service.predict_sentiment(text)
    
    # 假设 'negative' 对应模型预测的正确类别
    assert result == "negative", f"预期 'negative'，实际预测为 '{result}'"


def test_validation_neutral_or_mixed(real_sentiment_service):
    """
    测试中性或混合情感的例子。
    """
    text = "这个功能还行，但是价格有点贵。"
    result = real_sentiment_service.predict_sentiment(text)
    
    # 根据您模型的预期表现进行断言，可能需要检查是不是 'neutral' 或其他中间类别
    assert result in ["neutral", "negative"], f"预期 'neutral' 或 'negative'，实际预测为 '{result}'"