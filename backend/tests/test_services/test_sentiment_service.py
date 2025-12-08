# backend/tests/test_services/test_sentiment_service.py

import pytest
from unittest.mock import patch, MagicMock
from app.services.sentiment_service import SentimentService # 绝对导入
from app.core.config import settings
import torch

# ----------------------------------------------------
# 1. 设置夹具 (Fixtures) 来模拟依赖
# ----------------------------------------------------

@pytest.fixture
def mock_transformers(mocker):
    """
    模拟 (Mock) BertForSequenceClassification.from_pretrained 
    和 BertTokenizer.from_pretrained 的调用。
    """
    
    # 1. 创建模拟的模型和分词器实例
    mock_model_instance = MagicMock(spec=['eval', 'to', 'config']) # spec 确保有 eval 方法
    mock_tokenizer_instance = MagicMock()
    
    # 2. 直接模拟 from_pretrained 方法，并设置它们的返回值
    # 目标路径必须是 app.services.sentiment_service.BertForSequenceClassification.from_pretrained
    mock_model_from_pretrained = mocker.patch(
        'app.services.sentiment_service.BertForSequenceClassification.from_pretrained',
        return_value=mock_model_instance # 设置 from_pretrained 的返回值
    )
    
    mock_tokenizer_from_pretrained = mocker.patch(
        'app.services.sentiment_service.BertTokenizer.from_pretrained',
        return_value=mock_tokenizer_instance # 设置 from_pretrained 的返回值
    )
    
    # 返回 from_pretrained 的模拟对象和实例
    return (
        mock_model_from_pretrained, 
        mock_tokenizer_from_pretrained, 
        mock_model_instance, 
        mock_tokenizer_instance
    )

# ----------------------------------------------------
# 2. 单元测试 SentimentService
# ----------------------------------------------------
def test_sentiment_service_initialization(mock_transformers):
    """测试服务是否能正确初始化 (即调用 from_pretrained)"""
    # 接收新的返回值：from_pretrained 的模拟对象 和 实例对象
    mock_model_fp, mock_tokenizer_fp, mock_model_instance, _ = mock_transformers
    
    service = SentimentService(
        model_path="fake/model/path", 
        tokenizer_path="fake/tokenizer/path"
    )
    
    # 验证加载函数是否被调用 (使用 from_pretrained 的模拟对象来断言)
    mock_model_fp.assert_called_once_with("fake/model/path")
    mock_tokenizer_fp.assert_called_once_with("fake/tokenizer/path")
    
    # 验证模型实例是否被设置为 eval 模式 (使用实例对象来断言)
    mock_model_instance.eval.assert_called_once()

def test_predict_sentiment_positive(mock_transformers):
    """测试正向情感预测"""
    # 我们只需要实例对象来设置返回值
    _, _, mock_model_instance, mock_tokenizer_instance = mock_transformers
    
    # 1. 配置模拟分词器的返回值
    # 注意：模拟分词器本身 (mock_tokenizer_instance) 是可调用的
    mock_tokenizer_instance.return_value = { # 模拟调用 mock_tokenizer_instance(...) 的返回值
        'input_ids': torch.tensor([[101, 2023, 102]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    
    # 2. 配置模拟模型的返回值 (Outputs)
    # 模拟模型实例被调用 (mock_model_instance) 的返回值
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[5.0, -1.0, 0.5]]) # 5.0 是最高的
    mock_model_instance.return_value = mock_outputs # 模拟调用 mock_model_instance(...) 的返回值
    
    # 3. 初始化服务 (此时 from_pretrained 已经被模拟并返回了我们配置的实例)
    service = SentimentService(
        model_path=settings.MODEL_PATH, 
        tokenizer_path=settings.TOKENIZER_PATH
    )

    # 4. 执行预测
    result = service.predict_sentiment("This is a great day!")
    
    # 5. 断言结果
    assert result == "positive"
    
    
def test_predict_sentiment_negative(mock_transformers):
    """测试负向情感预测"""
    _, _, mock_model_instance, mock_tokenizer_instance = mock_transformers
    
    # ... 配置模拟分词器 ... 
    mock_tokenizer_instance.return_value = { # 模拟调用 mock_tokenizer_instance(...) 的返回值
        'input_ids': torch.tensor([[101, 2023, 102]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }

    # 模拟 logits，使得类别 1 (Negative) 的分数最高
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[0.1, 7.0, 0.5]]) # 7.0 是最高的
    mock_model_instance.return_value = mock_outputs 
    
    service = SentimentService(
        model_path=settings.MODEL_PATH, 
        tokenizer_path=settings.TOKENIZER_PATH
    )

    result = service.predict_sentiment("This is a terrible experience.")
    
    assert result == "negative"