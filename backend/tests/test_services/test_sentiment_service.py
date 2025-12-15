# backend/tests/test_services/test_sentiment_service.py

import pytest
from unittest.mock import patch, MagicMock
from app.services.sentiment_service import SentimentService # 绝对导入
from app.core.config import settings
import torch

# ----------------------------------------------------
# 1. Set up fixtures to simulate dependencies.
# ----------------------------------------------------

@pytest.fixture
def mock_transformers(mocker):
    """
    Mocking the calls to `BertForSequenceClassification.from_pretrained`
    and `BertTokenizer.from_pretrained`.
    """
    
    # 1. Create a mock model and tokenizer instance.
    mock_model_instance = MagicMock(spec=['eval', 'to', 'config']) # The specification ensures that an `eval` method exists.
    mock_tokenizer_instance = MagicMock()
    
    # 2. Directly mock the `from_pretrained` method and set its return values.
    # The target path must be app.services.sentiment_service.BertForSequenceClassification.from_pretrained
    mock_model_from_pretrained = mocker.patch(
        'app.services.sentiment_service.BertForSequenceClassification.from_pretrained',
        return_value=mock_model_instance # Set the return value of `from_pretrained`.
    )
    
    mock_tokenizer_from_pretrained = mocker.patch(
        'app.services.sentiment_service.BertTokenizer.from_pretrained',
        return_value=mock_tokenizer_instance # Set the return value of `from_pretrained`.
    )
    
    # Returns the mock object and instance from `from_pretrained`.
    return (
        mock_model_from_pretrained, 
        mock_tokenizer_from_pretrained, 
        mock_model_instance, 
        mock_tokenizer_instance
    )

# ----------------------------------------------------
# 2.  Unit tests for SentimentService
# ----------------------------------------------------
def test_sentiment_service_initialization(mock_transformers):
    """Test whether the service can be initialized correctly (i.e., by calling `from_pretrained`)."""
    # Receiving new return values: the mock object and instance object of `from_pretrained`.
    mock_model_fp, mock_tokenizer_fp, mock_model_instance, _ = mock_transformers
    
    service = SentimentService(
        model_path="fake/model/path", 
        tokenizer_path="fake/tokenizer/path"
    )
    
    # Verify that the loading function is called (using a mock object for `from_pretrained` to assert this).
    mock_model_fp.assert_called_once_with("fake/model/path")
    mock_tokenizer_fp.assert_called_once_with("fake/tokenizer/path")
    
    # Verify that the model instance is set to evaluation mode (using the instance object for assertion).
    mock_model_instance.eval.assert_called_once()

def test_predict_sentiment_positive(mock_transformers):
    """Testing positive emotion prediction."""
    # We only need an instance object to set the return value.
    _, _, mock_model_instance, mock_tokenizer_instance = mock_transformers
    
    # 1. Configure the return value of the simulated tokenizer.
    # Note: The mock tokenizer instance itself (mock_tokenizer_instance) is callable.
    mock_tokenizer_instance.return_value = { # Simulate the return value of calling mock_tokenizer_instance(...)
        'input_ids': torch.tensor([[101, 2023, 102]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    
    # 2. Configure the return values ​​(Outputs) of the simulation model.
    # The return value of the mocked model instance (mock_model_instance) when it is called.
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[5.0, -1.0, 0.5]]) # 5.0 is the highest.
    mock_model_instance.return_value = mock_outputs # Simulating the return value of calling `mock_model_instance(...)`
    
    # 3. Initializing the service (at this point, `from_pretrained` has already been simulated and returned the instance we configured).
    service = SentimentService(
        model_path=settings.MODEL_PATH, 
        tokenizer_path=settings.TOKENIZER_PATH
    )

    # 4. Execute prediction
    result = service.predict_sentiment("This is a great day!")
    
    # 5. Assertion result
    assert result == "positive"
    
    
def test_predict_sentiment_negative(mock_transformers):
    """Testing negative emotion prediction"""
    _, _, mock_model_instance, mock_tokenizer_instance = mock_transformers
    
    # ... Configure the simulated tokenizer. ... 
    mock_tokenizer_instance.return_value = { # Simulate the return value of calling mock_tokenizer_instance(...)
        'input_ids': torch.tensor([[101, 2023, 102]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }

    # Simulate logits such that the score for category 1 (Negative) is the highest.
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[0.1, 7.0, 0.5]]) # 7.0 is the highest.
    mock_model_instance.return_value = mock_outputs 
    
    service = SentimentService(
        model_path=settings.MODEL_PATH, 
        tokenizer_path=settings.TOKENIZER_PATH
    )

    result = service.predict_sentiment("This is a terrible experience.")
    
    assert result == "negative"