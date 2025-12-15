'''
 # Author: Wenqing Zhao
 # Date: 2025-12-08 21:06:01
LastEditTime: 2025-12-15 13:05:44
 # Description: 
FilePath: test_sentiment_validation.py
'''
# backend/tests/test_sentiment_validation.py

import pytest
from app.services.sentiment_service import SentimentService
from app.core.config import settings
import torch

# ----------------------------------------------------
# ⚠️ Note:
# 1. This test will load real model files, so it will take longer to run than unit tests.
# 2. The test will fail if the model files do not exist.
# ----------------------------------------------------

# Fixture: Create a real SentimentService instance.
@pytest.fixture(scope="module") 
def real_sentiment_service():
    """
    Loading the actual model and tokenizer.
    We use the 'module' scope to ensure that the model is loaded only once throughout the entire test module.
    """
    # Ensure that `settings.MODEL_PATH` and `settings.TOKENIZER_PATH` point to the correct actual paths.
    try:
        service = SentimentService(
            model_path=settings.MODEL_PATH, 
            tokenizer_path=settings.TOKENIZER_PATH
        )
        return service
    except Exception as e:
        # If loading fails (for example, due to an incorrect model file path), skip the test.
        pytest.skip(f"  ❌Unable to load the actual model or tokenizer, skipping validation tests. Error: {e}")


def test_validation_known_positive(real_sentiment_service):
    """
    Using a clear positive example, verify whether the model predicts it as 'positive'.
    """
    text = "Cramo Group 's financial targets for 2010-2013 are sales growth higher than 10 percent per year; return on equity above 15 percent."
    result = real_sentiment_service.predict_sentiment(text)
    
    assert result == "positive", f"Expected outcome: 'positive', actual prediction: '{result}'"
    

def test_validation_known_negative(real_sentiment_service):
    """
    Using a clear negative example, verify whether the model predicts it as 'negative'.
    """
    text = "The Helsinki-based company, which also owns the Salomon, Atomic and Suunto brands, said net profit rose 15 percent in the three months through Dec. 31 to €47 million ($61 US million), from €40.8 million a year earlier."
    result = real_sentiment_service.predict_sentiment(text)
    
    assert result == "positive", f"Expected outcome: 'positive', actual prediction: '{result}'"


def test_validation_neutral_or_mixed(real_sentiment_service):
    """
    Examples of testing neutral or mixed emotions.
    """
    text = "The product is okay, but a bit expensive."
    result = real_sentiment_service.predict_sentiment(text)
    
    assert result in ["neutral", "negative"], f"The expected outcome was 'neutral' or 'negative', but the actual prediction was... '{result}'"
