
import os

# ------Project Directory Settings ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', 'sentiment_intention_bert')

# ---- kaggle -----
KAGGLE_RAW_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw','kaggle')

KAGGLE_CLEAN_TRAIN = os.path.join(PROJECT_ROOT, 'data', 'processed', 'kaggle_split', 'train.csv')
KAGGLE_CLEAN_VAL = os.path.join(PROJECT_ROOT, 'data', 'processed', 'kaggle_split', 'val.csv')
KAGGLE_CLEAN_TEST = os.path.join(PROJECT_ROOT, 'data', 'processed', 'kaggle_split', 'test.csv')


# ---- intention path ------
INTENTION_RAW_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw','intention')
INTENTION_CLEAN_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'intention')

# ---- intention file ------
INTENTION_BERT_DATA = os.path.join(PROJECT_ROOT, 'data', 'processed', 'intention', 'intention_bert.csv')
INTENTION_FASTSET_DATA = os.path.join(PROJECT_ROOT, 'data', 'processed', 'intention', 'intention_fastset.csv')



# ------ Sentiment Analysis Model Training Settings ----
MODEL_NAME = 'bert-base-multilingual-cased'     # bert-base-multilingual-cased supported Chinese
NUM_LABELS = 3                                  # the number of sentiment classes: positive, negative, neutral
BATCH_SIZE = 8                                  # batch size for training and evaluation
EPOCHS = 4                                      # epochs for training
LEARNING_RATE = 2e-5                            # learning rate for optimizer
MAX_SEQ_LEN = 128                               # maximum sequence length for tokenization

NUM_SENTIMENT = 3                                  # the number of_sentiment
NUM_INTENT = 3                                  # the number of_intent
BERT_HIDDEN_SIZE = 768                          # the hidden size of bert
DROPOUT = 0.1