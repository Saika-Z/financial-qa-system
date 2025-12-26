
import os

# ------Project Directory Settings ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', 'sentiment_model')

# ---- kaggle -----
KAGGLE_RAW_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw','kaggle')

KAGGLE_CLEAN_TRAIN = os.path.join(PROJECT_ROOT, 'data', 'processed', 'kaggle_split', 'train.csv')
KAGGLE_CLEAN_VAL = os.path.join(PROJECT_ROOT, 'data', 'processed', 'kaggle_split', 'val.csv')
KAGGLE_CLEAN_TEST = os.path.join(PROJECT_ROOT, 'data', 'processed', 'kaggle_split', 'test.csv')

# ---- intention ------
INTENTION_RAW_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw','intention','kaggle_data.csv')
INTENTION_CLEAN_DATA = os.path.join(PROJECT_ROOT, 'data', 'processed', 'intention', 'clean_intention.csv')



# ------ Sentiment Analysis Model Training Settings ----
MODEL_NAME = 'bert-base-uncased'    # name of the pre-trained model
NUM_LABELS = 3                      # the number of sentiment classes: positive, negative, neutral
BATCH_SIZE = 8                      # batch size for training and evaluation
EPOCHS = 4                          # epochs for training
LEARNING_RATE = 2e-5                # learning rate for optimizer
MAX_SEQ_LEN = 128                   # maximum sequence length for tokenization