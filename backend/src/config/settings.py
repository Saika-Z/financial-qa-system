import os

# ------Project Directory Settings ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', 'sentiment_model')

TRAIN_DATA_PATH = os.path.join(DATA_ROOT, 'train.csv')
VAL_DATA_PATH = os.path.join(DATA_ROOT, 'val.csv')
TEST_DATA_PATH = os.path.join(DATA_ROOT, 'test.csv')


# ------Sentiment Analysis Model Training Settings ----
MODEL_NAME ='bert-base-uncased',    # Define constants for model training
NUM_LABELS = 3,                     # Positive, Neutral, Negative
BATCH_SIZE = 8,                     # Batch size for training and evaluation
EPOCHS =4,                          # Number of training epochs 
LEARNING_RATE = 2e-5,               # Learning rate for the optimizer
MAX_SEQ_LEN = 128                   # Path to save the trained model
