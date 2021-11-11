from pathlib import Path

DOWNLOAD_DATA = True
DOWNLOAD_DATA_URL = 'https://www.openml.org/data/get_csv/2419/electricity-normalized.arff'

DATA_PROCESSED_PATH = Path('data', 'processed')
DATA_RAW_PATH = Path('data', 'raw')
PIPELINE_PATH = Path('models')

RAW_FILE_NAME = 'raw.csv'
TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'

PIPELINE_FILE_NAME = 'model_pipeline.pkl'

TARGET_NAME = 'class'

TRAIN_RATIO = 0.5

RANDOM_STATE = 31
