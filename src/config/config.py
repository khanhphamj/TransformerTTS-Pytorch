import os

# Define data directories
RAW_DATA_DIR = 'raw'
PROCESSED_DATA_DIR = 'processed'

# Define dataset name
DATASET_NAME = 'doof-ferb/vlsp2020_vinai_100h'

# Define model directories
MODEL_DIR = 'models'

# Define training parameters
BATCH_SIZE = 8

# Define audio parameters
SAMPLE_RATE = 16000

# Define Mel spectrogram parameters
N_MELS = 80

# Define number of workers for data processing
NUM_WORKERS = os.cpu_count() - 8