import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
IMAGES_DIR = DATA_DIR / "images"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
LOG_DIR = BASE_DIR / "logs"

# Create directories
for directory in [DATA_DIR, DOCUMENTS_DIR, IMAGES_DIR, VECTOR_STORE_DIR, LOG_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight but effective
IMAGE_EMBEDDING_MODEL = "ViT-B/32"  # CLIP model
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Database settings
CHROMA_PERSIST_DIR = str(VECTOR_STORE_DIR / "chroma")
TEXT_COLLECTION_NAME = "documents"
IMAGE_COLLECTION_NAME = "images"

# Processing settings
SUPPORTED_TEXT_EXTENSIONS = ['.pdf', '.doc', '.docx', '.txt', '.ppt', '.pptx', '.xls', '.xlsx']
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
MAX_FILE_SIZE_MB = 50
BATCH_SIZE = 10

# Search settings
TOP_K_RESULTS = 10
SIMILARITY_THRESHOLD = 0.5
RERANK_ENABLED = True

# API settings
HOST = "0.0.0.0"
PORT = 8000
DEBUG = True
