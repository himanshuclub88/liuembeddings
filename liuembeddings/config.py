# liuembeddings/config.py

"""
Configuration settings for LiuEmbeddings framework.
"""

class LiuConfig:
    """Global configuration for LiuEmbeddings."""
    
    # # Embedding Model Settings
    # EMBEDDING_MODEL = "https://tfhub.dev/google/universal-sentence-encoder/4"
    # MODEL_DIMENSION = 512  # USE model output dimension
    
    # Available Models
    AVAILABLE_MODELS = {
        "USE": {
            "url": "https://tfhub.dev/google/universal-sentence-encoder/4",
            "dimension": 512,
            "name": "Universal Sentence Encoder"
        },
        "USEL": {
            "url": "https://tfhub.dev/google/universal-sentence-encoder-large/5",
            "dimension": 512,
            "name": "USE Large"
        }
    }
    
    # Chunking Settings
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    
    # Vector Store Settings
    DEFAULT_VECTOR_PATH = "./liu_db"
    DEFAULT_COLLECTION_NAME = "default_collection"
    DISTANCE_METRIC = "cosine"  # "cosine", "l2", "ip"
    
    # Search Settings
    DEFAULT_N_RESULTS = 3
    MAX_N_RESULTS = 100
    
    # Batch Processing
    DEFAULT_BATCH_SIZE = 100
    
    # Model Caching
    ENABLE_MODEL_CACHE = True
    
    # Logging
    LOG_LEVEL = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Similarity Score Threshold
    DEFAULT_SIMILARITY_SEARCH_SCORE_THRESHOLD = 0.45
    