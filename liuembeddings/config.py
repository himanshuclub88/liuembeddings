# liuembeddings/config.py

"""
Configuration settings for LiuEmbeddings framework.
"""

class LiuConfig:
    """Global configuration for LiuEmbeddings."""
    
    # Available Models
    AVAILABLE_MODELS = {
        "MiniLM": {
            "id": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384,
            "size": 22,  # MB
            "description": "Lightweight & fast general-purpose embedding model.",
            "accuracy": 0.78
        },
        "MPNetBase": {
            "id": "sentence-transformers/all-mpnet-base-v2",
            "dimension": 768,
            "size": 420,  # MB
            "description": "Balanced model: size vs accuracy, good quality.",
            "accuracy": 0.82
        },
        "USE": {
            "id": "intfloat/e5-base-v2",
            "dimension": 768,
            "size": 300,  # MB
            "description": "High-quality semantic embeddings, strong performance.",
            "accuracy": 0.84
        },
        "USEL": {
            "id": "BAAI/bge-base-en-v1.5",
            "dimension": 1024,
            "size": 1024,  # MB
            "description": "Premium model: larger dimension, top-tier accuracy.",
            "accuracy": 0.86
        }
    }
    
    # Chunking Settings
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    
    # Vector Store Settings
    DEFAULT_VECTOR_PATH = "./liu_db"
    DEFAULT_COLLECTION_NAME = "default_collection"
    DISTANCE_METRIC = "cosine"  
    
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
    DEFAULT_SIMILARITY_SEARCH_SCORE_THRESHOLD = 0.4
    