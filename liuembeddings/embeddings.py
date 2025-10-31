# liuembeddings/embeddings.py
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from .logger import setup_logger
from .config import LiuConfig

logger = setup_logger(__name__)

# Model cache
_MODEL_CACHE = {}

class LiuEmbeddings:
    """
    Transformer-based embeddings using Sentence-Transformers models.

    Supports model caching for performance optimization and multiple
    embedding models.
    
    Attributes:
        model: Loaded TensorFlow embedding model
        model_name: Name of the model being used
        model_dimension: Output dimension of embeddings
    
    Example:
        >>> embedder = LiuEmbeddings()
        >>> embedding = embedder.embed_query("Hello world")
        >>> embeddings = embedder.embed_documents(["Hello", "World"])

    """
    
    def __init__(self, model_name: str = "USE") -> None:
        """
        Initialize embeddings with specified model.
        
        Args:
            model_name: Model identifier from LiuConfig.AVAILABLE_MODELS

        Raises:
            ValueError: If model_name not found in available models
            RuntimeError: If model loading fails
        """
        self.model_name = model_name

        if model_name not in LiuConfig.AVAILABLE_MODELS:
            available = ", ".join(LiuConfig.AVAILABLE_MODELS.keys())
            raise ValueError(f"Model '{model_name}' not found. Available: {available}")

        model_info = LiuConfig.AVAILABLE_MODELS[model_name]
        model_id = model_info["id"]
        self.model_dimension = model_info["dimension"]

        try:
            # Use cache if enabled
            if LiuConfig.ENABLE_MODEL_CACHE and model_id in _MODEL_CACHE:
                logger.debug(f"Loading '{model_name}' from cache")
                self.model = _MODEL_CACHE[model_id]
            else:
                logger.info(f"Loading model '{model_name}' ({model_id})")
                self.model = SentenceTransformer(model_id)
                if LiuConfig.ENABLE_MODEL_CACHE:
                    _MODEL_CACHE[model_id] = self.model
                    logger.debug(f"Cached model '{model_name}'")

            logger.info(f"✅ Loaded '{model_name}' model (dim={self.model_dimension})")

        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    # ---------------------- Core Methods ----------------------

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.
        
        Args:
            text: Query text to embed
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            TypeError: If text is not a string
            ValueError: If text is empty
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            vec = self.model.encode(text, normalize_embeddings=True)
            logger.debug(f"Embedded single query (len={len(text)})")
            return vec.tolist()
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise RuntimeError(f"Embedding failed: {e}") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        if not isinstance(texts, list):
            raise TypeError(f"Expected list, got {type(texts).__name__}")
        if not texts:
            raise ValueError("Texts list cannot be empty")
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("All items must be strings")

        try:
            vectors = self.model.encode(texts, normalize_embeddings=True)
            logger.info(f"Embedded {len(texts)} documents")
            return [v.tolist() for v in vectors]
        except Exception as e:
            logger.error(f"Failed to embed documents: {e}")
            raise RuntimeError(f"Batch embedding failed: {e}") from e

    def embed_documents_batch(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """
        Embed documents in batches to manage memory efficiently.
        
        Args:
            texts: List of documents to embed
            batch_size: Size of each batch (default: LiuConfig.DEFAULT_BATCH_SIZE)
            
        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If batch_size is not positive
        """
        if batch_size is None:
            batch_size = LiuConfig.DEFAULT_BATCH_SIZE
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not texts:
            raise ValueError("Texts list cannot be empty")

        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        logger.info(f"Processing {len(texts)} docs in {total_batches} batches")

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            try:
                embeddings = self.embed_documents(batch)
                all_embeddings.extend(embeddings)
                logger.debug(f"Batch {batch_num}/{total_batches} complete")
            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {e}")
                raise

        logger.info(f"✅ Batch processing complete ({len(texts)} documents)")
        return all_embeddings

    # ---------------------- Model Info ----------------------

    @property
    def model_info(self) -> dict:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model metadata
        """
        info = LiuConfig.AVAILABLE_MODELS[self.model_name]
        return {
            "name": self.model_name,
            "id": info["id"],
            "dimension": info["dimension"],
            "size_mb": info.get("size"),
            "description": info.get("description", ""),
            "accuracy": info.get("accuracy")
        }
