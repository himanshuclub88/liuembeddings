# liuembeddings/__init__.py

"""
LiuEmbeddings - TensorFlow embeddings with ChromaDB vector store.

A lightweight Python framework for semantic search using TensorFlow
embeddings and ChromaDB vector database.
"""

__version__ = "0.1.0"
__author__ = "Himanshu Singh"
__license__ = "MIT"

from .embeddings import LiuEmbeddings
from .vectorstore import LiuVectorStore
from .utils import clean_text, split_text, validate_texts, batch_generator, clean
from .config import LiuConfig
from .logger import setup_logger
from .liu_search import fastquery

__all__ = [
    "LiuEmbeddings",
    "LiuVectorStore",
    "clean_text",
    "split_text",
    "validate_texts",
    "batch_generator",
    "LiuConfig",
    "setup_logger",
    "fastquery",
    "clean"
]


