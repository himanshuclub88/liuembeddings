# liuembeddings/utils.py

"""
Utility functions for text processing and chunking.
"""

import re
from typing import List
from .logger import setup_logger
from .config import LiuConfig

logger = setup_logger(__name__)


def clean(raw : dict):
    """
    Args:
        raw : raw ouput

    returns:
        clean raw output
    [   {'id': '', 'document': "", 'metadata': {'': ''}, 'distance': },...]

    Raises:
        TypeError: If raw is not a raw file

    """
    try:
        logger.info("Raw data cleaning........................")
        return [{"id": id,"document": doc, "metadata": meta, "distance": dist}
            for id, doc, meta, dist in zip(raw['ids'][0], raw['documents'][0], raw['metadatas'][0], raw['distances'][0])]

    except Exception as e:
        logger.error(f"please returned raw file: {str(e)}")
        raise


def clean_text(
    text: str, 
    lowercase: bool = False, 
    remove_extra_spaces: bool = True,
    remove_newlines: bool = True
) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Text to clean
        lowercase: Convert to lowercase
        remove_extra_spaces: Collapse multiple spaces
        remove_newlines: Remove newline characters
        
    Returns:
        Cleaned text
        
    Raises:
        TypeError: If text is not a string
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    
    try:
        if lowercase:
            text = text.lower()
        
        if remove_newlines:
            text = text.replace('\n', ' ').replace('\r', ' ')
        
        if remove_extra_spaces:
            text = re.sub(r'\s+', ' ', text).strip()
        
        logger.debug(f"Cleaned text (length: {len(text)})")
        return text
    
    except Exception as e:
        logger.error(f"Text cleaning failed: {str(e)}")
        raise


def split_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
    split_by_sentences: bool = True,
    clean_before_split: bool = True,
    lowercase: bool = False
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk in characters
        chunk_overlap: Number of overlapping characters between chunks
        split_by_sentences: Split at sentence boundaries first
        clean_before_split: Clean text before splitting
        
    Returns:
        List of text chunks
        
    Raises:
        ValueError: If parameters are invalid
        TypeError: If text is not a string
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    
    chunk_size = chunk_size or LiuConfig.DEFAULT_CHUNK_SIZE
    chunk_overlap = chunk_overlap or LiuConfig.DEFAULT_CHUNK_OVERLAP
    
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    if not (0 <= chunk_overlap < chunk_size):
        raise ValueError(f"chunk_overlap must be >= 0 and < chunk_size but provided chunk_overlap={chunk_overlap} and chunk_size={chunk_size}",
                         "Chunk overlap is larger than chunk size" if chunk_overlap >= chunk_size else "")
    
    if clean_before_split:
        text = clean_text(text, lowercase=lowercase)
    
    if not text.strip():
        raise ValueError("Text is empty after cleaning")
    
    try:
        chunks = []
        
        if split_by_sentences:
            # Split by sentences first
            sentences = re.split(r'(?<=[.!?])\s+', text)
            logger.debug(f"Split into {len(sentences)} sentences")
            
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += sentence + " "
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        
        else:
            # Character-based chunking with overlap
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
                
                if i + chunk_size >= len(text):
                    break
        
        logger.info(f"Split text into {len(chunks)} chunks "
                   f"(size: {chunk_size}, overlap: {chunk_overlap})")
        
        return chunks
    
    except Exception as e:
        logger.error(f"Text splitting failed: {str(e)}")
        raise


def validate_texts(texts: List[str], min_length: int = 1) -> bool:
    """
    Validate a list of texts.
    
    Args:
        texts: List of texts to validate
        min_length: Minimum text length required
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
        TypeError: If texts is not a list
    """
    if not isinstance(texts, list):
        raise TypeError(f"Expected list, got {type(texts).__name__}")
    
    if not texts:
        raise ValueError("Texts list cannot be empty")
    
    if not all(isinstance(t, str) for t in texts):
        raise TypeError("All items must be strings")
    
    if any(len(t.strip()) < min_length for t in texts):
        raise ValueError(f"All texts must have minimum length {min_length}")
    
    logger.debug(f"Validated {len(texts)} texts")
    return True


def batch_generator(items: List, batch_size: int):
    """
    Generator for processing items in batches.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
