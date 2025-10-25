# liuembeddings/logger.py

"""
Logging configuration for LiuEmbeddings.
"""

import logging
from .config import LiuConfig

def setup_logger(name: str) -> logging.Logger:
    """
    Setup a logger with consistent formatting.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.hasHandlers():
        return logger
    
    # Set level
    logger.setLevel(getattr(logging, LiuConfig.LOG_LEVEL))
    
    # Console handler
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, LiuConfig.LOG_LEVEL))
    
    # Formatter
    formatter = logging.Formatter(LiuConfig.LOG_FORMAT)
    handler.setFormatter(formatter)
    
    # Add handler
    logger.addHandler(handler)
    
    return logger
