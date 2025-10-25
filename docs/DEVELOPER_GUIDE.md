# DEVELOPER_GUIDE.md

# LiuEmbeddings Developer Guide

## Overview

This guide covers the architecture, design patterns, and best practices for the LiuEmbeddings framework.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Application                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    Fastquery() or
                    LiuVectorStore
                           │
┌──────────────────────────┴──────────────────────────────────┐
│              LiuEmbeddings Core Module                      │
├─────────────────────────────────────────────────────────────┤
│  Embeddings          VectorStore       Utils                │
│  • embed_query()     • add_texts()     • split_text()       │
│  • embed_documents() • query()         • clean_text()       │
│  • Model caching     • CRUD ops        • batch_generator()  │
└──────────────────────┬─────────────────┬────────────────────┘
                       │                 │
        ┌──────────────┴─────────────┐   │
        │                            │   │
┌───────▼──────────┐        ┌────────▼───▼────────┐
│  TensorFlow      │        │    ChromaDB         │
│  Hub             │        │    Vector DB        │
│  (Embeddings)    │        │    (Storage)        │
└──────────────────┘        └─────────────────────┘
```

## Module Description

### 1. `config.py`
- **Purpose**: Centralized configuration management
- **Contains**: Default values, model configs, paths
- **Usage**: Import and modify `LiuConfig` for customization

```python
from config import LiuConfig
LiuConfig.DEFAULT_CHUNK_SIZE = 100
```

### 2. `logger.py`
- **Purpose**: Logging infrastructure
- **Contains**: Setup function for consistent logging
- **Usage**: Get logger in each module

```python
from logger import setup_logger
logger = setup_logger(__name__)
logger.info("Message")
```

### 3. `embeddings.py`
- **Purpose**: TensorFlow embedding model wrapper
- **Features**: 
  - Model caching for performance
  - Batch processing
  - Multiple model support
  - Type validation
- **Public Methods**:
  - `embed_query(text: str)` - Single embedding
  - `embed_documents(texts: List[str])` - Batch embeddings
  - `embed_documents_batch(texts, batch_size)` - Managed batches

### 4. `vectorstore.py`
- **Purpose**: ChromaDB wrapper for semantic storage
- **Features**:
  - CRUD operations
  - Metadata filtering
  - Batch operations
  - Export/Import
- **Public Methods**:
  - `add_texts()` - Store documents
  - `query()` - Search
  - `search_by_id()` - Retrieve by ID
  - `update_by_id()` - Update document
  - `delete_by_id()` - Delete document

### 5. `utils.py`
- **Purpose**: Text processing utilities
- **Functions**:
  - `clean_text()` - Normalize text
  - `split_text()` - Chunk text with overlap
  - `validate_texts()` - Input validation
  - `batch_generator()` - Batch iteration

### 6. `__init__.py`
- **Purpose**: Package initialization and public API
- **Exports**: Main classes and functions
- **Contains**: `liu_search()` convenience function

## Design Patterns

### 1. Error Handling
All methods use try-catch with specific error types:

```python
try:
    # Operation
except TypeError:
    # Handle type error
except ValueError:
    # Handle value error
except Exception as e:
    # Handle unexpected error
    raise RuntimeError(f"Operation failed: {str(e)}") from e
```

### 2. Logging
Every module logs at appropriate levels:

```python
logger.debug()    # Development details
logger.info()     # General info
logger.warning()  # Warnings
logger.error()    # Errors
```

### 3. Type Hints
All functions have complete type annotations:

```python
def method(
    param1: str,
    param2: List[str],
    param3: Optional[Dict] = None
) -> List[float]:
    pass
```

### 4. Documentation
Every class and method has docstrings:

```python
def method(arg: str) -> str:
    """
    Short description.
    
    Long description if needed.
    
    Args:
        arg: Argument description
        
    Returns:
        Return description
        
    Raises:
        ValueError: When...
        
    Example:
        >>> method("test")
        "result"
    """
```

## Best Practices

### For Library Users

1. **Always use context managers** where applicable
2. **Check for errors** and handle them appropriately
3. **Use batch operations** for large datasets
4. **Enable caching** for repeated operations
5. **Organize documents** with metadata

### For Contributors

1. **Add tests** for new features
2. **Update documentation** with changes
3. **Follow type hints** consistently
4. **Use logging** instead of print()
5. **Handle all exceptions** explicitly
6. **Validate inputs** before processing
7. **Write docstrings** for public APIs

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=liuembeddings

# Specific test
pytest tests/test_embeddings.py::TestLiuEmbeddings::test_embed_query_basic -v
```

### Writing Tests

```python
import pytest
from liuembeddings import LiuEmbeddings

class TestFeature:
    @pytest.fixture
    def embedder(self):
        return LiuEmbeddings()
    
    def test_something(self, embedder):
        result = embedder.embed_query("test")
        assert result is not None
    
    def test_error_case(self, embedder):
        with pytest.raises(ValueError):
            embedder.embed_query("")
```

## Performance Optimization

### Memory Management
- Use batch processing for large datasets
- Set appropriate batch size (default: 100)
- Monitor memory usage with large models

### Caching
- Enable model caching in config
- Cache is per-model-URL
- Reduces initialization time

### Batch Operations
```python
# Good: Batch processing
embeddings = embedder.embed_documents_batch(texts, batch_size=50)

# Avoid: Processing all at once for large datasets
embeddings = embedder.embed_documents(texts)  # Large memory usage
```

## Adding New Features

### New Embedding Model

1. Add to `LiuConfig.AVAILABLE_MODELS`:
```python
AVAILABLE_MODELS = {
        "USE": {
            "url": "https://tfhub.dev/google/universal-sentence-encoder/4",
            "dimension": 512,
            "name": "Universal Sentence Encoder"
        },
        "BERT": {
            "url": "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/3",
            "dimension": 768,
            "name": "BERT Uncased"
        }
    }
```

2. Update `LiuEmbeddings.__init__()` if needed
3. Add tests in `test_embeddings.py`

### New Vector Store Method

1. Implement in `LiuVectorStore` class
2. Include error handling and logging
3. Add docstring with examples
4. Add tests in `test_vectorstore.py`
5. Update README with usage example

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Model load slow | First-time download | Run once, then cached |
| High memory | Large batch | Reduce batch size |
| Search no results | Poor chunking | Adjust chunk_size |
| Errors not clear | Missing validation | Check input types |

## Deployment

### Production Checklist

- [ ] All tests passing
- [ ] No warnings in linting
- [ ] Documentation complete
- [ ] Error handling tested
- [ ] Performance benchmarked
- [ ] Version bumped
- [ ] Changelog updated

### PyPI Upload

```bash
# Build package
python setup.py sdist bdist_wheel

# Upload to PyPI
twine upload dist/*
```

## Versioning

Follow Semantic Versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

## Code Quality Tools

```bash
# Format code
black liuembeddings/ tests/

# Lint
flake8 liuembeddings/ tests/

# Type checking
mypy liuembeddings/

# All checks
black . && flake8 . && mypy . && pytest .
```

## License

MIT License - See LICENSE file

## Support

- Create GitHub issues for bugs
- Submit pull requests for features
- Check discussions for Q&A
