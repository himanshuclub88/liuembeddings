# Developer & Contributor Guide

This section combines developer setup, architecture overview, and migration information for developers working with LiuEmbeddings.

## Developer Setup

### Prerequisites

- Python 3.8+
- pip package manager
- Git for version control

### Local Development Installation

Clone the repository and install in development mode:

```bash
git clone https://github.com/himanshuclub88/liuembeddings.git
cd liuembeddings
pip install -e ".[dev]"
```

### Running Tests

Run the test suite to verify your environment:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=liuembeddings

# Run specific test file
pytest tests/test_embeddings.py -v
```

### Code Quality Tools

Format and validate code:

```bash
# Format code
black liuembeddings/

# Check style
flake8 liuembeddings/

# Type checking
mypy liuembeddings/

# All checks at once
black liuembeddings/ && flake8 liuembeddings/ && mypy liuembeddings/
```

---

## Architecture Overview

### Core Modules

**config.py** - Central configuration management
- Model registry and metadata
- Default values for all settings
- Logging configuration
- Key Class: `LiuConfig`

**logger.py** - Logging infrastructure
- Setup and configuration
- Consistent logging format
- Log level management

**embeddings.py** - HuggingFace embeddings wrapper
- Universal Sentence Encoder support
- Multiple model support
- Model caching and batch processing
- Key Class: `LiuEmbeddings`

**vectorstore.py** - ChromaDB integration
- Vector storage and retrieval
- CRUD operations
- Metadata filtering and batch operations
- Key Class: `LiuVectorStore`

**utils.py** - Text processing utilities
- Text splitting with overlap
- Text cleaning and normalization
- Batch generators
- Key Functions: `split_text()`, `clean_text()`

### Module Dependencies

```
config.py
    ↓
logger.py (depends on config)
    ↓
embeddings.py (depends on logger, config)
vectorstore.py (depends on logger, config, embeddings)
utils.py (depends on logger, config)
    ↓
__init__.py (depends on all above)
```

---

## Migration Guide

### Overview

`liuembeddings` **v2.0.0** introduces a major upgrade — moving from **TensorFlow Hub (USE)** to **Hugging Face Sentence-Transformers** for faster, lightweight, and modern embeddings.

### What Changed

| Area                 | v1.x (Old)                     | v2.0.0 (New)                                |
| -------------------- | ------------------------------ | ------------------------------------------- |
| **Backend**          | TensorFlow Hub (USE)           | Hugging Face Sentence-Transformers          |
| **Dependencies**     | `tensorflow`, `tensorflow_hub` | `sentence-transformers`, `torch`            |
| **Default Model**    | `USE`                          | `MiniLM`                                    |
| **Performance**      | Slow, large dependencies       | Faster, smaller, and GPU/CPU friendly       |
| **API**              | Mostly the same                | Same API, different model loading mechanism |
| **Supported Models** | Only USE / USE-Large           | MiniLM, MPNet, E5, BGE                      |
| **Model Cache**      | TensorFlow in-memory           | Hugging Face in-memory                      |

### New Dependencies

Uninstall TensorFlow-related packages:

```bash
pip uninstall tensorflow tensorflow-hub -y
```

### Model List in v2.0.0

| Model Name  | ID                                        | Dimension | Size (MB) | Accuracy |
| ----------- | ----------------------------------------- | --------- | --------- | -------- |
| `MiniLM`    | `sentence-transformers/all-MiniLM-L6-v2`  | 384       | 22        | 0.78     |
| `MPNetBase` | `sentence-transformers/all-mpnet-base-v2` | 768       | 420       | 0.82     |
| `USE`       | `intfloat/e5-base-v2`                     | 768       | 300       | 0.84     |
| `USEL`      | `BAAI/bge-base-en-v1.5`                   | 1024      | 1024      | 0.86     |

### Code Migration Example

**Old (v1.x)**

```python
from liuembeddings.embeddings import LiuEmbeddings

embedder = LiuEmbeddings("USE")
vec = embedder.embed_query("Hello world")
```

**New (v2.0.0)**

```python
from liuembeddings.embeddings import LiuEmbeddings

embedder = LiuEmbeddings("USE")  # or "MiniLM" for speed
vec = embedder.embed_query("Hello world")
```

✅ **Same API**, just use one of the new supported model names.

### Config Changes

**Old** - `LiuConfig.AVAILABLE_MODELS` contained only TensorFlow Hub models (`USE`, `USE-Large`).

**New** - `LiuConfig.AVAILABLE_MODELS` now includes modern transformer models with metadata:

```python
{
  "MiniLM": {...},
  "MPNetBase": {...},
  "USE": {...},
  "USEL": {...}
}
```

### Removed TensorFlow Environment Settings

In v1.x we had:

```python
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
```

This is no longer needed. ✅ All models now run directly via PyTorch or CPU using Sentence-Transformers.

### Benefits of v2.0.0

- 🚀 **30–50× faster installation** (no TensorFlow)
- 🧠 **Higher accuracy** via modern transformer encoders
- 💾 **Smaller dependency footprint**
- 🔄 **Same API for easy migration**
- 🧩 **Future support for multilingual and domain-specific models**

### Verification

Run a quick embedding test:

```python
from liuembeddings import LiuEmbeddings

embedder = LiuEmbeddings("E5Base")
print(len(embedder.embed_query("Migration complete!")))
```

Output should show a vector of **768 dimensions** without TensorFlow logs.

### Migration Summary

| Action                  | Description                                                          |
| ----------------------- | -------------------------------------------------------------------- |
| 🧹 Uninstall TensorFlow | `pip uninstall tensorflow tensorflow-hub`                            |
| 📦 Install Transformers | `pip install sentence-transformers`                                  |
| 🔄 Update Code          | Change `"USE-Large"` → `"E5Base"` or `"MiniLM"`                      |
| ⚙️ No API Change        | Keep using `embed_query`, `embed_documents`, `embed_documents_batch` |

---

## Testing Guidelines

### Writing Tests

Create tests in the `tests/` directory following the existing patterns:

```python
import pytest
from liuembeddings import LiuEmbeddings, LiuVectorStore

def test_embeddings_basic():
    embedder = LiuEmbeddings(model_name="USE")
    result = embedder.embed_query("test")
    assert isinstance(result, list)
    assert len(result) > 0
```

### Test Coverage

Maintain high code coverage (target: >80%):

```bash
pytest tests/ --cov=liuembeddings --cov-report=html
```

---

## Performance Tips

1. **Batch Processing** - Use `embed_documents_batch()` for large document sets
2. **Model Selection** - Choose MiniLM for speed, MPNet for accuracy
3. **Chunk Size** - Balance context and precision (typical: 400-500 tokens)
4. **Metadata Filtering** - Filter before similarity search when possible

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes and add tests
4. Run code quality checks
5. Commit with clear messages
6. Push to your fork and create a Pull Request

---

## Package Structure

```
liuembeddings/
├── liuembeddings/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── utils.py
│   ├── config.py
│   └── logger.py
├── tests/
│   ├── test_embeddings.py
│   └── test_vectorstore.py
├── docs/
├── setup.py
├── requirements.txt
└── README.md
```

---
