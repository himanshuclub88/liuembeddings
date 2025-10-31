# Liu Embeddings

Welcome to Liu Embeddings—a lightweight semantic search framework that combines **embedding generation** with **vector storage**.

## 💡 What is Liu Embeddings?

Liu Embeddings = embedding + storage

**Save your money on expensive embedding models**

Built on HuggingFace embeddings and ChromaDB vector storage, it provides a unified solution for small to medium projects requiring efficient embedding, storage, and retrieval operations.

## 🚀 Why Choose Liu Embeddings?

### ⚡ Easy to Use

- **Minimal setup** - Get started in minutes, not hours
- **Zero configuration** - Works out of the box with sensible defaults
- **Automatic dependency management** - No manual ChromaDB or HuggingFace setup needed

### 💰 Cost-Effective

- **No expensive API calls** - Use open-source HuggingFace models instead of paid embedding services
- **Self-hosted solution** - Avoid recurring costs associated with cloud-based embedding APIs

### 🏗️ Production Ready

- **Integrated logging** - Comprehensive logging for debugging and monitoring
- **Validation & error handling** - Robust input validation and error messages
- **Batch operations** - Efficient batch ingestion and export to JSON

## 📚 Quick Navigation

| Section | Description |
|---------|-------------|
| [Introduction](introduction.md) | Learn what Liu Embeddings offers |
| [Quick Start](quickstart.md) | Get up and running in 5 minutes |
| [API Reference](api-reference.md) | Complete API documentation |
| [Examples](examples.md) | Real-world usage examples |
| [Developer Guide](developer-guide.md) | Architecture, setup, migration info |

## 🎯 Core Features

- **Unified API**: Single interface for both embedding generation and vector storage
- **Multiple Models**: Support for MiniLM, MPNet, E5, and BGE transformers
- **Semantic Search**: Fast, accurate similarity search with metadata filtering
- **CRUD Operations**: Complete Create, Read, Update, Delete capabilities
- **Batch Processing**: Handle large datasets efficiently
- **Text Processing**: Intelligent chunking with overlap preservation

## 📦 Installation

```bash
pip install liuembeddings
```

## ⚡ Quick Start

```python
from liuembeddings import LiuEmbeddings, LiuVectorStore

# Initialize
embedder = LiuEmbeddings(model_name="USE")
store = LiuVectorStore(embedder, collection_name="my_docs")

# Add documents
store.add_texts([
    "Python is a programming language",
    "JavaScript is for web development"
])

# Search
results, documents = store.similarity_search("What is Python?", n_results=1)
print(documents)
```

## 🧪 Try It Now

Looking for hands-on examples? Check out:

- [Quick Start Guide](quickstart.md) - Step-by-step tutorials
- [Examples & Workflows](examples.md) - Complete working examples
- [API Reference](api-reference.md) - Detailed method documentation

## 👨‍💻 For Developers

- [Developer Guide](developer-guide.md) - Architecture, setup, and migration info
- [GitHub Repository](https://github.com/himanshuclub88/liuembeddings)
- [Package on PyPI](https://pypi.org/project/liuembeddings/)

## 📄 License

MIT License - Free to use in personal and commercial projects

## 🤝 Contribute

Found a bug? Have a feature request? Contributions are welcome! Visit our [GitHub repository](https://github.com/himanshuclub88/liuembeddings).

---

**Ready to get started?** Head over to the [Quick Start Guide](quickstart.md)!
