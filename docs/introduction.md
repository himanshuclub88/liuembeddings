# Introduction - With Navigation Links

LiuEmbedding is a lightweight semantic search framework that combines **embedding generation** with **vector storage**. Built on HuggingFace embeddings and ChromaDB vector storage, it provides a unified solution for small to medium projects requiring efficient embedding, storage, and retrieval operations.

## LiuEmbedding = embedding + storage

**Save your money on expensive embedding models**

## Core Architecture

### Embedding Layer
- **HuggingFace-based embedding generation** with consistent interface
- Model information exposure for debugging and observability
- Support for various pre-trained models and custom implementations

### Storage Layer
- **ChromaDB-backed vector storage** with persistent HNSW indexing
- Metadata filtering for efficient similarity search and organization
- Comprehensive CRUD operations and batch ingestion capabilities

## Key Features

- **Unified API**: Single interface for both embedding generation and vector storage
- **Production Ready**: Integrated logging, validation, and error handling
- **Batch Operations**: Efficient batch ingestion and export to JSON for data portability
- **Text Processing**: Chunking with overlap and document packing for optimal retrieval
- **Lightweight**: Minimal dependencies while maintaining full functionality

---

## 📚 Next Steps

Ready to get started? Follow these sections in order:

### 1️⃣ **[Quick Start Guide](quickstart.md)**
Get up and running in 5 minutes with step-by-step installation and examples.

### 2️⃣ **[API Reference](api-reference.md)**
Complete documentation of all classes, methods, and configuration options.

### 3️⃣ **[Examples & Workflows](examples.md)**
Real-world usage patterns, CRUD operations, batch processing, and text utilities.

### 4️⃣ **[Developer & Contributor Guide](developer-guide.md)**
Architecture overview, development setup, migration from v1.x, testing guidelines, and performance optimization tips.

---

## 🎯 Choose Your Path

### 👤 I'm New to LiuEmbedding
**Recommended:** [Quick Start](quickstart.md) → [Examples](examples.md) → [API Reference](api-reference.md)

Start with quick start to see it in action, then explore real examples, and finally use the API reference for detailed documentation.

### 👨‍💻 I'm a Developer
**Recommended:** [API Reference](api-reference.md) → [Examples](examples.md) → [Developer Guide](developer-guide.md)

Jump to the API reference for complete method documentation, see examples, and check the developer guide for architecture and setup.

### 🚀 I'm Upgrading from v1.x
**Recommended:** [Developer Guide](developer-guide.md) → [Quick Start](quickstart.md) → [API Reference](api-reference.md)

Read the migration guide first, then follow quick start with the new API, and refer to the API reference as needed.

### 📊 I Need Specific Information
- **Getting started?** → [Quick Start](quickstart.md)
- **Need code examples?** → [Examples & Workflows](examples.md)
- **Looking up a method?** → [API Reference](api-reference.md)
- **Integrating into my project?** → [Developer Guide](developer-guide.md)
- **Migrating from v1.x?** → [Developer Guide](developer-guide.md) (Migration section)

---

## 📖 Documentation Structure

### **Quick Start** [quickstart.md](quickstart.md)
- Installation instructions
- 5 different quick start examples
- Configuration guide
- ⏱️ Reading time: 5-10 minutes

### **API Reference** [api-reference.md](api-reference.md)
- Complete class documentation
- All methods with signatures
- Parameter types and defaults
- Return value specifications
- Error handling
- ⏱️ Reading time: 15-20 minutes (reference)

### **Examples & Workflows** [examples.md](examples.md)
- Complete working examples
- Design overview
- Retrieval questions example
- CRUD operations
- Batch processing
- Text utilities
- ⏱️ Reading time: 10-15 minutes

### **Developer Guide** [developer-guide.md](developer-guide.md)
- Developer setup
- Architecture overview
- Module descriptions
- Migration guide (v1.x → v2.0.0)
- Testing guidelines
- Performance tips
- Contributing guidelines
- ⏱️ Reading time: 20-30 minutes

---

## 💡 Common Use Cases

### 📝 **Document Search**
Add your documents to LiuEmbedding and instantly search semantically across them.

```python
from liuembeddings import LiuEmbeddings, LiuVectorStore

embedder = LiuEmbeddings()
store = LiuVectorStore(embedder, "documents")
store.add_texts(["Your document here..."])
results = store.query("Search query")
```

👉 **Next:** [Quick Start](quickstart.md)

### 🤖 **AI Applications**
Build semantic search into your AI/ML pipeline for better retrieval.

👉 **Next:** [Examples & Workflows](examples.md)

### 🔧 **Integration**
Integrate LiuEmbedding into your existing Python application.

👉 **Next:** [API Reference](api-reference.md)

### 📦 **Data Processing**
Process large datasets with batch operations and metadata filtering.

👉 **Next:** [Examples & Workflows](examples.md)

---

## 🔑 Key Concepts

### Embeddings
Mathematical representations of text that capture semantic meaning. LiuEmbedding uses HuggingFace Sentence-Transformers for state-of-the-art embeddings.

### Vector Storage
Efficient storage and retrieval of high-dimensional vectors. LiuEmbedding uses ChromaDB with persistent storage.

### Semantic Search
Finding documents similar to a query based on meaning, not keywords. Perfect for question-answering and retrieval augmented generation (RAG).

### Metadata
Additional information attached to documents for filtering and organization. Perfect for tracking source, date, category, etc.

---

## 🚀 Installation

Get started in seconds:

```bash
# Install LiuEmbeddings
pip install liuembeddings

# Verify installation
python -c "from liuembeddings import LiuEmbeddings; print('✓ Ready!')"
```

👉 **Next:** [Quick Start](quickstart.md)

---

## 📞 Need Help?

### 📖 Documentation
- [Quick Start Guide](quickstart.md) - Getting started
- [API Reference](api-reference.md) - Complete reference
- [Examples](examples.md) - Code examples
- [Developer Guide](developer-guide.md) - Deep dive

### 🐛 Issues
Visit [GitHub Issues](https://github.com/himanshuclub88/liuembeddings/issues) to report bugs or request features.

### 💬 Discussion
Join our community discussions on [GitHub Discussions](https://github.com/himanshuclub88/liuembeddings/discussions).

---

## 🎓 Learning Path

```
Start Here (Introduction)
        ↓
[Quick Start] ← Basic usage & examples
        ↓
[Examples] ← Real-world patterns
        ↓
[API Reference] ← Detailed methods
        ↓
[Developer Guide] ← Advanced topics & architecture
```

---

## ⚡ TL;DR

1. **Install:** `pip install liuembeddings`
2. **Quick Start:** Read [Quick Start](quickstart.md) (5 min)
3. **Try Examples:** Check [Examples](examples.md) (10 min)
4. **Reference:** Use [API Reference](api-reference.md) when needed
5. **Deep Dive:** Read [Developer Guide](developer-guide.md) for architecture

---

## 🎉 Ready?

👉 **Next Step:** Go to [Quick Start Guide](quickstart.md)

Or jump directly to:
- [📚 API Reference](api-reference.md) - If you prefer reading documentation
- [💡 Examples](examples.md) - If you prefer learning by example
- [👨‍💻 Developer Guide](developer-guide.md) - If you want to understand architecture

---

**Let's build amazing things with LiuEmbedding!** ✨



**← [Introduction](introduction.md) | [quickstart](quickstart.md) →**
