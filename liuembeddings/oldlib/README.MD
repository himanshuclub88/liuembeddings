Absolutely! Here’s a **ready-to-use README.md** tailored for your `liuembeddings` pip module, focusing on simplicity for end users and documenting all features clearly.

---

# LiuEmbeddings

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**LiuEmbeddings** is an **open-source Python module** for generating embeddings from text and performing **semantic search** using **TensorFlow embeddings** and **Chroma vector database**.

It is designed to be **simple, intuitive, and ready for AI applications**, allowing users to embed, store, and query text with minimal code.

---

## Features

* TensorFlow embeddings (Universal Sentence Encoder by default)
* Split long documents into **chunks with optional overlap**
* Store embeddings in **local Chroma vector database**
* Perform **semantic search** on stored chunks
* Simple API for end users — no TensorFlow/NumPy knowledge required
* Fully open-source and free

---

## Installation

```bash
pip install liuembeddings
```

**Dependencies:**

* Python 3.8+
* tensorflow
* tensorflow-hub
* chromadb
* langchain (optional, for advanced chunking)

---

## Quick Start

```python
from liuembeddings import LiuEmbeddings, LiuVectorStore, split_text

# 1️⃣ Split long text into chunks
long_text = "LangChain is an open-source framework..." * 50
chunks = split_text(long_text)  # default chunk_size=1000, overlap=200

# 2️⃣ Load embedding model (Universal Sentence Encoder)
emb_model = LiuEmbeddings()

# 3️⃣ Create Chroma vector store
vectorstore = LiuVectorStore(embedding_model=emb_model, collection_name="my_collection")

# 4️⃣ Add chunks to vector store
vectorstore.add_texts(chunks)

# 5️⃣ Perform semantic search
query = "What is LangChain used for?"
results = vectorstore.query(query, n_results=3)

for i, r in enumerate(results, 1):
    print(f"Result {i}:", r[:300])
```

---
# Query Options

## Vector Store Methods

| Method | Purpose |
|--------|---------|
| `search_by_id()` | Get document by ID |
| `search_by_metadata()` | Filter by metadata |
| `similarity_search()` | Search with similarity scores |
| `get_all()` | Retrieve all documents |
| `delete_by_id()` | Remove a document |
| `update_by_id()` | Update document content |
| `count_documents()` | Get collection size |

## Method Details

### `search_by_id(doc_id)`
Retrieve a specific document using its ID.

**Example:**
```python
result = vector_store.search_by_id("doc1")
# Returns: {"id": "doc1", "document": "...", "metadata": {...}}
```

---

### `search_by_metadata(metadata_filter)`
Filter documents by metadata criteria.

**Example:**
```python
results = vector_store.search_by_metadata({"category": "news"})
# Returns: List of documents matching the metadata filter
```

---

### `similarity_search(query_text, n_results=3, with_scores=False)`
Perform semantic similarity search.

**Example:**
```python
results = vector_store.similarity_search("What is Python?", n_results=5, with_scores=True)
# Returns: List of documents with similarity scores
```

---

### `get_all()`
Retrieve all documents in the collection.

**Example:**
```python
all_docs = vector_store.get_all()
# Returns: List of all documents with metadata
```

---

### `delete_by_id(doc_id)`
Delete a document by its ID.

**Example:**
```python
vector_store.delete_by_id("doc3")
# Document is permanently removed
```

---

### `update_by_id(doc_id, new_text, new_metadata=None)`
Update a document's content and/or metadata.

**Example:**
```python
vector_store.update_by_id("doc1", "Updated content", {"category": "updated"})
```

---

### `count_documents()`
Get the total number of documents in the collection.

**Example:**
```python
count = vector_store.count_documents()
# Returns: Integer count
```


## API Reference

### `LiuEmbeddings`

* `LiuEmbeddings(model_url=...)`
  Load a TensorFlow embedding model. Defaults to USE v4.

* `embed_query(text: str)` → list of floats
  Convert a **single string** into an embedding vector.

* `embed_documents(texts: List[str])` → list of vectors
  Convert **multiple chunks** into embeddings.

---

### `LiuVectorStore`

* `LiuVectorStore(embedding_model, collection_name="default_collection")`
  Create a Chroma vector store using your embedding model.

* `add_texts(texts, metadatas=None, ids=None)`
  Add chunks to the vector store. Automatically embeds them.

* `query(query_text, n_results=3)` → list of strings
  Perform a semantic search and return top matching chunks.

---

### `split_text(text, chunk_size=1000, chunk_overlap=200, clean=True)`

* Split long text into chunks with optional overlap.
* Keeps **sentences intact** and supports basic **text cleaning**.
* Returns a list of text chunks.

---

## Example: Adding metadata

```python
metadatas = [{"page": i+1} for i in range(len(chunks))]
ids = [str(i) for i in range(len(chunks))]

vectorstore.add_texts(chunks, metadatas=metadatas, ids=ids)
```

---

## Advanced Options

* Use a **different TensorFlow model**:

```python
emb_model = LiuEmbeddings(model_url="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/3")
```

* Store Chroma DB locally:

```python
from chromadb.config import Settings
vectorstore.client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))
```

* Batch embedding for speed:

```python
vectors = emb_model.embed_documents(chunks)  # all at once
```

---

## Contribution

Contributions welcome!

1. Fork the repository
2. Create a feature branch
3. Commit and push changes
4. Open a Pull Request

---

## License

MIT License — free to use and modify.

---

### ✅ Summary

`liuembeddings` is a **ready-to-use embedding + vector store module**, perfect for:

* Semantic search
* RAG pipelines
* AI applications that require fast, local embeddings

---

If you want, I can also **add a nice pipeline diagram** at the top of the README showing **Text → Chunking → Embeddings → Chroma → Query → Results**, which makes it visually very beginner-friendly.

Do you want me to create that diagram for the README?
