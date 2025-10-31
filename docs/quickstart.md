# Quick Start Guide

## Installation

Install LiuEmbeddings from PyPI:

```bash
pip install liuembeddings
```

Ensure you have **Python 3.8+** installed.

## Basic Quick Start

Initialize an embedder and vector store, then add documents and run a similarity search to retrieve relevant results.

```python
from liuembeddings import LiuEmbeddings, LiuVectorStore

embedder = LiuEmbeddings(model_name="USE")
store = LiuVectorStore(embedder, collection_name="my_docs")

store.add_texts([
    "Python is a programming language",
    "JavaScript is for web development"
])

results, documents = store.similarity_search(
    "What is Python?", n_results=1
)

print(documents)
```

The example shows a minimal flow: initialize, add, and search to get back matching texts quickly.

## Split Text Example

Chunk a long text, add to the vector store, ask a question, and iterate on results:

```python
from liuembeddings import LiuEmbeddings, LiuVectorStore, split_text

# Initialize
embedder = LiuEmbeddings(model_name="USE")
store = LiuVectorStore(embedder, collection_name="ml_knowledge")

# Long text
long_text = """
Machine learning is a powerful and rapidly growing method of data analysis...
Feature engineering is crucial for model performance...
"""

# Chunk and add
chunks = split_text(long_text, chunk_size=400, chunk_overlap=50)
store.add_texts(chunks)

# Ask a question
raw, docs = store.query("What techniques improve model accuracy?", n_results=2)

# Show the matched chunks
for i, d in enumerate(docs, 1):
    print(f"Answer {i}: {d[:250]}...")
```

## One-Liner Semantic Search

Use the vector store search method to combine chunking, ingestion, and querying in a single call:

```python
from liuembeddings import LiuEmbeddings, LiuVectorStore

embedder = LiuEmbeddings()
store = LiuVectorStore(embedder, collection_name="my_docs")

long_doc = "Machine learning is a subset of AI. Deep learning uses neural networks."

# Ingest chunks and then search
store.search(
    text_document=long_doc,
    chunk_size=250,
    chunk_overlap=100
)

raw, docs = store.search(
    query="What is machine learning?",
    n_results=2
)

for d in docs:
    print(d)
```

## Minimalistic Quick Start with fastquery

Initialize an embedder and vector store internally without manual definition:

```python
from liuembeddings import fastquery

# Simple use: embed and search in 3 lines
text = "New York is the largest city in the United States. Washington D.C. is the capital. California is a state."

fastquery(text_document=text)

raw, results = fastquery(
    query="Capital of USA?",
    n_results=2   
)

for chunk in results:
    print(chunk)
```

## Configuration

Use `LiuConfig` to change default variables for your entire app or pass them manually during function calls:

```python
from liuembeddings import LiuConfig as l

l.DEFAULT_BATCH_SIZE = 32
l.DEFAULT_CHUNK_SIZE = 2000
l.DEFAULT_COLLECTION_NAME = 'test_collection'
```

**← [Introduction](introduction.md) | [API Reference](api-reference.md) →**
  