# Examples & Workflows

Complete real-world usage patterns and workflows for LiuEmbeddings.

---

## 📚 Table of Contents

1. [Basic Example Workflow](#basic-example-workflow)
2. [Retrieval Question Example](#retrieval-question-example)
3. [CRUD and Scored Search](#crud-and-scored-search-example)
4. [Batch Ingestion](#batch-ingestion-and-metadata-filtering)
5. [Text Chunking](#text-utilities)
6. [Embeddings Generation](#embedding-generation)
7. [fastquery Quick Search](#fastquery-quick-search)
8. [One-liner Semantic Search](#one-liner-semantic-search)
9. [Cleaning Raw Output](#cleaning-raw-output)
10. [External Embedding Models](#external-embedding-models)
11. [Advanced Workflows](#advanced-workflows)

---

## Basic Example Workflow

Create embeddings, initialize vector store, add documents, and perform search:

```python
from liuembeddings import LiuEmbeddings, LiuVectorStore

# Create embeddings
embedder = LiuEmbeddings()

# Initialize vector store
store = LiuVectorStore(embedder, collection_name="knowledge_base")

# Add documents
texts = ["AI is transforming industries.", "Data engineering powers analytics."]
store.add_texts(texts)

# Perform search
_, docs = store.similarity_search("What is AI?")
print("Search results:", docs)

# Get collection info
print(store.info)
```

**Output:**
```
Search results: [{'id': 'doc_0_...', 'document': 'AI is transforming industries.', 'metadata': {...}, 'similarity_score': 0.92}]
```

---

## Retrieval Question Example

Chunk a long text, add to vector store, ask a question, and iterate on results:

```python
from liuembeddings import LiuEmbeddings, LiuVectorStore, split_text

# Initialize
embedder = LiuEmbeddings(model_name="USE")
store = LiuVectorStore(embedder, collection_name="ml_knowledge")

# Long text
long_text = """
Machine learning is a powerful and rapidly growing method of data analysis.
Feature engineering is crucial for model performance.
Neural networks are inspired by biological systems.
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

---

## CRUD and Scored Search Example

Retrieve scored results for CRUD, update one by id, read it back, and list the total count:

```python
from liuembeddings import LiuEmbeddings, LiuVectorStore

embedder = LiuEmbeddings()
store = LiuVectorStore(embedder, collection_name="tech_docs")

# Add some documents
texts = [
    "Python is a high-level programming language",
    "Machine learning requires data processing",
    "Feature engineering improves model accuracy"
]
store.add_texts(texts)

# Scored search for a targeted update
raw, first = store.similarity_search(
    "What improves model performance?",
    n_results=1,
)

print(f"Found: {first[0]['document']}")

# Update by ID
store.update_by_id(first[0]["id"], "Feature engineering drives machine learning success")

# Verify by id
found = store.search_by_id(first[0]["id"])
print(f"Updated: {found['document']}")

# Count all
print(f"Total documents: {store.count_documents()}")
```

---

## Batch Ingestion and Metadata Filtering

Use batch ingestion to process large inputs, assign consistent metadata, and fetch subsets:

```python
from liuembeddings import LiuEmbeddings, LiuVectorStore

embedder = LiuEmbeddings()
store = LiuVectorStore(embedder, collection_name="batch_collection")

# Prepare 100 documents with metadata
texts = [f"Document {i+1}: This is sample text for document {i+1}." for i in range(100)]
metas = [{"source": "Batch Source", "index": i} for i in range(100)]

# Ingest in batches of 10
store.add_texts_batch(texts, batch_size=10, metadatas=metas)

# Filter by metadata
subset = store.search_by_metadata({"source": "Batch Source"})
print(f"Found {len(subset)} documents with metadata filter")

# Show first 3
for x in subset[:3]:
    print(f"- {x['id']}: {x['document'][:60]}...")

# Search within metadata
results = store.search_by_metadata({"index": 50})
print(f"\nDocument at index 50: {results[0]['document']}")
```

**Key Features:**
- ✅ Efficient batch processing for large datasets
- ✅ Metadata-based filtering for organization
- ✅ Automatic progress logging

---

## Text Utilities

Use `split_text` for chunking long content with overlap to preserve context:

```python
from liuembeddings import split_text, clean_text

# Sample long text
long_text = """
Machine learning is a subset of artificial intelligence.
It focuses on algorithms that learn from data.
Neural networks are inspired by the brain.
Deep learning uses multiple layers.
"""

# Chunk text with overlap
chunks = split_text(
    text=long_text,
    chunk_size=150,
    chunk_overlap=30,
    split_by_sentences=True
)

print(f"Created {len(chunks)} chunks:")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")

# Clean messy text
messy = "   Hello   WORLD!!!  \n\n  How are YOU?   "
cleaned = clean_text(messy, lowercase=True)
print(f"\nOriginal: {repr(messy)}")
print(f"Cleaned: {repr(cleaned)}")
```

**Output:**
```
Created 3 chunks:
Chunk 1: Machine learning is a subset of artificial intelligence. It focuses on algorithms...
Chunk 2: algorithms that learn from data. Neural networks are inspired by the brain. Deep...
Chunk 3: Deep learning uses multiple layers.

Original: '   Hello   WORLD!!!  \n\n  How are YOU?   '
Cleaned: 'hello world how are you'
```

---

## Embedding Generation

Generate embeddings for single queries and multiple documents:

### Single Query Embedding

```python
from liuembeddings import LiuEmbeddings

print("Initializing embedding model")
embedder = LiuEmbeddings(model_name="USE")

query_embedding = embedder.embed_query("What is machine learning?")

print(f"Query embedding dimension: {len(query_embedding)}")
print(f"First 5 values: {query_embedding[:5]}")
```

**Output:**
```
Query embedding dimension: 768
First 5 values: [-0.004198566, -0.072232731, -0.060910277, -0.007246587, -0.022054186]
```

---

### Multiple Documents Embedding

```python
from liuembeddings import LiuEmbeddings

embedder = LiuEmbeddings()  # DEFAULT: USE

documents = [
    "quickly bring the cash",
    "rush and get the money",
    "this boy loves potato"
]

doc_embeddings = embedder.embed_documents(documents)

print(f"Embedded {len(doc_embeddings)} documents")
for i, emb in enumerate(doc_embeddings):
    print(f"Doc {i+1} embedding (first 5): {emb[:5]}")
```

**Output:**
```
Embedded 3 documents
Doc 1 embedding (first 5): [-0.044405017, -0.059026677, 0.012156504, 0.035481732, 0.064193733]
Doc 2 embedding (first 5): [0.045886047, -0.074626729, 0.077477388, 0.004644655, 0.070818394]
Doc 3 embedding (first 5): [0.058161151, 0.025409227, 0.001942495, 0.029804586, -0.035508242]
```

---

### Batch Embedding for Large Datasets

```python
from liuembeddings import LiuEmbeddings

embedder = LiuEmbeddings(model_name="MiniLM")  # Faster model

# Generate 1000 documents
documents = [f"Document {i}: Sample text for document {i}." for i in range(1000)]

# Embed in batches (memory efficient)
embeddings = embedder.embed_documents_batch(
    texts=documents,
    batch_size=100
)

print(f"Successfully embedded {len(embeddings)} documents")
print(f"Embedding dimension: {len(embeddings[0])}")
```

---

## fastquery Quick Search

For rapid prototyping with minimal setup:

### Adding Documents

```python
from liuembeddings import fastquery

document = """
Luna loves exploring the night sky. Every weekend, she sets up her telescope on the rooftop.
Her favorite constellation is Orion.
Last month, she discovered a small comet passing near Jupiter.
"""

# Add document to collection
fastquery(
    text_document=document,
    n_results=5,
    collection_name="story_collection"
)

print("Document added to collection")
```

---

### Querying Documents

```python
from liuembeddings import fastquery

# Query the collection
raw, answers = fastquery(
    query="What celestial object did Luna discover?",
    collection_name="story_collection",
    n_results=2
)

for i, answer in enumerate(answers, 1):
    print(f"Answer {i}: {answer}")
```

**Output:**
```
Answer 1: Last month, she discovered a small comet passing near Jupiter and recorded its movement.
```

---

### Getting Similarity Scores

```python
from liuembeddings import fastquery

raw, results = fastquery(
    query="What celestial object did Luna discover?",
    with_score=0.5,  # Return detailed results with scores
    collection_name="story_collection",
    n_results=2
)

for item in results:
    print(f"ID: {item['id']}")
    print(f"Document: {item['document']}")
    print(f"Score: {item['similarity_score']:.2f}")
    print(f"Metadata: {item['metadata']}\n")
```

**Output:**
```
ID: doc_1_1761378265744
Document: Last month, she discovered a small comet passing near Jupiter...
Score: 0.89
Metadata: {'source': 'story_collection'}
```

---

### Filtering Results by Score

```python
from liuembeddings import fastquery

raw, results = fastquery(
    query="What is astronomy?",
    with_score=0.5,
    collection_name="story_collection",
    n_results=5
)

# Filter for high-relevance results only
high_relevance = [item for item in results if item['similarity_score'] > 0.7]

print(f"High relevance results (> 0.7): {len(high_relevance)}")
for item in high_relevance:
    print(f"- {item['document'][:80]}... ({item['similarity_score']:.2f})")
```

---

## One-liner Semantic Search

Combine chunking, ingestion, and querying in a single call:

```python
from liuembeddings import LiuEmbeddings, LiuVectorStore

embedder = LiuEmbeddings()
store = LiuVectorStore(embedder, collection_name="quick_search")

long_doc = """
Machine learning is a subset of AI.
Deep learning uses neural networks.
Feature engineering improves model performance.
"""

# Ingest and search in one call
raw, docs = store.search(
    text_document=long_doc,
    query="What improves models?",
    chunk_size=250,
    chunk_overlap=100,
    n_results=1
)

for d in docs:
    print(d)
```

---

## Cleaning Raw Output

Convert raw ChromaDB output to clean, usable format:

```python
from liuembeddings import LiuEmbeddings, LiuVectorStore, split_text, clean

# Initialize and add documents
embedder = LiuEmbeddings(model_name="USE")
store = LiuVectorStore(embedder, collection_name="ml_knowledge")

long_text = """
Machine learning is powerful. Feature engineering is crucial.
Neural networks learn patterns.
"""

chunks = split_text(long_text, chunk_size=400, chunk_overlap=50)
store.add_texts(chunks)

# Query
raw, docs = store.query("What is feature engineering?", n_results=2)

# Clean raw output
clean_output = clean(raw)

print("Clean Output:")
for item in clean_output:
    print(f"ID: {item['id']}")
    print(f"Document: {item['document']}")
    print(f"Metadata: {item['metadata']}")
    print(f"Distance: {item['distance']:.4f}\n")
```

**Output:**
```
Clean Output:
ID: doc_0_1761401259744_1b0cf4
Document: Machine learning is powerful. Feature engineering is crucial.
Metadata: {'source': 'ml_knowledge'}
Distance: 0.7754
```

---

## External Embedding Models

Add custom embedding models from HuggingFace:

```python
from liuembeddings import LiuEmbeddings, LiuVectorStore, LiuConfig

# Add a custom embedding model
LiuConfig.AVAILABLE_MODELS['MPNetMini'] = {
    'id': "sentence-transformers/all-mpnet-base-v2",
    'dimension': 384,
    'full_name': 'MPNet Mini',
    'size': 90,
    'description': 'Smaller MPNet variant, faster than full base',
    'accuracy': 0.80
}

# Initialize with custom model
custom_embedder = LiuEmbeddings('MPNetMini')
custom_store = LiuVectorStore(
    embedding_model=custom_embedder,
    collection_name="knowledge_base"
)

# Use it normally
documents = [
    "all boys love winning prizes",
    "all boys love money",
    "all boys love protein",
    "lorem ipsum text"
]

custom_store.add_texts(documents)

raw, docs = custom_store.search('what do all the boys love')

print("Search results:")
for doc in docs:
    print(f"- {doc}")
```

---

## Advanced Workflows

### Multi-Document Processing Pipeline

```python
from liuembeddings import LiuEmbeddings, LiuVectorStore, split_text

# Initialize
embedder = LiuEmbeddings(model_name="USE")
store = LiuVectorStore(embedder, collection_name="document_pipeline")

# Multiple long documents
documents = {
    "python_guide": """
    Python is versatile. It's used in web, data science, and automation.
    Libraries like pandas and numpy are essential for data work.
    """,
    "ml_basics": """
    Machine learning requires understanding algorithms.
    Supervised learning includes classification and regression.
    Unsupervised learning includes clustering.
    """
}

# Process each document
for name, content in documents.items():
    chunks = split_text(content, chunk_size=200, chunk_overlap=40)
    metadata = [{"source": name} for _ in chunks]
    store.add_texts(chunks, metadatas=metadata)

# Query across all
raw, results = store.similarity_search("What is machine learning?", n_results=3)

for r in results:
    print(f"Source: {r['metadata']['source']} | Score: {r['similarity_score']:.2f}")
    print(f"Content: {r['document'][:80]}...\n")
```

---

### Question-Answering System

```python
from liuembeddings import LiuEmbeddings, LiuVectorStore, split_text

embedder = LiuEmbeddings()
store = LiuVectorStore(embedder, collection_name="qa_system")

# Knowledge base
knowledge_base = """
Q: What is Python?
A: Python is a high-level programming language known for simplicity.

Q: How does machine learning work?
A: Machine learning learns patterns from data without explicit programming.

Q: What is deep learning?
A: Deep learning uses neural networks with multiple layers to learn complex patterns.
"""

chunks = split_text(knowledge_base, chunk_size=200, chunk_overlap=50)
store.add_texts(chunks)

# Answer user questions
questions = [
    "Tell me about Python",
    "How does ML work?",
    "What is deep learning?"
]

for q in questions:
    raw, answers = store.similarity_search(q, n_results=1)
    print(f"Q: {q}")
    print(f"A: {answers[0]['document']}\n")
```

---

## Configuration & Custom Settings

Modify default settings for your project:

```python
from liuembeddings import LiuConfig, LiuEmbeddings, LiuVectorStore

# Customize configuration
LiuConfig.DEFAULT_BATCH_SIZE = 32
LiuConfig.DEFAULT_CHUNK_SIZE = 2000
LiuConfig.DEFAULT_COLLECTION_NAME = 'test_collection'
LiuConfig.DEFAULT_N_RESULTS = 5

# Now all components use these settings
embedder = LiuEmbeddings()
store = LiuVectorStore(embedder)  # Uses test_collection

print(f"Collection: {LiuConfig.DEFAULT_COLLECTION_NAME}")
print(f"Chunk Size: {LiuConfig.DEFAULT_CHUNK_SIZE}")
```

---

## Design Overview

| Feature                | Description                                                 |
| ---------------------- | ----------------------------------------------------------- |
| **Persistence**        | Uses `chromadb.PersistentClient` for disk-based collections |
| **Batch Support**      | Handles large ingestions efficiently                        |
| **CRUD Operations**    | Add, Update, Delete, Retrieve                               |
| **Semantic Search**    | Embedding-based similarity using `LiuEmbeddings`            |
| **Metadata Filtering** | Query subsets via structured filters                        |
| **Export**             | JSON serialization for backups or migration                 |
| **Text Processing**    | Automatic chunking with overlap preservation                |
| **Model Flexibility**   | Support for multiple embedding models                       |

---

## Tips & Best Practices

✅ **Always define collection_name** - Avoids accidental data mixing
✅ **Use batch processing** - For large datasets (>1000 documents)
✅ **Set consistent models** - Don't switch models for same collection
✅ **Leverage metadata** - For filtering and organization
✅ **Test locally first** - Before deploying to production
✅ **Monitor collection size** - Use `count_documents()` to track growth

---

**← [API Reference](api-reference.md) | [Docs](docs.md) →** 