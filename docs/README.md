
# LiuEmbeddings

LiuEmbeddings is a lightweight framework for semantic search built around TensorFlow-based embeddings and a ChromaDB-backed vector store. It targets small to medium projects that need fast embedding, storage, and retrieval with clear CRUD and batch APIs plus robust logging and validation out of the box.

### Features

- TensorFlow embeddings with a simple, consistent interface and model info exposure for debugging and observability.
- ChromaDB vector storage with persistence, HNSW indexing, and metadata filtering for efficient similarity search and organization.
- Comprehensive vector store CRUD operations, batch ingestion, and export to JSON for portability.
- Text preprocessing utilities like chunking with overlap to optimize retrieval quality and packing for longer documents.
- Validations, clear error messages, and integrated logging across components to support production readiness.


### Installation

- pip install liuembeddings for the latest published package version, or install from source using pip install liuembeddings . inside the cloned repository root.
- Python 3.8+ and recent TensorFlow, ChromaDB, and NumPy versions are required as listed in requirements.txt to ensure compatibility and performance.

```python
    pip install liuembeddings
```


### Quick start

- Initialize an embedder and vector store, then add documents and run a similarity search to retrieve relevant results.[^2]

```python
from liuembeddings import LiuEmbeddings, LiuVectorStore

embedder = LiuEmbeddings(model_name="USE")
store = LiuVectorStore(embedder, collection_name="my_docs")

store.add_texts([
    "Python is a programming language",
    "JavaScript is for web development"
])

results, documents = store.similarity_search(
    "What is Python?", n_results=2
)

print(documents)
```

The example shows a minimal flow: initialize, add, and search to get back matching texts quickly.


### Quick start: split_text

The following example mirrors the structure from example_usage.py and demonstrates chunking a long text, adding to the vector store, asking a question, and iterating on results while using the documented return shapes.[^2]

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


### One‑liner semantic search

- Use the vector store search method to combine chunking, ingestion, and querying in a single call for rapid prototyping.[^2]

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

This mirrors the end‑to‑end pattern shown in the examples and uses the text_document parameter name used by the vector store’s search method implementation and examples.[^2]


### Minimilistic Quick Start

- Initialize an embedder and vector store intenally not need to define but we can change model and collection.[^1]
- REFER end of the page to learn about fastquery in detailed [Go to Fastquery](#fastquery)


```python
from liuembeddings import fastquery

# Simple use: embed and search in 3 lines
text = "New York is the largest city in the United States. Washington D.C. is the capital. California is a state."


fastquery(text_document=text,)

results = fastquery(
    query="Capital of USA?",
    n_results=2   
)

for chunk in results:
    print(chunk)
```

## config.py

- use this to change deafult variable for entire app or you can give it manully during function calls

```python
from liuembeddings import LiuConfig as l

l.DEFAULT_BATCH_SIZE=32
l.DEFAULT_CHUNK_SIZE=2000
l.DEFAULT_COLLECTION_NAME='test_collection'

```


### Vector store API

Here’s a **professional and detailed `README.md`** section for your `liuembeddings/vectorstore.py` module — written in GitHub-style Markdown with clean code formatting, proper examples, and explanations of all vector store methods:

---

# 🧠 `LiuVectorStore` — Semantic Vector Storage with ChromaDB

`LiuVectorStore` is a high-level wrapper around **ChromaDB** that integrates seamlessly with **TensorFlow-based embeddings** (`LiuEmbeddings`).
It provides a complete CRUD + semantic search interface with optional batch ingestion, persistence, and metadata filtering.

---

## ⚙️ Initialization

```python
from liuembeddings import LiuEmbeddings
from liuembeddings import LiuVectorStore

# Create embeddings and initialize the vector store
embedder = LiuEmbeddings()
store = LiuVectorStore(embedder, collection_name="my_collection")
```

✅ Automatically:

* Ensures the persistence path exists
* Creates or opens a ChromaDB collection
* Logs document count and health

---

## 📥 Adding Data

### `add_texts(texts, metadatas=None, ids=None)`

Adds one or more text documents to the vector store.

```python
texts = [
    "Python is a high-level programming language.",
    "Apache Spark is a distributed data processing framework."
]
store.add_texts(texts)
```

**Features**

* Auto-generates unique IDs if not provided
* Accepts optional metadata for each document
* Performs validation on inputs

```python
store.add_texts(
    texts=["Document A", "Document B"],
    metadatas=[{"topic": "A"}, {"topic": "B"}],
    ids=["docA", "docB"]
)
```

---

### `add_texts_batch(texts, batch_size=None, metadatas=None, ids=None)`

Adds large datasets in batches to manage memory efficiently.

```python
# Example: Add 20 texts in batches of 5
data = [f"Sample document {i}" for i in range(20)]
store.add_texts_batch(data, batch_size=5)
```

🧩 Automatically splits your dataset and logs batch progress.

---

## 🔍 Querying & Search

### `query(query_text, n_results=None) -> (raw_results, documents)`

Performs **semantic search** and returns both raw and simplified results.

```python
raw, docs = store.query("What is Spark?")
print(docs)
```

Returns:

```python
(['Apache Spark is a distributed data processing framework.'])
```

---

### `similarity_search(query_text, n_results=None, with_scores=False)`

Returns the most similar documents with optional similarity scores.

```python
# Without scores
_, docs = store.similarity_search("Python language")
print(docs)

# With similarity scores
results, _ = store.similarity_search("Python language", with_scores=True)
for r in results:
    print(r["id"], r["similarity_score"], r["document"])
```

📈 When `with_scores=True`, each result includes:

```python
{
    "id": "...",
    "document": "...",
    "metadata": {...},
    "similarity_score": 0.93
}
```

---

## 🧾 Document Management

### `search_by_id(doc_id) -> dict | None`

Fetch a document and metadata by its unique ID.

```python
result = store.search_by_id("docA")
print(result)
```

Returns:

```python
{
  "id": "docA",
  "document": "Document A",
  "metadata": {"topic": "A"}
}
```

---

### `search_by_metadata(metadata_filter) -> list[dict]`

Find documents matching a specific metadata filter.

```python
docs = store.search_by_metadata({"topic": "B"})
```

Returns a list of `{id, document, metadata}` objects.

---

### `get_all() -> list[dict]`

Retrieve **all** documents and metadata from the collection.

```python
all_docs = store.get_all()
```

Useful for exporting or debugging.

---

### `update_by_id(doc_id, new_text, new_metadata=None)`

Replace a document’s text (and optionally metadata).

```python
store.update_by_id("docA", "Updated Document A", {"topic": "Updated"})
```

✔️ Preserves the same ID — ideal for maintaining references.

---

### `delete_by_id(doc_id)`

Remove a specific document by ID.

```python
store.delete_by_id("docB")
```

---

## 📊 Collection Info & Utilities

### `count_documents() -> int`

Get the total number of stored documents.

```python
print("Document count:", store.count_documents())
```

---

### `save(path)`

Export all documents and metadata to a `.json` file.

```python
store.save("backup_my_collection.json")
```

---

### `info` (property)

Quick collection overview:

```python
print(store.info)
```

Output example:

```python
{
    "name": "my_collection",
    "document_count": 42,
    "embedding_model": "TensorFlow Universal Sentence Encoder"
}
```

---

## 🪄 One-Call Convenience Search

### `search(query=None, text_document=None, chunk_size=None, chunk_overlap=None, n_results=None, collection_name=None, model_name="USE")`

End-to-end helper that:

1. Splits long documents into chunks
2. Embeds and stores them
3. Performs similarity search in one call

```python
document = """
Python is a versatile language used for data engineering,
machine learning, and automation. Spark integrates easily with Python.
"""

store.search(
    query="How does Spark work with Python?",
    text_document=document,
    chunk_size=200,
    n_results=3
)
```

---

## 🧩 Example Workflow

```python
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

---

## 🧱 Design Overview

| Feature                | Description                                                 |
| ---------------------- | ----------------------------------------------------------- |
| **Persistence**        | Uses `chromadb.PersistentClient` for disk-based collections |
| **Batch Support**      | Handles large ingestions efficiently                        |
| **CRUD Operations**    | Add, Update, Delete, Retrieve                               |
| **Semantic Search**    | Embedding-based similarity using `LiuEmbeddings`            |
| **Metadata Filtering** | Query subsets via structured filters                        |
| **Export**             | JSON serialization for backups or migration                 |

---

### Worked example: solve a retrieval question

The following example mirrors the structure from example_usage.py and demonstrates chunking a long text, adding to the vector store, asking a question, and iterating on results while using the documented return shapes.[^2]

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

This demonstrates selecting only the documents from the returned tuple after calling query, exactly as shown in the example where ans = ans is used to extract the document list.[^2]

### CRUD and scored search example

- Retrieve scored results for CRUD, update one by id, read it back, and list the total count, following the patterns used in the example.[^2]

```python
# Scored search for a targeted update
scored = store.similarity_search(
    "What techniques improve model accuracy?",
    n_results=1,
    with_scores=True
)

first = scored[^0]
store.update_by_id(first["id"], "Machine learning drives innovation and efficiency")

# Verify by id
found = store.search_by_id(first["id"])
print(found["document"])

# Count all
print("Total documents:", store.count_documents())
```

The with_scores=True shape is a list of dicts, which enables direct access to id, document, metadata, and similarity_score for downstream operations as used in the example.[^2]

### Batch ingestion and metadata filtering

- Use add_texts_batch to process large inputs, assign consistent metadata for later filtering, and fetch the subset with search_by_metadata for targeted work.[^2]

```python
# Prepare 100 documents with metadata
texts = [f"Document {i+1}: This is sample text for document {i+1}." for i in range(100)]
metas = [{"source": "Batch of 5"} for _ in range(100)]

# Ingest in batches of 10
store.add_texts_batch(texts, batch_size=10, metadatas=metas)

# Filter by metadata
subset = store.search_by_metadata({"source": "Batch of 5"})
for x in subset[:3]:
    print(x["id"], x["document"][:60], "...")
```

This mirrors the example’s approach to batched add and subsequent metadata filtering to isolate a logical group of documents.[^2]

### Text utilities

- Use split_text for chunking long content with overlap to preserve context across chunk boundaries, and clean_text if included in your setup to normalize inputs prior to embedding as shown in the README’s text utilities section and example file.[^2]



# Fastquery

LiuEmbeddings includes an advanced utility function for rapid prototyping and streamlined semantic search, `fastquery`. This section adds full documentation and usage examples for `fastquery`, and clarifies key usage expectations such as embedding model consistency and API behaviors. All major methods, including `fastquery`, are now documented with coding blocks and concise explanations for every function.

***

## 🚀 Quick Embedding (`fastquery`)

- Always define collection do not rely on default collection
- Alaways rely on default embedings or use only 1 for all

**`fastquery`** provides the fastest workflow for embedding and semantic search. It is designed for scenarios where users need to process a document and execute queries immediately—no manual collection or embedding model setup required.

**Key Features:**

- **Default Model:** Uses the `"USE"` Universal Sentence Encoder by default for embeddings.
- **Model Consistency:** The embedding model is fixed _per vector store instance_. Once texts are embedded with a given model, you cannot switch models for the same collection.
- **Single-call API:** Combines text chunking, embedding, storage, and querying in one function.
- **Minimal Setup:** No need to initialize LiuEmbeddings or LiuVectorStore directly—simply provide your text and query.

***

### Function Documentation

```python
def fastquery(
    query: str = None,
    text_document: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    n_results: int = None,
    with_scores: bool = False,
    collection_name: str = None,
    model_name: str = "USE"
) -> list:
    """
    Rapid semantic search with minimal configuration.

    - Uses a dedicated embedding model and vector store per call.
    - The embedding model is fixed for the life of the vector store/collection.
    - Ideal for quick prototyping and temporary workloads.

    Args:
        text_document: The input text/document.
        query: The string to search for.
        chunk_size: Document chunk size (default from config).
        chunk_overlap: Overlap between chunks (default from config).
        n_results: Number of top results (default from config).
        with_scores: If True, returns similarity scores.
        collection_name: Optional custom name for collection.
        model_name: Embedding model to use (default "USE").

    Returns:
        List of most relevant document chunks OR dicts if with_scores=True.

    Raises:
        ValueError/RuntimeError: For invalid input or failures.
    """
```


***

### Quickstart Example

```python
from liuembeddings import fastquery

# Simple use: embed and search in 3 lines
text = "New York is the largest city in the United States. Washington D.C. is the capital. California is a state."

fastquery.collection_name="minimal_collection"

fastquery(text_document=text,)

results = fastquery(
    query="Capital of USA?",
    n_results=2
)

for chunk in results:
    print(chunk)
```


***

### Using `fastquery` with Custom Settings

```python
# Custom chunk size and overlap

fastquery(
    text_document="Deep learning uses neural networks. Machine learning is a subset of AI.",
    chunk_size=80,
    chunk_overlap=15,
)


results = fastquery(
    query="What is machine learning?",
    n_results=1
)
print("Best answer:", results[1][0])
```


***

### Scores and Metadata

```python
# Extract results with similarity scores = 0.3

fastquery(text_document="Do pandas eat bamboo? Yes, they primarily eat bamboo.")

scored_results = fastquery(
    query="What do pandas eat?",
    n_results=1,
    with_scores=True
)

for item in scored_results[0]:
    print(f"Chunk: {item['document']}")
    print(f"Similarity score: {item['similarity_score']}")


for item in scored_results[0]:
    if item['similarity_score'] < 0.3:
        print(f"Chunk: {item['document']}")
```


***

### Notes on Model and Collection Management

- **Model Switching:** Once a vector store or collection is created with an embedding model, you cannot switch to another model for embedding/search in that collection. If you need to use a new model (e.g., USE/BERT), create a new collection:

```python
fastquery(
    text_document="...", 
    query="...", 
    model_name="USE",
    collection_name="my_new_collection"
)
```

Attempting to switch models within the same collection will result in an error.

***

## Summary Table: Quick Embedding API

| Function | Purpose | Returns | Model Switching | Use Case |
| :-- | :-- | :-- | :-- | :-- |
| fastquery | Rapid embed \& search | Chunks/results | Not allowed | Quick prototyping, temporary |
| LiuEmbeddings | Manual embed model | Embedding vectors | At instantiation | Advanced/custom workflows |
| LiuVectorStore | Full CRUD/search | Document batches | At initialization | Persistent/high-volume apps |


***

## Complete Example: End-to-End Embedding and Immediate Query

```python
from liuembeddings import fastquery

long_doc = """
The solar system includes the Sun and the objects that orbit it, such as planets,
asteroids, and comets. Planets like Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus,
and Neptune revolve around the Sun.
"""

#storing and query at same 
#use with CAUTION re ingesting same data leads to data dublication add ones and Query multiple time
results = fastquery(
    text_document=long_doc,
    query="Which planets orbit the sun?",
    n_results=3
)

for answer in results:
    print(answer)
```


***

## Final Tips

- **Use `fastquery` for fast, disposable vector stores and quick searches.**
- **Switch models only by creating new collections—existing data uses a single embedding model.**
- For larger or persistent applications, use the full LiuEmbeddings and LiuVectorStore APIs documented above for manual control, persistence, batch processing, and advanced CRUD.

***

This advanced quick embedding tutorial section is now integrated into your README for maximum developer clarity and speed in onboarding and prototyping. All function blocks provided are ready for direct copy-paste and exploration.
<span style="display:none">[^2]</span>



[^2]: liu_search.py





### Configuration

- LiuConfig exposes defaults such as chunk sizes, overlaps, n_results, and logging levels, which you can adjust at runtime to tune quality and performance for your domain.


### Requirements and project structure

- Requirements include Python 3.8+, TensorFlow 2.8+, ChromaDB 0.3+, and NumPy 1.20+, and the repository layout includes embeddings.py, vectorstore.py, utils, config, logger modules, tests, and packaging files as shown in the current README.


### Notes on return shapes and usage patterns

- query and similarity_search return a tuple of (raw_results, documents) when with_scores is False, and your code should access the second element to iterate over just the text matches, as demonstrated by assigning ans = ans before printing chunks.[^2]
- similarity_search with with_scores=True returns a list of dicts where id can be fed into update_by_id and search_by_id to perform targeted modifications and retrieval as illustrated in the example.[^2]
- search composes splitting, ingestion, and similarity search and returns the same shape as similarity_search in the default mode, enabling quick prototyping without wiring multiple calls in your application code.[^2]


### Existing examples

- The repository examples cover basic embedding, text processing, CRUD, batch operations, one‑liner search, and error handling, and the updated examples above align with those flows while clarifying parameter names and return shapes.[^2]


### Contributing

- Fork the repository, create a feature branch, commit changes, push, and open a pull request following the guidelines already present in the README to keep contributions consistent and reviewable.


### License and citation

- The project is MIT‑licensed, and a BibTeX entry is provided in the README if you cite LiuEmbeddings in academic work, keeping attribution straightforward and standardized.


### Changelog and roadmap

- The initial release includes core embedding, vector store, utilities, tests, and documentation, and the roadmap lists future enhancements like additional embedding models, REST, Docker, and advanced filtering to guide community contributions.

If you’d like this as a drop‑in replacement file, the sections above can fully replace the current README’s Quick Start and API portions while keeping badges, installation, testing, and contribution policy intact from the existing document.[^2]

<div align="center">⁂</div>

[^1]: example_usage.py

