# API Reference - LiuEmbeddings & LiuVectorStore 
---

## 📑 Table of Contents


1. [LiuVectorStore](#liuvectorstore-class)
2. [LiuEmbeddings](#liuembeddings-class)
3. [Utilities](#utility-functions)
4. [Configuration](#liuconfig)
5. [fastquery](#fastquery)

---

## LiuVectorStore Class

**Module:** `liuembeddings.vectorstore`

ChromaDB-based vector store with persistent storage, CRUD operations, and semantic search.

### Initialization

```python
from liuembeddings import LiuEmbeddings, LiuVectorStore

embedder = LiuEmbeddings(model_name="MiniLM")
store = LiuVectorStore(
    embedding_model=embedder,
    collection_name="my_documents",
    persist_path="./chroma_data"
)
```

**Parameters:**
- `embedding_model` (LiuEmbeddings) - Embedder instance (required)
- `collection_name` (str) - Collection name. Default: from config
- `persist_path` (str, optional) - Path for ChromaDB storage. Default: from config

**Raises:**
- `TypeError` - If embedding_model is invalid or missing `embed_documents`
- `RuntimeError` - If ChromaDB initialization fails

---

### Adding Data

#### `add_texts(texts: List[str], metadatas: List[dict] = None, ids: List[str] = None) -> List[str]`

Add documents to the vector store.

```python
texts = [
    "Python is a programming language",
    "JavaScript runs in browsers"
]

# Simple add
doc_ids = store.add_texts(texts)

# With metadata
store.add_texts(
    texts=texts,
    metadatas=[
        {"topic": "Programming", "lang": "Python"},
        {"topic": "Web", "lang": "JavaScript"}
    ],
    ids=["doc1", "doc2"]  # Optional custom IDs
)
```

**Parameters:**
- `texts` (List[str]) - Documents to add (required, non-empty)
- `metadatas` (List[dict], optional) - Metadata for each document. If None, adds default metadata
- `ids` (List[str], optional) - Custom IDs. Auto-generated if None

**Returns:**
- `List[str]` - Document IDs

**Raises:**
- `ValueError` - If texts is empty or lengths don't match
- `TypeError` - If texts contain non-strings
- `RuntimeError` - If addition fails

---

#### `add_texts_batch(texts: List[str], batch_size: int = 32, metadatas: List[dict] = None, ids: List[str] = None) -> None`

Add texts in batches (memory efficient for large datasets).

```python
docs = [f"Document {i}: Sample..." for i in range(1000)]
metas = [{"index": i, "source": "batch"} for i in range(1000)]

store.add_texts_batch(
    texts=docs,
    batch_size=100,
    metadatas=metas
)
```

**Parameters:**
- `texts` (List[str]) - Documents to add
- `batch_size` (int) - Documents per batch. Default: 32
- `metadatas` (List[dict], optional) - Metadata for each document
- `ids` (List[str], optional) - Custom IDs

**Returns:**
- None

**Raises:**
- `ValueError` - If batch_size <= 0
- `RuntimeError` - If batch addition fails

---

### Searching & Querying

#### `query(query_text: str, n_results: int = None) -> Tuple[dict, List[str]]`

Simple semantic search returning only document texts.

```python
query = "What is Python?"
raw, documents = store.query(query, n_results=2)

for doc in documents:
    print(doc)
# Output:
# "Python is a programming language"
# "Python is used for web development"
```

**Parameters:**
- `query_text` (str) - Search query (non-empty)
- `n_results` (int, optional) - Number of results. Default: from config

**Returns:**
- `Tuple[dict, List[str]]` - (raw ChromaDB output, list of document strings)

**Raises:**
- `ValueError` - If query is empty or n_results invalid
- `RuntimeError` - If query fails

---

#### `similarity_search(query_text: str, n_results: int = None, with_score: float = None) -> Tuple[dict, List[dict]]`

Semantic search with similarity scores and metadata.

```python
query = "Tell me about Python"
raw, results = store.similarity_search(
    query_text=query,
    n_results=2,
    with_score=0.4  # Filter results with score > 0.4
)

for result in results:
    print(f"ID: {result['id']}")
    print(f"Score: {result['similarity_score']:.2f}")
    print(f"Text: {result['document']}")
    print(f"Metadata: {result['metadata']}\n")
```

**Parameters:**
- `query_text` (str) - Search query
- `n_results` (int, optional) - Number of results. Default: from config
- `with_score` (float, optional) - Minimum similarity threshold (0-1). Default: 0.4

**Returns:**
- `Tuple[dict, List[dict]]` - (raw output, list of results with id, document, metadata, similarity_score)

**Example Result:**
```python
[
    {
        'id': 'doc_0_abc123',
        'document': 'Python is a high-level programming language',
        'metadata': {'topic': 'Programming', 'lang': 'Python'},
        'similarity_score': 0.89
    },
    {
        'id': 'doc_1_def456',
        'document': 'Python is used for machine learning',
        'metadata': {'topic': 'ML', 'lang': 'Python'},
        'similarity_score': 0.76
    }
]
```

**Raises:**
- `ValueError` - If inputs invalid
- `RuntimeError` - If search fails

---

#### `search_by_id(doc_id: str) -> Optional[dict]`

Get a specific document by ID.

```python
doc = store.search_by_id("doc_0_abc123")

if doc:
    print(f"ID: {doc['id']}")
    print(f"Text: {doc['document']}")
    print(f"Metadata: {doc['metadata']}")
else:
    print("Document not found")
```

**Parameters:**
- `doc_id` (str) - Document ID

**Returns:**
- `dict` - Document with {id, document, metadata} or None

---

#### `search_by_metadata(metadata_filter: dict) -> List[dict]`

Find documents by metadata.

```python
# Find all programming documents
results = store.search_by_metadata({"topic": "Programming"})

for doc in results:
    print(f"{doc['id']}: {doc['document']}")
```

**Parameters:**
- `metadata_filter` (dict) - Metadata criteria

**Returns:**
- `List[dict]` - Matching documents

---

#### `get_all() -> List[dict]`

Get all documents in collection.

```python
all_docs = store.get_all()

print(f"Total documents: {len(all_docs)}")
for doc in all_docs:
    print(f"{doc['id']}: {doc['document'][:50]}...")
```

**Returns:**
- `List[dict]` - All documents with id, document, metadata

---

### Document Management

#### `update_by_id(doc_id: str, new_text: str, new_metadata: dict = None) -> None`

Update a document.

```python
store.update_by_id(
    doc_id="doc_0_abc123",
    new_text="Python: A powerful programming language",
    new_metadata={"updated": True, "version": "2"}
)
```

**Parameters:**
- `doc_id` (str) - Document ID
- `new_text` (str) - New document text
- `new_metadata` (dict, optional) - Updated metadata

**Raises:**
- `ValueError` - If new_text empty or doc not found
- `RuntimeError` - If update fails

---

#### `delete_by_id(doc_id: str) -> None`

Delete a document.

```python
try:
    store.delete_by_id("doc_0_abc123")
    print("Deleted successfully")
except:
    print("Document not found")
```

**Parameters:**
- `doc_id` (str) - Document ID

**Raises:**
- `RuntimeError` - If deletion fails

---

### Collection Info

#### `count_documents() -> int`

Get total documents in collection.

```python
total = store.count_documents()
print(f"Collection has {total} documents")
```

**Returns:**
- `int` - Document count

---

#### `info -> dict`

Get collection information.

```python
info = store.info

print(f"Collection: {info['name']}")
print(f"Documents: {info['document_count']}")
print(f"Model: {info['embedding_model']}")
```

**Returns:**
```python
{
    'name': str,
    'document_count': int,
    'embedding_model': dict  # model_info
}
```

---

#### `save(path: str) -> None`

Export collection to JSON.

```python
store.save("backup_collection.json")
```

**Parameters:**
- `path` (str) - File path (.json)

**Raises:**
- `RuntimeError` - If export fails

---

### Advanced Search

#### `search(query: str = None, text_document: str = None, chunk_size: int = None, chunk_overlap: int = None, n_results: int = None) -> Tuple`

Combined chunking, ingestion, and search operation.

**Mode 1: Add documents**
```python
long_doc = """
Machine learning is powerful. Feature engineering improves models. 
Deep learning uses neural networks.
"""

store.search(
    text_document=long_doc,
    chunk_size=150,
    chunk_overlap=30
)
```

**Mode 2: Search only**
```python
raw, docs = store.search(
    query="What improves model performance?",
    n_results=2
)
```

**Mode 3: Add and search**
```python
raw, docs = store.search(
    query="machine learning",
    text_document=long_doc,
    chunk_size=150,
    chunk_overlap=30,
    n_results=1
)
```

**Parameters:**
- `query` (str, optional) - Search query
- `text_document` (str, optional) - Text to chunk and add
- `chunk_size` (int, optional) - Chunk size. Default: from config
- `chunk_overlap` (int, optional) - Chunk overlap. Default: from config
- `n_results` (int, optional) - Results count. Default: from config

**Returns:**
- `Tuple[dict, List[str]]` - (raw, documents)

---


## LiuEmbeddings Class

**Module:** `liuembeddings.embeddings`

Transformer-based embeddings using Sentence-Transformers. Features model caching for performance and multiple model support.

### Initialization

```python
from liuembeddings import LiuEmbeddings

# Default model (USE - 768 dimensions)
embedder = LiuEmbeddings()

# Custom model
embedder = LiuEmbeddings(model_name="MiniLM")
```

**Parameters:**
- `model_name` (str) - Model to use. Options: `MiniLM`, `MPNetBase`, `USE`, `USEL`. Default: `"USE"`

**Raises:**
- `ValueError` - If model_name not in available models
- `RuntimeError` - If model loading fails

### Available Models

| Model | ID | Dimension | Size (MB) | Accuracy | Best For |
|-------|----|-----------|-----------|-----------|---------| 
| `MiniLM` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | 22 | 0.78 | ⚡ Speed |
| `MPNetBase` | `sentence-transformers/all-mpnet-base-v2` | 768 | 420 | 0.82 | ⚖️ Balanced |
| `USE` | `intfloat/e5-base-v2` | 768 | 300 | 0.84 | 🎯 Quality (default) |
| `USEL` | `BAAI/bge-base-en-v1.5` | 1024 | 1024 | 0.86 | 🏆 Premium |

---


### External Embeddings Model

You can add a new embedding model by modifying `LiuConfig.AVAILABLE_MODELS`. While you can use any embedding model of your choice, it is recommended to use the predefined models like **USE** or **USEL** for compatibility.

#### Adding an External Embedding Model

1. Ensure the model is compatible with HuggingFace Hub.
2. Provide the model URL, embedding dimension, and a custom name.

```python
from liuembeddings import LiuEmbeddings, LiuVectorStore, LiuConfig

# Add a custom external embedding model
LiuConfig.AVAILABLE_MODELS['MPNetMini'] = {
    'id': "sentence-transformers/all-mpnet-base-v2",  # HuggingFace MPNet variant
    'dimension': 384,
    'full_name': 'MPNet Mini', # OPTIONAL
    'size': 90,  # OPTIONAL MB
    'description': 'Smaller MPNet variant, faster than full base', # OPTIONAL
    'accuracy': 0.80 # OPTIONAL
}

# Initialize the custom embedder
custom_embedder = LiuEmbeddings('MPNetMini')


custom_vector = LiuVectorStore(
    embedding_model=custom_embedder,
    collection_name="knowledge-NNLM"
)

# Multiple documents
documents = [
    "all boy's lovewin prize",
    "all boy's love money",
    "all boy's love protein",
    "lorem ipsum lorem ipsum",
    "loremipsum lorem iplorem"
]

custom_vector.add_texts(documents,)

raw,docs=custom_vector.search('what all the boys love')

print("answer:")
for i in docs:
    print(i)

>>> answer:
    all boy's love money
    all boy's love protein
    all boy's lovewin prize

print(raw)
```


### Properties

#### `model_info -> dict`

Get information about the loaded model.

```python
info = embedder.model_info

print(info)
# {
#     'name': 'USE',
#     'id': 'intfloat/e5-base-v2',
#     'dimension': 768,
#     'size_mb': 300,
#     'description': 'High-quality semantic embeddings...',
#     'accuracy': 0.84
# }
```

**Returns:**
```python
{
    'name': str,           # Model name
    'id': str,             # HuggingFace model ID
    'dimension': int,      # Output dimension
    'size_mb': int,        # File size in MB
    'description': str,    # Model description
    'accuracy': float      # Accuracy score
}
```

---




## Utility Functions

**Module:** `liuembeddings.utils`

**`split_text(...) -> List[str]`**

Split text into overlapping chunks.

```python
from liuembeddings import split_text

text = "Machine learning is powerful. Feature engineering improves performance."

chunks = split_text(
    text=text,
    chunk_size=150,
    chunk_overlap=30,
    split_by_sentences=True,
    clean_before_split=True,
    lowercase=False
)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk}")
```

**Parameters:**
- `text` (str) - Text to split
- `chunk_size` (int) - Characters per chunk. Default: 1000
- `chunk_overlap` (int) - Overlap between chunks. Default: 200
- `split_by_sentences` (bool) - Split at sentences first. Default: True
- `clean_before_split` (bool) - Clean text first. Default: True
- `lowercase` (bool) - Convert to lowercase. Default: False

**Returns:**
- `List[str]` - Text chunks

**Raises:**
- `TypeError` - If text is not string
- `ValueError` - If parameters invalid or text empty



**`clean_text(...) -> str`**

Normalize and clean text.

```python
from liuembeddings import clean_text

messy = "   Hello   WORLD!!! \n\n How are YOU?   "
cleaned = clean_text(messy, lowercase=True)
# Output: "hello world how are you"
```

**Parameters:**
- `text` (str) - Text to clean
- `lowercase` (bool) - Convert to lowercase. Default: False
- `remove_extra_spaces` (bool) - Collapse spaces. Default: True
- `remove_newlines` (bool) - Remove newlines. Default: True

**Returns:**
- `str` - Cleaned text



**`clean(raw: dict) -> List[dict]`**

Convert raw ChromaDB output to clean dictionary format.

```python
raw, _ = store.similarity_search("query")
clean_results = clean(raw)

for result in clean_results:
    print(f"ID: {result['id']}")
    print(f"Document: {result['document']}")
    print(f"Metadata: {result['metadata']}")
    print(f"Distance: {result['distance']}\n")
```

**Parameters:**
- `raw` (dict) - Raw ChromaDB query output

**Returns:**
- `List[dict]` - Cleaned results with {id, document, metadata, distance}



**`validate_texts(texts: List[str], min_length: int = 1) -> bool`**

Validate a list of texts.

```python
from liuembeddings import validate_texts

texts = ["Hello", "World"]
is_valid = validate_texts(texts, min_length=1)
print(is_valid)  # True
```

**Parameters:**
- `texts` (List[str]) - Texts to validate
- `min_length` (int) - Minimum length required. Default: 1

**Returns:**
- `bool` - True if valid

**Raises:**
- `TypeError` - If not list or items not strings
- `ValueError` - If empty or items too short



***`batch_generator(items: List, batch_size: int)`***
Generator for batching items.

```python
from liuembeddings import batch_generator

items = list(range(100))
for batch in batch_generator(items, batch_size=10):
    print(f"Batch: {batch}")
```

**Parameters:**
- `items` (List) - Items to batch
- `batch_size` (int) - Batch size

**Yields:**
- List - Batches of items

---

## LiuConfig

**Module:** `liuembeddings.config`

Global configuration class.

```python
from liuembeddings import LiuConfig

# View models
print(LiuConfig.AVAILABLE_MODELS)

# Change defaults
LiuConfig.DEFAULT_CHUNK_SIZE = 2000
LiuConfig.DEFAULT_BATCH_SIZE = 64
```

### Configuration Constants

```python
# Models
AVAILABLE_MODELS = {
    "MiniLM": {...},
    "MPNetBase": {...},
    "USE": {...},
    "USEL": {...}
}

# Chunking
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Vector Store
DEFAULT_VECTOR_PATH = "./liu_db"
DEFAULT_COLLECTION_NAME = "default_collection"
DISTANCE_METRIC = "cosine"

# Search
DEFAULT_N_RESULTS = 3
MAX_N_RESULTS = 100

# Batch Processing
DEFAULT_BATCH_SIZE = 100

# Model Caching
ENABLE_MODEL_CACHE = True

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Similarity
DEFAULT_SIMILARITY_SEARCH_SCORE_THRESHOLD = 0.4
```

---

## fastquery

**Module:** `liuembeddings.liu_search`

## Fastquery

LiuEmbeddings includes an advanced utility function for rapid prototyping and streamlined semantic search, `fastquery`. This section adds full documentation and usage examples for `fastquery`, and clarifies key usage expectations such as embedding model consistency and API behaviors. All major methods, including `fastquery`, are now documented with coding blocks and concise explanations for every function.

***

### 🚀 Quick Embedding (`fastquery`)

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
fastquery(
        query: str=None,
        text_document: str,
        chunk_size: int, #deafult from config file
        chunk_overlap: int, #deafult from config file
        n_results: int, #deafult from config file
        with_score: float = None, #deafult from config file
        collection_name: str = fastquery.collection_name,
        model_name: str = fastquery.model_name 
) -> list:
        """
        One-line semantic search function.

        note -> use the same model for embedding and searching.
        by default its use "USE" model.
        
        Combines chunking, embedding, storage, and search in a single call.
        Perfect for quick prototyping and small applications.
        
        Args:
            text_document: Long text or document to search within
            query: Query string to search for
            chunk_size: Size of text chunks (default: from config)
            chunk_overlap: Overlap between chunks (default: from config)
            n_results: Number of results to return (default: from config)
            with_score: give smilarity search like answer
            collection_name: Name for the vector store collection (default: from config)
            model_name: Embedding model to use (default: "USE")
            
        Returns:
            List of most similar chunks from the document
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If operation failsalueError/RuntimeError: For invalid input or failures.
    """
```


***

### ⚡ Quickstart Example

- The fastquery utility provides a minimal setup for embedding and querying text within your vector database.
- It automatically handles model loading, text chunking, and search retrieval in just a few lines of code.


```python
from liuembeddings import fastquery

# Simple use: embed and search in 3 lines
text = "New York is the largest city in the United States. Washington D.C. is the capital. California is a state."

fastquery.collection_name="minimal_collection"

fastquery(text_document=text,)

raw,results = fastquery(
    query="Capital of USA?",
    n_results=2
)

for chunk in results:
    print(chunk)
```

**🔹 Using Class Variables**
- You can configure fastquery globally before calling it.
- These class variables act as persistent defaults until changed or overridden.
You can configure **`fastquery`** globally **before calling it**.
These class variables act as **persistent defaults** until they are changed or overridden.

You can customize `fastquery` behavior in **three ways**:

| Method                            | Description                                                                          | Recommended Use                        |
| :-------------------------------- | :----------------------------------------------------------------------------------- | :------------------------------------- |
| **Class Variables**               | Set once and apply globally for all future calls.                                    | ✅ *Easy and Recommended*              |
| **Function Parameters**           | Define per call — overrides both class and global defaults.                          | Use for temporary or dynamic settings. |
| **Global Defaults (`LiuConfig`)** | Automatically used when neither class variables nor function parameters are defined. | Used as fallback configuration.        |

```python
from liuembeddings import fastquery
fastquery.collection_name='liu-collection'  
fastquery.model_name='USE'
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


raw,results = fastquery(
    query="What is machine learning?",
    n_results=1
)
print("Best answer:", results[1][0])
```


### Scores and Metadata

**Adding Documents to a Collection** collection for later querying.

```python
from liuembeddings import fastquery

document = """
Luna loves exploring the night sky. Every weekend, she sets up her telescope on the rooftop to watch distant galaxies.
Her favorite constellation is Orion, and she can identify it even without a telescope.
Last month, she discovered a small comet passing near Jupiter and recorded its movement in her astronomy journal.
"""

# Add document to collection "story_collection"
fastquery(
    text_document=document,
    n_results=5,       
    collection_name="story_collection"
)
```

> ⚠️ **Note:** `n_results` specifies the maximum number of similar results to retrieve when querying.



**Querying the Collection**

When `with_score=0.4` (default), `fastquery` returns:
* `raw`: the raw output from the database or retrieval engine.
* `document`: a **list of matching documents**.

```python
raw, ans = fastquery(
    query="What celestial object did Luna discover?",
    collection_name="story_collection"
)

for item in ans:
    print(f"Answer: {item}")
```

**Example Output:**

```
Answer: Last month, she discovered a small comet passing near Jupiter and recorded its movement in her astronomy journal.
```

When `with_score=.5`, `fastquery` returns:
* `raw`: the raw retrieval output.
* `ans`: a **list of dictionaries**, each containing:

  * `id` – the document ID in the collection
  * `document` – the text content
  * `metadata` – metadata associated with the document
  * `similarity_score` – similarity between the query and the document

```python
raw, ans = fastquery(
    query="What celestial object did Luna discover?",
    with_score=0.5,
    collection_name="story_collection"
)

for item in ans:
    print(f"id: {item['id']}")
    print(f"Document: {item['document']}")
    print(f"Metadata: {item['metadata']}")
    print(f"Similarity score: {item['similarity_score']}")
```

**Example Output:**

```
id: doc_1_1761378265744
Document: Last month, she discovered a small comet passing near Jupiter and recorded its movement in her astronomy journal.
Metadata: {'source': 'story_collection'}
Similarity score: 0.41
```


- You can filter results by similarity score to get only the most relevant documents:

```python
for item in ans:
    if item['similarity_score'] < 0.5:
        print(f"Answer: {item['document']}")
```
**Example Output:**
```
Answer: Last month, she discovered a small comet passing near Jupiter and recorded its movement in her astronomy journal.
```

> This allows you to exclude low-relevance documents from your results.



### Summary

* **Adding documents:** `fastquery(text_document, collection_name)`
* **Querying documents:** `fastquery(query, collection_name)`
* **Optional similarity scores:** Use `with_score` to get IDs, metadata, and similarity values.
* **Filtering:** You can filter results by similarity score for more precise retrieval.

This function is particularly useful for **quick semantic search**, **QA over text collections**, and **vector database integrations**.

---

### Notes on Model and Collection Management

- **Model Switching:** Once a vector store or collection is created with an embedding model, you cannot switch to another model for embedding/search in that collection. If you need to use a new model (e.g., USE/USEL), create a new collection:

```python
fastquery(
    text_document="...", 
    query="...", 
    model_name="USE",
    collection_name="my_new_collection"
)
```

Attempting to switch models within the same collection will result in an error.



### Summary Table: Quick Embedding API

| Function | Purpose | Returns | Model Switching | Use Case |
| :-- | :-- | :-- | :-- | :-- |
| fastquery | Rapid embed \& search | Chunks/results | Not allowed | Quick prototyping, temporary |
| LiuVectorStore | Full CRUD/search | Document batches | At initialization | Persistent/high-volume apps |



### Complete Example: End-to-End Embedding and Immediate Query

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
> ⚠️ **Note:** Make sure the embedding model is compatible with the hub and that the dimensions match your configuration.


### Final Tips

- **Use `fastquery` for fast, disposable vector stores and quick searches.**
- **Switch models only by creating new collections—existing data uses a single embedding model.**
- For larger or persistent applications, use the full LiuEmbeddings and LiuVectorStore APIs documented above for manual control, persistence, batch processing, and advanced CRUD.



## Advance Embedding 

**Methods used by Liuemedding Internally to work with vector DB**
- **we can also use for embedding visulization if needed**

**It has three function**
- `embed_query(text: str) -> List[float]`
- `embed_documents(texts: List[str]) -> List[List[float]]`
- `embed_documents_batch(texts: List[str], batch_size: int = None) -> List[List[float]]`

### embed_query
Generate embedding for a single query string.
- `embed_query(text: str) -> List[float]`

```python
embedder = LiuEmbeddings(model_name="USE")
embedding = embedder.embed_query("What is machine learning?")

print(len(embedding))  # 768 (for USE model)
print(embedding[:5])   # [-0.004, -0.072, -0.060, -0.007, -0.022]
```


### embed_documents
Embed multiple documents at once.
- `embed_documents(texts: List[str]) -> List[List[float]]`

```python
documents = [
    "Python is a programming language",
    "Machine learning requires data",
    "Data science uses statistics"
]

embeddings = embedder.embed_documents(documents)

print(f"Generated {len(embeddings)} embeddings")
print(f"Each embedding has {len(embeddings[0])} dimensions")
```


---

## Complete Workflow Example

```python
from liuembeddings import (
    LiuEmbeddings,
    LiuVectorStore,
    split_text,
    LiuConfig
)

# 1. Initialize embedder
embedder = LiuEmbeddings(model_name="MiniLM")

# 2. Create vector store
store = LiuVectorStore(embedder, "knowledge_base")

# 3. Add documents
docs = [
    "Python is a programming language",
    "Machine learning requires data",
    "Data science combines stats and coding"
]
store.add_texts(docs, metadatas=[{"type": "intro"} for _ in docs])

# 4. Search
query = "What is Python?"
raw, results = store.similarity_search(query, n_results=2)

# 5. Process results
for result in results:
    print(f"Score: {result['similarity_score']:.2f}")
    print(f"Text: {result['document']}\n")

# 6. Update
store.update_by_id(results[0]['id'], "Python is a powerful language")

# 7. Get info
print(f"Total docs: {store.count_documents()}")
print(f"Collection: {store.info}")
```

---

## Error Handling

```python
from liuembeddings import LiuEmbeddings, LiuVectorStore

try:
    embedder = LiuEmbeddings(model_name="InvalidModel")
except ValueError as e:
    print(f"Model error: {e}")

try:
    store = LiuVectorStore(None, "test")  # Invalid embedder
except TypeError as e:
    print(f"Type error: {e}")

try:
    results = store.query("")  # Empty query
except ValueError as e:
    print(f"Query error: {e}")
```

---

## Performance Tips

1. **Choose right model:** MiniLM for speed, USE for quality
2. **Use batch methods:** `embed_documents_batch()` for large sets
3. **Optimize chunks:** Balance context vs precision (typical: 200-500 chars)
4. **Metadata filtering:** Reduce search space before similarity search
5. **Model caching:** Embedder caches models automatically
6. **Batch search:** Process multiple queries at once when possible

---

**Version:** 2.0.0  
**Last Updated:** October 31, 2025 
**Status:** ✅ Verified against source code

 **← [quickstart](quickstart.md) | [Examples & Workflows](examples.md) →**
