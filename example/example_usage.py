# example_usage.py

"""
Complete example demonstrating LiuEmbeddings framework.
"""

from liuembeddings import LiuEmbeddings
from liuembeddings import LiuVectorStore
from liuembeddings import split_text, clean_text


# Example 1: Basic embedding
print("=" * 60)
print("Example 1: Basic Embedding")
print("=" * 60)

print("\nInitializing embedding model and vector store...")

embedder = LiuEmbeddings(model_name="USE")

vector_store = LiuVectorStore(
    embedding_model=embedder,
    collection_name="ml_knowledge"
)

# Single query embedding
query_embedding = embedder.embed_query("What is machine learning?")
print(f"Query embedding dimension: {len(query_embedding)}")
print(f"First 5 values: {query_embedding[:5]}")

# Multiple documents
documents = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "Natural language processing handles text"
]

doc_embeddings = embedder.embed_documents(documents)
print(f"\nEmbedded {len(doc_embeddings)} documents")


# Example 2: Text Processing
print("\n" + "=" * 60)
print("Example 2: Text Processing & Chunking")
print("=" * 60)

long_text = """
Machine learning is a powerful and rapidly growing method of data analysis that automates the process of building analytical models. It belongs to the broader field of artificial intelligence (AI), which focuses on creating systems that can simulate aspects of human intelligence. The central idea behind machine learning is that computers can be trained to learn from data, recognize patterns, and make informed decisions with little or no direct human intervention.

In traditional programming, humans explicitly write rules and logic for a computer to follow. However, in machine learning, these rules are not hard-coded. Instead, the system uses algorithms that allow it to learn and improve automatically from experience. The more data it receives, the better it becomes at identifying patterns, relationships, and insights that may not be obvious to humans. This ability to adapt and refine itself makes machine learning extremely valuable for handling complex and large-scale data problems.

The learning process begins with observations or data — these could come from examples, historical records, direct measurements, or real-world experiences. The model examines the data to detect recurring trends, correlations, or hidden patterns. Using this knowledge, it develops a mathematical representation that can be applied to new data to predict outcomes or make decisions. Over time, as more examples are processed, the model continuously updates and becomes more accurate.

Machine learning can be applied to an astonishing variety of real-world scenarios. For example, in healthcare, it helps predict diseases, recommend treatments, and analyze medical images. In finance, it detects fraudulent transactions and supports algorithmic trading. In e-commerce, machine learning powers recommendation engines that suggest products based on user behavior. Even in daily life, it appears in voice assistants like Siri or Alexa, which improve through constant interaction with users.

There are several types of machine learning approaches, including supervised learning, unsupervised learning, semi-supervised learning, and reinforcement learning. Supervised learning involves training a model with labeled data, where the correct output is known, while unsupervised learning deals with unlabeled data, letting the system find structure on its own. Reinforcement learning, on the other hand, trains models through trial and error, rewarding successful outcomes and penalizing mistakes. Each approach has unique strengths and is suited to different kinds of problems.

Ultimately, the goal of machine learning is to enable computers to make better and more autonomous decisions over time, based on the information they encounter. By continuously learning from new data, these systems can adjust their strategies, correct errors, and enhance performance without explicit reprogramming. This capability represents a major step toward the development of intelligent systems that can assist humans in solving complex challenges and making data-driven decisions with unprecedented speed and precision.

""" 


# Split into chunks
chunks = split_text(long_text, chunk_size=400, chunk_overlap=50)
print(f"\nSplit into {len(chunks)} chunks:")
for i, chunk in enumerate(chunks[:2], 1):
    print(f"  Chunk {i}: {chunk[:60]}...")


#add cleaned chunks to vector store
vector_store.add_texts(chunks)

# Query the vector store
print("\nQuerying the vector store...")
ans =vector_store.query("What techniques improve model accuracy?", n_results=2)

ans=ans[1]  #getting only documents from returned tuple


for i, chunk in enumerate(ans, 1):
    print(f"  answer {i}: {chunk[:250]}...")



# Example 3: Vector Store Operations
print("\n" + "=" * 60)
print("Example 3: Vector Store CRUD Operations")
print("=" * 60)



# Add documents
docs_to_store = [
    "Machine learning is used for predictive analysis",
    "Deep learning requires large amounts of data",
    "Feature engineering is crucial for model performance"
]
vector_store.add_texts(docs_to_store)
print(f"Added {vector_store.count_documents()} documents")

# Search
raw,results = vector_store.similarity_search(
    "What techniques improve model accuracy?",
    n_results=1,
    with_score=.3
)

print(f"\nSearch results:")
for i, result in enumerate(results, 1):
    print(f"  Result {i} (score: {result['similarity_score']:.3f})")
    print(f"    {result['document'][:60]}...")

#print(results)




# Update document
print("\nUpdating first document...")
vector_store.update_by_id(
    results[0]['id'],
    "Machine learning drives innovation and efficiency"
)

# Search by ID
print(f"\nRetrieving by ID: {results[0]['id']}")
retrieved = vector_store.search_by_id(results[0]['id'])
print(f"  {retrieved['document']}")

# Get all documents
all_docs = vector_store.get_all()
print(f"\nTotal documents in store: {len(all_docs)}")


# Example 4: Batch Processing
print("\n" + "=" * 60)
print("Example 4: Batch Processing Large Documents")
print("=" * 60)

# Create many documents
large_doc_set = [f"Document number {i} with content about topic {i % 5}" 
                 for i in range(25)]

embedder_batch = LiuEmbeddings()
embeddings_batch = embedder_batch.embed_documents_batch(
    large_doc_set,
    batch_size=10
)
print(f"Processed {len(embeddings_batch)} documents in batches")


#batch storing (batch of 10 in each run 10 document will be added)
# 20 sample documents
texts = [f"Document {i+1}: This is sample text for document {i+1}." for i in range(100)]

# Optional metadata
metadatas = [{"source":"Batch of 5"} for i in range(100)]

vector_store.add_texts_batch(
    texts,
    batch_size=10,
    metadatas=metadatas
)

results=vector_store.search_by_metadata({"source":"Batch of 5"})

for i, result in enumerate(results, 1):
    print(f"   id: {result['id']}")
    print(f"    {result['document'][:60]}...")





# Example 5: One-Line Search one function ultimate usage
print("\n" + "=" * 60)
print("Example 5: One-Line Semantic Search")
print("=" * 60)

print("\nPerforming semantic search using liu_search... \n all in embedding solution")

document = """
Python is a high-level programming language known for its simplicity.
JavaScript is used primarily for web development.
Java is popular for enterprise applications.
C++ is known for high performance and system programming.
""" * 2

#try to always  pass text in text_document parameter while adding 

#adding and searching
raw,ans = vector_store.search(
    "What language is best for web development?",
    text_document=document,
    n_results=1
)

for i in range(len(ans)):
    print(f"  Result {i+1}: {ans[i][:90]}...")




print("\nPerforming another semantic search using search... \n only adding document")
document = """
my name is himanshu i am a data engineer working in tcs india pvt ltd.
i have experience in spark,hadoop,python,sql,azure,aws,tableau,power bi etc.
i love to work on data and build data pipelines and dashboards.
python is a high-level programming language known for its simplicity but it is not simple :).
""" 


#adding only
vector_store.search(
    text_document=document,
    chunk_size=250,
    chunk_overlap=100,
)


print("\nPerforming another semantic search using liu_search... \n only searching")
#searching only
search_results = vector_store.search(
    query="What himanshu does for a living?",
    n_results=1
)

# Example 6: Error Handling
print("\n" + "=" * 60)
print("Example 6: Error Handling")
print("=" * 60)

try:
    # This will raise an error
    embedder.embed_query("")
except ValueError as e:
    print(f"✓ Caught error: {e}")

try:
    # Invalid model
    LiuEmbeddings(model_name="INVALID")
except ValueError as e:
    print(f"✓ Caught error: {e}")

try:
    # Empty documents list
    vector_store.add_texts([])
except ValueError as e:
    print(f"✓ Caught error: {e}")


print("\n" + "=" * 60)
print("Examples completed successfully! ✅")
print("=" * 60)
