 
# liuembeddings/__init__.py
from lib.embeddings import LiuEmbeddings
from lib.vectorstore import LiuVectorStore
from lib.utils import split_text

def liu_search(
    text_document,       # long text or document string
    query,               # query string
    chunk_size=1000, 
    chunk_overlap=200, 
    n_results=3,
    collection_name="default_collection"
):
    """
    Minimalistic function to:
    1. Chunk text
    2. Create embeddings
    3. Store in Chroma
    4. Query and return top matching chunks
    """
    # Step 1: Split text into chunks
    chunks = split_text(text_document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Step 2: Load embedding model
    emb_model = LiuEmbeddings()

    # Step 3: Create Chroma vector store
    vectorstore = LiuVectorStore(embedding_model=emb_model, collection_name=collection_name)

    # Step 4: Add chunks to vector store
    vectorstore.add_texts(chunks)

    # Step 5: Query
    results = vectorstore.query(query, n_results=n_results)

    return results
