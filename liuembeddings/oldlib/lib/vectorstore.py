import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


class TFEmbeddingWrapper(EmbeddingFunction[Documents]):
    """
    Wrapper to make TensorFlow embeddings compatible with ChromaDB
    """
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        ChromaDB calls this method with a list of text documents
        """
        return self.embedding_model.embed_documents(input)


class LiuVectorStore:
    def __init__(self, embedding_model, collection_name="default_collection", persist_path="./chroma_db"):
        
        self.client = chromadb.PersistentClient(path=persist_path)
        self.embedding_function = TFEmbeddingWrapper(embedding_model)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Use get_or_create_collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}  # Set distance metric at creation
        )
        
        print(f"✅ Chroma collection '{collection_name}' is ready")

    def add_texts(self, texts, metadatas=None, ids=None):
        """
        Add chunks/documents to Chroma
        """
        if metadatas is None:
            metadatas = [{"name":self.collection_name} for _ in texts]
        if ids is None:
            ids = [str(i) for i in range(len(texts))]
        
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ Added {len(texts)} documents to Chroma")

    def query(self, query_text, n_results=3):
        """
        Perform semantic search using query_texts (recommended)
        """
        # Option 1: Let ChromaDB embed the query automatically
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results['documents'][0]
    
    def query_with_embeddings(self, query_text, n_results=3):
        """
        Alternative: Manually embed query and use query_embeddings
        """
        q_vec = self.embedding_model.embed_query(query_text)
        results = self.collection.query(
            query_embeddings=[q_vec],
            n_results=n_results
        )
        return results['documents'][0]
