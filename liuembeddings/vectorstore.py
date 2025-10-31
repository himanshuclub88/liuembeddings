# liuembeddings/vectorstore.py

"""
Vector store implementation using ChromaDB with HuggingFace  embeddings.
"""

from typing import List, Dict, Optional, Any, Tuple
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import os
import json
from .logger import setup_logger
from .config import LiuConfig
from .embeddings import LiuEmbeddings
from .utils import split_text,clean
import time, uuid

logger = setup_logger(__name__)


class TFEmbeddingWrapper(EmbeddingFunction[Documents]):
    """
    Wrapper to make HuggingFace  embeddings compatible with ChromaDB.
    
    This class adapts the LiuEmbeddings interface to ChromaDB's
    EmbeddingFunction protocol.
    """
    
    def __init__(self, embedding_model):
        """
        Initialize wrapper with embedding model.
        
        Args:
            embedding_model: LiuEmbeddings instance
        """
        self.embedding_model = embedding_model
        logger.debug("Initialized TFEmbeddingWrapper")
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for documents.
        
        Args:
            input: List of documents to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.embedding_model.embed_documents(input)
            logger.debug(f"Generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Embedding wrapper error: {str(e)}")
            raise


class LiuVectorStore:
    """
    Vector store for embedding and searching documents using ChromaDB.
    
    Features:
        - Store embeddings with metadata
        - Semantic search
        - CRUD operations (Create, Read, Update, Delete)
        - Batch operations
        - Persistence to disk
        - Metadata filtering
    
    Attributes:
        client: ChromaDB client
        collection: Active ChromaDB collection
        collection_name: Name of current collection
        embedding_model: Reference to embedding model
    
    Example:
        >>> embedder = LiuEmbeddings()
        >>> store = LiuVectorStore(embedder, "my_docs")
        >>> store.add_texts(["Hello", "World"])
        >>> results = store.similarity_search("Greetings")
    """
    
    def __init__(
        self, 
        embedding_model, 
        collection_name: str,
        persist_path: str = None
    ) -> None:
        """
        Initialize vector store.
        
        Args:
            embedding_model: LiuEmbeddings instance
            collection_name: Name for the collection (default: from config)
            persist_path: Path to save ChromaDB (default: from config)
            
        Raises:
            TypeError: If embedding_model is invalid
            RuntimeError: If ChromaDB initialization fails
        """
        if embedding_model is None:
            raise TypeError("embedding_model cannot be None")
        
        if not hasattr(embedding_model, 'embed_documents'):
            raise TypeError("embedding_model must have embed_documents method")
        
        self.embedding_model = embedding_model
        self.collection_name = collection_name or LiuConfig.DEFAULT_COLLECTION_NAME
        persist_path = persist_path or LiuConfig.DEFAULT_VECTOR_PATH
        
        try:
            # Ensure persist path exists
            os.makedirs(persist_path, exist_ok=True)
            
            logger.info(f"Initializing ChromaDB at {persist_path}")
            self.client = chromadb.PersistentClient(path=persist_path)
            
            # Create embedding wrapper
            self.embedding_function = TFEmbeddingWrapper(embedding_model)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": LiuConfig.DISTANCE_METRIC}
            )
            
            count = self.collection.count()
            logger.info(f"✅ Initialized collection '{self.collection_name}' (documents: {count})")
        
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise RuntimeError(f"Vector store initialization failed: {str(e)}") from e
    
    def add_texts(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict]] = None, 
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add text documents to the vector store.
        
        Args:
            texts: List of text documents
            metadatas: Optional list of metadata dicts for each text
            ids: Optional custom IDs for each document
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If addition fails
        """
        if not texts or len(texts) == 0:
            raise ValueError("texts list cannot be empty")
        
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("All items in texts must be strings")
        
        if metadatas is not None:
            if len(metadatas) != len(texts):
                raise ValueError("metadatas length must match texts length")
        else:
            metadatas = [{"source": self.collection_name} for _ in texts]
        
        if ids is not None:
            if len(ids) != len(texts):
                raise ValueError("ids length must match texts length")
        else:
            ids = [f"doc_{i}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}" for i in range(len(texts))]

        
        try:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"✅ Added {len(texts)} documents to '{self.collection_name}'")
        
        except Exception as e:
            logger.error(f"Failed to add texts: {str(e)}")
            raise RuntimeError(f"Add texts failed: {str(e)}") from e
    
    def add_texts_batch(
        self, 
        texts: List[str],
        batch_size: int = None,
        metadatas: Optional[List[Dict]] = None, 
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add texts in batches to manage memory efficiently.
        
        Args:
            texts: List of documents to add
            batch_size: Number of documents per batch
            metadatas: Optional metadata for each document
            ids: Optional custom IDs
            
        Raises:
            ValueError: If batch_size is invalid
        """
        if batch_size is None:
            batch_size = LiuConfig.DEFAULT_BATCH_SIZE
        
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        total_batches = (len(texts) + batch_size - 1) // batch_size
        logger.info(f"Adding {len(texts)} documents in {total_batches} batches")
        
        for i in range(0, len(texts), batch_size):
            batch_num = i // batch_size + 1
            batch_texts = texts[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size] if metadatas else None
            batch_ids = [f"doc_{i}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"] if ids else None
            
            try:
                self.add_texts(batch_texts, batch_meta, batch_ids)
                logger.debug(f"Batch {batch_num}/{total_batches} complete")
            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {str(e)}")
                raise
        
        logger.info(f"✅ Batch addition complete")
    
    def query(
        self, 
        query_text: str, 
        n_results: int = None
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Perform semantic search on stored documents.
        
        Args:
            query_text: Query string
            n_results: Number of results to return
            
        Returns:
            List of most similar raw,documents
            raw : is dict with id,document,metadata
            documents : is list of document similar strings

        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text must be non-empty string")
        
        n_results = n_results or LiuConfig.DEFAULT_N_RESULTS
        
        if n_results <= 0 or n_results > LiuConfig.MAX_N_RESULTS:
            raise ValueError(f"n_results must be between 1 and {LiuConfig.MAX_N_RESULTS}")
        
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            logger.info(f"Query returned {len(results['documents'][0])} results")
            return results,results['documents'][0]
        
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise RuntimeError(f"Query failed: {str(e)}") from e
    
    def similarity_search(
        self, 
        query_text: str, 
        n_results: int = None,
        with_score: float = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Semantic similarity search with optional scores.
        
        Args:
            query_text: Query string
            n_results: Number of results to return
            with_score: (Deafult: 0.4) Include similarity of .4 scores in results
            
        Returns:
            List of most similar raw,documents
            raw : is dict with id,document,metadata ,source and all
            documents : is list of dict id,document,metadata
        """
        n_results = n_results or LiuConfig.DEFAULT_N_RESULTS
        with_score = with_score or LiuConfig.DEFAULT_SIMILARITY_SEARCH_SCORE_THRESHOLD
        
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            

            return results,[{
                    "id": results['ids'][0][i],
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "similarity_score": float(1 - results['distances'][0][i])
            } for i in range(len(results['ids'][0])) if float(1 - results['distances'][0][i])> with_score]
            
        
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            raise RuntimeError(f"Similarity search failed: {str(e)}") from e
    
    def search_by_id(self, doc_id: str) -> Optional[Dict]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Dict with id, document, metadata or None if not found
        """
        try:
            result = self.collection.get(ids=[doc_id])
            if result['documents']:
                return {
                    "id": doc_id,
                    "document": result['documents'][0],
                    "metadata": result['metadatas'][0]
                }
            logger.warning(f"Document with ID '{doc_id}' not found")
            return None
        
        except Exception as e:
            logger.error(f"Search by ID failed: {str(e)}")
            raise RuntimeError(f"Search by ID failed: {str(e)}") from e
    
    def search_by_metadata(self, metadata_filter: Dict) -> List[Dict]:
        """
        Search documents by metadata.
        
        Args:
            metadata_filter: Metadata filter dict (ChromaDB where format)
            
        Returns:
            List of matching documents
            
        Example:
            >>> store.search_by_metadata({"category": "news"})
        """
        try:
            results = self.collection.get(where=metadata_filter)
            docs = [{
                "id": results['ids'][i],
                "document": results['documents'][i],
                "metadata": results['metadatas'][i]
            } for i in range(len(results['ids']))]
            logger.info(f"Metadata search returned {len(docs)} results")
            return docs
        
        except Exception as e:
            logger.error(f"Metadata search failed: {str(e)}")
            raise RuntimeError(f"Metadata search failed: {str(e)}") from e
    
    def get_all(self) -> List[Dict]:
        """
        Retrieve all documents in collection.
        
        Returns:
            List of all documents with metadata
        """
        try:
            results = self.collection.get()
            docs = [{
                "id": results['ids'][i],
                "document": results['documents'][i],
                "metadata": results['metadatas'][i]
            } for i in range(len(results['ids']))]
            logger.info(f"Retrieved {len(docs)} total documents")
            return docs
        
        except Exception as e:
            logger.error(f"Get all failed: {str(e)}")
            raise RuntimeError(f"Get all failed: {str(e)}") from e
    
    def delete_by_id(self, doc_id: str) -> None:
        """
        Delete a document by ID.
        
        Args:
            doc_id: Document ID to delete
        """
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"✅ Deleted document '{doc_id}'")
        
        except Exception as e:
            logger.error(f"Delete failed: {str(e)}")
            raise RuntimeError(f"Delete failed: {str(e)}") from e
    
    def update_by_id(
        self, 
        doc_id: str, 
        new_text: str,
        new_metadata: Optional[Dict] = None
    ) -> None:
        """
        Update a document by ID.
        
        Args:
            doc_id: Document ID to update
            new_text: New document text
            new_metadata: Optional new metadata
        """
        if not isinstance(new_text, str) or not new_text.strip():
            raise ValueError("new_text must be non-empty string")
        
        try:
            # Get existing metadata if not provided
            if new_metadata is None:
                result = self.collection.get(ids=[doc_id])
                if not result['documents']:
                    raise ValueError(f"Document '{doc_id}' not found")
                new_metadata = result['metadatas'][0]
            
            # Delete and re-add
            self.collection.delete(ids=[doc_id])
            self.collection.add(
                documents=[new_text],
                metadatas=[new_metadata],
                ids=[doc_id]
            )
            logger.info(f"✅ Updated document '{doc_id}'")
        
        except Exception as e:
            logger.error(f"Update failed: {str(e)}")
            raise RuntimeError(f"Update failed: {str(e)}") from e
    
    def count_documents(self) -> int:
        """
        Get total number of documents in collection.
        
        Returns:
            Document count
        """
        try:
            count = self.collection.count()
            logger.debug(f"Collection has {count} documents")
            return count
        
        except Exception as e:
            logger.error(f"Count failed: {str(e)}")
            raise RuntimeError(f"Count failed: {str(e)}") from e
    
    def save(self, path: str) -> None:
        """
        Export collection data to JSON.
        
        Args:
            path: File path to save to
        """
        try:
            docs = self.get_all()
            with open(path, 'w') as f:
                json.dump(docs, f, indent=2)
            logger.info(f"✅ Saved collection to {path}")
        
        except Exception as e:
            logger.error(f"Save failed: {str(e)}")
            raise RuntimeError(f"Save failed: {str(e)}") from e
    
    @property
    def info(self) -> Dict:
        """Get collection information."""
        return {
            "name": self.collection_name,
            "document_count": self.count_documents(),
            "embedding_model": self.embedding_model.model_info
        }
    
    
    def search(
        self,
        query: str=None,
        text_document: str=None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        n_results: int = None,
        ) -> Tuple[Dict[str, Any], List[str]]:
        """
        One-line semantic search function.
        
        Combines chunking, embedding, storage, and search in a single call.
        Perfect for quick prototyping and small applications.
        
        Args:
            text_document: Long text or document to search within
            query: Query string to search for
            chunk_size: Size of text chunks (default: from config)
            chunk_overlap: Overlap between chunks (default: from config)
            n_results: Number of results to return (default: from config)
+

        Returns:
            List of most similar raw,documents
            [raw : is dict with id,document,metadata]
            [documents : is list of document similar strings]
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If operation fails
            
        Example:
            >>> #adding to docs
            >>> vectore.search(
            ...     long_text,
            ...     chunk_size=500,
            ... )
            >>> for result in docs:
            ...     print(result)
            ...   
            >>> #return only documents
            >>> raw,docs = vector.search(
            ...     "What is the capital?",
            ...     chunk_size=500,
            ...     n_results=3
            ... )
            >>> for result in docs:
            ...     print(result)
        """
        chunk_size = chunk_size or LiuConfig.DEFAULT_CHUNK_SIZE
        chunk_overlap = chunk_overlap or LiuConfig.DEFAULT_CHUNK_OVERLAP
        n_results = n_results or LiuConfig.DEFAULT_N_RESULTS
        
        logger = setup_logger(__name__)
        
        try:
            if text_document:
                logger.info("Starting liu_search operation")
                
                # Step 1: Split text into chunks
                logger.info(f"Splitting text (size: {chunk_size}, overlap: {chunk_overlap})")
                chunks = split_text(
                    text_document,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                # Step 4: Add chunks to vector store
                logger.info(f"Adding {len(chunks)} chunks to vector store")
                self.add_texts(chunks)
            
            # Step 5: Query
            if query:
                logger.info(f"Searching for: '{query}'")
                results = self.query(query, n_results=n_results)
                logger.info(f"✅ Search complete. Found {len(results)} results")
                return results
        
        except Exception as e:
            logger.error(f"liu_search failed: {str(e)}")
            raise RuntimeError(f"Search operation failed: {str(e)}") from e
