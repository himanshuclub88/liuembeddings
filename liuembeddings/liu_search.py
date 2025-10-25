from .embeddings import LiuEmbeddings
from .vectorstore import LiuVectorStore 
from .utils import split_text
from .config import LiuConfig
from .logger import setup_logger

class fastquery:
     collection_name = LiuConfig.DEFAULT_COLLECTION_NAME
     model_name = "USE"


def fastquery(
        query: str=None,
        text_document: str=None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        n_results: int = None,
        with_scores: bool = False,
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
            with_scores: give smilarity search like answer
            collection_name: Name for the vector store collection (default: from config)
            model_name: Embedding model to use (default: "USE")
            
        Returns:
            List of most similar chunks from the document
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If operation fails
            
        Example:
        >>> #adding to docs
        >>> fastquery(
        ...     long_text,
        ...     chunk_size=500,
        ... )
        >>> for result in docs:
        ...     print(result)
        ...   
        >>> #searching in docs
        >>> raw,docs = fastquery(
        ...     "What is the capital?",
        ...     chunk_size=500,
        ...     n_results=3
        ... )
        >>> for result in docs:
        ...     print(result)
        >>>
        >>> #return only documents
        """
        chunk_size = chunk_size or LiuConfig.DEFAULT_CHUNK_SIZE
        chunk_overlap = chunk_overlap or LiuConfig.DEFAULT_CHUNK_OVERLAP
        n_results = n_results or LiuConfig.DEFAULT_N_RESULTS
        collection_name = collection_name or LiuConfig.DEFAULT_COLLECTION_NAME
        
        logger = setup_logger(__name__)

        # Step 2: Load embedding model
        logger.info(f"Loading embedding model: {model_name}")
        emb_model = LiuEmbeddings(model_name=model_name)
                
        # Step 3: Create vector store
        logger.info(f"Creating vector store: {collection_name}")
        vectorstore = LiuVectorStore(
                embedding_model=emb_model,
                collection_name=collection_name)
        
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
                vectorstore.add_texts(chunks)
            
            # Step 5: Query
            if query:
                if not with_scores:
                    logger.info(f"Searching for: '{query}'")
                    results = vectorstore.query(query, n_results=n_results)
                else:
                    logger.info(f"Searching for: '{query}'")
                    results = vectorstore.similarity_search(query, n_results=n_results)
                     
            
                logger.info(f"✅ Search complete. Found {len(results)} results")
                return results
        
        except Exception as e:
            logger.error(f"fastquery failed: {str(e)}")
            raise RuntimeError(f"Search operation failed: {str(e)}") from e
