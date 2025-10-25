# tests/test_vectorstore.py

"""
Unit tests for LiuVectorStore.
"""
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pytest
import tempfile
import shutil
from liuembeddings import LiuEmbeddings
from liuembeddings import LiuVectorStore


class TestLiuVectorStore:
    """Test suite for LiuVectorStore."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def embedder(self):
        """Create embedder instance."""
        return LiuEmbeddings(model_name="USE")
    
    @pytest.fixture
    def vectorstore(self, embedder, temp_dir):
        """Create vector store instance."""
        return LiuVectorStore(
            embedding_model=embedder,
            collection_name="test_collection",
            persist_path=temp_dir
        )
    
    def test_initialization(self, vectorstore):
        """Test vector store initialization."""
        assert vectorstore.collection_name == "test_collection"
        assert vectorstore.collection is not None
        assert vectorstore.count_documents() == 0
    
    def test_invalid_embedding_model_raises(self, temp_dir):
        """Test that invalid embedding model raises error."""
        with pytest.raises(TypeError):
            LiuVectorStore(None, persist_path=temp_dir)
    
    def test_add_texts_basic(self, vectorstore):
        """Test adding texts to vector store."""
        texts = ["Hello world", "Test document"]
        vectorstore.add_texts(texts)
        
        assert vectorstore.count_documents() == 2
    
    def test_add_texts_with_metadata(self, vectorstore):
        """Test adding texts with metadata."""
        texts = ["Hello", "World"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]
        
        vectorstore.add_texts(texts, metadatas=metadatas)
        
        assert vectorstore.count_documents() == 2
    
    def test_add_texts_empty_raises(self, vectorstore):
        """Test that empty texts list raises error."""
        with pytest.raises(ValueError):
            vectorstore.add_texts([])
    
    def test_query_basic(self, vectorstore):
        """Test basic semantic search."""
        texts = [
            "Python is a programming language",
            "Machine learning is AI",
            "Weather is sunny today"
        ]
        vectorstore.add_texts(texts)
        
        results = vectorstore.query("What is Python?", n_results=2)
        
        assert len(results) == 2
        assert isinstance(results[0], str)
    
    def test_similarity_search_with_scores(self, vectorstore):
        """Test similarity search with scores."""
        texts = ["Hello world", "Hello there"]
        vectorstore.add_texts(texts)
        
        results = vectorstore.similarity_search(
            "Hello",
            n_results=2,
            with_scores=True
        )
        
        assert len(results) == 2
        assert all("similarity_score" in r for r in results)
        assert all(0 <= r["similarity_score"] <= 1 for r in results)
    
    def test_search_by_id(self, vectorstore):
        """Test search by document ID."""
        texts = ["Document one"]
        ids = ["doc_1"]
        
        vectorstore.add_texts(texts, ids=ids)
        result = vectorstore.search_by_id("doc_1")
        
        assert result is not None
        assert result["id"] == "doc_1"
        assert result["document"] == "Document one"
    
    def test_search_by_id_not_found(self, vectorstore):
        """Test search by ID returns None for missing document."""
        result = vectorstore.search_by_id("nonexistent")
        assert result is None
    
    def test_get_all(self, vectorstore):
        """Test retrieving all documents."""
        texts = ["Doc 1", "Doc 2", "Doc 3"]
        vectorstore.add_texts(texts)
        
        all_docs = vectorstore.get_all()
        
        assert len(all_docs) == 3
        assert all("id" in d and "document" in d for d in all_docs)
    
    def test_delete_by_id(self, vectorstore):
        """Test deleting document by ID."""
        texts = ["Delete me"]
        ids = ["delete_doc"]
        
        vectorstore.add_texts(texts, ids=ids)
        assert vectorstore.count_documents() == 1
        
        vectorstore.delete_by_id("delete_doc")
        assert vectorstore.count_documents() == 0
    
    def test_update_by_id(self, vectorstore):
        """Test updating document by ID."""
        texts = ["Original content"]
        ids = ["update_doc"]
        
        vectorstore.add_texts(texts, ids=ids)
        vectorstore.update_by_id("update_doc", "Updated content")
        
        result = vectorstore.search_by_id("update_doc")
        assert result["document"] == "Updated content"
    
    def test_add_texts_batch(self, vectorstore):
        """Test batch addition of texts."""
        texts = [f"Document {i}" for i in range(10)]
        vectorstore.add_texts_batch(texts, batch_size=3)
        
        assert vectorstore.count_documents() == 10
    
    def test_collection_info(self, vectorstore):
        """Test getting collection info."""
        vectorstore.add_texts(["Test"])
        
        info = vectorstore.info
        assert "name" in info
        assert "document_count" in info
        assert "embedding_model" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
