# tests/test_embeddings.py

"""
Unit tests for LiuEmbeddings.
"""

import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pytest
from liuembeddings import LiuEmbeddings


class TestLiuEmbeddings:
    """Test suite for LiuEmbeddings."""
    
    @pytest.fixture
    def embedder(self):
        """Create embedder instance."""
        return LiuEmbeddings(model_name="USE")
    
    def test_initialization(self, embedder):
        """Test embedder initialization."""
        assert embedder.model_name == "USE"
        assert embedder.model_dimension == 512
        assert embedder.model is not None
    
    def test_embed_query_basic(self, embedder):
        """Test embedding a single query."""
        embedding = embedder.embed_query("Hello world")
        assert isinstance(embedding, list)
        assert len(embedding) == 512
        assert all(isinstance(x, float) for x in embedding)
    
    def test_embed_query_empty_raises(self, embedder):
        """Test that empty query raises error."""
        with pytest.raises(ValueError):
            embedder.embed_query("")
    
    def test_embed_query_whitespace_raises(self, embedder):
        """Test that whitespace-only query raises error."""
        with pytest.raises(ValueError):
            embedder.embed_query("   ")
    
    def test_embed_query_non_string_raises(self, embedder):
        """Test that non-string input raises error."""
        with pytest.raises(TypeError):
            embedder.embed_query(123)
    
    def test_embed_documents_basic(self, embedder):
        """Test embedding multiple documents."""
        texts = ["Hello", "World", "Test"]
        embeddings = embedder.embed_documents(texts)
        
        assert len(embeddings) == 3
        assert all(len(e) == 512 for e in embeddings)
    
    def test_embed_documents_empty_raises(self, embedder):
        """Test that empty documents list raises error."""
        with pytest.raises(ValueError):
            embedder.embed_documents([])
    
    def test_embed_documents_non_strings_raises(self, embedder):
        """Test that non-string items raise error."""
        with pytest.raises(TypeError):
            embedder.embed_documents(["Hello", 123])
    
    def test_embed_documents_batch(self, embedder):
        """Test batch embedding."""
        texts = [f"Document {i}" for i in range(10)]
        embeddings = embedder.embed_documents_batch(texts, batch_size=3)
        
        assert len(embeddings) == 10
        assert all(len(e) == 512 for e in embeddings)
    
    def test_model_info_property(self, embedder):
        """Test model info property."""
        info = embedder.model_info
        
        assert "name" in info
        assert "dimension" in info
        assert info["name"] == "USE"
        assert info["dimension"] == 512
    
    def test_invalid_model_raises(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError):
            LiuEmbeddings(model_name="INVALID_MODEL")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
