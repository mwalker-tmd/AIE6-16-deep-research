import pytest
import json
import numpy as np
from open_deep_research.cache import HybridCache
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

@pytest.fixture
def mock_embed_fn():
    """Mock embedding function that returns a fixed vector."""
    def embed(text: str) -> list[float]:
        # Return a fixed vector for testing
        return [0.1] * 384
    return embed

@pytest.fixture
def mock_qdrant():
    """Mock Qdrant client."""
    with patch('open_deep_research.cache.QdrantClient') as mock:
        # Create a mock instance
        mock_instance = MagicMock()
        mock_instance.url = "http://test:6333"
        
        # Mock the recreate_collection method to do nothing
        mock_instance.recreate_collection = MagicMock(return_value=None)
        
        # Mock the search method to return empty results
        mock_instance.search = MagicMock(return_value=[])
        
        # Mock the upsert method to do nothing
        mock_instance.upsert = MagicMock(return_value=None)
        
        # Mock the delete_collection method to do nothing
        mock_instance.delete_collection = MagicMock(return_value=None)
        
        # Return the mock instance when QdrantClient is instantiated
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def cache(mock_qdrant):
    """Create a HybridCache instance for testing."""
    return HybridCache(embed_fn=lambda x: [0.0] * 768)

def test_exact_cache_operations(cache):
    """Test exact cache operations (get/put)."""
    # Test putting and getting a value
    cache.put_exact("test_key", "test_value")
    assert cache.get_exact("test_key") == "test_value"
    
    # Test updating a value
    cache.put_exact("test_key", "new_value")
    assert cache.get_exact("test_key") == "new_value"
    
    # Test non-existent key
    assert cache.get_exact("non_existent") is None

def test_semantic_cache_operations(cache, mock_qdrant):
    """Test semantic cache operations (get/put)."""
    # Test putting and getting a value
    cache.put_semantic("test_query", "test_value")
    mock_qdrant.search.return_value = [MagicMock(score=0.9, payload={"value": json.dumps("test_value")})]
    result = cache.get_semantic("test_query")
    assert result == "test_value"
    
    # Test updating a value
    cache.put_semantic("test_query", "new_value")
    mock_qdrant.search.return_value = [MagicMock(score=0.9, payload={"value": json.dumps("new_value")})]
    result = cache.get_semantic("test_query")
    assert result == "new_value"
    
    # Test non-existent query
    mock_qdrant.search.return_value = []
    assert cache.get_semantic("non_existent") is None

def test_cache_without_embed_fn(mock_qdrant):
    """Test cache behavior without embedding function."""
    cache = HybridCache()
    
    # Semantic operations should return None without embed_fn
    assert cache.get_semantic("test_query") is None
    cache.put_semantic("test_query", "test_value")  # Should do nothing
    
    # Exact operations should still work
    cache.put_exact("test_key", "test_value")
    assert cache.get_exact("test_key") == "test_value"

def test_hash_consistency(cache):
    """Test that hash function produces consistent results."""
    text = "test text"
    hash1 = cache._hash(text)
    hash2 = cache._hash(text)
    assert hash1 == hash2
    
    # Different text should produce different hash
    hash3 = cache._hash("different text")
    assert hash1 != hash3 