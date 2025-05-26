import pytest
import numpy as np
from open_deep_research.cache import HybridCache
import tempfile
import os
from unittest.mock import Mock, patch

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
    mock_client = Mock()
    # Mock the search method to return a list with one hit
    mock_client.search.return_value = [
        Mock(score=0.9, payload={"value": "test_value"})
    ]
    return mock_client

@pytest.fixture
def cache(mock_embed_fn, mock_qdrant):
    """Create a HybridCache instance with a temporary SQLite database and mocked Qdrant."""
    with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as tmp:
        db_path = tmp.name
    
    with patch('open_deep_research.cache.QdrantClient', return_value=mock_qdrant):
        cache = HybridCache(
            db_path=db_path,
            qdrant_url="http://localhost:6333",
            collection_name="test_cache",
            embed_fn=mock_embed_fn
        )
    
    yield cache
    
    # Cleanup
    cache.conn.close()
    os.unlink(db_path)

def test_exact_cache_operations(cache):
    """Test exact cache operations (get/put)."""
    # Test putting and getting a value
    cache.put_exact("test_key", "test_value")
    assert cache.get_exact("test_key") == "test_value"
    
    # Test getting non-existent value
    assert cache.get_exact("non_existent") is None
    
    # Test updating existing value
    cache.put_exact("test_key", "new_value")
    assert cache.get_exact("test_key") == "new_value"

def test_semantic_cache_operations(cache, mock_qdrant):
    """Test semantic cache operations (get/put)."""
    # Test putting and getting a value
    cache.put_semantic("test_query", "test_value")
    result = cache.get_semantic("test_query")
    assert result == "test_value"
    
    # Verify Qdrant client was called correctly
    mock_qdrant.search.assert_called_once()
    mock_qdrant.upsert.assert_called_once()
    
    # Test getting non-existent value
    mock_qdrant.search.return_value = []  # No hits
    assert cache.get_semantic("non_existent") is None
    
    # Test updating existing value
    mock_qdrant.search.return_value = [Mock(score=0.9, payload={"value": "new_value"})]
    cache.put_semantic("test_query", "new_value")
    result = cache.get_semantic("test_query")
    assert result == "new_value"

def test_cache_without_embed_fn(mock_qdrant):
    """Test cache behavior when no embedding function is provided."""
    with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as tmp:
        db_path = tmp.name
    
    with patch('open_deep_research.cache.QdrantClient', return_value=mock_qdrant):
        cache = HybridCache(
            db_path=db_path,
            qdrant_url="http://localhost:6333",
            collection_name="test_cache_no_embed"
        )
    
    # Semantic operations should return None when no embed_fn is provided
    assert cache.get_semantic("test") is None
    cache.put_semantic("test", "value")  # Should do nothing
    
    # Verify Qdrant client was not called
    mock_qdrant.search.assert_not_called()
    mock_qdrant.upsert.assert_not_called()
    
    # Cleanup
    cache.conn.close()
    os.unlink(db_path)

def test_hash_consistency(cache):
    """Test that the hash function produces consistent results."""
    text = "test_text"
    hash1 = cache._hash(text)
    hash2 = cache._hash(text)
    assert hash1 == hash2
    assert hash1 != cache._hash("different_text") 