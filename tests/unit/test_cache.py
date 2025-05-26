import pytest
import os
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock
from open_deep_research.utils import HybridCache, HFEmbedder

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary directory for cache files."""
    return str(tmp_path)

@pytest.fixture
def mock_qdrant():
    """Create a mock Qdrant client."""
    with patch('open_deep_research.cache.QdrantClient') as mock:
        # Create a mock instance
        mock_instance = MagicMock()
        mock_instance.url = "http://test:6333"  # Set the URL attribute
        
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
def cache(temp_cache_dir, mock_qdrant):
    """Create a HybridCache instance with temporary storage."""
    db_path = os.path.join(temp_cache_dir, "test_cache.sqlite")
    return HybridCache(
        db_path=db_path,
        qdrant_url="http://localhost:6333",
        collection_name="test_cache",
        embed_fn=lambda x: [0.0] * 768  # Mock embedding function
    )

def test_cache_exact_match(cache):
    """Test exact match caching functionality."""
    query = "test query"
    result = {"title": "Test Result", "content": "Test Content"}
    
    # Test putting and getting exact match
    cache.put_exact(query, result)
    retrieved = cache.get_exact(query)
    assert retrieved == result
    
    # Test non-existent query
    assert cache.get_exact("non-existent") is None

def test_cache_environment_variables(temp_cache_dir, monkeypatch, mock_qdrant):
    """Test that cache configuration respects environment variables."""
    # Set environment variables
    monkeypatch.setenv("CACHE_DB_PATH", os.path.join(temp_cache_dir, "env_cache.sqlite"))
    monkeypatch.setenv("QDRANT_URL", "http://test:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "test_collection")
    monkeypatch.setenv("QDRANT_TIMEOUT", "60")
    monkeypatch.setenv("CACHE_TTL", "3600")
    
    # Create cache with environment variables
    cache = HybridCache(
        db_path=os.getenv("CACHE_DB_PATH"),
        qdrant_url=os.getenv("QDRANT_URL"),
        collection_name=os.getenv("QDRANT_COLLECTION"),
        embed_fn=lambda x: [0.0] * 768,  # Mock embedding function
        timeout=float(os.getenv("QDRANT_TIMEOUT")),
        ttl=int(os.getenv("CACHE_TTL"))
    )
    
    # Verify configuration
    assert cache.db_path == os.path.join(temp_cache_dir, "env_cache.sqlite")
    assert cache.qdrant.url == "http://test:6333"
    assert cache.collection == "test_collection"
    assert cache.ttl == 3600 