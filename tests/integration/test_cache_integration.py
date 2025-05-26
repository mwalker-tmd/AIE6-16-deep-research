import pytest
import os
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock
from open_deep_research.utils import get_embedder
from open_deep_research.cache import HybridCache

@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables."""
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "test_api_key")
    monkeypatch.setenv("HUGGINGFACE_ENDPOINT_URL", "https://test.endpoint.huggingface.cloud")

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
def mock_embedder():
    """Create a mock embedder."""
    with patch('open_deep_research.utils.get_embedder') as mock:
        mock_instance = MagicMock()
        mock_instance.encode = MagicMock(return_value=[0.0] * 768)
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def cache(mock_embedder, mock_qdrant):
    """Create a cache instance for testing."""
    return HybridCache(embed_fn=mock_embedder.encode)

def test_cache_with_embedder(mock_embedder, cache, mock_qdrant):
    """Test that cache works correctly with embedder."""
    # Test exact match
    cache.put_exact("test_key", "test_value")
    assert cache.get_exact("test_key") == "test_value"
    
    # Test semantic match
    mock_embedder.encode.return_value = [0.1] * 768
    mock_qdrant.search.return_value = [MagicMock(score=0.9, payload={"value": json.dumps("test_result")})]
    cache.put_semantic("test_query", "test_result")
    result = cache.get_semantic("similar query")
    assert result == "test_result"
    
    # Verify embedder was called
    assert mock_embedder.encode.call_count >= 2

def test_cache_without_embedder(mock_qdrant):
    """Test that cache works without embedder."""
    cache = HybridCache()
    
    # Test exact match
    cache.put_exact("test_key", "test_value")
    assert cache.get_exact("test_key") == "test_value"
    
    # Test semantic match (should return None without embedder)
    assert cache.get_semantic("test_query") is None

def test_cache_embedding_consistency(mock_embedder, cache, mock_qdrant):
    """Test that embeddings are consistent for the same input."""
    # First call
    mock_embedder.encode.return_value = [0.1] * 768
    mock_qdrant.search.return_value = [MagicMock(score=0.9, payload={"value": json.dumps("test_result")})]
    cache.put_semantic("test_query", "test_result")
    
    # Second call with same input
    mock_embedder.encode.return_value = [0.1] * 768
    result = cache.get_semantic("test_query")
    assert result == "test_result"
    
    # Verify embedder was called with same input
    assert mock_embedder.encode.call_count >= 2
    assert all(call[0][0] == "test_query" for call in mock_embedder.encode.call_args_list)

def test_cache_embedding_different_inputs(mock_embedder, cache, mock_qdrant):
    """Test that different inputs get different embeddings."""
    # First call
    mock_embedder.encode.return_value = [0.1] * 768
    mock_qdrant.search.return_value = [MagicMock(score=0.9, payload={"value": json.dumps("result1")})]
    cache.put_semantic("query1", "result1")
    
    # Second call with different input
    mock_embedder.encode.return_value = [0.2] * 768
    mock_qdrant.search.return_value = [MagicMock(score=0.5, payload={"value": json.dumps("result1")})]  # Below threshold
    result = cache.get_semantic("query2")
    assert result is None  # Should not match due to different embedding
    
    # Verify embedder was called with different inputs
    assert mock_embedder.encode.call_count >= 2
    assert mock_embedder.encode.call_args_list[0][0][0] == "query1"
    assert mock_embedder.encode.call_args_list[1][0][0] == "query2"

def test_cache_integration(cache, mock_qdrant):
    """Test the full caching workflow with both exact and semantic matches."""
    # Test data
    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "Explain neural networks"
    ]
    results = [
        {"title": "ML Overview", "content": "Machine learning is a field of AI"},
        {"title": "DL Guide", "content": "Deep learning uses neural networks"},
        {"title": "NN Basics", "content": "Neural networks are computing systems"}
    ]
    
    # Store results
    for query, result in zip(queries, results):
        cache.put_exact(query, result)
        cache.put_semantic(query, result)
    
    # Test exact matches
    for query, result in zip(queries, results):
        assert cache.get_exact(query) == result
    
    # Test semantic matches with similar queries
    similar_queries = [
        "Tell me about machine learning",
        "What is deep learning?",
        "How do neural networks function?"
    ]
    
    # Mock semantic search results
    mock_qdrant.search.side_effect = [
        [MagicMock(score=0.9, payload={"value": json.dumps(results[0])})],
        [MagicMock(score=0.9, payload={"value": json.dumps(results[1])})],
        [MagicMock(score=0.9, payload={"value": json.dumps(results[2])})]
    ]
    
    for similar_query, original_result in zip(similar_queries, results):
        retrieved = cache.get_semantic(similar_query)
        assert retrieved is not None
        assert retrieved == original_result

def test_cache_persistence(temp_cache_dir, mock_embedder, mock_qdrant):
    """Test that cache persists between instances."""
    db_path = os.path.join(temp_cache_dir, "persistent_cache.sqlite")
    
    # Create first cache instance and store data
    cache1 = HybridCache(
        db_path=db_path,
        embed_fn=mock_embedder.encode
    )
    
    query = "test query"
    result = {"title": "Test Result", "content": "Test Content"}
    cache1.put_exact(query, result)
    cache1.put_semantic(query, result)
    
    # Create second cache instance and verify data persistence
    cache2 = HybridCache(
        db_path=db_path,
        embed_fn=mock_embedder.encode
    )
    
    assert cache2.get_exact(query) == result
    mock_qdrant.search.return_value = [MagicMock(score=0.9, payload={"value": json.dumps(result)})]
    assert cache2.get_semantic(query) == result 