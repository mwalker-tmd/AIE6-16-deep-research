import pytest
from unittest.mock import patch, MagicMock
from open_deep_research.utils import HFEmbedder, get_embedder

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables."""
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "test_api_key")
    monkeypatch.setenv("HUGGINGFACE_ENDPOINT_URL", "https://test.endpoint.huggingface.cloud")
    monkeypatch.setenv("EMBEDDING_MODEL", "test/model")

def test_embedder_initialization(mock_env_vars):
    """Test that the embedder initializes correctly."""
    embedder = get_embedder()
    assert embedder is not None
    assert embedder.model_name == "test/model"
    assert embedder.api_key == "test_api_key"
    assert embedder.endpoint_url == "https://test.endpoint.huggingface.cloud"

def test_embedder_missing_api_key(monkeypatch):
    """Test that the embedder raises an error when API key is missing."""
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="HUGGINGFACE_API_KEY environment variable is required"):
        get_embedder()

def test_embedder_missing_endpoint_url(monkeypatch):
    """Test that the embedder raises an error when endpoint URL is missing."""
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "test_api_key")
    monkeypatch.delenv("HUGGINGFACE_ENDPOINT_URL", raising=False)
    with pytest.raises(ValueError, match="HUGGINGFACE_ENDPOINT_URL environment variable is required"):
        get_embedder()

def test_embedder_encoding(mock_env_vars):
    """Test that the embedder produces valid embeddings."""
    with patch('open_deep_research.utils.requests.post') as mock_post:
        # Mock the response from the HuggingFace API
        mock_response = MagicMock()
        mock_response.json.return_value = [0.1] * 768  # Return a flat list instead of list of lists
        mock_post.return_value = mock_response

        embedder = get_embedder()
        result = embedder.encode("test text")
        
        assert len(result) == 768
        assert all(isinstance(x, float) for x in result)
        assert all(x == 0.1 for x in result)

@patch('requests.post')
def test_embedder_encoding_list_response(mock_post, mock_env_vars):
    """Test embedder with list response from API."""
    # Mock API response
    mock_response = MagicMock()
    mock_response.json.return_value = [0.1] * 768
    mock_post.return_value = mock_response
    
    embedder = get_embedder()
    text = "This is a test sentence."
    embedding = embedder.encode(text)
    
    # Verify API call
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[0][0] == "https://test.endpoint.huggingface.cloud"
    assert call_args[1]["headers"]["Authorization"] == "Bearer test_api_key"
    assert call_args[1]["json"]["inputs"] == text
    
    # Verify response
    assert isinstance(embedding, list)
    assert len(embedding) == 768
    assert all(isinstance(x, float) for x in embedding)

@patch('requests.post')
def test_embedder_encoding_dict_response(mock_post, mock_env_vars):
    """Test embedder with dict response from API."""
    # Mock API response
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1] * 768}
    mock_post.return_value = mock_response
    
    embedder = get_embedder()
    text = "This is a test sentence."
    embedding = embedder.encode(text)
    
    # Verify response
    assert isinstance(embedding, list)
    assert len(embedding) == 768
    assert all(isinstance(x, float) for x in embedding)

@patch('requests.post')
def test_embedder_encoding_error(mock_post, mock_env_vars):
    """Test embedder with unexpected API response format."""
    # Mock API response with unexpected format
    mock_response = MagicMock()
    mock_response.json.return_value = {"unexpected": "format"}
    mock_post.return_value = mock_response
    
    embedder = get_embedder()
    text = "This is a test sentence."
    
    with pytest.raises(ValueError, match="Unexpected response format from HuggingFace API"):
        embedder.encode(text)

def test_embedder_custom_model(mock_env_vars):
    """Test HFEmbedder with a custom model."""
    model_name = "BAAI/bge-small-en-v1.5"  # Using a smaller model for testing
    embedder = HFEmbedder(model_name=model_name)
    assert embedder.model_name == model_name

def test_embedder_environment_variable(monkeypatch):
    """Test that embedder respects environment variable configuration."""
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "test_api_key")
    monkeypatch.setenv("HUGGINGFACE_ENDPOINT_URL", "https://test.endpoint.huggingface.cloud")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    embedder = get_embedder()
    assert embedder.model_name == "BAAI/bge-small-en-v1.5" 