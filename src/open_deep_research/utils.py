import os
import asyncio
import requests
import random 
import concurrent
import aiohttp
import httpx
import time
from typing import List, Optional, Dict, Any, Union, Literal
from urllib.parse import unquote

from exa_py import Exa
from linkup import LinkupClient
from tavily import AsyncTavilyClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient as AsyncAzureAISearchClient
import asyncio
import os
from duckduckgo_search import DDGS 
from bs4 import BeautifulSoup
from markdownify import markdownify

from langchain_community.retrievers import ArxivRetriever
from langchain_community.utilities.pubmed import PubMedAPIWrapper
from langchain_core.tools import tool

from langsmith import traceable

from open_deep_research.state import Section
from open_deep_research.cache import HybridCache
import requests

from .rate_limiting import with_rate_limit, with_retry, RetryPolicy

class HFEmbedder:
    def __init__(self, model_name=None):
        # Use environment variable or default to BGE model
        self.model_name = model_name or os.getenv('EMBEDDING_MODEL', 'BAAI/bge-base-en-v1.5')
        self.api_key = os.getenv('HUGGINGFACE_API_KEY')
        self.endpoint_url = os.getenv('HUGGINGFACE_ENDPOINT_URL')
        
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY environment variable is required")
        if not self.endpoint_url:
            raise ValueError("HUGGINGFACE_ENDPOINT_URL environment variable is required")

    def encode(self, text: str) -> list[float]:
        """Get embeddings from HuggingFace Inference API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": text,
            "options": {"wait_for_model": True}
        }
        
        response = requests.post(self.endpoint_url, headers=headers, json=payload)
        response.raise_for_status()
        
        # The response format depends on the model, but typically returns a list of floats
        embeddings = response.json()
        if isinstance(embeddings, list):
            return embeddings
        elif isinstance(embeddings, dict) and "embedding" in embeddings:
            return embeddings["embedding"]
        else:
            raise ValueError(f"Unexpected response format from HuggingFace API: {embeddings}")

# Initialize the cache with configuration from environment variables
def get_embedder():
    """Lazy initialization of the embedder."""
    return HFEmbedder()

def get_cache():
    """Lazy initialization of the cache."""
    embedder = get_embedder()
    cache_config = {
        'db_path': os.getenv('CACHE_DB_PATH', 'cache.sqlite'),
        'qdrant_url': os.getenv('QDRANT_URL', 'http://localhost:6333'),
        'collection_name': os.getenv('QDRANT_COLLECTION', 'search_cache'),
        'embed_fn': embedder.encode
    }

    # Add optional configuration parameters if specified
    if os.getenv('QDRANT_API_KEY'):
        cache_config['qdrant_api_key'] = os.getenv('QDRANT_API_KEY')

    if os.getenv('QDRANT_TIMEOUT'):
        cache_config['timeout'] = float(os.getenv('QDRANT_TIMEOUT'))

    if os.getenv('CACHE_TTL'):
        cache_config['ttl'] = int(os.getenv('CACHE_TTL'))

    return HybridCache(**cache_config)

def get_config_value(value):
    """
    Helper function to handle string, dict, and enum cases of configuration values
    """
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def get_search_params(search_api: str, search_api_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Filters the search_api_config dictionary to include only parameters accepted by the specified search API.

    Args:
        search_api (str): The search API identifier (e.g., "exa", "tavily").
        search_api_config (Optional[Dict[str, Any]]): The configuration dictionary for the search API.

    Returns:
        Dict[str, Any]: A dictionary of parameters to pass to the search function.
    """
    # Define accepted parameters for each search API
    SEARCH_API_PARAMS = {
        "exa": ["max_characters", "num_results", "include_domains", "exclude_domains", "subpages"],
        "tavily": ["max_results", "topic"],
        "perplexity": [],  # Perplexity accepts no additional parameters
        "arxiv": ["load_max_docs", "get_full_documents", "load_all_available_meta"],
        "pubmed": ["top_k_results", "email", "api_key", "doc_content_chars_max"],
        "linkup": ["depth"],
        "googlesearch": ["max_results"],
    }

    # Get the list of accepted parameters for the given search API
    accepted_params = SEARCH_API_PARAMS.get(search_api, [])

    # If no config provided, return an empty dict
    if not search_api_config:
        return {}

    # Filter the config to only include accepted parameters
    return {k: v for k, v in search_api_config.items() if k in accepted_params}

def deduplicate_and_format_sources(search_response, max_tokens_per_source=5000, include_raw_content=True):
    """
    Takes a list of search responses and formats them into a readable string.
    Limits the raw_content to approximately max_tokens_per_source tokens.
 
    Args:
        search_responses: List of search response dicts, each containing:
            - query: str
            - results: List of dicts with fields:
                - title: str
                - url: str
                - content: str
                - score: float
                - raw_content: str|None
        max_tokens_per_source: int
        include_raw_content: bool
            
    Returns:
        str: Formatted string with deduplicated sources
    """
     # Collect all results
    sources_list = []
    for response in search_response:
        sources_list.extend(response['results'])
    
    # Deduplicate by URL
    unique_sources = {source['url']: source for source in sources_list}

    # Format output
    formatted_text = "Content from sources:\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"{'='*80}\n"  # Clear section separator
        formatted_text += f"Source: {source['title']}\n"
        formatted_text += f"{'-'*80}\n"  # Subsection separator
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
        formatted_text += f"{'='*80}\n\n" # End section separator
                
    return formatted_text.strip()

def format_sections(sections: list[Section]) -> str:
    """ Format a list of sections into a string """
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
{'='*60}
Section {idx}: {section.name}
{'='*60}
Description:
{section.description}
Requires Research: 
{section.research}

Content:
{section.content if section.content else '[Not yet written]'}

"""
    return formatted_str

@traceable
async def tavily_search_async(search_queries, max_results: int = 5, topic: Literal["general", "news", "finance"] = "general", include_raw_content: bool = True):
    """
    Performs concurrent web searches with the Tavily API

    Args:
        search_queries (List[str]): List of search queries to process
        max_results (int): Maximum number of results to return
        topic (Literal["general", "news", "finance"]): Topic to filter results by
        include_raw_content (bool): Whether to include raw content in the results

    Returns:
            List[dict]: List of search responses from Tavily API:
                {
                    'query': str,
                    'follow_up_questions': None,      
                    'answer': None,
                    'images': list,
                    'results': [                     # List of search results
                        {
                            'title': str,            # Title of the webpage
                            'url': str,              # URL of the result
                            'content': str,          # Summary/snippet of content
                            'score': float,          # Relevance score
                            'raw_content': str|None  # Full page content if available
                        },
                        ...
                    ]
                }
    """
    tavily_async_client = AsyncTavilyClient()
    search_tasks = []
    for query in search_queries:
            search_tasks.append(
                tavily_async_client.search(
                    query,
                    max_results=max_results,
                    include_raw_content=include_raw_content,
                    topic=topic
                )
            )

    # Execute all searches concurrently
    search_docs = await asyncio.gather(*search_tasks)
    return search_docs

@traceable
@with_rate_limit(rate=10.0, burst=5)  # 10 requests per second, burst of 5
@with_retry(RetryPolicy(
    max_retries=3,
    initial_delay=1.0,
    max_delay=10.0,
    backoff_factor=2.0,
    jitter=True
))
async def azureaisearch_search_async(search_queries: list[str], max_results: int = 5, topic: str = "general", include_raw_content: bool = True) -> list[dict]:
    """
    Performs concurrent web searches using the Azure AI Search API.

    Args:
        search_queries (List[str]): list of search queries to process
        max_results (int): maximum number of results to return for each query
        topic (str): semantic topic filter for the search.
        include_raw_content (bool)

    Returns:
        List[dict]: list of search responses from Azure AI Search API, one per query.
    """
    # configure and create the Azure Search client
    # ensure all environment variables are set
    if not all(var in os.environ for var in ["AZURE_AI_SEARCH_ENDPOINT", "AZURE_AI_SEARCH_INDEX_NAME", "AZURE_AI_SEARCH_API_KEY"]):
        raise ValueError("Missing required environment variables for Azure Search API which are: AZURE_AI_SEARCH_ENDPOINT, AZURE_AI_SEARCH_INDEX_NAME, AZURE_AI_SEARCH_API_KEY")
    endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
    index_name = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
    credential = AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_API_KEY"))

    reranker_key = '@search.reranker_score'

    async with AsyncAzureAISearchClient(endpoint, index_name, credential) as client:
        async def do_search(query: str) -> dict:
            # search query 
            paged = await client.search(
                search_text=query,
                vector_queries=[{
                    "fields": "vector",
                    "kind": "text",
                    "text": query,
                    "exhaustive": True
                }],
                semantic_configuration_name="fraunhofer-rag-semantic-config",
                query_type="semantic",
                select=["url", "title", "chunk", "creationTime", "lastModifiedTime"],
                top=max_results,
            )
            # async iterator to get all results
            items = [doc async for doc in paged]
            # Umwandlung in einfaches Dict-Format
            results = [
                {
                    "title": doc.get("title"),
                    "url": doc.get("url"),
                    "content": doc.get("chunk"),
                    "score": doc.get(reranker_key),
                    "raw_content": doc.get("chunk") if include_raw_content else None
                }
                for doc in items
            ]
            return {"query": query, "results": results}

        # parallelize the search queries
        tasks = [do_search(q) for q in search_queries]
        return await asyncio.gather(*tasks)


@traceable
@with_rate_limit(rate=5.0, burst=3)  # 5 requests per second, burst of 3
@with_retry(RetryPolicy(
    max_retries=3,
    initial_delay=1.0,
    max_delay=10.0,
    backoff_factor=2.0,
    jitter=True
))
async def perplexity_search(search_queries):
    """Search the web using the Perplexity API.
    
    Args:
        search_queries (List[SearchQuery]): List of search queries to process
  
    Returns:
        List[dict]: List of search responses from Perplexity API, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,      
                'answer': None,
                'images': list,
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the search result
                        'url': str,              # URL of the result
                        'content': str,          # Summary/snippet of content
                        'score': float,          # Relevance score
                        'raw_content': str|None  # Full content or None for secondary citations
                    },
                    ...
                ]
            }
    """
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"
    }
    
    search_docs = []
    for query in search_queries:
        payload = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": "Search the web and provide factual information with sources."
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        }
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Parse the response
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        citations = data.get("citations", ["https://perplexity.ai"])
        
        # Create results list for this query
        results = []
        
        # First citation gets the full content
        results.append({
            "title": f"Perplexity Search, Source 1",
            "url": citations[0],
            "content": content,
            "raw_content": content,
            "score": 1.0  # Adding score to match Tavily format
        })
        
        # Add additional citations without duplicating content
        for i, citation in enumerate(citations[1:], start=2):
            results.append({
                "title": f"Perplexity Search, Source {i}",
                "url": citation,
                "content": "See primary source for full content",
                "raw_content": None,
                "score": 0.5  # Lower score for secondary sources
            })
        
        # Format response to match Tavily structure
        search_docs.append({
            "query": query,
            "follow_up_questions": None,
            "answer": None,
            "images": [],
            "results": results
        })
    
    return search_docs

@traceable
@with_rate_limit(rate=2.0, burst=1)  # 2 requests per second, burst of 1
@with_retry(RetryPolicy(
    max_retries=3,
    initial_delay=1.0,
    max_delay=10.0,
    backoff_factor=2.0,
    jitter=True
))
async def exa_search(search_queries, top_k_results=5, api_key=None, doc_content_chars_max=4000):
    """
    Performs concurrent searches using the Exa API.

    Args:
        search_queries (List[str]): List of search queries
        top_k_results (int, optional): Maximum number of documents to return per query. Default is 5.
        api_key (str, optional): Exa API key. Required for API access.
        doc_content_chars_max (int, optional): Maximum characters for document content. Default is 4000.

    Returns:
        List[dict]: List of search responses from Exa, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,      
                'answer': None,
                'images': [],
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the search result
                        'url': str,              # URL of the search result
                        'content': str,          # Snippet or description
                        'score': float,          # Relevance score (approximated)
                        'raw_content': str       # Full content if available
                    },
                    ...
                ]
            }
    """
    
    async def process_single_query(query):
        try:
            if not api_key:
                raise ValueError("Exa API key is required")
            
            # Create Exa wrapper
            wrapper = ExaSearchAPIWrapper(
                exa_api_key=api_key,
                top_k_results=top_k_results,
                doc_content_chars_max=doc_content_chars_max
            )
            
            # Run the synchronous wrapper in a thread pool
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, lambda: list(wrapper.lazy_load(query)))
            
            print(f"Query '{query}' returned {len(docs)} results")
            
            results = []
            # Assign decreasing scores based on the order
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0
            
            for i, doc in enumerate(docs):
                result = {
                    'title': doc.get('title', ''),
                    'url': doc.get('url', ''),
                    'content': doc.get('text', ''),
                    'score': base_score - (i * score_decrement),
                    'raw_content': doc.get('text', '')
                }
                results.append(result)
            
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': results
            }
        except Exception as e:
            error_msg = f"Error processing Exa query '{query}': {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            }
    
    # Process all queries concurrently
    tasks = [process_single_query(query) for query in search_queries]
    return await asyncio.gather(*tasks)

@traceable
async def arxiv_search_async(search_queries, load_max_docs=5, get_full_documents=True, load_all_available_meta=True):
    """
    Performs concurrent searches on arXiv using the ArxivRetriever.

    Args:
        search_queries (List[str]): List of search queries or article IDs
        load_max_docs (int, optional): Maximum number of documents to return per query. Default is 5.
        get_full_documents (bool, optional): Whether to fetch full text of documents. Default is True.
        load_all_available_meta (bool, optional): Whether to load all available metadata. Default is True.

    Returns:
        List[dict]: List of search responses from arXiv, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,      
                'answer': None,
                'images': [],
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the paper
                        'url': str,              # URL (Entry ID) of the paper
                        'content': str,          # Formatted summary with metadata
                        'score': float,          # Relevance score (approximated)
                        'raw_content': str|None  # Full paper content if available
                    },
                    ...
                ]
            }
    """
    
    async def process_single_query(query):
        try:
            # Create retriever for each query
            retriever = ArxivRetriever(
                load_max_docs=load_max_docs,
                get_full_documents=get_full_documents,
                load_all_available_meta=load_all_available_meta
            )
            
            # Run the synchronous retriever in a thread pool
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, lambda: retriever.invoke(query))
            
            results = []
            # Assign decreasing scores based on the order
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0
            
            for i, doc in enumerate(docs):
                # Extract metadata
                metadata = doc.metadata
                
                # Use entry_id as the URL (this is the actual arxiv link)
                url = metadata.get('entry_id', '')
                
                # Format content with all useful metadata
                content_parts = []

                # Primary information
                if 'Summary' in metadata:
                    content_parts.append(f"Summary: {metadata['Summary']}")

                if 'Authors' in metadata:
                    content_parts.append(f"Authors: {metadata['Authors']}")

                # Add publication information
                published = metadata.get('Published')
                published_str = published.isoformat() if hasattr(published, 'isoformat') else str(published) if published else ''
                if published_str:
                    content_parts.append(f"Published: {published_str}")

                # Add additional metadata if available
                if 'primary_category' in metadata:
                    content_parts.append(f"Primary Category: {metadata['primary_category']}")

                if 'categories' in metadata and metadata['categories']:
                    content_parts.append(f"Categories: {', '.join(metadata['categories'])}")

                if 'comment' in metadata and metadata['comment']:
                    content_parts.append(f"Comment: {metadata['comment']}")

                if 'journal_ref' in metadata and metadata['journal_ref']:
                    content_parts.append(f"Journal Reference: {metadata['journal_ref']}")

                if 'doi' in metadata and metadata['doi']:
                    content_parts.append(f"DOI: {metadata['doi']}")

                # Get PDF link if available in the links
                pdf_link = ""
                if 'links' in metadata and metadata['links']:
                    for link in metadata['links']:
                        if 'pdf' in link:
                            pdf_link = link
                            content_parts.append(f"PDF: {pdf_link}")
                            break

                # Join all content parts with newlines 
                content = "\n".join(content_parts)
                
                result = {
                    'title': metadata.get('Title', ''),
                    'url': url,  # Using entry_id as the URL
                    'content': content,
                    'score': base_score - (i * score_decrement),
                    'raw_content': doc.page_content if get_full_documents else None
                }
                results.append(result)
                
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': results
            }
        except Exception as e:
            # Handle exceptions gracefully
            print(f"Error processing arXiv query '{query}': {str(e)}")
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            }
    
    # Process queries sequentially with delay to respect arXiv rate limit (1 request per 3 seconds)
    search_docs = []
    for i, query in enumerate(search_queries):
        try:
            # Add delay between requests (3 seconds per ArXiv's rate limit)
            if i > 0:  # Don't delay the first request
                await asyncio.sleep(3.0)
            
            result = await process_single_query(query)
            search_docs.append(result)
        except Exception as e:
            # Handle exceptions gracefully
            print(f"Error processing arXiv query '{query}': {str(e)}")
            search_docs.append({
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            })
            
            # Add additional delay if we hit a rate limit error
            if "429" in str(e) or "Too Many Requests" in str(e):
                print("ArXiv rate limit exceeded. Adding additional delay...")
                await asyncio.sleep(5.0)  # Add a longer delay if we hit a rate limit
    
    return search_docs

@traceable
@with_rate_limit(rate=3.0, burst=2)  # 3 requests per second, burst of 2
@with_retry(RetryPolicy(
    max_retries=3,
    initial_delay=1.0,
    max_delay=10.0,
    backoff_factor=2.0,
    jitter=True
))
async def pubmed_search_async(search_queries, top_k_results=5, email=None, api_key=None, doc_content_chars_max=4000):
    """
    Performs concurrent searches on PubMed using the PubMedAPIWrapper.

    Args:
        search_queries (List[str]): List of search queries
        top_k_results (int, optional): Maximum number of documents to return per query. Default is 5.
        email (str, optional): Email address for PubMed API. Required by NCBI.
        api_key (str, optional): API key for PubMed API for higher rate limits.
        doc_content_chars_max (int, optional): Maximum characters for document content. Default is 4000.

    Returns:
        List[dict]: List of search responses from PubMed, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,      
                'answer': None,
                'images': [],
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the paper
                        'url': str,              # URL to the paper on PubMed
                        'content': str,          # Formatted summary with metadata
                        'score': float,          # Relevance score (approximated)
                        'raw_content': str       # Full abstract content
                    },
                    ...
                ]
            }
    """
    
    async def process_single_query(query):
        try:
            # Create PubMed wrapper for the query
            wrapper = PubMedAPIWrapper(
                top_k_results=top_k_results,
                doc_content_chars_max=doc_content_chars_max,
                email=email if email else "your_email@example.com",
                api_key=api_key if api_key else ""
            )
            
            # Run the synchronous wrapper in a thread pool
            loop = asyncio.get_event_loop()
            
            # Use wrapper.lazy_load instead of load to get better visibility
            docs = await loop.run_in_executor(None, lambda: list(wrapper.lazy_load(query)))
            
            print(f"Query '{query}' returned {len(docs)} results")
            
            results = []
            # Assign decreasing scores based on the order
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0
            
            for i, doc in enumerate(docs):
                # Format content with metadata
                content_parts = []
                
                if doc.get('Published'):
                    content_parts.append(f"Published: {doc['Published']}")
                
                if doc.get('Copyright Information'):
                    content_parts.append(f"Copyright Information: {doc['Copyright Information']}")
                
                if doc.get('Summary'):
                    content_parts.append(f"Summary: {doc['Summary']}")
                
                # Generate PubMed URL from the article UID
                uid = doc.get('uid', '')
                url = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/" if uid else ""
                
                # Join all content parts with newlines
                content = "\n".join(content_parts)
                
                result = {
                    'title': doc.get('Title', ''),
                    'url': url,
                    'content': content,
                    'score': base_score - (i * score_decrement),
                    'raw_content': doc.get('Summary', '')
                }
                results.append(result)
            
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': results
            }
        except Exception as e:
            # Handle exceptions with more detailed information
            error_msg = f"Error processing PubMed query '{query}': {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())  # Print full traceback for debugging
            
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            }
    
    # Process all queries concurrently
    tasks = [process_single_query(query) for query in search_queries]
    return await asyncio.gather(*tasks)

@traceable
async def linkup_search(search_queries, depth: Optional[str] = "standard"):
    """
    Performs concurrent web searches using the Linkup API.

    Args:
        search_queries (List[SearchQuery]): List of search queries to process
        depth (str, optional): "standard" (default)  or "deep". More details here https://docs.linkup.so/pages/documentation/get-started/concepts

    Returns:
        List[dict]: List of search responses from Linkup API, one per query. Each response has format:
            {
                'results': [            # List of search results
                    {
                        'title': str,   # Title of the search result
                        'url': str,     # URL of the result
                        'content': str, # Summary/snippet of content
                    },
                    ...
                ]
            }
    """
    client = LinkupClient()
    search_tasks = []
    for query in search_queries:
        search_tasks.append(
                client.async_search(
                    query,
                    depth,
                    output_type="searchResults",
                )
            )

    search_results = []
    for response in await asyncio.gather(*search_tasks):
        search_results.append(
            {
                "results": [
                    {"title": result.name, "url": result.url, "content": result.content}
                    for result in response.results
                ],
            }
        )

    return search_results

@traceable
@with_rate_limit(rate=2.0, burst=1)  # 2 requests per second, burst of 1
@with_retry(RetryPolicy(
    max_retries=3,
    initial_delay=1.0,
    max_delay=10.0,
    backoff_factor=2.0,
    jitter=True
))
async def google_search_async(search_queries, top_k_results=5, api_key=None, cx=None, doc_content_chars_max=4000):
    """
    Performs concurrent searches on Google using either the Google Search API or web scraping.

    Args:
        search_queries (List[str]): List of search queries
        top_k_results (int, optional): Maximum number of documents to return per query. Default is 5.
        api_key (str, optional): Google Search API key. If not provided, falls back to web scraping.
        cx (str, optional): Google Custom Search Engine ID. Required if api_key is provided.
        doc_content_chars_max (int, optional): Maximum characters for document content. Default is 4000.

    Returns:
        List[dict]: List of search responses from Google, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,      
                'answer': None,
                'images': [],
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the search result
                        'url': str,              # URL of the search result
                        'content': str,          # Snippet or description
                        'score': float,          # Relevance score (approximated)
                        'raw_content': str       # Full content if available
                    },
                    ...
                ]
            }
    """
    
    async def process_single_query(query):
        try:
            if api_key and cx:
                # Use Google Search API
                wrapper = GoogleSearchAPIWrapper(
                    google_api_key=api_key,
                    google_cse_id=cx,
                    top_k_results=top_k_results,
                    doc_content_chars_max=doc_content_chars_max
                )
            else:
                # Use web scraping
                wrapper = GoogleSearchAPIWrapper(
                    top_k_results=top_k_results,
                    doc_content_chars_max=doc_content_chars_max
                )
            
            # Run the synchronous wrapper in a thread pool
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, lambda: list(wrapper.lazy_load(query)))
            
            print(f"Query '{query}' returned {len(docs)} results")
            
            results = []
            # Assign decreasing scores based on the order
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0
            
            for i, doc in enumerate(docs):
                result = {
                    'title': doc.get('title', ''),
                    'url': doc.get('link', ''),
                    'content': doc.get('snippet', ''),
                    'score': base_score - (i * score_decrement),
                    'raw_content': doc.get('snippet', '')
                }
                results.append(result)
            
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': results
            }
        except Exception as e:
            error_msg = f"Error processing Google query '{query}': {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            }
    
    # Process all queries concurrently
    tasks = [process_single_query(query) for query in search_queries]
    return await asyncio.gather(*tasks)

async def scrape_pages(titles: List[str], urls: List[str]) -> str:
    """
    Scrapes content from a list of URLs and formats it into a readable markdown document.
    
    This function:
    1. Takes a list of page titles and URLs
    2. Makes asynchronous HTTP requests to each URL
    3. Converts HTML content to markdown
    4. Formats all content with clear source attribution
    
    Args:
        titles (List[str]): A list of page titles corresponding to each URL
        urls (List[str]): A list of URLs to scrape content from
        
    Returns:
        str: A formatted string containing the full content of each page in markdown format,
             with clear section dividers and source attribution
    """
    
    # Create an async HTTP client
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        pages = []
        
        # Fetch each URL and convert to markdown
        for url in urls:
            try:
                # Fetch the content
                response = await client.get(url)
                response.raise_for_status()
                
                # Convert HTML to markdown if successful
                if response.status_code == 200:
                    # Handle different content types
                    content_type = response.headers.get('Content-Type', '')
                    if 'text/html' in content_type:
                        # Convert HTML to markdown
                        markdown_content = markdownify(response.text)
                        pages.append(markdown_content)
                    else:
                        # For non-HTML content, just mention the content type
                        pages.append(f"Content type: {content_type} (not converted to markdown)")
                else:
                    pages.append(f"Error: Received status code {response.status_code}")
        
            except Exception as e:
                # Handle any exceptions during fetch
                pages.append(f"Error fetching URL: {str(e)}")
        
        # Create formatted output 
        formatted_output = f"Search results: \n\n"
        
        for i, (title, url, page) in enumerate(zip(titles, urls, pages)):
            formatted_output += f"\n\n--- SOURCE {i+1}: {title} ---\n"
            formatted_output += f"URL: {url}\n\n"
            formatted_output += f"FULL CONTENT:\n {page}"
            formatted_output += "\n\n" + "-" * 80 + "\n"
        
    return  formatted_output

@tool
async def duckduckgo_search(search_queries: List[str]):
    """Perform searches using DuckDuckGo with retry logic to handle rate limits
    
    Args:
        search_queries (List[str]): List of search queries to process
        
    Returns:
        List[dict]: List of search results
    """
    
    async def process_single_query(query):
        # Execute synchronous search in the event loop's thread pool
        loop = asyncio.get_event_loop()
        
        def perform_search():
            max_retries = 3
            retry_count = 0
            backoff_factor = 2.0
            last_exception = None
            
            while retry_count <= max_retries:
                try:
                    results = []
                    with DDGS() as ddgs:
                        # Change query slightly and add delay between retries
                        if retry_count > 0:
                            # Random delay with exponential backoff
                            delay = backoff_factor ** retry_count + random.random()
                            print(f"Retry {retry_count}/{max_retries} for query '{query}' after {delay:.2f}s delay")
                            time.sleep(delay)
                            
                            # Add a random element to the query to bypass caching/rate limits
                            modifiers = ['about', 'info', 'guide', 'overview', 'details', 'explained']
                            modified_query = f"{query} {random.choice(modifiers)}"
                        else:
                            modified_query = query
                        
                        # Execute search
                        ddg_results = list(ddgs.text(modified_query, max_results=5))
                        
                        # Format results
                        for i, result in enumerate(ddg_results):
                            results.append({
                                'title': result.get('title', ''),
                                'url': result.get('href', ''),
                                'content': result.get('body', ''),
                                'score': 1.0 - (i * 0.1),  # Simple scoring mechanism
                                'raw_content': result.get('body', '')
                            })
                        
                        # Return successful results
                        return {
                            'query': query,
                            'follow_up_questions': None,
                            'answer': None,
                            'images': [],
                            'results': results
                        }
                except Exception as e:
                    # Store the exception and retry
                    last_exception = e
                    retry_count += 1
                    print(f"DuckDuckGo search error: {str(e)}. Retrying {retry_count}/{max_retries}")
                    
                    # If not a rate limit error, don't retry
                    if "Ratelimit" not in str(e) and retry_count >= 1:
                        print(f"Non-rate limit error, stopping retries: {str(e)}")
                        break
            
            # If we reach here, all retries failed
            print(f"All retries failed for query '{query}': {str(last_exception)}")
            # Return empty results but with query info preserved
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(last_exception)
            }
            
        return await loop.run_in_executor(None, perform_search)

    # Process queries with delay between them to reduce rate limiting
    search_docs = []
    urls = []
    titles = []
    for i, query in enumerate(search_queries):
        # Add delay between queries (except first one)
        if i > 0:
            delay = 2.0 + random.random() * 2.0  # Random delay 2-4 seconds
            await asyncio.sleep(delay)
        
        # Process the query
        result = await process_single_query(query)
        search_docs.append(result)
        
        # Safely extract URLs and titles from results, handling empty result cases
        if result['results'] and len(result['results']) > 0:
            for res in result['results']:
                if 'url' in res and 'title' in res:
                    urls.append(res['url'])
                    titles.append(res['title'])
    
    # If we got any valid URLs, scrape the pages
    if urls:
        return await scrape_pages(titles, urls)
    else:
        # Return a formatted error message if no valid URLs were found
        return "No valid search results found. Please try different search queries or use a different search API."

@tool
async def tavily_search(queries: List[str], max_results: int = 5, topic: Literal["general", "news", "finance"] = "general") -> str:
    """
    Fetches results from Tavily search API.
    
    Args:
        queries (List[str]): List of search queries
        max_results (int): Maximum number of results to return
        topic (Literal["general", "news", "finance"]): Topic to filter results by
        
    Returns:
        str: A formatted string of search results
    """
    # Use tavily_search_async with include_raw_content=True to get content directly
    search_results = await tavily_search_async(
        queries,
        max_results=5,
        topic="general",
        include_raw_content=True
    )

    # Format the search results directly using the raw_content already provided
    formatted_output = f"Search results: \n\n"
    
    # Deduplicate results by URL
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = result
    
    # Format the unique results
    for i, (url, result) in enumerate(unique_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        if result.get('raw_content'):
            formatted_output += f"FULL CONTENT:\n{result['raw_content'][:30000]}"  # Limit content size
        formatted_output += "\n\n" + "-" * 80 + "\n"
    
    if unique_results:
        return formatted_output
    else:
        return "No valid search results found. Please try different search queries or use a different search API."


@tool
async def azureaisearch_search(queries: List[str], max_results: int = 5, topic: str = "general") -> str:
    """
    Fetches results from Azure AI Search API.
    
    Args:
        queries (List[str]): List of search queries
        
    Returns:
        str: A formatted string of search results
    """
    # Use azureaisearch_search_async with include_raw_content=True to get content directly
    search_results = await azureaisearch_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True
    )

    # Format the search results directly using the raw_content already provided
    formatted_output = f"Search results: \n\n"
    
    # Deduplicate results by URL
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = result
    
    # Format the unique results
    for i, (url, result) in enumerate(unique_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        if result.get('raw_content'):
            formatted_output += f"FULL CONTENT:\n{result['raw_content'][:30000]}"  # Limit content size
        formatted_output += "\n\n" + "-" * 80 + "\n"
    
    if unique_results:
        return formatted_output
    else:
        return "No valid search results found. Please try different search queries or use a different search API."





async def select_and_execute_search(search_api: str, query_list: list[str], params_to_pass: dict) -> str:
    """Select and execute the appropriate search API.
    
    Args:
        search_api: Name of the search API to use
        query_list: List of search queries to execute
        params_to_pass: Parameters to pass to the search API
        
    Returns:
        Formatted string containing search results
        
    Raises:
        ValueError: If an unsupported search API is specified
    """
    print(f"query_list: {query_list} params_to_pass: {params_to_pass}")
    
    # Get cache instance
    cache = get_cache()
    
    # Try to get results from cache first
    cached_results = []
    uncached_queries = []
    
    for query in query_list:
        # Try exact match first
        exact_result = cache.get_exact(query)
        if exact_result:
            cached_results.append(exact_result)
        else:
            # Try semantic match
            semantic_result = cache.get_semantic(query)
            if semantic_result:
                cached_results.append(semantic_result)
            else:
                uncached_queries.append(query)
    
    # If all queries were cached, return combined results
    if not uncached_queries:
        return deduplicate_and_format_sources(cached_results, max_tokens_per_source=4000)
    
    # Execute search for uncached queries
    search_results = []
    if search_api == "tavily":
        # Tavily search tool used with both workflow and agent 
        search_results = await tavily_search.ainvoke({'queries': uncached_queries}, **params_to_pass)
    elif search_api == "duckduckgo":
        # DuckDuckGo search tool used with both workflow and agent 
        search_results = await duckduckgo_search.ainvoke({'search_queries': uncached_queries})
    elif search_api == "perplexity":
        search_results = perplexity_search(uncached_queries, **params_to_pass)
    elif search_api == "exa":
        search_results = await exa_search(uncached_queries, **params_to_pass)
    elif search_api == "arxiv":
        search_results = await arxiv_search_async(uncached_queries, **params_to_pass)
    elif search_api == "pubmed":
        search_results = await pubmed_search_async(uncached_queries, **params_to_pass)
    elif search_api == "linkup":
        search_results = await linkup_search(uncached_queries, **params_to_pass)
    elif search_api == "googlesearch":
        search_results = await google_search_async(uncached_queries, **params_to_pass)
    elif search_api == "azureaisearch":
        search_results = await azureaisearch_search_async(uncached_queries, **params_to_pass)
    else:
        raise ValueError(f"Unsupported search API: {search_api}")
    
    # Cache the new results
    for query, result in zip(uncached_queries, search_results):
        cache.put_exact(query, result)
        cache.put_semantic(query, result)
    
    # Combine cached and new results
    all_results = cached_results + search_results
    
    return deduplicate_and_format_sources(all_results, max_tokens_per_source=4000)
