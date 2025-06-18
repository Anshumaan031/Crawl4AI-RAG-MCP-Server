"""
Utility functions for the Crawl4AI MCP server.
"""
import os
from typing import List, Dict, Any, Optional
import json
from supabase import create_client, Client
from urllib.parse import urlparse
import openai
import uuid
from datetime import datetime
import requests
import tempfile
from pathlib import Path

# Load OpenAI API key for embeddings
openai.api_key = os.getenv("OPENAI_API_KEY")

# Imports for semantic chunking
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document
from llama_index.llms.gemini import Gemini

def get_supabase_client() -> Client:
    """
    Get a Supabase client with the URL and key from environment variables.
    
    Returns:
        Supabase client instance
    """
    # Prioritize local Supabase environment variables if they exist
    local_url = os.getenv("SUPABASE_URL_LOCAL")
    local_key = os.getenv("SUPABASE_SERVICE_KEY_LOCAL")

    if local_url and local_key:
        print("Connecting to local Supabase instance.")
        return create_client(local_url, local_key)
    else:
        # Fallback to remote Supabase environment variables
        remote_url = os.getenv("SUPABASE_URL")
        remote_key = os.getenv("SUPABASE_SERVICE_KEY")

        if not remote_url or not remote_key:
            raise ValueError("SUPABASE_URL/SUPABASE_SERVICE_KEY or SUPABASE_URL_LOCAL/SUPABASE_SERVICE_KEY_LOCAL must be set in environment variables")
        
        print("Connecting to remote Supabase instance.")
        return create_client(remote_url, remote_key)

def semantic_chunk_text(text: str, chunk_size: int = 1024, chunk_overlap: int = 20) -> List[str]:
    """
    Split text into semantically meaningful chunks using LlamaIndex's SemanticSplitterNodeParser.
    
    Args:
        text: The text content to chunk
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks preserving semantic meaning
    """
    # Check for GOOGLE_API_KEY in env
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not found in environment. Semantic chunking will fail.")
    
    # Check for OpenAI API key - needed for embeddings
    if not openai.api_key:
        print("Warning: OPENAI_API_KEY not found. Semantic chunking requires embeddings.")
        # Fall back to standard chunking
        from utils import smart_chunk_markdown
        return smart_chunk_markdown(text, chunk_size)
        
    # Initialize the LLM for semantic splitting
    chunking_llm = Gemini(model="models/gemini-2.0-flash", api_key=api_key)
    
    try:
        # Create an embedding model using OpenAI (already imported and set up)
        from llama_index.embeddings.openai import OpenAIEmbedding
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=openai.api_key
        )
        
        # Create the semantic splitter
        splitter = SemanticSplitterNodeParser(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            llm=chunking_llm,
            embed_model=embed_model
        )
        
        # Create a document from the text
        document = Document(text=text)
        
        # Parse the document into nodes
        nodes = splitter.get_nodes_from_documents([document])
        
        # Extract text from each node
        chunks = [node.text for node in nodes]
        
        if not chunks:
            print("Warning: Semantic chunking produced no chunks. Falling back to standard chunking.")
            from utils import smart_chunk_markdown
            return smart_chunk_markdown(text, chunk_size)
        
        return chunks
        
    except Exception as e:
        print(f"Error during semantic chunking: {e}. Falling back to standard chunking.")
        # Fall back to the standard chunking method if semantic chunking fails
        from utils import smart_chunk_markdown
        return smart_chunk_markdown(text, chunk_size)

def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts in a single API call.
    
    Args:
        texts: List of texts to create embeddings for
        
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []
        
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small", # Hardcoding embedding model for now, will change this later to be more dynamic
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Error creating batch embeddings: {e}")
        # Return empty embeddings if there's an error
        return [[0.0] * 1536 for _ in range(len(texts))]

def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using OpenAI's API.
    
    Args:
        text: Text to create an embedding for
        
    Returns:
        List of floats representing the embedding
    """
    try:
        embeddings = create_embeddings_batch([text])
        return embeddings[0] if embeddings else [0.0] * 1536
    except Exception as e:
        print(f"Error creating embedding: {e}")
        # Return empty embedding if there's an error
        return [0.0] * 1536

def add_documents_to_supabase(
    client: Client, 
    urls: List[str], 
    chunk_numbers: List[int],
    contents: List[str], 
    metadatas: List[Dict[str, Any]],
    batch_size: int = 20
) -> None:
    """
    Add documents to the Supabase crawled_pages table in batches.
    Deletes existing records with the same URLs before inserting to prevent duplicates.
    
    Args:
        client: Supabase client
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of document metadata
        batch_size: Size of each batch for insertion
    """
    # Get unique URLs to delete existing records
    unique_urls = list(set(urls))
    
    # Delete existing records for these URLs in a single operation
    try:
        if unique_urls:
            # Use the .in_() filter to delete all records with matching URLs
            client.table("crawled_pages").delete().in_("url", unique_urls).execute()
    except Exception as e:
        print(f"Batch delete failed: {e}. Trying one-by-one deletion as fallback.")
        # Fallback: delete records one by one
        for url in unique_urls:
            try:
                client.table("crawled_pages").delete().eq("url", url).execute()
            except Exception as inner_e:
                print(f"Error deleting record for URL {url}: {inner_e}")
                # Continue with the next URL even if one fails
    
    # Process in batches to avoid memory issues
    for i in range(0, len(contents), batch_size):
        batch_end = min(i + batch_size, len(contents))
        
        # Get batch slices
        batch_urls = urls[i:batch_end]
        batch_chunk_numbers = chunk_numbers[i:batch_end]
        batch_contents = contents[i:batch_end]
        batch_metadatas = metadatas[i:batch_end]
        
        # Create embeddings for the entire batch at once
        batch_embeddings = create_embeddings_batch(batch_contents)
        
        batch_data = []
        for j in range(len(batch_contents)):
            # Generate a unique ID for the document
            # Using a combination of UUID and timestamp to ensure uniqueness
            unique_id = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}-{uuid.uuid4()}"

            # Extract metadata fields
            chunk_size = len(batch_contents[j])
            
            # Prepare data for insertion
            data = {
                "id": unique_id, # Include the generated unique ID
                "url": batch_urls[j],
                "chunk_number": batch_chunk_numbers[j],
                "content": batch_contents[j],
                "metadata": {
                    "chunk_size": chunk_size,
                    **batch_metadatas[j]
                },
                "embedding": batch_embeddings[j]
            }
            
            batch_data.append(data)
        
        # Insert batch into Supabase
        try:
            client.table("crawled_pages").insert(batch_data).execute()
        except Exception as e:
            print(f"Error inserting batch into Supabase: {e}")

def search_documents(
    client: Client, 
    query: str, 
    match_count: int = 10, 
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search for documents in Supabase using vector similarity.
    
    Args:
        client: Supabase client
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        
    Returns:
        List of matching documents
    """
    # Create embedding for the query
    query_embedding = create_embedding(query)
    
    # Execute the search using the match_crawled_pages function
    try:
        # Only include filter parameter if filter_metadata is provided and not empty
        params = {
            'query_embedding': query_embedding,
            'match_count': match_count
        }
        
        # Only add the filter if it's actually provided and not empty
        if filter_metadata:
            params['filter'] = filter_metadata  # Pass the dictionary directly, not JSON-encoded
        
        result = client.rpc('match_crawled_pages', params).execute()
        
        return result.data
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []

def download_pdf_to_temp_file(url: str, temp_dir: Path) -> str:
    """
    Downloads a PDF from a URL to a temporary file and returns its path.

    Args:
        url (str): The URL of the PDF file.
        temp_dir (Path): The directory where the temporary file should be stored.

    Returns:
        str: The path to the downloaded temporary PDF file.

    Raises:
        requests.exceptions.RequestException: If the download fails.
        ValueError: If the downloaded content is not a PDF.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
            raise ValueError(f"URL does not appear to contain a PDF file. Content-Type: {content_type}")

        # Create a temporary file to save the PDF bytes
        # The delete=False is used because we need the file to persist for LlamaParse
        # and we'll manually delete it later.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=temp_dir) as tmp_file:
            tmp_file.write(response.content)
            temp_pdf_path = tmp_file.name
        
        return temp_pdf_path

    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"Failed to download PDF from {url}: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid content type for {url}: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred during PDF download: {e}")
