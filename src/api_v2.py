"""
FastAPI wrapper for the Crawl4AI MCP server functionality with PDF processing support.

This module exposes the MCP server tools as REST API endpoints for testing with Postman.
Now includes support for processing PDF files directly.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from dataclasses import dataclass
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from dotenv import load_dotenv
from supabase import Client
from pathlib import Path
import requests
import asyncio
import json
import os
import re
import platform
import tempfile
from llama_index.core.llms import ChatMessage
from fastapi_mcp import FastApiMCP

# PDF processing imports
import PyPDF2
import fitz  # PyMuPDF - better PDF text extraction
from io import BytesIO
from pdf_parser import parse_pdf_to_markdown # Import LlamaParse utility

from llama_index.llms.gemini import Gemini

# Get API key from environment variables instead of hardcoding
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    print("Warning: GEMINI_API_KEY not found in environment variables. Using default value.")
    gemini_api_key = "YOUR_API_KEY" # Replace with placeholder instead of real key

llm = Gemini(
    model="models/gemini-2.0-flash",
    api_key=gemini_api_key
)

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
from utils import get_supabase_client, add_documents_to_supabase, search_documents, download_pdf_to_temp_file

# Windows-specific asyncio fix
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)

# Set a custom temporary directory within the project root
custom_temp_dir = project_root / "temp"
if not custom_temp_dir.exists():
    custom_temp_dir.mkdir()
os.environ['TMPDIR'] = str(custom_temp_dir)

# Global context to hold crawler and supabase client
@dataclass
class AppContext:
    """Context for the FastAPI application."""
    crawler: AsyncWebCrawler
    supabase_client: Client

app_context: Optional[AppContext] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application lifecycle.
    """
    global app_context
    
    try:
        # Create browser configuration with Windows-friendly settings
        browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            # Add Windows-specific browser arguments
            browser_type="chromium",  # Explicitly use chromium
            extra_args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-extensions",
                "--disable-plugins",
                "--disable-images",  # Speed up crawling
                "--disable-javascript",  # If you don't need JS execution
            ] if platform.system() == "Windows" else None
        )
        
        # Initialize the crawler
        crawler = AsyncWebCrawler(config=browser_config)
        await crawler.__aenter__()
        
        # Initialize Supabase client
        supabase_client = get_supabase_client()
        
        # Store in global context
        app_context = AppContext(
            crawler=crawler,
            supabase_client=supabase_client
        )
        
        yield
        
    except Exception as e:
        print(f"Error during startup: {e}")
        raise
    finally:
        # Clean up the crawler
        if app_context and app_context.crawler:
            try:
                await app_context.crawler.__aexit__(None, None, None)
            except Exception as e:
                print(f"Error during cleanup: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Crawl4AI RAG API",
    description="REST API for web crawling and RAG queries with Crawl4AI, now with PDF support",
    version="1.1.0",
    lifespan=lifespan
)

mcp = FastApiMCP(app)
# Mount the MCP server directly to your FastAPI app
mcp.mount()

# Pydantic models for request/response
class CrawlSinglePageRequest(BaseModel):
    url: str = Field(..., description="URL of the web page or PDF to crawl")

class SmartCrawlRequest(BaseModel):
    url: str = Field(..., description="URL to crawl (can be a regular webpage, sitemap.xml, .txt file, or PDF)")
    max_depth: int = Field(3, description="Maximum recursion depth for regular URLs")
    max_concurrent: int = Field(10, description="Maximum number of concurrent browser sessions")
    chunk_size: int = Field(5000, description="Maximum size of each content chunk in characters")

class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="The search query")
    source: Optional[str] = Field(None, description="Optional source domain to filter results")
    match_count: int = Field(5, description="Maximum number of results to return")

class APIResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# PDF processing functions
def is_pdf_url(url: str) -> bool:
    """Check if a URL points to a PDF file."""
    return url.lower().endswith('.pdf') or 'pdf' in url.lower()

def extract_text_from_pdf_path(pdf_path: str, method: str = "llama_parse") -> str:
    """
    Extract text from a PDF file using different methods.
    
    Args:
        pdf_path: The path to the PDF file.
        method: Either "llama_parse" (default), "pymupdf" (fallback) or "pypdf2" (fallback).
    
    Returns:
        Extracted text as string.
    """
    text = ""
    
    try:
        if method == "llama_parse":
            try:
                # Check for LLAMA_CLOUD_API_KEY
                if not os.getenv("LLAMA_CLOUD_API_KEY"):
                    print("LLAMA_CLOUD_API_KEY not set. Falling back to PyMuPDF.")
                    return extract_text_from_pdf_path(pdf_path, "pymupdf")
                    
                # --- DEBUGGING: Print temp directory paths ---
                print(f"TEMP environment variable: {os.getenv('TEMP')}")
                print(f"TMP environment variable: {os.getenv('TMP')}")
                print(f"Default temp directory (tempfile.gettempdir()): {tempfile.gettempdir()}")
                # ----------------------------------------------

                # Parse using LlamaParse
                text = parse_pdf_to_markdown(pdf_path)
                    
                if not text.strip():
                    raise ValueError("LlamaParse extracted no text. Falling back to PyMuPDF.")
                
            except Exception as e:
                print(f"Error extracting text from PDF using LlamaParse: {e}. Falling back to PyMuPDF.")
                return extract_text_from_pdf_path(pdf_path, "pymupdf")
            
        elif method == "pymupdf":
            # Using PyMuPDF (fitz) - generally better text extraction
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                page_text = page.get_text()
                text += f"\n\n--- Page {page_num + 1} ---\n\n"
                text += page_text
                
            pdf_document.close()
            
        elif method == "pypdf2":
            # Using PyPDF2 as fallback
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            pdf_file = BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += f"\n\n--- Page {page_num + 1} ---\n\n"
                text += page_text
                
    except Exception as e:
        print(f"Error extracting text from PDF using {method}: {e}")
        # If primary method fails, try the next one in the fallback chain
        if method == "llama_parse":
            return extract_text_from_pdf_path(pdf_path, "pymupdf")
        elif method == "pymupdf":
            return extract_text_from_pdf_path(pdf_path, "pypdf2")
        else:
            raise e
    
    return text

async def process_pdf_url(url: str) -> Dict[str, Any]:
    """
    Download and process a PDF from a URL.
    
    Returns:
        Dict with 'url' and 'markdown' keys, similar to crawler results
    """
    temp_pdf_path = None
    try:
        # Download the PDF to a temporary file
        temp_pdf_path = await asyncio.to_thread(download_pdf_to_temp_file, url, custom_temp_dir)
        
        # Extract text from PDF
        extracted_text = extract_text_from_pdf_path(temp_pdf_path)
        
        if not extracted_text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        return {
            'url': url,
            'markdown': extracted_text,
            'content_type': 'pdf'
        }
        
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to download PDF: {e}")
    except Exception as e:
        raise ValueError(f"Failed to process PDF: {e}")
    finally:
        # Clean up the temporary PDF file
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
                print(f"Cleaned up temporary PDF file: {temp_pdf_path}")
            except Exception as e:
                print(f"Error cleaning up temporary PDF file {temp_pdf_path}: {e}")

# Helper functions (copied from original MCP server)
def is_sitemap(url: str) -> bool:
    """Check if a URL is a sitemap, supporting both remote and local files."""
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path or Path(url).is_file() and url.endswith('sitemap.xml')

def is_txt(url: str) -> bool:
    """Check if a URL is a text file."""
    return url.endswith('.txt')

def parse_sitemap(sitemap_path_or_url: str) -> List[str]:
    """Parse a sitemap and extract URLs, supporting both local files and remote URLs."""
    urls = []
    try:
        if Path(sitemap_path_or_url).is_file():
            # It's a local file
            with open(sitemap_path_or_url, 'rb') as f:
                tree = ElementTree.parse(f)
                urls = [loc.text for loc in tree.findall('.//{*}loc')]
        else:
            # Assume it's a URL
            resp = requests.get(sitemap_path_or_url, stream=True)
            resp.raise_for_status()
            try:
                tree = ElementTree.parse(resp.raw)
                urls = [loc.text for loc in tree.findall('.//{*}loc')]
            finally:
                resp.close()
    except FileNotFoundError:
        print(f"Local sitemap file not found: {sitemap_path_or_url}")
    except ElementTree.ParseError as e:
        print(f"Error parsing sitemap XML from {sitemap_path_or_url}: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching sitemap {sitemap_path_or_url}: {e}")
    return urls

def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs, with robust sanitization and whitespace normalization."""
    # Sanitize text: remove all non-printable characters
    sanitized_text = ''.join(char for char in text if char.isprintable() or char in ('\n', '\r', '\t'))
    
    # Normalize whitespace: replace multiple spaces/newlines with single space, then strip
    normalized_text = re.sub(r'\s+', ' ', sanitized_text).strip()
    
    chunks = []
    start = 0
    text_length = len(normalized_text)

    while start < text_length:
        end = start + chunk_size

        if end >= text_length:
            final_chunk = normalized_text[start:].strip()
            if final_chunk: # Only add if not empty after stripping
                chunks.append(final_chunk)
            break

        chunk = normalized_text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block
        elif '\n\n' in chunk:
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:
                end = start + last_break
        elif '. ' in chunk:
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1

        chunk_to_add = normalized_text[start:end].strip()
        if chunk_to_add: # Only add if not empty after stripping
            chunks.append(chunk_to_add)

        start = end

    return chunks

def extract_section_info(chunk: str) -> Dict[str, Any]:
    """Extracts headers and stats from a chunk."""
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }

async def crawl_markdown_file(crawler: AsyncWebCrawler, url: str) -> List[Dict[str, Any]]:
    """Crawl a .txt or markdown file."""
    crawl_config = CrawlerRunConfig()
    result = await crawler.arun(url=url, config=crawl_config)
    if result.success and result.markdown:
        return [{'url': url, 'markdown': result.markdown}]
    else:
        print(f"Failed to crawl {url}: {result.error_message}")
        return []

async def crawl_batch(crawler: AsyncWebCrawler, urls: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """Batch crawl multiple URLs in parallel."""
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    results = []
    for url in urls:
        try:
            # Check if it's a PDF
            if is_pdf_url(url):
                pdf_result = await process_pdf_url(url)
                results.append(pdf_result)
            else:
                result = await crawler.arun(url=url, config=crawl_config, dispatcher=dispatcher)
                if result.success and result.markdown:
                    results.append({'url': result.url, 'markdown': result.markdown})
                else:
                    print(f"Failed to crawl {url}: {result.error_message}")
        except Exception as e:
            print(f"Error crawling {url}: {e}")
    return results

async def crawl_recursive_internal_links(crawler: AsyncWebCrawler, start_urls: List[str], max_depth: int = 3, max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """Recursively crawl internal links from start URLs up to a maximum depth."""
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    visited = set()

    def normalize_url(url):
        return urldefrag(url)[0]

    current_urls = set([normalize_url(u) for u in start_urls])
    results_all = []

    for depth in range(max_depth):
        urls_to_crawl = [normalize_url(url) for url in current_urls if normalize_url(url) not in visited]
        if not urls_to_crawl:
            break

        results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
        next_level_urls = set()

        for result in results:
            norm_url = normalize_url(result.url)
            visited.add(norm_url)

            if result.success and result.markdown:
                results_all.append({'url': result.url, 'markdown': result.markdown})
                for link in result.links.get("internal", []):
                    next_url = normalize_url(link["href"])
                    if next_url not in visited:
                        next_level_urls.add(next_url)

        current_urls = next_level_urls

    visited.clear()
    return results_all

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Crawl4AI RAG API with PDF Support",
        "version": "1.1.0",
        "endpoints": {
            "POST /crawl/single": "Crawl a single web page or PDF",
            "POST /crawl/smart": "Smart crawl based on URL type (now supports PDFs)",
            "POST /query/rag": "Perform RAG query on stored content",
            "GET /sources": "Get available sources"
        },
        "new_features": [
            "PDF processing support",
            "Automatic PDF text extraction",
            "Enhanced content type detection"
        ]
    }

@app.post("/crawl/single", response_model=APIResponse, operation_id="perform_single_crawl")
async def crawl_single_page_endpoint(request: CrawlSinglePageRequest):
    """
    Crawl a single web page or PDF and store its content in Supabase.
    
    This endpoint now supports both web pages and PDF files. For PDFs, it will
    download and extract the text content. The content is stored in Supabase
    for later retrieval and querying.
    """
    print("Crawling single page/PDF:", request.url)
    if not app_context:
        raise HTTPException(status_code=500, detail="Application context not initialized")
    
    try:
        crawler = app_context.crawler
        supabase_client = app_context.supabase_client
        
        content_result = None
        content_type = "webpage"
        
        # Check if it's a PDF and handle accordingly
        if is_pdf_url(request.url):
            print("Detected PDF URL, processing with PDF extractor...")
            content_result = await process_pdf_url(request.url)
            content_type = "pdf"
        else:
            # Configure the crawl for regular web pages
            run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
            
            # Crawl the page
            result = await crawler.arun(url=request.url, config=run_config)
            
            if result.success and result.markdown:
                content_result = {
                    'url': result.url,
                    'markdown': result.markdown
                }
            else:
                return APIResponse(
                    success=False,
                    error=result.error_message
                )
        
        if content_result and content_result.get('markdown'):
            # Robustly clean the markdown content before chunking
            cleaned_markdown = content_result['markdown'].encode('utf-8', errors='ignore').decode('utf-8')
            
            # Chunk the content
            chunks = smart_chunk_markdown(cleaned_markdown)
            
            # Prepare data for Supabase
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                urls.append(request.url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = request.url
                meta["source"] = urlparse(request.url).netloc
                meta["content_type"] = content_type
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)
            
            # Add to Supabase
            add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas)
            
            return APIResponse(
                success=True,
                data={
                    "url": request.url,
                    "content_type": content_type,
                    "chunks_stored": len(chunks),
                    "content_length": len(content_result['markdown']),
                    "links_count": {
                        "internal": 0 if content_type == "pdf" else "N/A",
                        "external": 0 if content_type == "pdf" else "N/A"
                    }
                }
            )
        else:
            return APIResponse(
                success=False,
                error="No content could be extracted from the URL"
            )
    except Exception as e:
        print(f"Error in crawl_single_page_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/crawl/smart", response_model=APIResponse, operation_id="perform_smart_crawl")
async def smart_crawl_endpoint(request: SmartCrawlRequest):
    """
    Intelligently crawl a URL based on its type and store content in Supabase.
    
    This endpoint automatically detects the URL type and applies the appropriate crawling method:
    - For PDFs: Downloads and extracts text content
    - For sitemaps: Extracts and crawls all URLs in parallel
    - For text files (llms.txt): Directly retrieves the content
    - For regular webpages: Recursively crawls internal links up to the specified depth
    
    All crawled content is chunked and stored in Supabase for later retrieval and querying.
    """
    if not app_context:
        raise HTTPException(status_code=500, detail="Application context not initialized")
    
    try:
        crawler = app_context.crawler
        supabase_client = app_context.supabase_client
        
        crawl_results = []
        crawl_type = "webpage"
        
        # Detect URL type and use appropriate crawl method
        if is_pdf_url(request.url):
            pdf_result = await process_pdf_url(request.url)
            crawl_results = [pdf_result]
            crawl_type = "pdf"
        elif is_txt(request.url):
            crawl_results = await crawl_markdown_file(crawler, request.url)
            crawl_type = "text_file"
        elif is_sitemap(request.url):
            sitemap_urls = parse_sitemap(request.url)
            if not sitemap_urls:
                return APIResponse(
                    success=False,
                    error="No URLs found in sitemap"
                )
            crawl_results = await crawl_batch(crawler, sitemap_urls, max_concurrent=request.max_concurrent)
            crawl_type = "sitemap"
        else:
            crawl_results = await crawl_recursive_internal_links(
                crawler, [request.url], 
                max_depth=request.max_depth, 
                max_concurrent=request.max_concurrent
            )
            crawl_type = "webpage"
        
        if not crawl_results:
            return APIResponse(
                success=False,
                error="No content found"
            )
        
        # Process results and store in Supabase
        urls = []
        chunk_numbers = []
        contents = []
        metadatas = []
        chunk_count = 0
        
        for doc in crawl_results:
            source_url = doc['url']
            md = doc['markdown']
            chunks = smart_chunk_markdown(md, chunk_size=request.chunk_size)
            
            for i, chunk in enumerate(chunks):
                urls.append(source_url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = source_url
                meta["source"] = urlparse(source_url).netloc
                meta["crawl_type"] = crawl_type
                meta["content_type"] = doc.get('content_type', 'webpage')
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)
                
                chunk_count += 1
        
        # Add to Supabase
        batch_size = 20
        add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas, batch_size=batch_size)
        
        return APIResponse(
            success=True,
            data={
                "url": request.url,
                "crawl_type": crawl_type,
                "pages_crawled": len(crawl_results),
                "chunks_stored": chunk_count,
                "urls_crawled": [doc['url'] for doc in crawl_results][:5] + (["..."] if len(crawl_results) > 5 else [])
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/query/rag", response_model=APIResponse, operation_id="perform_rag_query")
async def rag_query_endpoint(request: RAGQueryRequest):
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content.
    
    This endpoint searches the vector database for content relevant to the query,
    uses the matching documents as context, and generates an answer using a Gemini LLM.
    Optionally filter by source domain. Now works with PDF content as well.
    """
    if not app_context:
        raise HTTPException(status_code=500, detail="Application context not initialized")
    
    try:
        supabase_client = app_context.supabase_client
        
        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if request.source and request.source.strip():
            filter_metadata = {"source": request.source}
        
        # Perform the search to retrieve relevant documents (chunks)
        results = search_documents(
            client=supabase_client,
            query=request.query,
            match_count=request.match_count,
            filter_metadata=filter_metadata
        )

        # Extract content from the search results to form the context for the LLM
        context_chunks = []
        for result in results:
            content = result.get("content")
            if content:
                context_chunks.append(content)

        # If no relevant content is found, return an appropriate response
        if not context_chunks:
            return APIResponse(
                success=True, # Still a success, just no content to answer with
                data={
                    "query": request.query,
                    "source_filter": request.source,
                    "answer": "No relevant information found in the knowledge base to answer your query.",
                    "results_count": 0
                }
            )

        # Combine the context chunks into a single string for the LLM
        context = "\n\n".join(context_chunks)

        # Prepare messages for the Gemini LLM
        messages = [
            ChatMessage(role="system", 
                        content=
                        f"""
                        "You are a knowledge expert for Data Security Council of India, NASSCOM. Your task is to provide a direct and precise answer to the user's question about the organization and its activities, based on the provided context:\n\n{context}",
                        "Do not use conversational intros or phrases like Based on the information provided, According to the context, or The document states.",
                        "Your The answer must begin directly.",
                        "Try giving longer answers covering all the conxtual infro from the context provided."
                        """
                        ),
            ChatMessage(role="user", 
                        content=request.query)
        ]

        # Call the Gemini LLM
        resp = llm.chat(messages)

        # Return the AI-generated response
        return APIResponse(
            success=True,
            data={
                "query": request.query,
                "source_filter": request.source,
                "answer": resp.message.content,
                "results_count": len(results)
            }
        )
    except Exception as e:
        # Log the exception for debugging purposes
        print(f"Error during RAG query: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during the RAG query: {str(e)}")

@app.get("/sources", response_model=APIResponse, operation_id="show_all_endpoints")
async def get_available_sources_endpoint():
    """
    Get all available sources based on unique source metadata values.
    
    This endpoint returns a list of all unique sources (domains) that have been crawled and stored
    in the database. This is useful for discovering what content is available for querying.
    """
    if not app_context:
        raise HTTPException(status_code=500, detail="Application context not initialized")
    
    try:
        supabase_client = app_context.supabase_client
        
        result = supabase_client.from_('crawled_pages')\
            .select('metadata')\
            .not_.is_('metadata->>source', 'null')\
            .execute()
            
        # Use a set to efficiently track unique sources
        unique_sources = set()
        
        # Extract the source values from the result using a set for uniqueness
        if result.data:
            for item in result.data:
                source = item.get('metadata', {}).get('source')
                if source:
                    unique_sources.add(source)
        
        # Convert set to sorted list for consistent output
        sources = sorted(list(unique_sources))
        
        return APIResponse(
            success=True,
            data={
                "sources": sources,
                "count": len(sources)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
mcp.setup_server()

if __name__ == "__main__":
    import uvicorn
    
    # Windows-specific asyncio configuration
    if platform.system() == "Windows":
        # Set the event loop policy before running
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8051"))
    
    # Use the current event loop for Windows compatibility
    try:
        uvicorn.run(app, host=host, port=port, loop="asyncio")
    except Exception as e:
        print(f"Error starting server: {e}")
        # Fallback to basic uvicorn run
        uvicorn.run(app, host=host, port=port)