<h1 align="center">Crawl4AI RAG MCP Server</h1>

<p align="center">
  <em>Web Crawling and RAG Capabilities for AI Agents and AI Coding Assistants</em>
</p>

A powerful implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) integrated with [Crawl4AI](https://crawl4ai.com) and [Supabase](https://supabase.com/) for providing AI agents and AI coding assistants with advanced web crawling and RAG capabilities. This project also includes `api_v2.py`, a FastAPI wrapper that exposes these functionalities as REST API endpoints, offering enhanced PDF processing, flexible chunking strategies, and AI-powered RAG.

With this MCP server and its FastAPI interface, you can <b>scrape anything</b> (including PDFs), process it intelligently, and then <b>use that knowledge anywhere</b> for RAG.


## Overview

This MCP server provides tools that enable AI agents to crawl websites (including PDF documents), store content in a vector database (Supabase), and perform RAG over the crawled content. The `api_v2.py` script offers a RESTful interface to these tools with additional features like multiple chunking strategies and advanced PDF parsing.

## Features

- **Smart URL Detection**: Automatically detects and handles different URL types (regular webpages, sitemaps, text files, and PDF documents).
- **PDF Processing**: Directly crawls and extracts text from PDF URLs using LlamaParse (with PyMuPDF/PyPDF2 as fallbacks).
- **Recursive Crawling**: Follows internal links to discover content from websites.
- **Parallel Processing**: Efficiently crawls multiple pages simultaneously.
- **Flexible Content Chunking**:
    - **Standard Chunking**: Intelligently splits content by headers and size using an improved rule-based method (`smart_chunk_markdown`).
    - **Semantic Chunking**: Employs AI-based techniques (`semantic_chunk_text`) for more contextually relevant content division.
- **Vector Search & RAG**: Performs RAG over crawled content using Gemini LLM for response generation, optionally filtering by data source for precision.
- **Source Retrieval**: Retrieve sources available for filtering to guide the RAG process.
- **FastAPI Interface**: `api_v2.py` provides REST API endpoints for all major functionalities, facilitating easier testing and integration.

## Tools

The server provides four essential web crawling and search tools. These tools have been enhanced with PDF processing capabilities and flexible chunking options, especially when accessed via the `api_v2.py` FastAPI interface.

1. **`crawl_single_page`**: Quickly crawl a single web page or PDF document and store its content in the vector database.
    - Supports a `chunking_method` parameter (`standard` or `semantic`) via `api_v2.py`.
2. **`smart_crawl_url`**: Intelligently crawl a URL based on its type (sitemap, llms-full.txt, PDF document, or a regular webpage that needs to be crawled recursively).
    - Supports a `chunking_method` parameter (`standard` or `semantic`) via `api_v2.py`.
3. **`get_available_sources`**: Get a list of all available sources (domains) in the database.
4. **`perform_rag_query`**: Search for relevant content using semantic search, generate an answer using the Gemini LLM, with optional source filtering. Works with content extracted from web pages and PDFs.

## Prerequisites

- [Docker/Docker Desktop](https://www.docker.com/products/docker-desktop/) if running the MCP server as a container (recommended)
- [Python 3.12+](https://www.python.org/downloads/) if running the MCP server directly through uv
- [Supabase](https://supabase.com/) (database for RAG)
 - [OpenAI API key](https://platform.openai.com/api-keys) (for generating embeddings for Supabase)
 - [Gemini API key](https://aistudio.google.com/app/apikey) (for RAG query processing via `api_v2.py`)
 - Optional: [LlamaCloud API key](https://cloud.llamaindex.ai/) (for LlamaParse PDF processing via `api_v2.py`)

## Installation

### Using Docker (Recommended)

1. Clone this repository:
   ```bash
   git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
   cd mcp-crawl4ai-rag
   ```

2. Build the Docker image:
   ```bash
   docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .
   ```

3. Create a `.env` file based on the configuration section below

### Using uv directly (no Docker)

1. Clone this repository:
   ```bash
   git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
   cd mcp-crawl4ai-rag
   ```

2. Install uv if you don't have it:
   ```bash
   pip install uv
   ```

3. Create and activate a virtual environment:
   ```bash
   uv venv
   .venv\Scripts\activate
   # on Mac/Linux: source .venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   uv pip install -e .
   crawl4ai-setup
   ```

5. Create a `.env` file based on the configuration section below

### Running Supabase Locally with Docker (optional)

To run Supabase locally using Docker, follow these steps:

1.  **Get the Supabase code:**
    ```bash
    git clone --depth 1 https://github.com/supabase/supabase
    ```

2.  **Create your new Supabase project directory:**
    ```bash
    mkdir supabase-project
    ```

3.  **Copy the compose files to your project:**
    ```bash
    cp -rf supabase/docker/* supabase-project
    ```

4.  **Copy the fake environment variables:**
    ```bash
    cp supabase/docker/.env.example supabase-project/.env
    ```

5.  **Switch to your project directory:**
    ```bash
    cd supabase-project
    ```

6.  **Pull the latest images:**
    ```bash
    docker compose pull
    ```

7.  **Start the services (in detached mode):**
    ```bash
    docker compose up -d
    ```

After starting Supabase locally, ensure you configure your `.env` file in this project with the correct `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` pointing to your local Supabase instance. Typically, for a local setup, these would be:
SUPABASE_URL=http://localhost:54321
SUPABASE_SERVICE_KEY=your_local_anon_key_or_service_role_key
(The service_role_key can be found in the supabase/.env file after you run `docker compose up -d`. It's usually a long JWT token.)


## Database Setup

Before running the server, you need to set up the database with the pgvector extension:

1. Go to the SQL Editor in your Supabase dashboard (create a new project first if necessary)

2. Create a new query and paste the contents of `crawled_pages.sql`

3. Run the query to create the necessary tables and functions

## Configuration

Create a `.env` file in the project root with the following variables:

```
# MCP Server Configuration
HOST=0.0.0.0
PORT=8051
TRANSPORT=sse # Used by crawl4ai_mcp.py

# OpenAI API Configuration (for embeddings)
OPENAI_API_KEY=your_openai_api_key

# Supabase Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_key

# Gemini API Configuration (for RAG in api_v2.py)
GEMINI_API_KEY=your_gemini_api_key

# LlamaCloud API Configuration (Optional, for LlamaParse PDF processing in api_v2.py)
LLAMA_CLOUD_API_KEY=your_llamacloud_api_key

# Example local Supabase config (if you run Supabase locally)
# SUPABASE_URL=http://localhost:54321
# SUPABASE_SERVICE_KEY=your_local_anon_key
```

## Running the Server

### Using Docker

```bash
docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag
```

### Using Python

```bash
uv run src/crawl4ai_mcp.py
```

The server will start and listen on the configured host and port. This runs the original MCP server.

### Using Python (for `api_v2.py` with FastAPI)

If you want to run the FastAPI version (`api_v2.py`) which includes PDF processing and advanced features:
```bash
uvicorn src.api_v2:app --host 0.0.0.0 --port 8051 --reload
```
This server also listens on the configured host and port (defaults to 8051 if not set in `.env` or overridden by command line). The `--reload` flag is useful for development.

## Integration with MCP Clients

### SSE Configuration

Once you have the server running with SSE transport, you can connect to it using this configuration:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "transport": "sse",
      "url": "http://localhost:8051/sse"
    }
  }
}
```

> **Note for Windsurf users**: Use `serverUrl` instead of `url` in your configuration:
> ```json
> {
>   "mcpServers": {
>     "crawl4ai-rag": {
>       "transport": "sse",
>       "serverUrl": "http://localhost:8051/sse"
>     }
>   }
> }
> ```
>
> **Note for Docker users**: Use `host.docker.internal` instead of `localhost` if your client is running in a different container. This will apply if you are using this MCP server within n8n!

### Stdio Configuration

Add this server to your MCP configuration for Claude Desktop, Windsurf, or any other MCP client:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "python",
      "args": ["path/to/crawl4ai-mcp/src/crawl4ai_mcp.py"],
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_openai_api_key",
        "SUPABASE_URL": "your_supabase_url",
        "SUPABASE_SERVICE_KEY": "your_supabase_service_key"
      }
    }
  }
}
```

### Docker with Stdio Configuration

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "docker",
      "args": ["run", "--rm", "-i", 
               "-e", "TRANSPORT", 
               "-e", "OPENAI_API_KEY", 
               "-e", "SUPABASE_URL", 
               "-e", "SUPABASE_SERVICE_KEY", 
               "mcp/crawl4ai"],
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_openai_api_key",
        "SUPABASE_URL": "your_supabase_url",
        "SUPABASE_SERVICE_KEY": "your_supabase_service_key"
      }
    }
  }
}
```

## API v2 (FastAPI Interface - `src/api_v2.py`)

The `src/api_v2.py` script provides a RESTful API interface built with FastAPI for the server's web crawling and RAG functionalities. It enhances the core MCP server features with direct PDF processing, flexible chunking strategies, and uses Gemini for RAG. This version is ideal for testing with tools like Postman or for integrations that prefer HTTP APIs.

It uses `fastapi-mcp` to mount the MCP server, making MCP tools available via FastAPI.

### Key Endpoints in `api_v2.py`

-   **`POST /crawl/single`**: Crawl a single web page or PDF.
    -   Request Body Example:
        ```json
        {
          "url": "https://example.com/document.pdf",
          "chunking_method": "semantic", // "standard" or "semantic"
          "chunk_size": 1000
        }
        ```
-   **`POST /crawl/smart`**: Smart crawl based on URL type (webpage, sitemap, .txt file, or PDF).
    -   Request Body Example:
        ```json
        {
          "url": "https://example.com/sitemap.xml",
          "max_depth": 2,
          "max_concurrent": 5,
          "chunk_size": 1500,
          "chunking_method": "standard" // "standard" or "semantic"
        }
        ```
-   **`POST /query/rag`**: Perform a RAG query on stored content.
    -   Request Body Example:
        ```json
        {
          "query": "What are the key features of product X?",
          "source": "example.com", // Optional
          "match_count": 3
        }
        ```
-   **`GET /sources`**: Get a list of all available crawled sources (domains).
-   **`GET /`**: Root endpoint with API information and a list of available endpoints.

Refer to `src/api_v2.py` for detailed request/response models and further information.

## Building Your Own Server

This implementation provides a foundation for building more complex MCP servers with web crawling capabilities. To build your own:

1. Add your own tools by creating methods with the `@mcp.tool()` decorator (in `crawl4ai_mcp.py`) or FastAPI endpoints (in `api_v2.py`).
2. Create your own lifespan function to add your own dependencies.
3. Modify the `utils.py` file for any helper functions you need.
4. Extend the crawling capabilities by adding more specialized crawlers or PDF parsing methods.