# Tech Context

## Technologies Used

*   Python 3.12+: Programming language.
*   FastAPI: Web framework for building the API.
*   Pydantic: Data validation and settings management.
*   Requests: HTTP library for making web requests.
*   LlamaIndex: Data framework for LLM applications, used for RAG.
*   Crawl4AI: Web crawling library.
*   MCP: Model Context Protocol library.
*   Supabase: Database for storing crawled content and performing vector similarity search.
*   OpenAI API: For generating embeddings.
*   uv: Python package installer and virtual environment manager.
*   Docker: Containerization platform.

## Development Setup

1.  Clone the repository.
2.  Install uv: `pip install uv`
3.  Create and activate a virtual environment: `uv venv` and `.venv\\Scripts\\activate` (or `source .venv/bin/activate` on Mac/Linux).
4.  Install dependencies: `uv pip install -e .`
5.  Create a `.env` file with the necessary environment variables (see Configuration section in README.md).
    *   **For Local Supabase**: If using a locally hosted Supabase instance, ensure your `.env` file includes `SUPABASE_URL_LOCAL` and `SUPABASE_SERVICE_KEY_LOCAL` pointing to your local setup. The application will prioritize these if present.
6.  Run the server: `uv run src/crawl4ai_mcp.py`

## Technical Constraints

*   Requires a Supabase account and OpenAI API key.
*   Limited by the rate limits of the OpenAI API.
*   Requires a stable internet connection for crawling websites.

## Dependencies

*   See `pyproject.toml` for a list of dependencies.

## Tool Usage Patterns

*   Use `fastapi` for defining API endpoints and handling HTTP requests.
*   Use `pydantic` for request and response data validation.
*   Use `requests` for making HTTP requests (e.g., parsing sitemaps from URLs).
*   Use `llama_index` for RAG (Retrieval Augmented Generation) queries with LLMs.
*   Use `crawl4ai` library for crawling websites.
*   Use `supabase` library for interacting with the Supabase database.
*   Use `openai` library for creating embeddings.
*   Use `mcp` library for creating the MCP server.
*   **Note:** There are no explicit usage limits implemented in the current version of the code.
