# Progress

## What works
- Initial setup of FastAPI application.
- Basic web crawling for single pages.
- RAG query endpoint for retrieving and generating answers from stored content.
- Parsing of sitemap.xml files from hosted URLs.
- Parsing of local sitemap.xml files.
- Robust sanitization and whitespace normalization of text content before creating embeddings to prevent `$.input` invalid errors from OpenAI API.
- Aggressive text cleaning (UTF-8 encode/decode with error ignore) for content extracted from PDFs in the single page crawler endpoint to further mitigate embedding issues.
- **Support for local Supabase instance**: The application can now connect to a locally hosted Supabase instance by prioritizing `SUPABASE_URL_LOCAL` and `SUPABASE_SERVICE_KEY_LOCAL` environment variables.

## What's left to build
- Further testing of the smart crawl functionality with various sitemap and text file types.
- Comprehensive error handling and logging.
- Performance optimizations for large-scale crawling and RAG queries.

## Current status
The core functionality for web crawling and RAG queries is implemented. The system can now handle both remote and local sitemap.xml files, and connect to either remote or local Supabase instances based on environment variable configuration. All necessary project dependencies are now declared in `pyproject.toml`. A critical bug related to embedding creation due to invalid input characters from scraped content (especially PDFs) has been addressed by implementing robust text sanitization and an aggressive text cleaning step. The URL normalization issue for duplicate keys was explicitly deprioritized by the user.

## Known issues
- Duplicate key value violates unique constraint "crawled_pages_url_chunk_number_key" (user has deprioritized fixing this for now).

## Evolution of project decisions
- **Initial decision:** Implement basic web crawling and RAG.
- **Decision to support local sitemaps:** User requested the ability to parse local `sitemap.xml` files, which was implemented by modifying `is_sitemap` and `parse_sitemap` functions in `src/api.py`.
- **Decision to sanitize text for embeddings:** An `Error creating batch embeddings: $.input' is invalid` error was encountered when processing content, particularly from PDFs. This led to the implementation of a robust text sanitization and whitespace normalization step in `smart_chunk_markdown` in `src/api.py`.
- **Decision to add aggressive PDF text cleaning:** The embedding error persisted for PDFs, leading to an additional aggressive text cleaning step (UTF-8 encode/decode with error ignore) in `crawl_single_page_endpoint` in `src/api.py` to handle potentially malformed characters from PDF extraction.
- **Decision to deprioritize URL normalization:** User explicitly requested to "forget the URL normalization" for the duplicate key error.
- **Decision to support local Supabase:** Implemented logic in `src/utils.py` to prioritize connection to a local Supabase instance using `SUPABASE_URL_LOCAL` and `SUPABASE_SERVICE_KEY_LOCAL` environment variables.
- **Decision to update dependencies:** Added `fastapi`, `pydantic`, `requests`, `llama-index-core`, and `llama-index-llms-gemini` to `pyproject.toml` to ensure all necessary project dependencies are declared.
