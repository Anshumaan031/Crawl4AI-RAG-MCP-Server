# Progress

## What Works

*   The MCP server can be started and run.
*   The server can crawl websites using Crawl4AI.
*   The server can store crawled content in Supabase.
*   The server can perform RAG queries on the stored content.
*   The server supports different URL types (regular webpages, sitemaps, text files).
*   The server provides efficient crawling and storage of web content.
*   The server allows filtering RAG queries by data source.

## What's Left to Build

*   More specialized crawlers for different types of websites.
*   Improved error handling and logging.
*   More comprehensive documentation.
*   Automated testing.

## Current Status

The project is in the documentation phase.
Crawled the pages of pydantic ai documentation: https://ai.pydantic.dev/llms-full.txt
Created `docs/database.md` file explaining the Supabase SQL data schema in depth.

## Known Issues

*   The server may not be able to crawl all websites due to various reasons (e.g., rate limits, anti-scraping measures).
*   The quality of the RAG results depends on the quality of the crawled content and the embeddings.

## Evolution of Project Decisions

*   The project initially used a different web crawling library, but Crawl4AI was chosen due to its flexibility and ease of use.
*   The project initially used a different database, but Supabase was chosen due to its ease of use and scalability.
