# Product Context

## Why This Project Exists

This project exists to provide AI agents and AI coding assistants with the ability to crawl websites and use the crawled content for RAG. This allows AI agents to access and utilize information from the web, enabling them to perform tasks that require up-to-date information or knowledge from specific websites.

## Problems It Solves

*   Lack of access to real-time information for AI agents.
*   Difficulty in integrating web crawling capabilities into AI workflows.
*   Need for efficient storage and retrieval of crawled content.
*   Requirement for RAG capabilities to answer questions based on crawled data.

## How It Should Work

The MCP server should provide tools that allow AI agents to:

*   Crawl websites based on different URL types (regular webpages, sitemaps, text files).
*   Store the crawled content in a vector database (Supabase).
*   Perform semantic search over the crawled content to find relevant information.
*   Filter search results by source domain.

## User Experience Goals

*   Easy integration with AI agents and AI coding assistants.
*   Simple and intuitive tools for crawling and querying web content.
*   Efficient and reliable crawling and storage of web data.
*   Accurate and relevant search results.
