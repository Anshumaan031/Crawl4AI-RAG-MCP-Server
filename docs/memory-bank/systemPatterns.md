# System Patterns

## System Architecture

The Crawl4AI RAG MCP Server follows a microservice architecture, with the MCP server acting as the main entry point for AI agents and AI coding assistants. The server uses Crawl4AI for web crawling, Supabase for storing crawled content and performing vector similarity search, and OpenAI's API for creating embeddings.

## Key Technical Decisions

*   Using Crawl4AI for web crawling due to its flexibility and ability to handle different URL types.
*   Using Supabase for storing crawled content and performing vector similarity search due to its ease of use and scalability.
*   Using OpenAI's API for creating embeddings due to its high quality and availability.
*   Using the Model Context Protocol (MCP) for communication between the server and AI agents.

## Design Patterns in Use

*   **Microservice Architecture:** The server is designed as a microservice, making it easy to integrate with other systems.
*   **Dependency Injection:** The server uses dependency injection to manage its dependencies, making it easy to test and maintain.
*   **Asynchronous Programming:** The server uses asynchronous programming to handle multiple requests concurrently.

## Component Relationships

*   The MCP server receives requests from AI agents.
*   The MCP server uses Crawl4AI to crawl websites.
*   The MCP server uses Supabase to store crawled content and perform vector similarity search.
*   The MCP server uses OpenAI's API to create embeddings.

## Critical Implementation Paths

*   Crawling a website: The server receives a URL, uses Crawl4AI to crawl the website, chunks the content, creates embeddings, and stores the content in Supabase.
*   Performing a RAG query: The server receives a query, creates an embedding for the query, searches Supabase for relevant content, and returns the results to the AI agent.
