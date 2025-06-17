# Supabase SQL Data Schema for Crawled Pages

This document describes the SQL data schema used in Supabase to store crawled web pages and their embeddings.

## Tables

### `crawled_pages`

This table stores the crawled web page content, metadata, and embeddings.

#### Columns

*   `id` (`bigserial`): The primary key for the table, automatically generated.
*   `url` (`varchar`): The URL of the crawled web page.
*   `chunk_number` (`integer`): The chunk number of the content, used to split large pages into smaller chunks.
*   `content` (`text`): The actual content of the crawled web page chunk.
*   `metadata` (`jsonb`): A JSONB column storing metadata about the crawled page, such as the source domain.
*   `embedding` (`vector(1536)`): A vector embedding of the content, used for semantic search. The dimension is 1536 because OpenAI embeddings are used.
*   `created_at` (`timestamp with time zone`): The timestamp when the record was created, automatically set to the current time in UTC.

#### Constraints

*   `primary key (id)`: The `id` column is the primary key for the table.
*   `unique(url, chunk_number)`: This unique constraint prevents duplicate chunks for the same URL.

#### Data Handling for Duplicate URLs

When a URL is crawled, either individually or as part of a batch, the application logic in `src/utils.py` ensures that existing data for that specific URL is updated rather than duplicated. Before inserting new content chunks for a URL, any previously stored records associated with that URL in the `crawled_pages` table are deleted. This effectively means that if the same URL is crawled again, its content in the database will be **replaced** with the latest crawled data, preventing redundant entries and maintaining data freshness.

#### Indexes

*   `crawled_pages_embedding_idx`: An index on the `embedding` column using the `ivfflat` algorithm for efficient vector similarity search.
*   `idx_crawled_pages_metadata`: A GIN index on the `metadata` column for faster filtering based on metadata values.
*   `idx_crawled_pages_source`: An index on the `source` field within the `metadata` column.

## Functions

### `match_crawled_pages`

This function performs a vector similarity search on the `crawled_pages` table.

#### Parameters

*   `query_embedding` (`vector(1536)`): The vector embedding of the search query.
*   `match_count` (`int`, default `10`): The maximum number of results to return.
*   `filter` (`jsonb`, default `'{}'::jsonb`): A JSONB object used to filter the results based on metadata.

#### Returns

A table with the following columns:

*   `id` (`bigint`): The ID of the matching crawled page chunk.
*   `url` (`varchar`): The URL of the crawled page.
*   `chunk_number` (`integer`): The chunk number of the content.
*   `content` (`text`): The content of the crawled page chunk.
*   `metadata` (`jsonb`): The metadata associated with the crawled page chunk.
*   `similarity` (`float`): The similarity score between the query embedding and the crawled page chunk embedding.

## Row Level Security (RLS)

Row Level Security (RLS) is enabled on the `crawled_pages` table to control access to the data.

### Policies

*   `Allow public read access`: Allows anyone to read the data in the table.
*   `Allow authenticated users to insert`: Allows authenticated users to insert data into the table.
