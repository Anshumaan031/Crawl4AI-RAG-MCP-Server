# Semantic Chunking Test

This test demonstrates the difference between standard length-based text chunking and semantic chunking using LlamaIndex's SemanticSplitterNodeParser.

## Overview

The test compares two chunking methods:

1. **Standard Length-Based Chunking**: Splits text based on character count with basic sentence boundary detection
2. **Semantic Chunking**: Uses LlamaIndex's SemanticSplitterNodeParser with Gemini to create semantically coherent chunks

## Requirements

- Python 3.7+
- LlamaIndex
- Google Gemini API key

## Installation

1. Set up a Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a `.env` file with your API key:
   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```
3. Install required packages:
   ```
   pip install llama-index google-generativeai python-dotenv langchain
   ```

## Running the Test

### Option 1: Using the script directly

```bash
python test_semantic_chunking.py
```

### Option 2: Using the convenience shell script

```bash
chmod +x run_semantic_test.sh
./run_semantic_test.sh
```

## Integration with Crawl4AI-RAG-MCP-Server

To integrate semantic chunking into the main application:

1. Add the `semantic_chunk_text()` function from the test script to `utils.py`
2. Update `api_v2.py` to replace `smart_chunk_markdown()` with `semantic_chunk_text()` where PDFs are processed
3. Configure chunk size and overlap parameters as needed

Example integration in `api_v2.py`:
```python
# Replace this line in api_v2.py
chunks = smart_chunk_markdown(cleaned_markdown)

# With this line
chunks = semantic_chunk_text(cleaned_markdown, chunk_size=5000, chunk_overlap=50)
```

## Benefits of Semantic Chunking

1. **Contextual Boundaries**: Creates chunks based on semantic meaning rather than arbitrary character counts
2. **Improved RAG Performance**: Better chunks lead to more relevant context for retrievals
3. **Topic Cohesion**: Each chunk tends to contain complete topics or concepts
4. **Reduced Context Fragmentation**: Reduces cases where important contextual information is split across chunks

## Sample Output

The test prints a side-by-side comparison showing how each method chunks the same content, with statistics about the number and size of chunks. 

## Behind the Scenes (How SemanticSplitterNodeParser Works):

1.The text is first split into smaller pieces (usually sentences)
2.The embedding model converts these pieces into vector representations
3.The LLM analyzes the content to find natural topic boundaries
4.Pieces are combined into larger chunks based on semantic similarity
5.The LLM ensures each chunk maintains coherent meaning
6.Each chunk attempts to capture a complete thought or topic