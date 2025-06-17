# Semantic Chunking Test

This test script demonstrates the difference between standard length-based text chunking and semantic chunking using LlamaIndex's SemanticSplitterNodeParser.

## Requirements

- Python 3.7+
- LlamaIndex
- Google Gemini API key
- OpenAI API key (for embeddings)

## Setup

1. Create a `.env` file in the `test` directory with the following variables:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

2. Install the required packages:
   ```bash
   pip install llama-index google-generativeai python-dotenv openai
   ```

## Running the Test

```bash
cd test
python test_semantic_chunking.py
```

## What the Test Does

1. Takes a sample markdown document about data security
2. Chunks it using two different methods:
   - Standard length-based chunking (simple character count with basic sentence boundary detection)
   - Semantic chunking (uses LlamaIndex's SemanticSplitterNodeParser with Gemini LLM)
3. Displays and compares the results, showing how semantic chunking creates more contextually coherent chunks

## Important Notes

The SemanticSplitterNodeParser requires both:
1. An LLM (Gemini in this case) for analyzing semantic boundaries
2. An embedding model (OpenAI embeddings in this case) for measuring semantic similarity

If you encounter errors related to missing required fields, make sure both API keys are set and that you have the latest version of LlamaIndex installed.

## Alternative Embedding Models

If you prefer not to use OpenAI for embeddings, you can modify the script to use alternative embedding models such as:

- HuggingFace embeddings
- Cohere embeddings
- Local embedding models

Just update the `embed_model` initialization in the `semantic_chunk_text` function. 