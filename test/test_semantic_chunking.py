"""
Test script to demonstrate semantic chunking using LlamaIndex's SemanticSplitterNodeParser
compared with standard length-based chunking.
"""

import os
import re
import json
from typing import List
from dotenv import load_dotenv
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.schema import Document
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment variables
load_dotenv()

# Check if Gemini API key is set
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY environment variable not set. Please set it for the Gemini LLM.")

# Check if OpenAI API key is set for embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable not set. Please set it for embeddings.")


def standard_chunk_markdown(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Split text into chunks based on simple length-based chunking.
    
    Args:
        text: The text content to chunk
        chunk_size: Target chunk size
    
    Returns:
        List of text chunks
    """
    # Sanitize text
    sanitized_text = ''.join(char for char in text if char.isprintable() or char in ('\n', '\r', '\t'))
    
    # Normalize whitespace
    normalized_text = re.sub(r'\s+', ' ', sanitized_text).strip()
    
    chunks = []
    start = 0
    text_length = len(normalized_text)

    while start < text_length:
        end = start + chunk_size

        if end >= text_length:
            final_chunk = normalized_text[start:].strip()
            if final_chunk:
                chunks.append(final_chunk)
            break

        chunk = normalized_text[start:end]
        
        # Try to break at a sentence boundary
        last_period = chunk.rfind('. ')
        if last_period > chunk_size * 0.5:
            end = start + last_period + 1

        chunk_to_add = normalized_text[start:end].strip()
        if chunk_to_add:
            chunks.append(chunk_to_add)

        start = end

    return chunks


def semantic_chunk_text(text: str, chunk_size: int = 1024, chunk_overlap: int = 20) -> List[str]:
    """
    Split text into semantically meaningful chunks using LlamaIndex's SemanticSplitterNodeParser.
    
    Args:
        text: The text content to chunk
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks preserving semantic meaning
    """
    # Initialize the LLM for semantic splitting
    chunking_llm = Gemini(model="models/gemini-2.0-flash", api_key=GEMINI_API_KEY)
    
    # Initialize the embedding model (required by SemanticSplitterNodeParser)
    embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)
    
    # Create the semantic splitter
    splitter = SemanticSplitterNodeParser(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        llm=chunking_llm,
        embed_model=embed_model
    )
    
    # Create a document from the text
    document = Document(text=text)
    
    # Parse the document into nodes
    nodes = splitter.get_nodes_from_documents([document])
    
    # Extract text from each node
    chunks = [node.text for node in nodes]
    
    return chunks


def save_chunks_to_file(chunks: List[str], filename: str) -> None:
    """
    Save chunks to a file for comparison.
    
    Args:
        chunks: List of text chunks
        filename: Name of the file to save to
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f"CHUNK {i+1}/{len(chunks)} ({len(chunk)} chars)\n")
            f.write("="*50 + "\n")
            f.write(chunk + "\n\n")
            f.write("-"*50 + "\n\n")


def save_chunks_to_json(chunks: List[str], filename: str) -> None:
    """
    Save chunks to a JSON file for programmatic comparison.
    
    Args:
        chunks: List of text chunks
        filename: Name of the file to save to
    """
    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            "index": i,
            "size": len(chunk),
            "text": chunk
        })
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(chunk_data, f, indent=2)


def compare_chunking_methods():
    """Compare semantic chunking with standard chunking using sample markdown."""
    
    # Sample markdown content with different topics/sections
    sample_markdown = """
# Data Security Best Practices

## Introduction to Data Security

Data security refers to protective measures applied to prevent unauthorized access to computers, 
databases and websites. Data security also protects data from corruption. 
Data security is an essential aspect of IT for organizations of every size and type.

Data security includes encryption, tokenization, and key management practices that protect 
data across all applications and platforms. In this document, we will discuss various aspects
of data security including best practices.

## Authentication and Authorization

Authentication is the process of verifying who someone is, whereas authorization is the process 
of verifying what specific applications, files, and data a user has access to. Strong authentication
mechanisms are critical to ensuring data security.

Multi-factor authentication (MFA) adds an additional layer of protection. When users input their 
username and password, they will be required to provide another piece of information, such as a code
from their phone or a fingerprint scan. This makes it much harder for attackers to gain unauthorized access.

## Data Encryption Standards

### Symmetric Encryption
Symmetric encryption uses a single key to encrypt and decrypt data. Common symmetric algorithms include:
- AES (Advanced Encryption Standard)
- DES (Data Encryption Standard)
- 3DES (Triple DES)
- Blowfish

### Asymmetric Encryption
Asymmetric encryption uses a pair of keys: a public key for encryption and a private key for decryption:
- RSA (Rivest–Shamir–Adleman)
- ECC (Elliptic Curve Cryptography)
- Diffie-Hellman
- DSA (Digital Signature Algorithm)

## Cloud Security Considerations

When storing data in the cloud, organizations must implement additional security measures:

1. Understand the shared responsibility model
2. Implement proper access controls
3. Encrypt data at rest and in transit
4. Regularly audit cloud configurations
5. Implement compliance controls for regulatory requirements

Cloud service providers like AWS, Azure, and GCP offer various security tools, but organizations
must configure them properly to ensure adequate protection.

## Incident Response Planning

Even with the best security measures, breaches can still occur. An effective incident response
plan should include the following steps:

1. Preparation
2. Identification
3. Containment
4. Eradication
5. Recovery
6. Lessons learned

Regular testing and updates to the incident response plan are essential to maintaining
organizational readiness for security incidents.
    """
    
    # Set chunk sizes (approximately equal for fair comparison)
    std_chunk_size = 500
    semantic_chunk_size = 512
    
    # Generate chunks using both methods
    std_chunks = standard_chunk_markdown(sample_markdown, chunk_size=std_chunk_size)
    semantic_chunks = semantic_chunk_text(sample_markdown, chunk_size=semantic_chunk_size, chunk_overlap=20)
    
    # Save chunks to files for comparison
    save_chunks_to_file(std_chunks, "standard_chunks.txt")
    save_chunks_to_file(semantic_chunks, "semantic_chunks.txt")
    
    # Save chunks to JSON for programmatic analysis
    save_chunks_to_json(std_chunks, "standard_chunks.json")
    save_chunks_to_json(semantic_chunks, "semantic_chunks.json")
    
    # Print results
    print(f"\n{'-'*50}\nSTANDARD CHUNKING (Length-based, {len(std_chunks)} chunks):\n{'-'*50}")
    for i, chunk in enumerate(std_chunks):
        print(f"\nCHUNK {i+1}/{len(std_chunks)} ({len(chunk)} chars):\n{'-'*30}\n{chunk[:100]}...\n")
    
    print(f"\n{'-'*50}\nSEMANTIC CHUNKING (LlamaIndex, {len(semantic_chunks)} chunks):\n{'-'*50}")
    for i, chunk in enumerate(semantic_chunks):
        print(f"\nCHUNK {i+1}/{len(semantic_chunks)} ({len(chunk)} chars):\n{'-'*30}\n{chunk[:100]}...\n")
    
    # Analyze and compare
    print(f"\n{'-'*50}\nANALYSIS:\n{'-'*50}")
    print(f"Standard Chunking: {len(std_chunks)} chunks with avg size {sum(len(c) for c in std_chunks)/len(std_chunks):.1f} chars")
    print(f"Semantic Chunking: {len(semantic_chunks)} chunks with avg size {sum(len(c) for c in semantic_chunks)/len(semantic_chunks):.1f} chars")
    print(f"\nOutput saved to:")
    print(f"  - standard_chunks.txt (human-readable)")
    print(f"  - semantic_chunks.txt (human-readable)")
    print(f"  - standard_chunks.json (machine-readable)")
    print(f"  - semantic_chunks.json (machine-readable)")


if __name__ == "__main__":
    print("SEMANTIC CHUNKING TEST")
    print("======================")
    print("This script compares standard length-based chunking with LlamaIndex semantic chunking.")
    try:
        compare_chunking_methods()
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"\nError during test: {e}") 