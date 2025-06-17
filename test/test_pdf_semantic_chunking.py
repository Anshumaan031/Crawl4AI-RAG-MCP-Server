"""
Test script to demonstrate downloading a PDF and applying semantic chunking to it.
This shows how to integrate semantic chunking into the PDF processing flow of Crawl4AI-RAG-MCP-Server.
"""

import os
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import urlparse

# Add project root to path so we can import modules from src
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Load environment variables
load_dotenv()

# Import functionality from the project
from src.utils import download_pdf_to_temp_file, semantic_chunk_text
from src.api_v2 import extract_text_from_pdf_path

# Check for required environment variable
if not os.getenv("GEMINI_API_KEY"):
    print("GEMINI_API_KEY not set. Please set it in .env file.")
    sys.exit(1)

# PDF URL to test with - using a public PDF about data security
PDF_URL = "https://www.ftc.gov/system/files/attachments/protecting-personal-information-guide-business/cybersecurity-small-business-basics-data-security.pdf"

def test_pdf_semantic_chunking(pdf_url=PDF_URL):
    """
    Test downloading a PDF and applying semantic chunking to the extracted text.
    
    Args:
        pdf_url: URL of a PDF to process
    """
    print(f"Testing PDF semantic chunking with {pdf_url}")
    
    # Create temporary directory for PDF download
    temp_dir = Path(tempfile.gettempdir()) / "pdf_test"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Download the PDF
        print("\n--- Downloading PDF ---")
        pdf_path = download_pdf_to_temp_file(pdf_url, temp_dir)
        print(f"PDF downloaded to: {pdf_path}")
        
        # Step 2: Extract text from PDF
        print("\n--- Extracting text from PDF ---")
        # Try LlamaParse first, fall back to PyMuPDF
        try:
            pdf_text = extract_text_from_pdf_path(pdf_path, "pymupdf")
        except Exception as e:
            print(f"Error with LlamaParse: {e}. Using PyMuPDF.")
            pdf_text = extract_text_from_pdf_path(pdf_path, "pymupdf")
        
        text_length = len(pdf_text)
        print(f"Extracted {text_length} characters from PDF")
        print(f"Sample text: {pdf_text[:300]}...\n")
        
        # Step 3: Apply semantic chunking
        print("\n--- Applying semantic chunking ---")
        chunks = semantic_chunk_text(pdf_text, chunk_size=1024, chunk_overlap=50)
        
        print(f"Created {len(chunks)} semantic chunks")
        print("\n--- Sample chunks ---")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"\nCHUNK {i+1}/{len(chunks)} ({len(chunk)} chars):")
            print(f"{chunk[:200]}...\n")
        
        # Step 4: Show metadata for chunks
        print("\n--- Chunk metadata ---")
        source = urlparse(pdf_url).netloc
        for i, chunk in enumerate(chunks[:3]):  # Just the first 3 chunks
            metadata = {
                "chunk_index": i,
                "url": pdf_url,
                "source": source,
                "chunk_size": len(chunk),
                "content_type": "pdf"
            }
            print(f"Chunk {i+1} metadata: {metadata}")
        
        print(f"\nTotal chunks: {len(chunks)}")
        print(f"Average chunk size: {sum(len(c) for c in chunks)/len(chunks):.1f} characters")
        
        # Step 5: Clean up
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            print(f"\nCleaned up temporary PDF file: {pdf_path}")
        
        return True
        
    except Exception as e:
        print(f"Error during test: {e}")
        return False

if __name__ == "__main__":
    print("PDF SEMANTIC CHUNKING TEST")
    print("==========================")
    
    if len(sys.argv) > 1:
        # Use custom PDF URL if provided
        test_pdf_semantic_chunking(sys.argv[1])
    else:
        # Use default PDF URL
        test_pdf_semantic_chunking()