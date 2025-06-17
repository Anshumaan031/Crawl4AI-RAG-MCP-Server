"""
Utility script to visualize and compare chunks from different chunking methods.

This script loads chunks from the JSON files created by test_semantic_chunking.py
and displays them side by side for comparison.
"""

import json
import os
import sys
from typing import List, Dict, Any
import textwrap

def load_chunks(filename: str) -> List[Dict[str, Any]]:
    """Load chunks from a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filename} not found. Run test_semantic_chunking.py first.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: File {filename} is not valid JSON.")
        sys.exit(1)

def show_chunk_summary(chunks: List[Dict[str, Any]], method_name: str) -> None:
    """Display a summary of chunks."""
    total_size = sum(chunk['size'] for chunk in chunks)
    avg_size = total_size / len(chunks) if chunks else 0
    
    print(f"\n{method_name} Summary:")
    print(f"  - Total chunks: {len(chunks)}")
    print(f"  - Total characters: {total_size}")
    print(f"  - Average chunk size: {avg_size:.1f} characters")
    
    # Size distribution
    size_ranges = {
        "< 300 chars": 0,
        "300-500 chars": 0,
        "500-700 chars": 0,
        "> 700 chars": 0
    }
    
    for chunk in chunks:
        size = chunk['size']
        if size < 300:
            size_ranges["< 300 chars"] += 1
        elif size < 500:
            size_ranges["300-500 chars"] += 1
        elif size < 700:
            size_ranges["500-700 chars"] += 1
        else:
            size_ranges["> 700 chars"] += 1
    
    print("  - Size distribution:")
    for range_name, count in size_ranges.items():
        print(f"    - {range_name}: {count} chunks ({count/len(chunks)*100:.1f}%)")

def compare_specific_chunk(std_chunks: List[Dict[str, Any]], sem_chunks: List[Dict[str, Any]], 
                           std_index: int, sem_index: int) -> None:
    """Compare specific chunks from each method side by side."""
    if std_index >= len(std_chunks) or sem_index >= len(sem_chunks):
        print("Error: Chunk index out of range")
        return
    
    std_chunk = std_chunks[std_index]
    sem_chunk = sem_chunks[sem_index]
    
    print("\n" + "="*100)
    print(f"COMPARING CHUNKS: Standard #{std_index+1} vs Semantic #{sem_index+1}")
    print("="*100)
    
    # Format and display chunks side by side
    print(f"\nSTANDARD CHUNK #{std_index+1} ({std_chunk['size']} chars)")
    print("-"*50)
    print(textwrap.fill(std_chunk['text'], width=80))
    
    print("\n\n")
    
    print(f"SEMANTIC CHUNK #{sem_index+1} ({sem_chunk['size']} chars)")
    print("-"*50)
    print(textwrap.fill(sem_chunk['text'], width=80))
    
    print("\n" + "="*100)

def find_chunks_with_keyword(chunks: List[Dict[str, Any]], keyword: str) -> List[int]:
    """Find chunks containing a specific keyword."""
    matches = []
    for i, chunk in enumerate(chunks):
        if keyword.lower() in chunk['text'].lower():
            matches.append(i)
    return matches

def main():
    """Main function to compare chunks."""
    # Check if files exist
    std_file = "standard_chunks.json"
    sem_file = "semantic_chunks.json"
    
    if not os.path.exists(std_file) or not os.path.exists(sem_file):
        print("Error: Chunk files not found. Run test_semantic_chunking.py first.")
        sys.exit(1)
    
    # Load chunks
    std_chunks = load_chunks(std_file)
    sem_chunks = load_chunks(sem_file)
    
    # Display summary
    print("\n" + "="*50)
    print("CHUNK COMPARISON ANALYSIS")
    print("="*50)
    
    show_chunk_summary(std_chunks, "Standard Chunking")
    show_chunk_summary(sem_chunks, "Semantic Chunking")
    
    # Interactive mode
    print("\n" + "="*50)
    print("INTERACTIVE COMPARISON")
    print("="*50)
    print("\nOptions:")
    print("1. Compare specific chunks")
    print("2. Find chunks containing keyword")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            std_idx = int(input(f"Enter standard chunk index (0-{len(std_chunks)-1}): "))
            sem_idx = int(input(f"Enter semantic chunk index (0-{len(sem_chunks)-1}): "))
            compare_specific_chunk(std_chunks, sem_chunks, std_idx, sem_idx)
            
        elif choice == "2":
            keyword = input("Enter keyword to search for: ")
            std_matches = find_chunks_with_keyword(std_chunks, keyword)
            sem_matches = find_chunks_with_keyword(sem_chunks, keyword)
            
            print(f"\nFound keyword '{keyword}' in:")
            print(f"- Standard chunks: {std_matches}")
            print(f"- Semantic chunks: {sem_matches}")
            
            if std_matches and sem_matches:
                compare = input("\nCompare first matches? (y/n): ")
                if compare.lower() == 'y':
                    compare_specific_chunk(std_chunks, sem_chunks, std_matches[0], sem_matches[0])
                    
        elif choice == "3":
            print("Exiting.")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main() 