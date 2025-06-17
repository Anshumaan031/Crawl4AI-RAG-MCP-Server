"""
Generate a markdown report comparing chunks from standard and semantic chunking.

This script creates a detailed report in markdown format, allowing for easy
visualization of the differences between chunking methods.
"""

import json
import os
import sys
from datetime import datetime

def load_chunks(filename):
    """Load chunks from JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filename} not found. Run test_semantic_chunking.py first.")
        sys.exit(1)

def get_section_headers(text):
    """Extract section headers from text."""
    headers = []
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('# '):
            headers.append(('h1', line[2:]))
        elif line.startswith('## '):
            headers.append(('h2', line[3:]))
        elif line.startswith('### '):
            headers.append(('h3', line[4:]))
    return headers

def generate_chunk_summary(chunks):
    """Generate summary statistics for chunks."""
    total_size = sum(chunk['size'] for chunk in chunks)
    avg_size = total_size / len(chunks) if chunks else 0
    sizes = [chunk['size'] for chunk in chunks]
    
    return {
        'count': len(chunks),
        'total_size': total_size,
        'avg_size': avg_size,
        'min_size': min(sizes) if sizes else 0,
        'max_size': max(sizes) if sizes else 0
    }

def generate_report(std_chunks, sem_chunks):
    """Generate markdown report comparing chunks."""
    # Get summary stats
    std_summary = generate_chunk_summary(std_chunks)
    sem_summary = generate_chunk_summary(sem_chunks)
    
    report = []
    
    # Add report header
    report.append("# Chunk Comparison Report")
    report.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    report.append("")
    
    # Add summary section
    report.append("## Summary")
    report.append("")
    report.append("| Metric | Standard Chunking | Semantic Chunking |")
    report.append("| ------ | ----------------- | ----------------- |")
    report.append(f"| Number of chunks | {std_summary['count']} | {sem_summary['count']} |")
    report.append(f"| Total characters | {std_summary['total_size']} | {sem_summary['total_size']} |")
    report.append(f"| Average chunk size | {std_summary['avg_size']:.1f} | {sem_summary['avg_size']:.1f} |")
    report.append(f"| Minimum chunk size | {std_summary['min_size']} | {sem_summary['min_size']} |")
    report.append(f"| Maximum chunk size | {std_summary['max_size']} | {sem_summary['max_size']} |")
    report.append("")
    
    # Add visualization of chunk sizes
    report.append("## Chunk Size Distribution")
    report.append("")
    report.append("### Standard Chunking")
    report.append("<pre>")
    for i, chunk in enumerate(std_chunks):
        # Create a bar chart showing relative sizes
        bar_length = int(chunk['size'] / 20)  # 1 character = 20 chars in the chunk
        report.append(f"Chunk {i+1}: {'█' * bar_length} ({chunk['size']} chars)")
    report.append("</pre>")
    report.append("")
    
    report.append("### Semantic Chunking")
    report.append("<pre>")
    for i, chunk in enumerate(sem_chunks):
        bar_length = int(chunk['size'] / 20)
        report.append(f"Chunk {i+1}: {'█' * bar_length} ({chunk['size']} chars)")
    report.append("</pre>")
    report.append("")
    
    # Add section showing topic distribution
    report.append("## Topic Distribution Across Chunks")
    report.append("")
    report.append("### Standard Chunking")
    report.append("")
    for i, chunk in enumerate(std_chunks):
        headers = get_section_headers(chunk['text'])
        report.append(f"**Chunk {i+1}** ({chunk['size']} chars):")
        if headers:
            report.append("<ul>")
            for level, title in headers:
                indent = "&nbsp;" * (4 if level == 'h2' else 8 if level == 'h3' else 0)
                report.append(f"<li>{indent}{title}</li>")
            report.append("</ul>")
        else:
            report.append("- *No section headers*")
        report.append("")
    
    report.append("### Semantic Chunking")
    report.append("")
    for i, chunk in enumerate(sem_chunks):
        headers = get_section_headers(chunk['text'])
        report.append(f"**Chunk {i+1}** ({chunk['size']} chars):")
        if headers:
            report.append("<ul>")
            for level, title in headers:
                indent = "&nbsp;" * (4 if level == 'h2' else 8 if level == 'h3' else 0)
                report.append(f"<li>{indent}{title}</li>")
            report.append("</ul>")
        else:
            report.append("- *No section headers*")
        report.append("")
    
    # Add detailed comparison section
    report.append("## Detailed Chunk Content")
    report.append("")
    report.append("*Note: This section shows the full content of each chunk.*")
    report.append("")
    
    # Standard chunks
    report.append("### Standard Chunking Content")
    report.append("")
    for i, chunk in enumerate(std_chunks):
        report.append(f"<details><summary>Chunk {i+1} ({chunk['size']} chars)</summary>")
        report.append("")
        report.append("```markdown")
        report.append(chunk['text'])
        report.append("```")
        report.append("")
        report.append("</details>")
        report.append("")
    
    # Semantic chunks
    report.append("### Semantic Chunking Content")
    report.append("")
    for i, chunk in enumerate(sem_chunks):
        report.append(f"<details><summary>Chunk {i+1} ({chunk['size']} chars)</summary>")
        report.append("")
        report.append("```markdown")
        report.append(chunk['text'])
        report.append("```")
        report.append("")
        report.append("</details>")
        report.append("")
    
    # Add conclusion
    report.append("## Observations")
    report.append("")
    report.append("### Key Differences")
    report.append("")
    report.append("- Standard chunking divides text based solely on character count with basic sentence boundary detection")
    report.append("- Semantic chunking attempts to keep related content together based on semantic meaning")
    report.append("- Semantic chunking is more likely to keep sections and subsections intact")
    report.append("- Standard chunking may split content mid-section or mid-topic")
    report.append("")
    
    return "\n".join(report)

def main():
    # Check if files exist
    std_file = "standard_chunks.json"
    sem_file = "semantic_chunks.json"
    
    if not os.path.exists(std_file) or not os.path.exists(sem_file):
        print("Error: Chunk files not found. Run test_semantic_chunking.py first.")
        sys.exit(1)
    
    # Load chunks
    std_chunks = load_chunks(std_file)
    sem_chunks = load_chunks(sem_file)
    
    # Generate report
    report = generate_report(std_chunks, sem_chunks)
    
    # Save report to file
    output_file = "chunk_comparison_report.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report generated successfully: {output_file}")

if __name__ == "__main__":
    main() 