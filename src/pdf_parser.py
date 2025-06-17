import os
from dotenv import load_dotenv
from llama_cloud_services import LlamaParse

def parse_pdf_to_markdown(pdf_path, output_path=None):
    """
    Parse a PDF file to markdown format using LlamaParse.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_path (str, optional): Path to save the markdown output. If None, prints to console.
    """
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment variable
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise ValueError("Please set LLAMA_CLOUD_API_KEY environment variable")
    
    # Initialize parser
    parser = LlamaParse(
        api_key=api_key,
        num_workers=1,
        verbose=True,
        language="en"
    )
    
    # Parse the PDF
    result = parser.parse(pdf_path)
    
    # Get markdown documents
    markdown_documents = result.get_markdown_documents(split_by_page=True)
    
    # Combine all pages into one markdown document
    full_markdown = "\n\n".join([doc.text for doc in markdown_documents])
    
    # Save to file if output_path is provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_markdown)
        print(f"Markdown saved to: {output_path}")
    else:
        print(full_markdown)
    
    return full_markdown

if __name__ == "__main__":
    import argparse
    
    # --- Configure input and output files directly here ---
    # Set your default PDF input file path
    DEFAULT_PDF_INPUT_PATH = "./sample.pdf" 
    # Set your default Markdown output file path (set to None if you want to print to console by default)
    DEFAULT_MARKDOWN_OUTPUT_PATH = "./output.md"
    # ------------------------------------------------------

    parser = argparse.ArgumentParser(description="Convert PDF to Markdown using LlamaParse")
    parser.add_argument(
        "pdf_path", 
        nargs='?', # Makes the argument optional
        default=DEFAULT_PDF_INPUT_PATH, 
        help=f"Path to the PDF file (default: {DEFAULT_PDF_INPUT_PATH})"
    )
    parser.add_argument(
        "--output", "-o", 
        default=DEFAULT_MARKDOWN_OUTPUT_PATH, 
        help=f"Path to save the markdown output (optional, default: {DEFAULT_MARKDOWN_OUTPUT_PATH if DEFAULT_MARKDOWN_OUTPUT_PATH else 'print to console'})"
    )
    
    args = parser.parse_args()
    
    parse_pdf_to_markdown(args.pdf_path, args.output) 