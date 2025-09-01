#!/usr/bin/env python3
"""
Run Claude-based PDF summarizer on pdfs directory
"""

import os
import sys
from pathlib import Path
from claude_pdf_summarizer import ClaudePDFProcessor

def main():
    # Import here to get the improved API key handling
    from claude_pdf_summarizer import get_api_key
    
    # Check for API key using the improved function
    api_key = get_api_key()
    if not api_key:
        print("ERROR: Please set your Anthropic API key")
        print("\nOptions:")
        print("1. Virtual environment variable (recommended):")
        print(r'   Activate venv: .\.venv\Scripts\Activate.ps1')
        print('   Set key: $env:ANTHROPIC_API_KEY = "your-api-key-here"')
        print("\n2. Create .env file in this directory:")
        print('   ANTHROPIC_API_KEY=your_key_here')
        print("\n3. Command line:")
        print('   python claude_pdf_summarizer.py pdfs --api-key YOUR_KEY')
        return 1
    
    # Initialize processor
    print("Initializing Claude PDF Processor...")
    processor = ClaudePDFProcessor(
        api_key=api_key,
        base_output_dir="claude_summaries"
    )
    
    # Process pdfs directory
    books_dir = Path("pdfs")
    if not books_dir.exists():
        print(f"ERROR: {books_dir} directory not found")
        return 1
    
    print(f"Processing PDFs in {books_dir}...")
    print("=" * 60)
    
    try:
        results = processor.process_directory(str(books_dir))
        
        if results:
            print("\nProcessing complete!")
            print(f"Check the 'claude_summaries' directory for results")
            print("\nEach PDF has its own folder containing:")
            print("  - extracted_text.txt (full text from PDF)")
            print("  - chunks/ (text split into processable chunks)")
            print("  - summaries/ (Claude summaries for each chunk)")
            print("  - all_chunk_summaries.txt (combined chunk summaries)")
            print("  - FINAL_SUMMARY.txt (meta-summary of entire document)")
        else:
            print("\nNo PDFs were processed")
            
    except Exception as e:
        print(f"\nError occurred: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 