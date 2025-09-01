#!/usr/bin/env python3
"""
Claude-based PDF Summarizer
Uses Claude 3.5 Haiku for intelligent text summarization with chunking
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import hashlib

# PDF processing
import PyPDF2
import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract

# Anthropic Claude API
from anthropic import Anthropic

# Text processing
import tiktoken  # For accurate token counting

# Progress bar
from tqdm import tqdm

# Environment configuration
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Setup logging
def setup_logging(log_dir: Path):
    """Setup logging configuration"""
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"summarizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class PDFTextExtractor:
    """Handles robust PDF text extraction"""
    
    @staticmethod
    def extract_with_pypdf2(pdf_path: str) -> str:
        """Extract text using PyPDF2"""
        text = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num in tqdm(range(total_pages), desc="Extracting with PyPDF2", leave=False):
                    page = pdf_reader.pages[page_num]
                    text.append(page.extract_text())
            return "\n".join(text)
        except Exception as e:
            logging.warning(f"PyPDF2 extraction failed: {e}")
            return ""
    
    @staticmethod
    def extract_with_pdfplumber(pdf_path: str) -> str:
        """Extract text using pdfplumber"""
        text = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in tqdm(pdf.pages, desc="Extracting with pdfplumber", leave=False):
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
            return "\n".join(text)
        except Exception as e:
            logging.warning(f"pdfplumber extraction failed: {e}")
            return ""
    
    @staticmethod
    def extract_with_pdfminer(pdf_path: str) -> str:
        """Extract text using pdfminer"""
        try:
            return pdfminer_extract(pdf_path)
        except Exception as e:
            logging.warning(f"pdfminer extraction failed: {e}")
            return ""
    
    @classmethod
    def extract_text(cls, pdf_path: str, method_preference: List[str] = None) -> str:
        """
        Extract text from PDF using multiple methods
        """
        if method_preference is None:
            method_preference = ["pypdf2", "pdfplumber", "pdfminer"]
        
        methods = {
            "pypdf2": cls.extract_with_pypdf2,
            "pdfplumber": cls.extract_with_pdfplumber,
            "pdfminer": cls.extract_with_pdfminer
        }
        
        logging.info(f"Starting text extraction from: {pdf_path}")
        
        for method_name in method_preference:
            if method_name in methods:
                logging.info(f"Trying {method_name}...")
                text = methods[method_name](pdf_path)
                
                if text and len(text.strip()) > 100:
                    logging.info(f"Successfully extracted with {method_name}")
                    logging.info(f"Extracted {len(text)} characters")
                    return text
        
        logging.error(f"Failed to extract meaningful text from {pdf_path}")
        return ""


class ClaudeChunkSummarizer:
    """Handles text chunking and summarization using Claude API"""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-haiku-20241022"):
        """
        Initialize Claude client
        
        Args:
            api_key: Anthropic API key
            model: Claude model to use
        """
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens_per_chunk = 100000  # Claude 3.5 Haiku context window
        self.max_summary_tokens = 4000  # Allow longer, more detailed summaries
        
        # Initialize tokenizer for counting
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")  # Use as approximation
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count approximate tokens in text"""
        return len(self.encoding.encode(text))
    
    def chunk_text(self, text: str, chunk_words: int = 3000) -> List[str]:
        """
        Split text into chunks based on word count with strict word limits
        
        Args:
            text: Full text to chunk
            chunk_words: Target number of words per chunk
        """
        # Split into sentences for precise control
        sentences = text.replace('\n\n', ' [PARAGRAPH_BREAK] ').split('. ')
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for i, sentence in enumerate(sentences):
            # Restore sentence ending
            if i < len(sentences) - 1:
                sentence += '.'
            
            # Restore paragraph breaks
            sentence = sentence.replace(' [PARAGRAPH_BREAK] ', '\n\n')
            
            sentence_word_count = len(sentence.split())
            
            # If adding this sentence would exceed limit, finalize current chunk
            if current_word_count + sentence_word_count > chunk_words and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                logging.info(f"Created chunk {len(chunks)}: {current_word_count} words, ~{self.count_tokens(chunk_text)} tokens")
                
                current_chunk = [sentence]
                current_word_count = sentence_word_count
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
            
            # If single sentence is too large, split it further
            if sentence_word_count > chunk_words:
                words = sentence.split()
                for j in range(0, len(words), chunk_words):
                    word_chunk = ' '.join(words[j:j + chunk_words])
                    chunks.append(word_chunk)
                    logging.info(f"Created chunk {len(chunks)}: {len(word_chunk.split())} words, ~{self.count_tokens(word_chunk)} tokens")
                current_chunk = []
                current_word_count = 0
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            logging.info(f"Created chunk {len(chunks)}: {current_word_count} words, ~{self.count_tokens(chunk_text)} tokens")
        
        total_words = sum(len(chunk.split()) for chunk in chunks)
        avg_words = total_words // len(chunks) if chunks else 0
        avg_tokens = sum(self.count_tokens(chunk) for chunk in chunks) / len(chunks) if chunks else 0
        
        logging.info(f"Created {len(chunks)} chunks from {total_words:,} words")
        logging.info(f"Average chunk size: {avg_words:,} words (~{avg_tokens:.0f} tokens)")
        logging.info(f"Estimated processing time: {len(chunks) * 15 // 60} minutes {len(chunks) * 15 % 60} seconds")
        
        return chunks
    
    def summarize_chunk(self, chunk: str, chunk_index: int, total_chunks: int, 
                       document_context: str = "") -> str:
        """
        Summarize a single chunk using Claude with robust error handling
        
        Args:
            chunk: Text chunk to summarize
            chunk_index: Current chunk number
            total_chunks: Total number of chunks
            document_context: Brief context about the document
        """
        prompt = f"""You are creating a detailed summary of part {chunk_index + 1} of {total_chunks} from a technical document.
{f'Document context: {document_context}' if document_context else ''}

IMPORTANT: This should be a DETAILED summary that preserves information, NOT a condensed overview. 

Please provide a comprehensive, detailed summary that includes:

1. **All Key Concepts**: Explain each important concept thoroughly with context
2. **Technical Details**: Preserve specific technical information, methods, algorithms, and examples
3. **Code Examples**: Include any code snippets, programming concepts, or technical implementations
4. **Explanations**: Maintain the author's explanations and reasoning
5. **Specific Information**: Include names, dates, specific technologies, methodologies mentioned
6. **Context and Relationships**: How concepts connect to each other and build upon previous ideas

DO NOT oversimplify or leave out important details. The goal is to create a detailed reference that someone could use to understand the material without reading the original.

Text to analyze and summarize in detail:
{chunk}

Provide a thorough, detailed summary that preserves the educational and technical value of the original text:"""
        
        # Retry logic with exponential backoff
        max_retries = 5
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_summary_tokens,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Validate response
                if not response.content or not response.content[0].text:
                    raise ValueError("Empty response from Claude API")
                
                summary = response.content[0].text.strip()
                
                # Validate summary quality
                if len(summary.split()) < 50:
                    logging.warning(f"Very short summary for chunk {chunk_index}, may indicate API issue")
                
                logging.info(f"Successfully summarized chunk {chunk_index + 1}: {len(summary.split())} words")
                return summary
                
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                
                # Handle specific error types
                if "rate_limit" in error_msg.lower():
                    wait_time = min(60 * (2 ** attempt), 300)  # Exponential backoff, max 5 minutes
                    logging.warning(f"Rate limit hit on chunk {chunk_index + 1}, attempt {attempt + 1}/{max_retries}")
                    logging.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                    
                elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                    wait_time = base_delay * (2 ** attempt)
                    logging.warning(f"Network issue on chunk {chunk_index + 1}, attempt {attempt + 1}/{max_retries}: {error_type}")
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                    
                elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
                    logging.error(f"Authentication error: {error_msg}")
                    return f"[Authentication Error - Check API Key]"
                    
                elif attempt == max_retries - 1:
                    # Last attempt failed
                    logging.error(f"Failed to summarize chunk {chunk_index + 1} after {max_retries} attempts: {error_type}: {error_msg}")
                    return f"[Failed to summarize chunk {chunk_index + 1} after {max_retries} attempts: {error_type}]"
                else:
                    # Generic retry
                    wait_time = base_delay * (2 ** attempt)
                    logging.warning(f"Error on chunk {chunk_index + 1}, attempt {attempt + 1}/{max_retries}: {error_type}")
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
        
        return f"[Error: Max retries exceeded for chunk {chunk_index + 1}]"
    
    def create_meta_summary(self, chunk_summaries: List[str], document_name: str) -> str:
        """
        Create a final summary from all chunk summaries
        
        Args:
            chunk_summaries: List of all chunk summaries
            document_name: Name of the document being summarized
        """
        combined_summaries = "\n\n---\n\n".join(
            [f"CHUNK {i+1} SUMMARY:\n{summary}" 
             for i, summary in enumerate(chunk_summaries)]
        )
        
        prompt = f"""You are creating a comprehensive final summary of "{document_name}" based on detailed summaries of its individual sections.

The document has been processed in {len(chunk_summaries)} detailed chunks. Your task is to create a thorough final summary that serves as a complete reference.

Please create a comprehensive, detailed final summary that:

1. **Document Structure**: Outline the main sections and how the document is organized
2. **Key Themes**: Identify and explain the central themes with supporting details
3. **Technical Content**: Preserve all important technical concepts, methodologies, and examples
4. **Progressive Learning**: Show how concepts build upon each other throughout the document
5. **Practical Applications**: Include specific examples, code snippets, or real-world applications mentioned
6. **Important Details**: Maintain specific information like names, dates, technologies, and methodologies
7. **Learning Value**: Ensure someone could use this summary as a study guide or reference

DO NOT oversimplify. This final summary should be detailed enough that someone could understand the document's full educational value without reading the original.

Detailed chunk summaries to synthesize:
{combined_summaries}

Create a thorough, comprehensive final summary that preserves the educational and technical depth of the original document:"""
        
        # Robust error handling for meta-summary
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=6000,  # Larger limit for comprehensive final summary
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                if not response.content or not response.content[0].text:
                    raise ValueError("Empty response from Claude API")
                
                final_summary = response.content[0].text.strip()
                
                if len(final_summary.split()) < 100:
                    logging.warning("Very short final summary, may indicate API issue")
                
                logging.info(f"Successfully created meta-summary: {len(final_summary.split())} words")
                return final_summary
                
            except Exception as e:
                error_msg = str(e)
                
                if "rate_limit" in error_msg.lower():
                    wait_time = 60 * (2 ** attempt)
                    logging.warning(f"Rate limit for meta-summary, attempt {attempt + 1}/{max_retries}")
                    logging.info(f"Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                
                elif attempt == max_retries - 1:
                    logging.error(f"Failed to create meta-summary after {max_retries} attempts: {e}")
                    return f"Error creating final summary after {max_retries} attempts: {type(e).__name__}"
                else:
                    wait_time = 10 * (attempt + 1)
                    logging.warning(f"Meta-summary error, attempt {attempt + 1}/{max_retries}: {type(e).__name__}")
                    time.sleep(wait_time)
                    continue
        
        return "Error: Failed to create final summary"


class ClaudePDFProcessor:
    """Main processor for PDF summarization workflow"""
    
    def __init__(self, api_key: str, base_output_dir: str = "claude_summaries"):
        """
        Initialize processor
        
        Args:
            api_key: Anthropic API key
            base_output_dir: Base directory for all outputs
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(self.base_output_dir / "logs")
        
        # Initialize components
        self.extractor = PDFTextExtractor()
        self.summarizer = ClaudeChunkSummarizer(api_key)
        
        # Progress tracking
        self.progress_file = self.base_output_dir / "progress.json"
        self.progress = self.load_progress()
    
    def load_progress(self) -> Dict:
        """Load progress from file"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_progress(self):
        """Save progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def get_document_hash(self, pdf_path: str) -> str:
        """Get hash of document for tracking"""
        return hashlib.md5(pdf_path.encode()).hexdigest()[:8]
    
    def create_document_folder(self, pdf_path: str) -> Path:
        """Create organized folder structure for document"""
        pdf_name = Path(pdf_path).stem
        doc_hash = self.get_document_hash(pdf_path)
        
        # Create folder structure
        doc_folder = self.base_output_dir / f"{pdf_name}_{doc_hash}"
        doc_folder.mkdir(exist_ok=True)
        
        # Create subfolders
        (doc_folder / "chunks").mkdir(exist_ok=True)
        (doc_folder / "summaries").mkdir(exist_ok=True)
        
        return doc_folder
    
    def process_pdf(self, pdf_path: str, skip_extraction: bool = False) -> bool:
        """
        Process a single PDF through the entire pipeline
        
        Args:
            pdf_path: Path to PDF file
            skip_extraction: Skip extraction if text file already exists
        
        Returns:
            Success status
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logging.error(f"PDF not found: {pdf_path}")
            return False
        
        doc_hash = self.get_document_hash(str(pdf_path))
        
        # Check if already processed
        if doc_hash in self.progress and self.progress[doc_hash].get("completed", False):
            logging.info(f"Already processed: {pdf_path.name}")
            return True
        
        # Initialize progress for this document
        self.progress[doc_hash] = {
            "name": pdf_path.name,
            "path": str(pdf_path),
            "started": datetime.now().isoformat(),
            "completed": False,
            "stages": {}
        }
        self.save_progress()
        
        # Create folder structure
        doc_folder = self.create_document_folder(str(pdf_path))
        logging.info(f"Processing {pdf_path.name} -> {doc_folder}")
        
        try:
            # Stage 1: Extract text
            text_file = doc_folder / "extracted_text.txt"
            
            if not skip_extraction or not text_file.exists():
                logging.info("Stage 1: Extracting text from PDF...")
                text = self.extractor.extract_text(str(pdf_path))
                
                if not text:
                    logging.error("Failed to extract text")
                    self.progress[doc_hash]["stages"]["extraction"] = "failed"
                    self.save_progress()
                    return False
                
                # Save extracted text
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                logging.info(f"Saved extracted text ({len(text)} chars) to {text_file}")
                self.progress[doc_hash]["stages"]["extraction"] = "completed"
                self.save_progress()
            else:
                logging.info("Loading existing extracted text...")
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            # Stage 2: Chunk text
            logging.info("Stage 2: Chunking text...")
            chunks = self.summarizer.chunk_text(text)
            
            # Save chunks
            chunks_dir = doc_folder / "chunks"
            for i, chunk in enumerate(chunks):
                chunk_file = chunks_dir / f"chunk_{i+1:03d}.txt"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    f.write(chunk)
            
            logging.info(f"Created {len(chunks)} chunks")
            self.progress[doc_hash]["stages"]["chunking"] = f"completed ({len(chunks)} chunks)"
            self.save_progress()
            
            # Stage 3: Summarize each chunk
            logging.info("Stage 3: Summarizing chunks with Claude...")
            chunk_summaries = []
            summaries_dir = doc_folder / "summaries"
            
            for i, chunk in enumerate(tqdm(chunks, desc="Summarizing chunks")):
                # Check if already summarized
                summary_file = summaries_dir / f"chunk_{i+1:03d}_summary.txt"
                
                if summary_file.exists():
                    logging.info(f"Loading existing summary for chunk {i+1}")
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary = f.read()
                else:
                    logging.info(f"\nSummarizing chunk {i+1}/{len(chunks)}...")
                    summary = self.summarizer.summarize_chunk(
                        chunk, i, len(chunks), 
                        document_context=pdf_path.name
                    )
                    
                    # Save chunk summary
                    with open(summary_file, 'w', encoding='utf-8') as f:
                        f.write(summary)
                    
                    # Print to terminal as requested
                    print(f"\n{'='*60}")
                    print(f"CHUNK {i+1}/{len(chunks)} SUMMARY:")
                    print(f"{'='*60}")
                    print(summary)
                    print(f"{'='*60}\n")
                    
                    # Rate limiting to avoid API throttling
                    if i < len(chunks) - 1:
                        time.sleep(15)  # 15 second delay - allows ~4 chunks per minute, safer for detailed summaries
                
                chunk_summaries.append(summary)
            
            self.progress[doc_hash]["stages"]["chunk_summaries"] = "completed"
            self.save_progress()
            
            # Stage 4: Combine chunk summaries
            logging.info("Stage 4: Creating combined chunk summaries file...")
            combined_file = doc_folder / "all_chunk_summaries.txt"
            with open(combined_file, 'w', encoding='utf-8') as f:
                f.write(f"CHUNK SUMMARIES FOR: {pdf_path.name}\n")
                f.write("=" * 80 + "\n\n")
                
                for i, summary in enumerate(chunk_summaries):
                    f.write(f"CHUNK {i+1}/{len(chunks)}\n")
                    f.write("-" * 40 + "\n")
                    f.write(summary)
                    f.write("\n\n")
            
            logging.info(f"Saved combined summaries to {combined_file}")
            
            # Stage 5: Create meta-summary
            logging.info("Stage 5: Creating final meta-summary with Claude...")
            meta_summary = self.summarizer.create_meta_summary(chunk_summaries, pdf_path.name)
            
            # Save final summary
            final_summary_file = doc_folder / "FINAL_SUMMARY.txt"
            with open(final_summary_file, 'w', encoding='utf-8') as f:
                f.write(f"FINAL SUMMARY\n")
                f.write(f"Document: {pdf_path.name}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                f.write(meta_summary)
                f.write("\n\n")
                f.write("=" * 80 + "\n")
                f.write(f"This summary was generated from {len(chunks)} chunks\n")
                f.write(f"Original document: {pdf_path.name}\n")
            
            logging.info(f"Saved final summary to {final_summary_file}")
            
            # Print final summary
            print(f"\n{'#'*60}")
            print(f"FINAL SUMMARY FOR: {pdf_path.name}")
            print(f"{'#'*60}")
            print(meta_summary)
            print(f"{'#'*60}\n")
            
            # Mark as completed
            self.progress[doc_hash]["completed"] = True
            self.progress[doc_hash]["completed_at"] = datetime.now().isoformat()
            self.save_progress()
            
            logging.info(f"Successfully processed {pdf_path.name}")
            return True
            
        except Exception as e:
            logging.error(f"Error processing {pdf_path}: {e}")
            self.progress[doc_hash]["error"] = str(e)
            self.save_progress()
            return False
    
    def process_directory(self, directory: str) -> Dict[str, bool]:
        """
        Process all PDFs in a directory
        
        Args:
            directory: Path to directory containing PDFs
        
        Returns:
            Dictionary of PDF names and success status
        """
        pdf_files = list(Path(directory).glob("**/*.pdf"))
        
        if not pdf_files:
            logging.warning(f"No PDF files found in {directory}")
            return {}
        
        logging.info(f"Found {len(pdf_files)} PDF files to process")
        
        results = {}
        for pdf_file in pdf_files:
            success = self.process_pdf(str(pdf_file))
            results[pdf_file.name] = success
        
        # Print summary
        successful = sum(1 for s in results.values() if s)
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"Successful: {successful}/{len(results)}")
        print(f"Results saved to: {self.base_output_dir}")
        print(f"{'='*60}\n")
        
        return results


def get_api_key() -> Optional[str]:
    """Get API key from environment variables or .env file"""
    # Try environment variable first
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key and DOTENV_AVAILABLE:
        # Try loading from .env file
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    return api_key


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Summarize PDFs using Claude AI")
    parser.add_argument("path", help="Path to PDF file or directory")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--output-dir", default="claude_summaries", 
                       help="Output directory for summaries")
    parser.add_argument("--skip-extraction", action="store_true",
                       help="Skip extraction if text file exists")
    
    args = parser.parse_args()
    
    # Get API key from various sources
    api_key = args.api_key or get_api_key()
    
    if not api_key:
        print("ERROR: No Anthropic API key found!")
        print("\nSet it one of these ways:")
        print("1. Environment variable: ANTHROPIC_API_KEY")
        print("2. Command line: --api-key YOUR_KEY")
        print("3. Create .env file with: ANTHROPIC_API_KEY=your_key_here")
        if not DOTENV_AVAILABLE:
            print("\nFor .env file support, install: pip install python-dotenv")
        return 1
    
    # Initialize processor
    processor = ClaudePDFProcessor(
        api_key=api_key,
        base_output_dir=args.output_dir
    )
    
    # Process path
    path = Path(args.path)
    if path.is_file() and path.suffix.lower() == '.pdf':
        processor.process_pdf(str(path), skip_extraction=args.skip_extraction)
    elif path.is_dir():
        processor.process_directory(str(path))
    else:
        print(f"Error: {path} is not a valid PDF file or directory")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 