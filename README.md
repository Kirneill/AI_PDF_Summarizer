# Claude PDF Summarizer

An intelligent PDF summarization tool that uses Claude 3.5 Haiku AI to create detailed, technical summaries of PDF documents through intelligent chunking and progressive summarization.

## Features

- **Multi-method PDF text extraction** (PyPDF2, pdfplumber, pdfminer)
- **Intelligent text chunking** (3000 words per chunk)
- **Detailed AI summarization** using Claude 3.5 Haiku
- **Robust error handling** with exponential backoff
- **Progress tracking** and resume capability
- **Organized file structure** with comprehensive logging
- **Rate limit management** for API efficiency

## Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd AI_PDF_Summarizer
```

### 2. Create Virtual Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure API Key

**Option A: Environment Variable (Recommended)**
```powershell
# While in activated virtual environment
$env:ANTHROPIC_API_KEY = "your-anthropic-api-key-here"
```

**Option B: .env File**
```bash
# Copy example file
copy env.example .env

# Edit .env and add your API key
# ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

### 5. Run Summarizer

```powershell
# Process all PDFs in a directory
python claude_pdf_summarizer.py "path/to/pdfs"

# Or use the simple runner
python run_claude_summarizer.py
```

## Usage

### Command Line Options

```bash
# Summarize single PDF
python claude_pdf_summarizer.py "document.pdf"

# Summarize directory with custom output
python claude_pdf_summarizer.py "pdfs/" --output-dir "my_summaries"

# Skip re-extraction if text already exists
python claude_pdf_summarizer.py "pdfs/" --skip-extraction

# Verbose logging
python claude_pdf_summarizer.py "pdfs/" --verbose
```

### Output Structure

```
claude_summaries/
├── logs/                           # Processing logs
├── progress.json                   # Resume tracking
└── DocumentName_hash/              # Per-document folder
    ├── extracted_text.txt          # Full extracted text
    ├── chunks/                     # 3000-word text chunks
    │   ├── chunk_001.txt
    │   └── ...
    ├── summaries/                  # Claude summaries
    │   ├── chunk_001_summary.txt
    │   └── ...
    ├── all_chunk_summaries.txt     # Combined summaries
    └── FINAL_SUMMARY.txt           # Meta-summary
```

## Key Features

### Intelligent Chunking
- **3000 words per chunk** for optimal Claude processing
- **Preserves sentence boundaries** 
- **~4000 tokens per chunk** (fits well within rate limits)
- **Automatic large paragraph splitting**

### Detailed Summarization
- **Preserves technical information** (measurements, dates, formulas)
- **Maintains educational value** for study and reference
- **Includes code examples** and technical implementations
- **4000-token summaries** for comprehensive detail

### Robust Error Handling
- **Exponential backoff** for rate limits (60s → 2min → 4min)
- **Network error recovery** with automatic retries
- **API validation** and response checking
- **Progress saving** after each successful chunk
- **Resume capability** from any interruption point

### Cost Efficiency
- **3000-word chunks** = ~$0.03 per chunk
- **Estimated**: ~$1.50 per 300-page technical book
- **Rate limit optimized**: 4 chunks per minute
- **No wasted compute** - skips completed work

## API Costs

**Claude 3.5 Haiku Pricing:**
- Input: $0.25 per million tokens
- Output: $1.25 per million tokens

**Typical Book (300 pages):**
- ~42 chunks of 3000 words each
- ~4000 tokens per chunk input
- ~500 tokens per chunk output
- **Total cost: ~$1.50 per book**

## Security

- ✅ **API keys never committed** (excluded in .gitignore)
- ✅ **Virtual environment isolated**
- ✅ **Environment variable support**
- ✅ **Secure .env file handling**

## Troubleshooting

**Rate Limits:**
- Script includes 15-second delays between chunks
- Automatic exponential backoff for 429 errors
- Can resume from any interruption point

**Memory Issues:**
- Processes one PDF at a time
- 3000-word chunks prevent memory overload
- Automatic cleanup of temporary data

**API Errors:**
- Comprehensive error handling for all API issues
- Clear error messages for authentication problems
- Automatic retry logic with smart backoff

## Development

### Project Structure
```
├── claude_pdf_summarizer.py       # Main summarizer script
├── run_claude_summarizer.py       # Simple runner
├── requirements.txt               # Python dependencies
├── CLAUDE_USAGE.txt              # Detailed usage guide
├── env.example                   # Environment template
└── README.md                     # This file
```

### Contributing
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all existing tests pass
5. Submit pull request

## License

MIT License - See LICENSE file for details 