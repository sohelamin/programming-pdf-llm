from pathlib import Path

# Directory configuration
BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "pdfs"
PERSIST_DIRECTORY = BASE_DIR / "db"

# Model configuration
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "llama3.2"

# Processing configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def get_pdf_files():
    """Get all PDF files from the pdfs directory"""
    if not PDF_DIR.exists():
        PDF_DIR.mkdir()
        print(f"Created PDF directory at {PDF_DIR}")
        return []
    
    return list(PDF_DIR.glob("*.pdf"))
