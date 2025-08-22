# Programming PDF LLM QA

A local question-answering (QA) system for programming PDFs using LangChain, ChromaDB, and Ollama (or other LLMs). This project lets you embed your programming PDFs and ask questions about their content interactively.

## Features
- Loads and splits PDFs into chunks
- Embeds chunks using llm models
- Stores embeddings in a persistent Chroma vector database
- Interactive QA

## Requirements
- Python 3.9+
- See `requirements.txt` for Python dependencies
- Ollama (for local LLM/embedding, or modify for OpenAI/DeepSeek)

## Setup
1. **Clone the repository**
2. **Install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Add your PDFs**
   - Place your programming PDFs in the `pdfs/` directory.

4. **Configure models (optional)**
   - Edit `config.py` to change the LLM or embedding model names.

## Usage

### 1. Build the vector database
```bash
python embed.py
```
- This will process all PDFs in `pdfs/`, split them, embed them, and store the vectors in `db/`.

### 2. Start the QA system
```bash
python main.py
```
- Ask questions about your PDFs interactively.
- Type `exit` to quit.

## Customization
- To use OpenAI or DeepSeek instead of Ollama, update the embedding and LLM imports and config in `embed.py` and `main.py`.
- The prompt for QA is customizable in `main.py`.

## File Structure
- `pdfs/` — Place your PDF files here
- `db/` — Vector database (auto-generated)
- `embed.py` — Script to process and embed PDFs
- `main.py` — Interactive QA script
- `config.py` — Configuration (paths, model names, chunk sizes)
- `requirements.txt` — Python dependencies

## License
MIT
