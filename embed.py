from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config import *
import os
import shutil

def process_documents():
    """Load and process all PDF documents in the pdfs directory"""
    pdf_files = get_pdf_files()
    
    if not pdf_files:
        print("No PDF files found in the pdfs directory")
        return None
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Load PDF documents
    documents = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents.extend(loader.load())
            print(f"Processed {pdf_path.name}")
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
    print(f"Loaded {len(documents)} documents")
    
    if not documents:
        print("No valid documents found")
        return None
    
    # Split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks")

    # Clear existing vector database
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
        print("Cleared existing vector database")

    print("Creating embeddings and vector store...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIRECTORY)
    )
    print(f"Created vector store with {len(texts)} chunks")
    return vectorstore

if __name__ == "__main__":
    process_documents()
