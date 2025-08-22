from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from config import *

def initialize_qa_system():
    """Initialize the QA system with existing vector store"""
    if not PERSIST_DIRECTORY.exists():
        print("Vector database not found. Please run python embed.py first.")
        return None
    
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = Chroma(
            persist_directory=str(PERSIST_DIRECTORY),
            embedding_function=embeddings
        )
        
        llm = OllamaLLM(model=LLM_MODEL, temperature=0)
        
        from langchain.prompts import PromptTemplate
        prompt_template = (
            "You are an expert programming assistant. "
            "Use the following context from programming PDFs to answer the user's question as accurately and simply as possible. "
            "If the answer is not in the context, say 'I don't know.'\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        qa_system = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        return qa_system
    except Exception as e:
        print(f"Error initializing QA system: {e}")
        return None

def interactive_query(qa_system):
    """Run interactive query loop"""
    print("\nReady to answer questions about your programming PDFs!")
    print(f"Loaded model: {LLM_MODEL}")
    print("Type 'exit' to quit.\n")
    
    while True:
        query = input("Your question: ").strip()
        if query.lower() in ['exit', 'quit']:
            break
        
        if not query:
            continue
        
        try:
            result = qa_system.invoke({"query": query})
            print("\nAnswer:", result["result"])
            
            # Show source documents if available
            if "source_documents" in result:
                print("\nSources:")
                for i, doc in enumerate(result["source_documents"], 1):
                    source = doc.metadata.get('source', 'unknown')
                    page = doc.metadata.get('page', 'N/A')
                    print(f"{i}. {Path(source).name} (page {page})")
            print()
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    qa_system = initialize_qa_system()
    if qa_system:
        interactive_query(qa_system)
