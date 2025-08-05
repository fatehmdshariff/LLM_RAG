from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

def load_pdf(path: str) -> List[Document]:
    """
    Load a PDF from the given file path and return a list of LangChain Documents.
    """
    loader = PyPDFLoader(path)
    docs: List[Document] = loader.load()
    return docs



# Simple test 
if __name__ == "__main__":
    import sys
    path =  r"D:\Projects_25\Gem_llm_rag\Data\LA 2nd floor vibuthipura Akit Abhishek Lease Deed___ (1).pdf"
    
    print(f"Loading PDF from: {path}")
    docs = load_pdf(path)
    
    print(f"✅ Loaded {len(docs)} document chunks (pages).")
    # Show a preview of the first chunk
    if docs:
        print("\n--- Preview of first document chunk ---")
        print(docs[0].page_content[:500].replace("\n", " "), "…")
