import os
import sys
from typing import List
from dotenv import load_dotenv

# Community FAISS import
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from ..config import VECTOR_STORE_PATH
from .embedder import get_gemini_embeddings

load_dotenv()

class GeminiEmbeddings:
    """Wraps get_gemini_embeddings in the LangChain Embeddings interface."""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return get_gemini_embeddings(texts)

    def __call__(self, text: str) -> List[float]:
        # This makes the instance itself callable for query embedding
        return get_gemini_embeddings([text])[0]

def build_vector_store(docs: List[Document]) -> FAISS:
    embedder = GeminiEmbeddings()
    return FAISS.from_documents(docs, embedder)

def save_vector_store(store: FAISS, path: str = VECTOR_STORE_PATH) -> None:
    os.makedirs(path, exist_ok=True)
    store.save_local(path)

def load_vector_store(path: str = VECTOR_STORE_PATH) -> FAISS:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"No FAISS index directory at {path}")
    return FAISS.load_local(
        path,
        GeminiEmbeddings(),
        allow_dangerous_deserialization=True
    )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m Scripts.Embedding.store <path-to-pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    from ..Ingestion.pdf_loader import load_pdf

    print(f"Loading PDF: {pdf_path}")
    docs = load_pdf(pdf_path)

    print(f"Building FAISS store for {len(docs)} docs...")
    store = build_vector_store(docs)

    print("Saving FAISS store...")
    save_vector_store(store)

    print("Reloading FAISS store...")
    reloaded = load_vector_store()

    query = "What is the lease term?"
    print(f"\nTop-3 chunks for query: {query}")
    results = reloaded.similarity_search(query, k=3)
    for i, doc in enumerate(results):
        snippet = doc.page_content.replace("\n", " ")[:200]
        print(f"[{i+1}] {snippet}…")
