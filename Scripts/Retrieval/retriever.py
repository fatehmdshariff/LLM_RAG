from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from ..config import VECTOR_STORE_PATH, TOP_K
from ..Embedding.store import load_vector_store

class RAGRetriever:
    """
    Simple RAG retriever: loads the FAISS index, retrieves top-k docs,
    and formats a prompt for your Gemini model.
    """
    def __init__(self):
        # Load (and cache) the vector store
        self.store: FAISS = load_vector_store(VECTOR_STORE_PATH)

    def retrieve(self, question: str, k: int = TOP_K) -> List[Document]:
        """
        Return the top-k most relevant Document chunks for the question.
        """
        return self.store.similarity_search(question, k=k)

def rag_query(question: str, k: int = TOP_K) -> str:
    """
    Given a user question, retrieve context and call your Gemini chat model.
    """
    from google import genai
    import os
    from dotenv import load_dotenv
    from ..config import GEMINI_API_KEY, GEMINI_MODEL

    # Load API key
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Retrieve context
    retriever = RAGRetriever()
    docs = retriever.retrieve(question, k=k)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Build prompt
    prompt = (
        f"Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    # Call Gemini
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    return resp.text


# Quick test 
if __name__ == "__main__":
    q = "What is the lease term?"
    print("RAG answer:\n", rag_query(q))
