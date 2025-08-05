# GenAI + RAG Demo

An end-to-end proof-of-concept showing how to build a Retrieval-Augmented Generation (RAG) pipeline with Google Gemini (via `google-generativeai`), FAISS for vector indexing, and Streamlit for a lightweight interactive UI. 

---


## 📁 Repository Structure

GEM_LLM_RAG/
├── Data/
│   ├── raw/                      # Uploaded PDFs/CSVs/TXTs at runtime
│   └── processed/                # Persisted FAISS index (VECTOR_STORE_PATH)
├── gen_llm_rag_venv/             # Python virtual environment (ignored by Git)
├── Scripts/
│   ├── Ingestion/                
│   │   ├── __init__.py           
│   │   ├── pdf_loader.py         # Load PDF → LangChain Documents
│   │   └── csv_loader.py         # Load CSV → LangChain Documents
│   ├── Embedding/                
│   │   ├── __init__.py           
│   │   ├── embedder.py           # Wraps google-generativeai embed_content
│   │   └── store.py              # Build / save / load FAISS index
│   ├── Retrieval/                
│   │   ├── __init__.py           
│   │   └── retriever.py          # RAGRetriever + rag_query() helper
│   └── UI/                       
│       ├── __init__.py           
│       └── app.py                # Streamlit app (upload, settings, Q&A)
├── .env                          # API keys & configuration overrides
├── .gitignore                    # Ignore venv, caches, data artifacts, etc.
├── requirements.txt              # pip dependencies
└── README.md                     # This file




---

## 🎯 Project Goal

I wanted a **simple**, **modular**, and **self-contained** RAG demo that anyone can:

1. **Upload** their own document (PDF, CSV, TXT)  
2. **Configure** chunk size, overlap, and result count  
3. **Ask** natural-language questions  
4. **Get** answers powered by Google Gemini, grounded in the source material  

By splitting responsibilities into clear modules—ingestion, chunking, embedding, indexing, retrieval, and UI—this project becomes easy to extend (swap in a new vector store, tweak the chunker, use a different LLM, etc.).  

---

## 🔍 Thought Process & Pipeline

1. **Ingestion**  
   - **Initial plan**: hardcode PDF only.  
   - **Final**: support PDF, CSV, and plain text via separate loader modules.

2. **Chunking**  
   - Documents can be larger than the model’s input limit.  
   - Use LangChain’s `RecursiveCharacterTextSplitter` with configurable `chunk_size` & `chunk_overlap` so each chunk fits comfortably and provides context windows.

3. **Embeddings**  
   - **Challenge**: conflicting versions between `langchain-google-genai` and the `google-generativeai` SDK, plus Application Default Credentials (ADC) requirements.  
   - **Solution**: call `genai.embed_content(...)` directly in our own `embedder.py`, handling varying response formats (`"embedding"` vs. `"embeddings"` vs. `"data"`).

4. **Vector Indexing**  
   - Wrap our embedder in a minimal `GeminiEmbeddings` class implementing both `embed_documents()` and `__call__()` so FAISS can build and query seamlessly.  
   - Persist with `allow_dangerous_deserialization=True` (only for our own trusted data).

5. **Retrieval & RAG**  
   - A `RAGRetriever` class loads the index once and returns top-k LangChain `Document` chunks.  
   - The `rag_query()` helper concatenates those chunks into a “context” prompt and calls Gemini’s chat model to generate a grounded answer.

6. **Streamlit UI**  
   - Single-page app for upload, parameter tuning, and query/answer.  
   - Automatically rebuilds the FAISS index on each upload so you can iterate quickly.

---

## 🛠️ Installation & Usage

1. **Clone** and **cd** into the repo  
2. **Create** & **activate** a virtual environment:
   ```bash
   python -m venv gen_llm_rag_venv
   source gen_llm_rag_venv/bin/activate  # or on Windows: .\gen_llm_rag_venv\Scripts\activat
3. **install dependecies** in the virtual environment:
    ```bash
    pip install -r requirements.txt
4. **Configure** your **.env**:
    GEMINI_API_KEY=ya29.YOUR_KEY
    GEMINI_MODEL=gemini-2.5-flash
    GEMINI_EMBED_MODEL=models/text-embedding-004
    VECTOR_STORE_PATH=Data/processed/faiss_index
    CHUNK_SIZE=1000
    CHUNK_OVERLAP=200
    TOP_K=3
5. **Run** The **Streamlit app**
    ```bash
    streamlit run Scripts/UI/app.py



✅ What I Learned & Next Steps
Modular design: clear separation improves maintainability and swap-ability.

SDK quirks: wrestling with Google’s GenAI SDK led me to build a thin wrapper to handle authentication and response shapes.

Vector stores: integrating FAISS via LangChain taught me about safe deserialization and embedding interfaces.

Deployment readiness: the Streamlit app is ready for Dockerization or cloud hosting.

Potential Extensions

Swap FAISS for Pinecone, Weaviate, Qdrant, etc.

Add user authentication or API-key quotas.

Implement caching or chunk-level relevance feedback.

Deploy to Heroku, GCP App Engine, or AWS ECS.