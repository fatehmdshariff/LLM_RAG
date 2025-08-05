import streamlit as st
from dotenv import load_dotenv
import os
import sys

# Ensure project root on sys.path
current_dir = os.path.dirname(__file__)
scripts_dir = os.path.abspath(os.path.join(current_dir, ".."))
project_root = os.path.abspath(os.path.join(scripts_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Scripts.Retrieval.retriever import rag_query
from Scripts.Embedding.store import build_vector_store, save_vector_store
from Scripts.Ingestion.pdf_loader import load_pdf
from Scripts.Ingestion.csv_loader import load_csv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load .env
load_dotenv(os.path.join(project_root, ".env"))

st.set_page_config(page_title="GenAI + RAG Demo", layout="wide")
st.title("📄 GenAI + RAG Interactive Demo")

st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload a document", type=["pdf", "csv", "txt"])
top_k = st.sidebar.slider("Number of context chunks (k)", 1, 10, 3)
chunk_size = st.sidebar.number_input("Chunk size (chars)", 500, 2000, 1000, step=100)
chunk_overlap = st.sidebar.number_input("Chunk overlap (chars)", 50, 500, 200, step=50)

if uploaded_file:
    # save raw file
    raw_dir = os.path.join(project_root, "Data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    file_path = os.path.join(raw_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Saved {uploaded_file.name}")

    # ingest
    ext = uploaded_file.name.lower().split(".")[-1]
    if ext == "pdf":
        docs = load_pdf(file_path)
    elif ext == "csv":
        docs = load_csv(file_path)
    elif ext == "txt":
        text = uploaded_file.getvalue().decode("utf-8")
        docs = [Document(page_content=text)]
    else:
        st.error("Unsupported file type.")
        st.stop()

    st.info(f"Ingested {len(docs)} raw document(s)")

    # chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap)
    )
    docs = splitter.split_documents(docs)
    st.info(f"Split into {len(docs)} chunks (size={chunk_size}, overlap={chunk_overlap})")

    # build & save index
    st.info("Building FAISS index…")
    store = build_vector_store(docs)
    save_vector_store(store)
    st.success("Index built and saved! You can now ask questions below.")

    # query
    question = st.text_input("Ask a question about your document:")
    if question:
        with st.spinner("Retrieving answer…"):
            answer = rag_query(question, k=top_k)
        st.subheader("Answer")
        st.write(answer)
