
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Gemini API 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")

# Chunking parameters 
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# Vector store 
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "data/processed/faiss_index")

# Retrieval 
TOP_K = int(os.getenv("TOP_K", 3))

# Logging & Misc 
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
