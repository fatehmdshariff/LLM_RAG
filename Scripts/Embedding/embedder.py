from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
import os

from ..config import GEMINI_API_KEY, GEMINI_EMBED_MODEL

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_gemini_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Embed texts using Gemini via google.generativeai.embed_content,
    handling various response key names.
    """
    resp = genai.embed_content(
        model=GEMINI_EMBED_MODEL,
        content=texts
    )
    # The SDK may return under 'embeddings', 'data', or 'embedding'
    if "embedding" in resp:
        return resp["embedding"]
    if "embeddings" in resp:
        return [e["embedding"] for e in resp["embeddings"]]
    if "data" in resp:
        return [e["embedding"] for e in resp["data"]]
    raise ValueError(f"No embeddings found in response: {resp}")

if __name__ == "__main__":
    print("Using embed model:", GEMINI_EMBED_MODEL)
    sample = ["Hello world", "Test sentence for embeddings"]
    embs = get_gemini_embeddings(sample)
    for i, v in enumerate(embs):
        print(f" Text {i}: {len(v)} dims")
