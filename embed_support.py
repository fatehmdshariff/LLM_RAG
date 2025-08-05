# embed_support.py

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("Available embedding models:")
for model in genai.list_models():
    name = getattr(model, "name", None)
    if name and "embed" in name.lower():
        print(" -", name)
