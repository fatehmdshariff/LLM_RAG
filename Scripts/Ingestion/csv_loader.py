from typing import List
import pandas as pd
from langchain.schema import Document

def load_csv(path: str) -> List[Document]:
    """
    Load a CSV from the given file path and return a list of LangChain Documents,
    one per row, with the row data concatenated into text.
    """
    df = pd.read_csv(path)
    docs: List[Document] = []
    for idx, row in df.iterrows():
        # Convert row to a simple text representation
        content = " | ".join(f"{col}: {row[col]}" for col in df.columns)
        docs.append(Document(page_content=content, metadata={"row": idx}))
    return docs


# Simple test 
if __name__ == "__main__":
    path =  r"D:\Projects_25\Gem_llm_rag\Data\height-weight.csv"
    print(f"Loading CSV from: {path}")
    docs = load_csv(path)
    print(f"✅ Loaded {len(docs)} rows as documents.")
    if docs:
        print("\n--- Preview of first document ---")
        print(docs[0].page_content)
