# src/rag/retriever.py
import os
import json
from typing import List, Dict
from rank_bm25 import BM25Okapi


def _load_docs(index_path: str) -> List[Dict]:
    """Load documents from the specified index path."""
    docs = []
    docs_jsonl = os.path.join(index_path, "docs.jsonl")
    with open(docs_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def _tokenize(text: str):
    """Tokenize the input text."""
    return [t.lower() for t in text.split()]


class Retriever:
    def __init__(self, index_path: str = "data/index", k: int = 5):
        """Initialize the Retriever with the index path and top-k value."""
        self.docs = _load_docs(index_path)
        bm25_json = os.path.join(index_path, "bm25.json")
        with open(bm25_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.bm25 = BM25Okapi(payload["corpus_tokenized"])
        self.k = k

        # Validate k
        if not isinstance(self.k, int) or self.k <= 0:
            raise ValueError(f"Invalid value for k: {self.k}. It must be a positive integer.")
        if self.k > len(self.docs):
            # Replace with logging if needed
            print(f"Warning: k ({self.k}) is greater than the number of documents ({len(self.docs)}). Using {len(self.docs)} instead.")
            self.k = len(self.docs)

    def search(self, query: str) -> List[Dict]:
        """Search for the top-k documents matching the query."""
        try:
            scores = self.bm25.get_scores(_tokenize(query))
            
            # Check if scores is empty
            if scores.size == 0:  # Use .size for NumPy arrays
                # Replace with logging if needed
                print("Warning: No scores returned for the query.")
                return []

            # top-k indices
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.k]

            return [{"id": self.docs[i]["id"], "text": self.docs[i]["text"], "score": float(scores[i])} for i in ranked]
        except Exception as e:
            print(f"Error in search: {e}")
            return []


if __name__ == "__main__":
    # Example usage
    r = Retriever(index_path="data/index", k=3)
    hits = r.search("policies about leave and benefits")
    for h in hits:
        print(f"{h['id']}: {h['score']:.3f}")