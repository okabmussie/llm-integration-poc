# src/rag/build_index.py
import os, json
from pathlib import Path
from rank_bm25 import BM25Okapi

CORPUS_DIR = Path("data/corpus")
INDEX_DIR = Path("data/index")
DOCS_JSONL = INDEX_DIR / "docs.jsonl"
BM25_JSON  = INDEX_DIR / "bm25.json"

def _tokenize(text: str):
    return [t.lower() for t in text.split()]

def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    docs = []
    tokenized = []

    # include .txt and .md, recursively
    files = list(CORPUS_DIR.rglob("*"))
    files = [p for p in files if p.suffix.lower() in {".txt", ".md"}]

    for p in sorted(files):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            continue
        doc_id = p.name                     # e.g., "pii_07.txt"
        source = str(p.relative_to(CORPUS_DIR))  # e.g., "pii_07.txt" or "folder/pii_07.txt"
        docs.append({"id": doc_id, "source": source, "text": text})
        tokenized.append(_tokenize(text))

    # Save docs
    with DOCS_JSONL.open("w", encoding="utf-8") as out:
        for d in docs:
            out.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Build BM25
    bm25 = BM25Okapi(tokenized)
    with BM25_JSON.open("w", encoding="utf-8") as out:
        json.dump({"corpus_tokenized": tokenized}, out)

    print(f"Indexed {len(docs)} documents into {INDEX_DIR}")

if __name__ == "__main__":
    main()