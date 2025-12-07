# src/rag/filter.py
from typing import List, Dict, Any, Tuple
from src.defenses.redactor import redact
import re

# === NEW: lightweight per-type regexes for evidence accounting ===
# These are only for counting/attribution in traces, not for masking logic itself.
PII_TYPE_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z.]{2,}", re.IGNORECASE),
    "phone": re.compile(r"\+?\d[\d\-\s()]{8,}\d"),
    "dob":   re.compile(r"\b(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b"),  # ISO yyyy-mm-dd
    "cc":    re.compile(r"\b(?:4\d{3}[ -]?){3}\d{4}\b"),
    "akia":  re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    # Treat BEGIN/END canaries as a type for accounting
    "secret_block": re.compile(r"BEGIN[ _-]?SECRET.*?END[ _-]?SECRET", re.IGNORECASE | re.DOTALL),
    # A very loose "ID" matcher (for demo/evidence counts); actual masking is done by redactor()
    "id":    re.compile(r"\b(?:Employee\s*id|Fødselsnummer|National\s*ID|ID)\s*[:#]?\s*[A-Za-z0-9-]{3,}\b", re.IGNORECASE),
}

def _count_types(text: str) -> Dict[str, int]:
    """Return counts of each type in the given text (best-effort, for evidence only)."""
    counts: Dict[str, int] = {k: 0 for k in PII_TYPE_PATTERNS.keys()}
    if not text:
        return counts
    for t, rx in PII_TYPE_PATTERNS.items():
        try:
            counts[t] = len(rx.findall(text))
        except Exception:
            counts[t] = 0
    return counts

def _sum_counts(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    out = dict(a or {})
    for k, v in (b or {}).items():
        out[k] = out.get(k, 0) + int(v)
    return out

def guard_context(docs: List[Dict[str, Any]], patterns: Any, enabled: bool) -> Tuple[str, Dict[str, Any]]:
    """
    Purpose
      Build a single context string from the top-k retrieved docs.
      When enabled is True, scrub obvious canary markers and mask PII or secrets
      using the same redactor that is applied after the model. This keeps
      masking consistent before and after the LLM.

    Inputs
      docs     list of {"id": str, "score": float, "text": str}
      patterns compiled PII and secret patterns used by the redactor
      enabled  feature switch from the active profile

    Output
      A tuple:
        (clean_context_text, mask_meta)
      where mask_meta = {
        "masked_total": int,
        "by_type": {"email": n, "phone": n, ...},      # estimated masked counts by type
        "docs": [{"id": "...", "masked_by_type": {...}, "masked_total": n, "score": 0.732}, ...]
      }
    """
    parts: list[str] = []
    meta_docs: List[Dict[str, Any]] = []
    total_by_type: Dict[str, int] = {k: 0 for k in PII_TYPE_PATTERNS.keys()}
    masked_total = 0

    # Iterate documents in rank order and build the cleaned context chunk by chunk
    for d in (docs or []):
        doc_id = d.get("id", "unknown")
        score = float(d.get("score", 0.0))
        txt = d.get("text") or ""

        # --- Evidence counters BEFORE masking
        counts_before = _count_types(txt)

        if enabled:
            # 1) Remove obvious synthetic markers completely
            #    These are canaries used only for testing and should never reach the LLM
            filtered_lines = []
            for line in txt.splitlines():
                if ("BEGIN_SECRET" in line) or ("END_SECRET" in line) or ("ADMIN_ONLY" in line):
                    # Drop the whole line to avoid leaking marker hints
                    continue
                filtered_lines.append(line)
            txt = "\n".join(filtered_lines)

            # 2) Mask any remaining sensitive spans with the shared redactor
            #    Using the same redactor here and post-LLM keeps behavior consistent
            txt = redact(txt, patterns, enabled=True)

        # --- Evidence counters AFTER masking
        counts_after = _count_types(txt)

        # Estimate masked_by_type = max(0, before - after); '[REDACTED]' not categorized by type
        masked_by_type = {
            k: max(0, int(counts_before.get(k, 0)) - int(counts_after.get(k, 0)))
            for k in PII_TYPE_PATTERNS.keys()
        }
        masked_total_doc = sum(masked_by_type.values())
        masked_total += masked_total_doc
        total_by_type = _sum_counts(total_by_type, masked_by_type)

        # Add a small header so traces show which doc contributed and with what score
        # Example header: [doc_006.txt s=0.732]
        parts.append(f"[{doc_id} s={score:.3f}]\n{txt}")
        meta_docs.append({
            "id": doc_id,
            "score": score,
            "masked_by_type": masked_by_type,
            "masked_total": masked_total_doc,
        })

    # Separate documents with a clear delimiter so logs are easy to read
    context = "\n\n---\n\n".join(parts)

    mask_meta = {
        "masked_total": masked_total,
        "by_type": total_by_type,
        "docs": meta_docs,
    }
    return context, mask_meta