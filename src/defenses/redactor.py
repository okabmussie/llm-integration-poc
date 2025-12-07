# src/defenses/redactor.py
import re
from typing import Any, Dict, List

# Single canonical mask so logs and metrics can be counted easily
REPLACEMENT = "[REDACTED]"

# Built-in high-signal patterns
# These catch the most important leaks in a PoC setting
DEFAULTS: Dict[str, re.Pattern] = {
    # Entire synthetic secret blocks used in the corpus
    "secret_block": re.compile(r"BEGIN_SECRET[\s\S]*?END_SECRET", re.IGNORECASE),
    # AWS-style key shape from the seeded patterns
    "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # Explicit PII tags used in synthetic documents
    "pii_tag": re.compile(r"\[PII_(?:EMAIL|PHONE|DOB|ADDRESS|ID|CC_TEST)\]"),
    # Value-level patterns that often appear in text
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"\+?\d[\d\s().-]{7,}\d"),                 # generic international-ish
    "dob":   re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b"),        # YYYY-MM-DD
    "cc":    re.compile(r"\b(?:4\d{3}[ -]?){3}\d{4}\b"),           # simple Visa-style test numbers
    "org_id":re.compile(r"\b(?:EMP|MED|CORP|TST|AA)-\d{4,8}\b"),   # the synthetic IDs
}


# Apply larger / more specific patterns first
# This prevents partial matches and double masking
ORDER = ["secret_block", "aws_key", "pii_tag", "email", "phone", "dob", "cc", "org_id"]


def _iter_patterns(user_patterns: Any) -> List[re.Pattern]:
    """
    Build the final list of regex patterns in a deterministic order.
    1) Start with our defaults in ORDER
    2) Append any user-provided patterns from seeds or config
       Accepts dicts of compiled regexes, lists of compiled regexes,
       or raw regex strings which are compiled here.
    The function is tolerant of bad entries and skips them.
    """
    pats: List[re.Pattern] = []

    # 1)------ Defaults in fixed order for deterministic behavior ------
    for name in ORDER:
        rx = DEFAULTS.get(name)
        if rx is not None:
            pats.append(rx)

    # 2)------ User-provided patterns (from utils.pii_patterns.load_pii_patterns) ------
    if isinstance(user_patterns, dict):
        # Try to preserve ORDER for matching keys
        for k in ORDER:
            rx = user_patterns.get(k)
            if isinstance(rx, re.Pattern):
                pats.append(rx)

        # Append any remaining compiled regexes that were not in ORDER
        for v in user_patterns.values():
            if isinstance(v, re.Pattern) and v not in pats:
                pats.append(v)

    elif isinstance(user_patterns, (list, tuple)):
        for v in user_patterns:
            try:
                if isinstance(v, re.Pattern):
                    pats.append(v)
                elif isinstance(v, dict) and "regex" in v and isinstance(v["regex"], str):
                    pats.append(re.compile(v["regex"]))
                elif isinstance(v, str) and v.strip():
                    # treat raw strings as regex (escape if needed in your seeds)
                    pats.append(re.compile(v))
            except Exception:
                # Never fail the pipeline because one pattern was malformed
                continue

    # else: nothing provided (we still use defaults)
    return pats


def redact(text: str, patterns: Any, enabled: bool = True, mask: str = REPLACEMENT) -> str:
    """
    Replace any sensitive spans with the single mask token.
    Deterministic, order-aware, and tolerant of bad regexes.
    If disabled or text is empty, return the text unchanged.
    """
    if not enabled or not text:
        return text

    out = text
    for rx in _iter_patterns(patterns):
        try:
            out = rx.sub(mask, out)
        except Exception:
            # Do not break output generation because of a single bad pattern
            continue

    return out