# src/defenses/output_policy.py
import re
from typing import List, Any, Dict

# Message shown when output is blocked due to a violation
REFUSAL = "[BLOCKED: output policy violation]"


# --- Built-in high-risk patterns ---
# These are heuristic safety rules beyond your normal PII or secret filters.
# They catch risky content that could harm a system if executed or followed.

# Detects full URLs (http/https)
RISKY_URL = re.compile(r"\bhttps?://[^\s]+", re.I)

# Detects dangerous shell-like commands
SHELL_LIKE = re.compile(r"(?:\brm\s+-rf\b|\bchmod\s+777\b|\bwget\s+http|\bcurl\s+http)", re.I)

def check_output_policies(text: str, patterns: List[Any]) -> Dict[str, Any]:
    """
    Check the model's final text against all safety policies.

    - If the text contains seeded secrets or PII markers, flag as "seeded_or_pii".
    - If it contains risky URLs, flag as "risky_url".
    - If it contains shell commands, flag as "shell_like".

    Returns a dict with a list of violations found.
    Example: {"violations": ["risky_url", "shell_like"]}
    """
    if not text:
        return {"violations": []}

    v = []

    # 1. Check against PII/secret regex patterns loaded from seeds
    for rx in patterns or []:
        try:
            if rx.search(text):
                v.append("seeded_or_pii")
                break
        except Exception:
            # Continue if a pattern fails
            continue

    # 2. Check for risky URLs
    if RISKY_URL.search(text):
        v.append("risky_url")

    # 3. Check for dangerous shell commands
    if SHELL_LIKE.search(text):
        v.append("shell_like")

    return {"violations": v}

def enforce_output_policy(text: str, patterns: List[Any]) -> Dict[str, Any]:
    """
    Final enforcement step before showing the response.

    If any violation is found:
      - Block the message completely
      - Replace it with a refusal message "[BLOCKED: output policy violation]"
      - Return a dictionary with the reasons for blocking

    If everything is clean:
      - Pass the text unchanged

    This ensures the user never sees unsafe or leaking output.
    """
    res = check_output_policies(text, patterns)
    if res["violations"]:
        return {"blocked": True, "final": REFUSAL, "violations": res["violations"]}
    return {"blocked": False, "final": text, "violations": []}