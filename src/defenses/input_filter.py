# src/defenses/input_filter.py
import re
from typing import Union, Dict, Any


# The filter is intentionally small and high-precision
# Text is lowercased before matching so patterns are written in lowercase
# Each regex uses \b word boundaries so we only match whole phrases
# This avoids accidental matches inside longer words
_PATTERNS = [
    # Blocks classic jailbreak phrasing that tries to override safety or prior steps
    # Matches examples like:
    #   "ignore previous instructions"
    #   "please IGNORE previous instructions"
    r"\bignore previous instructions\b",

    # Blocks direct requests to expose hidden configuration
    # Matches:
    #   "reveal the system prompt"
    #   "could you reveal the system prompt"
    r"\breveal the system prompt\b",

    # Blocks prompts that ask for a special unsafe mode often used in jailbreaks
    # Matches:
    #   "enable developer mode"
    #   "respond in developer mode"
    r"\bdeveloper mode\b",
    
    # Blocks social-engineering phrasing that asks the model to role-play as someone else
    # Matches:
    #   "pretend to be an admin"
    #   "pretend to be the system"
    r"\bpretend to be\b",


    # explicit asks to ignore safety / previous instructions
    r"\bignore previous instructions\b",
    r"\bdisregard previous instructions\b",
    r"\bforget previous instructions\b",
    r"\bignore all (previous )?rules\b",

    # direct attempts to reveal hidden config / system prompt
    r"\breveal the system prompt\b",
    r"\bshow the system prompt\b",
    r"\bwhat is the system prompt\b",

    # "developer / jailbreak" modes
    r"\bdeveloper mode\b",
    r"\bjailbreak\b",
    r"\benable (developer|jailbreak) mode\b",
    r"\brespond as developer\b",

    # "do anything" style coercion
    r"\bdo anything\b",
    r"\bdo anything now\b",
    r"\bdo whatever it takes\b",

    # "respond only/first line/first non-empty line" coercion
    r"\breply only with\b",
    r"\brespond only with\b",
    r"\byour next step\b",
    r"\byour first non-?empty line\b",

    # tool-invocation attempts (explicit tool heads)
    r"\btool\s*:\s*[a-z0-9_-]+\b",        # matches 'tool: NAME' anywhere
    r"^\s*tool\s*:\s*[a-z0-9_-]+\b",      # anchored-first-line variant

    # escalation / role-change attempts
    r"\bescalat(e|ion)\b",
    r"\bgrant (me|user) admin\b",
    r"\bmake me an admin\b",
    r"\bchange my role to admin\b",
    r"\bbox to admin\b",

    # secret/canary markers (BEGIN/END variants)
    r"\bbegin[ _-]?secret\b",
    r"\bend[ _-]?secret\b",
    r"\bbegin[ _-]?canary\b",
    r"\bend[ _-]?canary\b",

    # social engineering / impersonation
    r"\bpretend to be\b",
    r"\bact as an? admin\b",
    r"\bact as system\b",

    # explicit bypass terms
    r"\bbypass (safety|filters|policy)\b",
    r"\bdisable (safety|filters|policy)\b",
    r"\bturn off (safety|filters|policy)\b"
    
    ]

# Optional extras, OFF by default
_TOOL_HEAD_RE   = re.compile(r"(?m)^\s*tool\s*:\s*[A-Za-z0-9_-]+")
_CANARY_RE      = re.compile(r"BEGIN[ _-]?SECRET.*END[ _-]?SECRET", re.I | re.S)


def apply_input_filter(user_text: str, enabled: Union[bool, Dict[str, Any]]):
    """
    First guard in the pipeline.

    Purpose
      Check the raw user text for obvious jailbreak phrases.
      If a risky pattern is found, block the request and return a short reason.
      If nothing risky is found, return the original text unchanged.

    Inputs
      user_text  the raw text typed by the user
      enabled    feature switch from the active profile

    Output shape
      dict with keys:
        - blocked: bool  True when a pattern matched
        - prompt:  str   the text that should continue through the pipeline
        - reason:  str   present only when blocked is True

    Notes
      The match is case-insensitive by lowercasing the input once,
      then checking each pattern. This keeps the code fast and simple.
    """


    # If the defense is disabled in the current profile, pass through unchanged
    if isinstance(enabled, dict):
        on = bool(enabled.get("on", True))
        block_tool_heads = bool(enabled.get("block_tool_heads", False))
        block_canaries   = bool(enabled.get("block_canaries", False))
    else:
        on = bool(enabled)
        block_tool_heads = False
        block_canaries   = False

    if not on:
        return {"blocked": False, "prompt": user_text}
    
    text_lower = (user_text or "").lower()

    # Base patterns (your precise jailbreak phrases)

    
    #if not enabled:
       # return {"blocked": False, "prompt": user_text}
    
    # Normalize case once to make the pattern checks case-insensitive
    #text = user_text.lower()

    # Scan for any of the risky patterns
    for pat in _PATTERNS:
        if re.search(pat, text_lower):
            # Stop at the first hit and explain why it was blocked
            return {"blocked": True, "reason": f"matched '{pat}'"}
        
    if block_tool_heads and _TOOL_HEAD_RE.search(user_text or ""):
        return {"blocked": True, "reason": "matched 'raw_tool_head'"}
    
    if block_canaries and _CANARY_RE.search(user_text or ""):
        return {"blocked": True, "reason": "matched 'secret_canary'"}

    # No risky pattern found, allow the request to continue
    return {"blocked": False, "prompt": user_text}
