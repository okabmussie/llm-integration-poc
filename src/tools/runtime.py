# src/tools/runtime.py
import re, shlex
from typing import Dict, Any, Optional, Tuple

# -------------- Role-based allow-list --------------------
# Role allow list
# which tools each role may call
ALLOWED: Dict[str, set] = {
    "viewer": {"search_docs"},
    "admin": {"search_docs", "create_user", "delete_user"},
}

# Detect "tool: name k=v k2='v two'" on the first line
_TOOL_LINE = re.compile(r'^\s*tool:\s*([a-zA-Z_][\w-]*)\s*(.*)$')

def _parse_tool_line(first_line: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """
    Parse a tool directive line into (name, args).
    Supports quoted values with spaces using shlex.
    """
    m = _TOOL_LINE.match(first_line or "")
    if not m:
        return None
    name, rest = m.group(1), m.group(2)

    args: Dict[str, str] = {}
    for tok in shlex.split(rest):
        if "=" in tok:
            k, v = tok.split("=", 1)
            # strip surrounding quotes if present (shlex already unquotes, but be safe)
            if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
                v = v[1:-1]
            args[k] = v
    return name, args


# ----------   Schema validation -----------
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
USER_ID_RE = re.compile(r"^\d{1,12}$")  # 1–12 digits
BLOCKED_QUERY_RE = re.compile(r"(BEGIN[ _-]?SECRET|AKIA[0-9A-Z]{16}|\[PII_[A-Z_]+\])", re.IGNORECASE)

def validate_tool_args(name: str, role: str, args: Dict[str, str]) -> Dict[str, Any]:
    """
    Schema check per tool
    Return a dict with allowed and optional reason and normalized args
    """
    args = {k: ("" if v is None else str(v)) for k, v in (args or {}).items()}

    if name == "delete_user":
        uid = args.get("id", "").strip()
        if not USER_ID_RE.match(uid):
            return {"allowed": False, "reason": "invalid id format (expected 1–12 digits)"}
        return {"allowed": True, "normalized": {"id": uid}}

    if name == "create_user":
        email = args.get("email", "").strip()
        if not EMAIL_RE.match(email):
            return {"allowed": False, "reason": "invalid email address"}
        nm = args.get("name", "").strip()
        if nm and len(nm) < 2:
            return {"allowed": False, "reason": "name too short"}
        data = {"email": email}
        if nm:
            data["name"] = nm
        return {"allowed": True, "normalized": data}

    if name == "search_docs":
        q = args.get("q", "").strip()
        if len(q) < 2:
            return {"allowed": False, "reason": "q required (min length 2)"}
        if BLOCKED_QUERY_RE.search(q):
            return {"allowed": False, "reason": "query matches blocked patterns"}
        return {"allowed": True, "normalized": {"q": q}}

    # Unknown tools are rejected by schema
    return {"allowed": False, "reason": f"unknown tool '{name}'"}

# ----- Execution shim used by the pipeline -----
def maybe_execute_tool(first_line: str, role: str, allowlist_enabled: bool) -> Dict[str, Any]:
    """
    Decision flow for tool lines
      1 parse first line
      2 check role against the allow list
      3 run schema validation
      4 if both pass return a harmless NO-OP effect and log
    Returns fields for logging so the pipeline can report attempted blocked allowed and effects
    """
    parsed = _parse_tool_line(first_line)
    if not parsed:
        return {"attempted": False}

    name, args = parsed
    attempted = True

    # Gate 1 role allow list
    role_ok = (name in ALLOWED.get(role, set())) if allowlist_enabled else True
    if not role_ok:
        return {
            "attempted": attempted,
            "allowed": False,
            "blocked": True,
            "tool": name,
            "args": args,
            "reason": f"{name} not allowed for role {role}",
        }

    # Gate 2 schema validation
    vr = validate_tool_args(name, role, args)
    if not vr.get("allowed", False):
        return {
            "attempted": attempted,
            "allowed": False,
            "blocked": True,
            "tool": name,
            "args": args,
            "reason": vr.get("reason", "schema validation failed"),
        }

    norm = vr.get("normalized", args)

    # Mock effects only (no side effects)
    # Execute as a NO-OP so experiments are safe and reproducible
    if name == "delete_user":
        effect = f"NO-OP: would delete user id={norm.get('id', '?')}"
    elif name == "create_user":
        effect = f"NO-OP: would create user {norm}"
    elif name == "search_docs":
        effect = "NO-OP: would search docs"
    else:
        effect = "NO-OP: tool executed"

    return {
        "attempted": attempted,
        "allowed": True,
        "blocked": False,
        "tool": name,
        "args": norm,
        "effect": effect,
    }
