# src/tools/validate.py
from typing import Dict, Any
import re

def validate_tool_args(tool: str, role: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Purpose
      Decide if a proposed tool call is safe to run based on user role and a tiny schema.
      Return a dict with allowed flag and optional reason so the pipeline can log the outcome.

    Inputs
      tool   name of the tool to run like "send_email" or "get_admin_report"
      role   current user role like "viewer" or "admin"
      args   arguments extracted from the tool line already parsed as a dict

    Behavior
      - Block known dangerous or privileged tools for non admin roles
      - Validate argument shapes so obviously bad inputs do not execute
      - Return {"allowed": True} only when both role and schema look safe
    """
    args = args or {}

    # ---------------------------
    # Read-only tools (safe demos)
    # ---------------------------
    if tool == "search_docs":
        # Viewers are allowed to perform read-only searches.
        # Require a short 'q' query; reject very long / empty queries.
        q = str(args.get("q", "")).strip()
        if not q:
            return {"allowed": False, "reason": "missing query"}
        if len(q) > 100:
            return {"allowed": False, "reason": "query too long"}
        return {"allowed": True, "reason": ""}

    if tool == "fetch_doc":
        # Viewers may fetch a single document by ID (constrained format).
        doc_id = str(args.get("id", "")).strip()
        if not doc_id:
            return {"allowed": False, "reason": "missing id"}
        if not re.fullmatch(r"[A-Za-z0-9_\-\.]{1,64}", doc_id):
            return {"allowed": False, "reason": "invalid id"}
        return {"allowed": True, "reason": ""}

    # ---------------------------------------
    # Privileged / potentially destructive ops
    # ---------------------------------------
    if tool == "delete_user":
        # Only admins can delete users; require a small integer id.
        if role != "admin":
            return {"allowed": False, "reason": "admin-only operation"}
        try:
            uid = int(str(args.get("id", "")).strip())
        except Exception:
            return {"allowed": False, "reason": "invalid id"}
        if uid < 1:
            return {"allowed": False, "reason": "invalid id"}
        return {"allowed": True, "reason": ""}

    # Example policies and hard stop for privilege escalation
    if tool == "escalate":
        # Never allow the model to escalate role through a tool
        return {"allowed": False, "reason": "escalate is admin-only"}

    if tool == "get_admin_report":
        # Only admins may request this report
        if role != "admin":
            return {"allowed": False, "reason": "admin-only report"}
        # simple schema for the "period" argument
        period = str(args.get("period", ""))
        if period not in {"daily", "weekly", "monthly"}:
            return {"allowed": False, "reason": "invalid period"}
        return {"allowed": True, "reason": ""}

    if tool == "send_email":
        # Guard message size and restrict the recipient domain
        # example: forbid free-form body > 500 chars and require whitelisted domain
        to = str(args.get("to", ""))
        body = str(args.get("body", ""))
        if len(body) > 500:
            return {"allowed": False, "reason": "body too long"}
        if not to.endswith("@example.com"):
            return {"allowed": False, "reason": "recipient not allowed"}
        return {"allowed": True, "reason": ""}

    # Safe default policy for unknown tools can be deny or allow
    # Here we choose DENY by default (safer, and clearer for your thesis write-up).
    # If you want the old behavior, change to {"allowed": True, "reason": ""}.
    return {"allowed": False, "reason": "tool not whitelisted"}