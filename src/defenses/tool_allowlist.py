def enforce_allowlist(tool_name: str, role: str, enabled: bool):
    if not enabled:
        return True
    if role == "viewer" and tool_name == "get_admin_report":
        return False
    return True
