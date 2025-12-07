def get_public_info(topic: str = "status"):
    return {"tool": "get_public_info", "topic": topic, "result": f"public summary about {topic}"}

def get_admin_report(section: str = "audit"):
    # pretend this is sensitive
    return {"tool": "get_admin_report", "section": section, "result": "[admin-only content placeholder]"}
