from dataclasses import dataclass

@dataclass
class RunContext:
    profile_name: str
    defenses: dict
    paths: dict
    role: str = "viewer"   # default role for runs
    model_name: str = "gpt-4o-mini"
    top_k: int = 3
