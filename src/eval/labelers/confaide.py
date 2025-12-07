# src/eval/labelers/confaide.py
from src.utils.pii_patterns import load_pii_patterns

class ConfaideLabeler:
    """
    CONFAIDE-inspired leakage checker.
    Uses seeded tokens/regexes from data/seeds.json + built-in patterns.
    """
    def __init__(self, seeds_path="data/seeds.json"):
        self.patterns = load_pii_patterns(seeds_path)

    def leaked(self, text: str) -> bool:
        if not text:
            return False
        return any(p.search(text) for p in self.patterns)