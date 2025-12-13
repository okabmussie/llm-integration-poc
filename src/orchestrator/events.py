
# src/orchestrator/events.py
import json, time, os
from datetime import datetime, timezone
from typing import Any, Dict

# --- NEW: recursive sanitizer for any non-JSON-serializable values -----------
def _json_sanitize(x: Any):
    """
    Recursively convert any non-JSON-serializable types to strings.

    This protects the logger from crashes when payloads contain objects like
    re.Pattern, numpy scalars, Exceptions, SimpleNamespace, sets, tuples, etc.
    """
    # Basic JSON types pass through
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    # Dict → sanitize keys/values
    if isinstance(x, dict):
        return {str(k): _json_sanitize(v) for k, v in x.items()}
    # List/Tuple/Set → sanitize elements
    if isinstance(x, (list, tuple, set)):
        return [_json_sanitize(v) for v in x]
    # Last resort: string representation
    try:
        # Sometimes objects are already JSON-compatible (e.g., numpy scalar castable)
        json.dumps(x)
        return x
    except Exception:
        return str(x)


class EventLogger:
    def __init__(self, run_dir: str):
        self.path = os.path.join(run_dir, "events.jsonl")
        os.makedirs(run_dir, exist_ok=True)
        self.f = open(self.path, "a", encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _now(self):
        ts = time.time()
        return ts, datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    def log(self, type: str, **payload: Dict[str, Any]):
        """
        Write one JSON line. 'type' is a short label like 'adversarial', 'benign', 'run_config', etc.
        Any extra keyword args become fields in the record (e.g., defense_trace, attack_success).
        """
        ts, ts_iso = self._now()
        rec = {"ts": ts, "ts_iso": ts_iso, "type": type, **payload}
        try:
            self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            # Fallback: recursively sanitize the record so we never drop the event
            safe = _json_sanitize(rec)
            self.f.write(json.dumps(safe, ensure_ascii=False) + "\n")
        self.f.flush()

    def log_run_config(self, profile_name: str, config: Dict[str, Any], model: str = None):
        """Optional convenience to record config at the start of a run."""
        self.log("run_config", profile_name=profile_name, config=config, model=model)

    def close(self):
        if not self.f.closed:
            self.f.close()