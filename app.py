# app.py
import os, time
from pathlib import Path
from types import SimpleNamespace
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.orchestrator.pipeline import Pipeline

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))


RUNS_ROOT = Path("runs")
SEEDS = "data/seeds.json"
INDEX = "data/index"
MODEL = os.getenv("MODEL", "gpt-4o-mini")

app = FastAPI(title="LLM Integration POC")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

class ChatIn(BaseModel):
    prompt: str
    profile: str = "defended"            # "baseline" or "defended"
    defenses: dict | None = None         # optional explicit dict
    # quick flags (used if defenses is not provided)
    input_filter: bool | None = None
    rag_filter: bool | None = None
    tool_allowlist: bool | None = None
    redaction: bool | None = None


from fastapi.responses import FileResponse

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/chat")
def chat(body: ChatIn):
    # pick defenses
    if body.profile == "baseline":
        defenses = {}
    else:
        defenses = body.defenses or {
            "input_filter": True if body.input_filter is None else bool(body.input_filter),
            "rag_filter": True if body.rag_filter is None else bool(body.rag_filter),
            "tool_allowlist": True if body.tool_allowlist is None else bool(body.tool_allowlist),
            "redaction": True if body.redaction is None else bool(body.redaction),
        }

    # per-request run directory (keeps your JSONL logs)
    run_dir = RUNS_ROOT / time.strftime("%Y-%m-%d_%H-%M-%S") / body.profile
    run_dir.mkdir(parents=True, exist_ok=True)

    # build a ctx just like runner.py does
    ctx = SimpleNamespace(
        profile_name=body.profile,
        model_name=MODEL,
        top_k=3,
        dry_run=False,     # set True if you want to avoid real API calls
        defenses=defenses,
        role="viewer",
        use_rag=True,
    )

    pipe = Pipeline(ctx, run_dir=str(run_dir), seeds_path=SEEDS, index_dir=INDEX)
    out = pipe.process(body.prompt)
    pipe.close()

    return {
        "final": out.get("final", ""),
        "defense_trace": out.get("defense_trace", {}),
        "run_dir": str(run_dir),
    }
