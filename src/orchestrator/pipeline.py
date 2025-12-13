
# src/orchestrator/pipeline.py
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re  # ensure imported
import shlex  # for robust tool arg parsing
from src.orchestrator.events import EventLogger
from src.llm.client import LLMClient
from src.rag.retriever import Retriever
from src.rag.filter import guard_context
from src.defenses.input_filter import apply_input_filter
from src.defenses.redactor import redact
from src.defenses.output_policy import enforce_output_policy
from src.tools.runtime import maybe_execute_tool
from src.utils.pii_patterns import load_pii_patterns


class Pipeline:
    """
    User → Input filter → (RAG → RAG guard) → LLM → Tool allow-list → Redaction → Output policy → Final
    Every stage logs to runs/.../events.jsonl
    """

    def __init__(self, ctx, run_dir: str, seeds_path: str, index_dir: str):
        self.ctx = ctx
        self.log = EventLogger(run_dir)
        self.llm = LLMClient(model_name=ctx.model_name, dry_run=getattr(ctx, "dry_run", True))
        self.retriever = Retriever(index_dir)
        self.patterns = load_pii_patterns(seeds_path)

        self.top_k = getattr(ctx, "top_k", 3)
        self.use_rag = getattr(ctx, "use_rag", True)

    # ------------------------------ unified logger ---------------------------
    def _elog(self, type_: str, **payload):
        """Always attach prompt_file so jq filters work consistently."""
        pf = getattr(self.ctx, "prompt_file", "")
        payload.setdefault("prompt_file", pf)
        self.log.log(type_, **payload)

    # ------------------------------ helpers ---------------------------------
    def _normalize_docs(self, raw: List[Any], k: int) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        try:
            raw = list(raw)[: max(0, int(k))]
        except Exception:
            raw = list(raw)[:3]

        for item in raw:
            if isinstance(item, dict):
                doc_id = item.get("id") or item.get("doc_id")
                score = float(item.get("score", 0.0))
                text = item.get("text")
            else:
                try:
                    doc_id, score = item
                    score = float(score)
                except Exception:
                    continue
                text = None

            if not isinstance(doc_id, str):
                continue

            if text is None:
                doc_path = Path("data/corpus") / doc_id
                if doc_path.exists():
                    try:
                        text = doc_path.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        text = ""
                else:

                    text = ""

            docs.append({"id": doc_id, "score": score, "text": text})
        return docs

    def _topk_docs(self, query: str, k: int) -> List[Dict[str, Any]]:
        if hasattr(self.retriever, "topk"):
            raw = self.retriever.topk(query, k)
        elif hasattr(self.retriever, "search"):
            raw = self.retriever.search(query)
        else:
            raw = []
        return self._normalize_docs(raw, k)

    def _call_llm(self, prompt: str, context_text: str) -> str:
        try:
            return self.llm.generate(prompt, context=context_text)
        except Exception as e:
            self._elog("llm_error", error=str(e))
            return "I'm sorry, but I can't assist with that."

    def _run_input_filter(self, user_prompt: str, enabled: bool) -> Tuple[bool, str, str]:
        if not enabled:
            return False, "", user_prompt
        try:
            res = apply_input_filter(user_prompt, enabled)
            if isinstance(res, dict):
                blocked = bool(res.get("blocked", False))
                reason = str(res.get("reason", "")) if blocked else ""
                sanitized = res.get("prompt", user_prompt)
                return blocked, reason, sanitized
            if isinstance(res, tuple) and len(res) >= 2:
                blocked, reason = res[0], res[1]
                return bool(blocked), str(reason), user_prompt
        except Exception as e:
            self._elog("input_filter_error", error=str(e))
        return False, "", user_prompt

    # ---------------------- main pipeline -----------------------
    def process(self, user_prompt: str) -> Dict[str, Any]:
        d = getattr(self.ctx, "defenses", {}) or {}
        role = getattr(self.ctx, "role", "viewer")
        profile = getattr(self.ctx, "profile_name", "unknown")

        pf = getattr(self.ctx, "prompt_file", "")
        self._current_pf = pf   # <-- capture once for this request

        self._elog("pf_debug_start", saw=pf or "<EMPTY>")

        self._elog("pf_debug", note="start_process")

        defense_trace: Dict[str, Any] = {}
        retrieval_meta: Dict[str, Any] = {"top_k": self.top_k, "docs": []}
        rag_retrieval_on = bool(self.use_rag) and (int(self.top_k) > 0)

        # EVENT: request (with prompt_file)
        self._elog("request", profile=profile, role=role, prompt=user_prompt, defenses=d)

        # Early raw-first snapshot + breadcrumb
        def _first_line_raw(s: str) -> str:
            if not s:
                return ""
            s = s.replace("\r\n", "\n").replace("\r", "\n")
            ln = s.split("\n", 1)[0]
            return ln.lstrip("\ufeff").lstrip()  # BOM + spaces

        raw_first = _first_line_raw(user_prompt)
        raw_is_tool = bool(re.match(r'^\s*tool:\s*\w+\b', raw_first, flags=re.IGNORECASE))

        self._elog("raw_first_debug", raw_first=raw_first)
        self._elog(
            "tool_scan_snapshot",
            reply_has=False,
            prompt_has=False,
            raw_has=raw_is_tool,
            reply_tool_line="",
            prompt_tool_line="",
            raw_tool_line=raw_first if raw_is_tool else "",
            reply_preview="",
            prompt_preview="",
            raw_preview=(user_prompt or "")[:80] + ("..." if len(user_prompt or "") > 80 else ""),
            tool_enabled=bool(d.get("tool_allowlist", False)),
        )


        # 1) Input filter
        blocked, reason, prompt_for_model = self._run_input_filter(
            user_prompt,
            bool(d.get("input_filter", False)),
        )

        # Debug log AFTER values exist
        self._elog(
            "input_filter_debug",
            enabled=bool(d.get("input_filter", False)),
            blocked=blocked,
            reason=reason,
            sanitized_preview=(prompt_for_model or "")[:120],
        )

        defense_trace["input_filter"] = {
            "enabled": bool(d.get("input_filter", False)),
            "blocked": blocked,
        }
        if reason:
            defense_trace["input_filter"]["reason"] = reason

        if blocked:
            self._elog("blocked", stage="input_filter", reason=reason)
            final = "[BLOCKED] " + (reason or "input matched a restricted pattern")
            self._elog(
                "final",
                role=role,
                profile=profile,
                context_len=0,
                final=final,
                defense_trace=defense_trace,
                retrieval=retrieval_meta,
            )
            return {"final": final, "defense_trace": defense_trace}

        # 2) Retrieval (define `docs` before RAG guard)
        docs: List[Dict[str, Any]] = []
        if rag_retrieval_on:
            docs = self._topk_docs(prompt_for_model, self.top_k)
            retrieval_meta["docs"] = [
                {"id": d0["id"], "score": d0["score"], "source": d0.get("id")}
                for d0 in docs
            ]
            self._elog("retrieved", top_k=self.top_k, docs=retrieval_meta["docs"])

        # 3) RAG guard
        rag_enabled = bool(d.get("rag_filter", False))
        raw_context = ""
        if docs:
            try:
                chunks = []
                for dd in docs:
                    head = f"[{dd.get('id','?')}] "
                    txt = (dd.get("text") or "")[:800]
                    chunks.append(head + txt)
                raw_context = "\n---\n".join(chunks)
            except Exception:
                raw_context = ""

        context_text, mask_meta = guard_context(docs, self.patterns, rag_enabled)
        masked_in_context = int(mask_meta.get("masked_total", 0)) if rag_enabled else 0

        defense_trace["rag_filter"] = {
            "enabled": rag_enabled,
            "masked": masked_in_context,
            "by_type": mask_meta.get("by_type", {}),
            "docs": mask_meta.get("docs", []),
            "preview_before": (raw_context or "")[:600],
            "preview_after": ((context_text or "")[:600]) if rag_enabled else (raw_context or "")[:600],
        }

        # 4) Model
        reply = self._call_llm(prompt_for_model, context_text)
        self._elog("llm_output", text=reply)

        if bool(d.get("postmask_echo", False)):
            reply = prompt_for_model
            defense_trace.setdefault("llm", {})["demo_mode"] = "echo_prompt"
            self._elog("llm_output_demo_echo", note="postmask_echo enabled; using prompt as reply")



        # 5) TOOL ALLOW-LIST + SCHEMA VALIDATION
        tool_enabled = bool(d.get("tool_allowlist", False))

        # Robust: allow spaces around colon; allow hyphens/underscores in names
        TOOL_HEAD_RE     = re.compile(r'^\s*tool\s*:\s*([A-Za-z0-9_-]+)\b', re.IGNORECASE)
        TOOL_ANYWHERE_RE = re.compile(r'\btool\s*:\s*([A-Za-z0-9_-]+)\b', re.IGNORECASE)
        FENCE_RE         = re.compile(r'^```.*$', re.IGNORECASE)

        def _first_lines(s: str, n: int = 30) -> list:
            if not s:
                return []
            s = s.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
            return s.split("\n")[:max(0, n)]

        def _find_tool_line(lines: list) -> str:
            """Return the first usable 'tool:' line; scan fence and non-fence content."""
            in_fence = False
            for ln in lines or []:
                if FENCE_RE.match(ln or ""):
                    in_fence = not in_fence
                    # still allow a 'tool:' that sits on the same fence delimiter line
                if TOOL_ANYWHERE_RE.search(ln or ""):
                    # take the last 'tool:' segment on the line
                    m_all = list(TOOL_ANYWHERE_RE.finditer(ln))
                    seg = (ln or "")[m_all[-1].start():].lstrip()
                    return seg
            return ""

        def _normalize_tool_line(line: str) -> str:
            """Collapse duplicated heads like 'tool: tool: purge_logs …' → 'tool: purge_logs …'."""
            if not line:
                return line
            s = line.lstrip("\ufeff").lstrip()
            while True:
                if not s.lower().startswith("tool:"):
                    break
                tail = s.split(":", 1)[1].strip()
                if tail.lower().startswith("tool:"):
                    s = tail
                else:
                    s = f"tool: {tail}"
                    break
            if not s.lower().startswith("tool:"):
                s = f"tool: {s}"
            return s.strip()

        def _fallback_parse(line: str):
            """Parse 'tool: <name> key=val key=\"spaced val\"' via shlex; returns (name, args:dict)."""
            name, args = "(unknown)", {}
            try:
                tail = line.split(":", 1)[1].strip() if ":" in line else line.strip()
                if tail.lower().startswith("tool:"):
                    tail = tail.split(":", 1)[1].strip()
                toks = shlex.split(tail)
                if toks:
                    name = toks[0]
                    for tok in toks[1:]:
                        if "=" in tok:
                            k, v = tok.split("=", 1)
                            args[k] = v
            except Exception:
                pass
            return name, args

        # Grab first ~30 lines from reply/prompt/raw
        reply_lines  = _first_lines(reply, 30)
        prompt_lines = _first_lines(prompt_for_model, 30)
        raw_lines    = _first_lines(user_prompt, 30)

        reply_tool_line   = _find_tool_line(reply_lines)
        prompt_tool_line0 = _find_tool_line(prompt_lines)
        raw_tool_line     = _find_tool_line(raw_lines)

        # If reply didn’t have it, prefer sanitized prompt; else fall back to raw
        effective_prompt_tool_line = prompt_tool_line0 or (raw_tool_line or "")
        is_tool_in_reply  = bool(reply_tool_line)
        is_tool_in_prompt = (not is_tool_in_reply) and bool(effective_prompt_tool_line)

        tool_line = reply_tool_line if is_tool_in_reply else (effective_prompt_tool_line if is_tool_in_prompt else "")
        tool_src  = "reply" if is_tool_in_reply else ("prompt" if is_tool_in_prompt else "")

        # Normalize duplicated heads for clean logs
        tool_line = _normalize_tool_line(tool_line)

        # Debug: show what we scanned
        self._elog("tool_scan_lines_debug",
                   reply_first_lines=reply_lines[:6],
                   prompt_first_lines=prompt_lines[:6],
                   raw_first_lines=raw_lines[:6])

        # Single authoritative snapshot
        self._elog(
            "tool_scan_snapshot",
            reply_has=bool(reply_tool_line),
            prompt_has=bool(effective_prompt_tool_line),
            raw_has=bool(raw_tool_line),
            reply_tool_line=reply_tool_line,
            prompt_tool_line=effective_prompt_tool_line,
            raw_tool_line=raw_tool_line,
            reply_preview=(reply or "")[:80] + ("..." if len(reply or "") > 80 else ""),
            prompt_preview=(prompt_for_model or "")[:80] + ("..." if len(prompt_for_model or "") > 80 else ""),
            raw_preview=(user_prompt or "")[:80] + ("..." if len(user_prompt or "") > 80 else ""),
            tool_enabled=tool_enabled,
        )

        # Extra breadcrumb if raw had a tool head but final selection is empty
        if bool(raw_tool_line) and not bool(tool_line):
            self._elog(
                "tool_detection_miss",
                raw_first=raw_lines[0] if raw_lines else "",
                prompt_first=prompt_lines[0] if prompt_lines else "",
                reply_first=reply_lines[0] if reply_lines else "",
                note="raw shows tool:, but no tool_line selected"
            )

        # Initialize defense trace for allow-list
        defense_trace["tool_allowlist"] = {
            "enabled": tool_enabled,
            "attempted": bool(tool_line),
            "executed": False,
            "allowed": False,
            "blocked": False,
            "source": tool_src,
        }

        self._elog("tool_decision_debug", tool_line=tool_line, tool_src=tool_src, tool_enabled=tool_enabled)

        tool_text = ""
        if tool_line:
            n0, a0 = _fallback_parse(tool_line)

            # Always record the attempt
            self._elog("tool_attempt", name=n0, args=a0, role=role, source=tool_src or "unknown", line=tool_line)

            if not tool_enabled:
                defense_trace["tool_allowlist"]["blocked"] = True
                self._elog(
                    "tool_blocked",
                    name=n0, args=a0, role=role, source=tool_src or "unknown",
                    reason="allowlist_disabled",
                    attempted=True, executed=False, allowed=False, blocked=True,
                    line=tool_line,
                )
            else:
                try:
                    # Execute via your runtime gate (enforces role/schema)
                    tool_res = maybe_execute_tool(tool_line, role, True)
                    attempted    = bool(tool_res.get("attempted", True))
                    allowed      = bool(tool_res.get("allowed", False))
                    blocked_flag = bool(tool_res.get("blocked", not allowed))
                    executed     = attempted

                    defense_trace["tool_allowlist"].update({
                        "attempted": attempted,
                        "executed": executed,
                        "allowed": allowed,
                        "blocked": blocked_flag,
                    })

                    if tool_res.get("effect"):
                        tool_text = str(tool_res["effect"])

                    payload = {
                        "name": tool_res.get("name", n0),
                        "args": tool_res.get("args", a0),
                        "role": role,
                        "source": tool_src or "unknown",
                        "attempted": attempted,
                        "executed": executed,
                        "allowed": allowed,
                        "blocked": blocked_flag,
                        "line": tool_line,
                    }
                    if blocked_flag:
                        payload["reason"] = tool_res.get("reason", "blocked")
                        self._elog("tool_blocked", **payload)
                    else:
                        self._elog("tool_result", **payload)

                except Exception as e:
                    defense_trace["tool_allowlist"]["blocked"] = True
                    self._elog(
                        "tool_blocked",
                        name=n0, args=a0, role=role, source=tool_src or "unknown",
                        reason=str(e),
                        attempted=True, executed=False, allowed=False, blocked=True,
                        line=tool_line,
                    )




        # 6) Compose
        combined = reply if isinstance(reply, str) else ""
        if tool_text:
            combined = (combined + "\n" + tool_text).strip()

        # 7) Redaction
        redaction_enabled = bool(d.get("redaction", False))
        final = redact(combined, self.patterns, redaction_enabled)
        masked_count = final.count("[REDACTED]") if isinstance(final, str) else 0
        defense_trace["redaction"] = {"enabled": redaction_enabled, "masked": masked_count}

        # 8) Output policy
        output_policy_enabled = bool(d.get("output_policy", False))
        if output_policy_enabled:
            pol_enforced = enforce_output_policy(final, self.patterns)
            if pol_enforced.get("blocked"):
                self._elog("output_policy_violation", violations=pol_enforced.get("violations", []))
                final = pol_enforced.get("final", final)
            defense_trace["output_policy"] = {
                "enabled": True,
                "blocked": bool(pol_enforced.get("blocked")),
                "violations": pol_enforced.get("violations", []),
            }
        else:
            defense_trace["output_policy"] = {"enabled": False, "blocked": False, "violations": []}

        # EVENT: final
        self._elog(
            "final",
            role=role, profile=profile,
            context_len=len((context_text or "")),
            final=final, defense_trace=defense_trace, retrieval=retrieval_meta,
        )
        return {"final": final, "defense_trace": defense_trace}

    def close(self):
        self.log.close()


