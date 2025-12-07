
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









# # src/orchestrator/pipeline.py
# from pathlib import Path
# from typing import List, Dict, Any, Tuple

# from src.orchestrator.events import EventLogger
# from src.llm.client import LLMClient
# from src.rag.retriever import Retriever
# from src.rag.filter import guard_context
# from src.defenses.input_filter import apply_input_filter
# from src.defenses.redactor import redact
# from src.defenses.output_policy import enforce_output_policy
# from src.tools.runtime import maybe_execute_tool
# from src.utils.pii_patterns import load_pii_patterns


# class Pipeline:
#     """
#     End to end request path:
#     User → Input filter → (RAG retrieval → RAG guard) → LLM → (Tool allow list and schema checks) → Redaction → Output policy → Final
#     Every step writes a structured event to runs/.../events.jsonl through EventLogger
#     This makes runs auditable and easy to reproduce
#     """

#     def __init__(self, ctx, run_dir: str, seeds_path: str, index_dir: str):
#         # Context carries profile settings such as model name, defenses, role, and flags
#         self.ctx = ctx

#         # The event logger writes json lines to the run directory
#         self.log = EventLogger(run_dir)

#         # The LLM client is created once and can run in dry run mode for safe local tests
#         # Dry run returns a fake response and avoids any remote API call
#         self.llm = LLMClient(model_name=ctx.model_name, dry_run=getattr(ctx, "dry_run", True))

#         # The retriever wraps a small BM25 index used for top k retrieval
#         self.retriever = Retriever(index_dir)

#         # PII and secret patterns are loaded once and reused by guard and redactor
#         self.patterns = load_pii_patterns(seeds_path)

#         # Reasonable defaults are used if the context does not define these values
#         self.top_k = getattr(ctx, "top_k", 3)
#         self.use_rag = getattr(ctx, "use_rag", True)


#     # ------------------------------ helpers ---------------------------------
    

#     def _normalize_docs(self, raw: List[Any], k: int) -> List[Dict[str, Any]]:
#         """
#         Normalize retriever outputs to:
#             [{"id": "doc_001.txt", "score": float, "text": "..."}]
#         If text is missing, load it from data/corpus/<id>.
#         """

#         """
#         This helper converts retriever outputs to a standard shape
#         The shape is a list of dicts with id, score, and text fields
#         If text is missing, the helper loads the file from data/corpus
#         This keeps downstream code simple because it can expect one format
#         """

#         docs: List[Dict[str, Any]] = []
#         try:
#             raw = list(raw)[: max(0, int(k))]
#         except Exception:
#             # If something odd comes back, fall back to a small safe slice
#             raw = list(raw)[:3]

#         for item in raw:
#             # Support both dict items and tuple items from different retriever modes
#             if isinstance(item, dict):
#                 doc_id = item.get("id") or item.get("doc_id")
#                 score = float(item.get("score", 0.0))
#                 text = item.get("text")
#             else:
#                 # Tuple case such as (id, score)
#                 try:
#                     doc_id, score = item
#                     score = float(score)
#                 except Exception:
#                     continue
#                 text = None

#             if not isinstance(doc_id, str):
#                 # If the id is not a string then skip this record
#                 continue

#             # If no text was included then read it from the corpus folder
#             if text is None:
#                 doc_path = Path("data/corpus") / doc_id
#                 if doc_path.exists():
#                     try:
#                         text = doc_path.read_text(encoding="utf-8", errors="ignore")
#                     except Exception:
#                         text = ""
#                 else:
#                     text = ""

#             docs.append({"id": doc_id, "score": score, "text": text})
#         return docs

#     def _topk_docs(self, query: str, k: int) -> List[Dict[str, Any]]:
#         # The retriever might expose different method names
#         # This wrapper calls what is available and then normalizes the shape
#         if hasattr(self.retriever, "topk"):
#             raw = self.retriever.topk(query, k)
#         elif hasattr(self.retriever, "search"):
#             raw = self.retriever.search(query)  # may return many so it will be sliced later
#         else:
#             raw = []
#         return self._normalize_docs(raw, k)

#     def _call_llm(self, prompt: str, context_text: str) -> str:
#         # The LLM client injects a safe system message and deterministic settings
#         # Only cleaned prompt and redacted context are sent to the model
#         # === CHANGE: robust fallback to safe refusal if client raises ===
#         try:
#             return self.llm.generate(prompt, context=context_text)
#         except Exception as e:
#             self.log.log("llm_error", error=str(e))
#             return "I'm sorry, but I can't assist with that."

#     def _run_input_filter(self, user_prompt: str, enabled: bool) -> Tuple[bool, str, str]:
#         """ Run the input filter if enabled. 

#         Returns a tuple with blocked flag, reason text, and sanitized prompt. 
#         If anything goes wrong, the function fails open and does not block the user. 
#         All errors are logged for later review.
#         """ 

#         if not enabled:
#             return False, "", user_prompt
#         try:
#             res = apply_input_filter(user_prompt, enabled)
#             # The filter may return a dict or a tuple, so handle both shapes
#             if isinstance(res, dict):
#                 blocked = bool(res.get("blocked", False))
#                 reason = str(res.get("reason", "")) if blocked else ""
#                 sanitized = res.get("prompt", user_prompt)
#                 return blocked, reason, sanitized
#             if isinstance(res, tuple) and len(res) >= 2:
#                 blocked, reason = res[0], res[1]
#                 return bool(blocked), str(reason), user_prompt
#         except Exception as e:
#             # Do not block the user if the filter itself has an error
#             # Record the error so it can be fixed
#             self.log.log("input_filter_error", error=str(e))
#         return False, "", user_prompt


#     # ---------------------- main pipeline -----------------------

#     # -------------------------------------------------------------
#     # Main function: process()
#     """ This is the central function that runs the full defense pipeline. for one user request. It connects all 
#     components — from input filtering to the final safety check — and ensures every defense runs in the correct order.
#     """

#     """ Overall goal:
#     To take a user prompt, safely process it through all defenses, send it to the model, clean the model’s output, 
#     and finally return a safe and traceable response.
#     """
#     #
#     # The process runs in 8 main steps:
#     #
#     # 1) INPUT FILTER:
#     #       The prompt is checked for risky or jailbreak content.
#     #       If the filter finds something dangerous, the request
#     #       is blocked immediately and no model call happens.
#     #
#     # 2) RETRIEVAL (optional):
#     #       If retrieval is enabled, the system searches for the top
#     #       matching documents (top_k) from the local corpus and logs
#     #       their IDs and scores for traceability.
#     #
#     # 3) RAG GUARD / CONTEXT FILTER:
#     #       Before the retrieved documents are passed to the model,
#     #       this step scans them for secrets, PII, or canary markers
#     #       and replaces those with [REDACTED] to prevent leakage.
#     #
#     # 4) MODEL CALL:
#     #       The cleaned prompt and redacted context are then sent to
#     #       the LLM. The model produces its reply based only on safe
#     #       inputs and masked context.
#     #
#     # 5) TOOL ALLOW-LIST + SCHEMA VALIDATION:
#     #       If the model tries to call a “tool:” command (for example,
#     #       requesting a system action), this layer validates it against
#     #       an allow-list and checks the user’s role. Invalid or
#     #       disallowed commands are blocked here.
#     #
#     # 6) COMPOSITION:
#     #       The text from the model and any valid tool results are joined
#     #       into a single string so the next steps see one clean response.
#     #
#     # 7) POST-RESPONSE REDACTION:
#     #       After generation, another redaction pass is run to remove
#     #       any sensitive data that may still appear in the text before
#     #       it reaches the user.
#     #
#     # 8) OUTPUT POLICY ENFORCEMENT:
#     #       The final output is checked against safety policies. If any
#     #       remaining violations are detected (e.g., forbidden URLs or
#     #       code), the message is blocked or cleaned before returning.
#     #
#     # The function logs every stage (input, retrieval, model, redaction,
#     # and output) with detailed metadata so each run can be analyzed or
#     # reproduced later. It returns a dictionary with the final safe
#     # message and a complete defense trace for evaluation.
#     # -------------------------------------------------------------


#     # Main function: process()
#     """ This is the central function that runs the full defense pipeline. for one user request. It connects all 
#     components — from input filtering to the final safety check — and ensures every defense runs in the correct order.
#     """
#     """ Overall goal:
#     To take a user prompt, safely process it through all defenses, send it to the model, clean the model’s output, 
#     and finally return a safe and traceable response.
#     """
#     def process(self, user_prompt: str) -> Dict[str, Any]:
#         # Read defense switches, role, and profile name from the context
#         d = getattr(self.ctx, "defenses", {}) or {}
#         role = getattr(self.ctx, "role", "viewer")
#         profile = getattr(self.ctx, "profile_name", "unknown")

#         # These dicts hold structured information that will be logged and returned
#         defense_trace: Dict[str, Any] = {}
#         retrieval_meta: Dict[str, Any] = {"top_k": self.top_k, "docs": []}

#         # === CHANGE: normalize retrieval enablement ===
#         # If use_rag is False OR top_k <= 0, we will not retrieve.
#         rag_retrieval_on = bool(self.use_rag) and (int(self.top_k) > 0)

#         # EVENT  request received
#         # This creates the first breadcrumb in the trace
#         self.log.log("request", profile=profile, role=role, prompt=user_prompt, defenses=d)



#         # 1) USER → INPUT FILTER
#         # 1) INPUT FILTER:
#         #     The prompt is checked for risky or jailbreak content.
#         #     If the filter finds something dangerous, the request
#         #     is blocked immediately and no model call happens.
#         blocked, reason, prompt_for_model = self._run_input_filter(
#             user_prompt, bool(d.get("input_filter", False))
#         )
#         defense_trace["input_filter"] = {"enabled": bool(d.get("input_filter", False)), "blocked": blocked}
#         if reason:
#             defense_trace["input_filter"]["reason"] = reason

#         if blocked:
#             # If blocked here then nothing reaches the model
#             self.log.log("blocked", stage="input_filter", reason=reason)
#             final = "[BLOCKED] " + (reason or "input matched a restricted pattern")

#             # include retrieval stub for consistency
#             # Return a standard final record so downstream tools can still parse it
#             self.log.log("final", final=final, defense_trace=defense_trace, retrieval=retrieval_meta)
#             return {"final": final, "defense_trace": defense_trace}


#         # 2) (optional) RAG RETRIEVAL
#         # 2) RETRIEVAL (optional):
#         #       If retrieval is enabled, the system searches for the top
#         #       matching documents (top_k) from the local corpus and logs
#         #       their IDs and scores for traceability.
#         docs: List[Dict[str, Any]] = []
#         if rag_retrieval_on:
#             docs = self._topk_docs(prompt_for_model, self.top_k)
#             retrieval_meta["docs"] = [{"id": d0["id"], "score": d0["score"], "source": d0.get("id")} for d0 in docs]
#             self.log.log("retrieved", top_k=self.top_k, docs=retrieval_meta["docs"]) 


#         # 3) RAG GUARD / CONTEXT ASSEMBLY
#         # 3) RAG GUARD / CONTEXT FILTER:
#         #       Before the retrieved documents are passed to the model,
#         #       this step scans them for secrets, PII, or canary markers
#         #       and replaces those with [REDACTED] to prevent leakage.
#         rag_enabled = bool(d.get("rag_filter", False))

#         # === CHANGE: build a clear before/after context preview for evidence ===
#         raw_context = ""
#         if docs:
#             try:
#                 # join with doc headers to make diffs obvious; keep bounded size
#                 chunks = []
#                 for dd in docs:
#                     head = f"[{dd.get('id','?')}] "
#                     txt = (dd.get("text") or "")[:800]  # per-doc cap
#                     chunks.append(head + txt)
#                 raw_context = "\n---\n".join(chunks)
#             except Exception:
#                 raw_context = ""

#         context_text, mask_meta = guard_context(docs, self.patterns, rag_enabled)
#         masked_in_context = int(mask_meta.get("masked_total", 0)) if rag_enabled else 0

#         #masked_in_context = context_text.count("[REDACTED]") if rag_enabled else 0

#         # Previews are truncated to keep logs safe/compact
#         preview_before = (raw_context or "")[:600]
#         preview_after  = (context_text or "")[:600] if rag_enabled else (raw_context or "")[:600]

#         defense_trace["rag_filter"] = {
#             "enabled": rag_enabled,
#             "masked": masked_in_context,
#             # === CHANGE: previews support your Chapter 5 evidence tables ===
#             "by_type": mask_meta.get("by_type", {}),
#             "docs": mask_meta.get("docs", []),
#             "preview_before": preview_before,
#             "preview_after": preview_after,
#         }


#         # 4) MODEL CALL
#         # 4) MODEL CALL:
#         #       The cleaned prompt and redacted context are then sent to
#         #       the LLM. The model produces its reply based only on safe
#         #       inputs and masked context.
#         reply = self._call_llm(prompt_for_model, context_text)
#         self.log.log("llm_output", text=reply)

#         # Demo-only hook to force post-LLM masking (opt-in via config)
#         if bool(d.get("postmask_echo", False)):
#             reply = prompt_for_model
#             defense_trace.setdefault("llm", {})["demo_mode"] = "echo_prompt"
#             self.log.log("llm_output_demo_echo", note="postmask_echo enabled; using prompt as reply")













#         # 5) TOOL ALLOW-LIST + SCHEMA VALIDATION
#          # 5) TOOL ALLOW-LIST + SCHEMA VALIDATION
#         # 5) TOOL ALLOW-LIST + SCHEMA VALIDATION:
#         #       If the model tries to call a “tool:” command (for example,
#         #       requesting a system action), this layer validates it against
#         #       an allow-list and checks the user’s role. Invalid or
#         #       disallowed commands are blocked here.
#         tool_enabled = bool(d.get("tool_allowlist", False))

#         # Prefer a tool line from the model reply; if absent (e.g., dry-run/echo)
#         # fall back to the user prompt first line so demos are captured as attempts.
#         reply_first  = (reply or "").lstrip().splitlines()[0] if reply else ""
#         prompt_first = (prompt_for_model or "").lstrip().splitlines()[0] if prompt_for_model else ""

#         is_tool_in_reply  = reply_first.lower().startswith("tool:")
#         is_tool_in_prompt = (not is_tool_in_reply) and prompt_first.lower().startswith("tool:")

#         tool_line = reply_first if is_tool_in_reply else (prompt_first if is_tool_in_prompt else "")
#         tool_src  = "reply" if is_tool_in_reply else ("prompt" if is_tool_in_prompt else "")

#         defense_trace["tool_allowlist"] = {
#             "enabled": tool_enabled,
#             "attempted": bool(tool_line),
#             "executed": False,
#             "allowed": False,
#             "blocked": False,
#             "source": tool_src,   # where we got the line from
#         }

#         # tiny fallback parser so we can include {name,args} in the log even if runtime omits them
#         def _fallback_parse(line: str):
#             name, args = "(unknown)", {}
#             try:
#                 if line.lower().startswith("tool:"):
#                     tail = line.split(":", 1)[1].strip()
#                     parts = tail.split()
#                     if parts:
#                         name = parts[0]
#                         for p in parts[1:]:
#                             if "=" in p:
#                                 k, v = p.split("=", 1)
#                                 args[k] = v.strip().strip('"')
#             except Exception:
#                 pass
#             return name, args

#         tool_text = ""
#         self.log.log("tool_decision_debug",
#              tool_line=tool_line,
#              tool_src=tool_src,
#              tool_enabled=tool_enabled)
        
#         if tool_line:
#             if not tool_enabled:
#                 # LLM proposed a tool, but allowlist is OFF → treat as blocked
#                 defense_trace["tool_allowlist"]["blocked"] = True
#                 name, args = _fallback_parse(tool_line)
#                 self.log.log(
#                     "tool_blocked",
#                     name=name,
#                     args=args,
#                     role=role,
#                     source=tool_src or "unknown",
#                     reason="allowlist_disabled",
#                     attempted=True,
#                     executed=False,
#                     allowed=False,
#                     blocked=True,
#                     line=tool_line,
#                 )
#             else:
#                 try:
#                     # maybe_execute_tool handles parsing, role checks, and argument validation
#                     tool_res = maybe_execute_tool(tool_line, role, True)  # dict; should include name/args/effect/allowed/blocked
#                     attempted = bool(tool_res.get("attempted", True))
#                     allowed   = bool(tool_res.get("allowed", False))
#                     blocked   = bool(tool_res.get("blocked", not allowed))
#                     executed  = attempted

#                     defense_trace["tool_allowlist"].update({
#                         "attempted": attempted,
#                         "executed": executed,
#                         "allowed": allowed,
#                         "blocked": blocked,
#                     })

#                     if tool_res.get("effect"):
#                         tool_text = str(tool_res["effect"])

#                     # Ensure we always emit a concrete event you can jq later
#                     payload = {
#                         "name": tool_res.get("name", _fallback_parse(tool_line)[0]),
#                         "args": tool_res.get("args", _fallback_parse(tool_line)[1]),
#                         "role": role,
#                         "source": tool_src or "unknown",
#                         "attempted": attempted,
#                         "executed": executed,
#                         "allowed": allowed,
#                         "blocked": blocked,
#                         "line": tool_line,
#                     }
#                     if blocked:
#                         payload["reason"] = tool_res.get("reason", "blocked")
#                         self.log.log("tool_blocked", **payload)
#                     else:
#                         self.log.log("tool_result", **payload)

#                 except Exception as e:
#                     # Any failure here is recorded and the tool is considered blocked
#                     defense_trace["tool_allowlist"]["blocked"] = True
#                     name, args = _fallback_parse(tool_line)
#                     self.log.log(
#                         "tool_blocked",
#                         name=name,
#                         args=args,
#                         role=role,
#                         source=tool_src or "unknown",
#                         reason=str(e),
#                         attempted=True,
#                         executed=False,
#                         allowed=False,
#                         blocked=True,
#                         line=tool_line,
#                     )

#         # # 5) TOOL ALLOW-LIST + SCHEMA VALIDATION:
#         # #       If the model tries to call a “tool:” command (for example,
#         # #       requesting a system action), this layer validates it against
#         # #       an allow-list and checks the user’s role. Invalid or
#         # #       disallowed commands are blocked here.
#         # tool_enabled = bool(d.get("tool_allowlist", False))
#         # # Prefer a tool line from the model reply; if absent (e.g., dry-run/echo)
#         # # fall back to the user prompt first line so demos are captured as attempts.
#         # reply_first  = (reply or "").lstrip().splitlines()[0] if reply else ""
#         # prompt_first = (prompt_for_model or "").lstrip().splitlines()[0] if prompt_for_model else ""

#         # is_tool_in_reply  = reply_first.lower().startswith("tool:")
#         # is_tool_in_prompt = (not is_tool_in_reply) and prompt_first.lower().startswith("tool:")

#         # tool_line = reply_first if is_tool_in_reply else (prompt_first if is_tool_in_prompt else "")
#         # tool_src  = "reply" if is_tool_in_reply else ("prompt" if is_tool_in_prompt else "")


#         # #first_line = (reply or "").lstrip().splitlines()[0] if reply else ""
#         # #is_tool_line = first_line.lower().startswith("tool:")

#         # defense_trace["tool_allowlist"] = {
#         #     "enabled": tool_enabled,
#         #     "attempted": bool(tool_line),
#         #     "executed": False,
#         #     "allowed": False,
#         #     "blocked": False,
#         #     "source": tool_src,
#         # }

#         # tool_text = ""
#         # if tool_line:
#         #     if not tool_enabled:
#         #         # LLM proposed a tool, but allowlist is OFF → treat as blocked
#         #         defense_trace["tool_allowlist"]["blocked"] = True
#         #         self.log.log("tool_blocked", reason="allowlist_disabled", role=role, line=tool_line, source=tool_src)
#         #     else:
#         #         try:
#         #             # maybe_execute_tool handles parsing, role checks, and argument validation
#         #             tool_res = maybe_execute_tool(tool_line, role, True)
#         #             # tool_res is always a dict from our runtime
#         #             # Update the trace with the outcome in a consistent shape
#         #             defense_trace["tool_allowlist"].update({
#         #                 "attempted": bool(tool_res.get("attempted")),
#         #                 "executed": bool(tool_res.get("attempted")),
#         #                 "allowed": bool(tool_res.get("allowed")),
#         #                 "blocked": bool(tool_res.get("blocked")),
#         #             })   
#         #             # If the tool produced user visible text then append it later
#         #             if tool_res.get("effect"):
#         #                 tool_text = str(tool_res["effect"])
#         #             # log either a blocked or successful/result tool event
#         #             if tool_res.get("reason"):
#         #                 self.log.log("tool_blocked", source=tool_src, **tool_res)
#         #             else:
#         #                 self.log.log("tool_result", source=tool_src, **tool_res)
#         #         except Exception as e:
#         #             # Any failure here is recorded and the tool is considered blocked
#         #             defense_trace["tool_allowlist"]["blocked"] = True
#         #             self.log.log("tool_error", error=str(e), line=tool_line, source=tool_src)







#         # 6) COMPOSE
#         # 6) COMPOSITION:
#         #       The text from the model and any valid tool results are joined
#         #       into a single string so the next steps see one clean response.
#         combined = reply if isinstance(reply, str) else ""
#         if tool_text:
#             combined = (combined + "\n" + tool_text).strip()


#         # 7) POST-RESPONSE REDACTION
#         # 7) POST-RESPONSE REDACTION:
#         #       After generation, another redaction pass is run to remove
#         #       any sensitive data that may still appear in the text before
#         #       it reaches the user.
#         redaction_enabled = bool(d.get("redaction", False))
#         final = redact(combined, self.patterns, redaction_enabled)
#         masked_count = final.count("[REDACTED]") if isinstance(final, str) else 0
#         defense_trace["redaction"] = {"enabled": redaction_enabled, "masked": masked_count}


#         # 8) OUTPUT POLICY ENFORCEMENT (final safety net)
#         # 8) OUTPUT POLICY ENFORCEMENT:
#         #       The final output is checked against safety policies. If any
#         #       remaining violations are detected (e.g., forbidden URLs or
#         #       code), the message is blocked or cleaned before returning.
#         output_policy_enabled = bool(d.get("output_policy", False))
#         if output_policy_enabled:
#             pol_enforced = enforce_output_policy(final, self.patterns)
#             if pol_enforced.get("blocked"):
#                 self.log.log("output_policy_violation", violations=pol_enforced.get("violations", []))
#                 final = pol_enforced.get("final", final)
#             defense_trace["output_policy"] = {
#                 "enabled": True,
#                 "blocked": bool(pol_enforced.get("blocked")),
#                 "violations": pol_enforced.get("violations", []),
#             }
#         else:
#             defense_trace["output_policy"] = {"enabled": False, "blocked": False, "violations": []}

#         # EVENT: final (include retrieval meta so downstream jq can inspect it)
#         self.log.log(
#             "final",
#             role=role,
#             profile=profile,
#             context_len=len((context_text or "")),
#             final=final,
#             defense_trace=defense_trace,
#             retrieval=retrieval_meta,
#         )
#         return {"final": final, "defense_trace": defense_trace}
#     # The function logs every stage (input, retrieval, model, redaction,
#     # and output) with detailed metadata so each run can be analyzed or
#     # reproduced later. It returns a dictionary with the final safe
#     # message and a complete defense trace for evaluation.

#     def close(self):
#         self.log.close()










        # # 5) Tool gate
        # # 5) TOOL ALLOW-LIST + SCHEMA VALIDATION
        # # # 5) TOOL ALLOW-LIST + SCHEMA VALIDATION
        # # 5) Tool gate
        # tool_enabled = bool(d.get("tool_allowlist", False))

        # # Accept tool: anywhere on the line (not just column 0)
        # TOOL_ANYWHERE_RE = re.compile(r'\btool\s*:\s*([A-Za-z0-9_-]+)\b', re.IGNORECASE)

        # def _first_lines(s: str, n: int = 12) -> list:
        #     if not s:
        #         return []
        #     s = s.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
        #     lines = s.split("\n")
        #     out = []
        #     for ln in lines[: max(0, n)]:
        #         out.append(ln.lstrip("\ufeff"))
        #     return out

        # def _extract_tool_line(lines: list) -> str:
        #     """
        #     Find the first line that contains 'tool:' and return the substring
        #     starting at 'tool:' (trim leading spaces), e.g.
        #        'Execute: tool: add_admin user=bob'
        #     -> 'tool: add_admin user=bob'
        #     """
        #     for ln in lines or []:
        #         if not ln:
        #             continue
        #         matches = list(TOOL_ANYWHERE_RE.finditer(ln))
        #         if matches:
        #             start = matches[-1].start()  # take the last 'tool:' match
        #             return ln[start:].lstrip()
        #     return ""

        # reply_lines  = _first_lines(reply, 12)
        # prompt_lines = _first_lines(prompt_for_model, 12)
        # raw_lines    = _first_lines(user_prompt, 12)

        # reply_tool_line  = _extract_tool_line(reply_lines)
        # prompt_tool_line = _extract_tool_line(prompt_lines)
        # raw_tool_line    = _extract_tool_line(raw_lines)

        # # Prefer reply, else sanitized prompt, else raw
        # tool_line = ""
        # tool_src  = ""
        # if reply_tool_line:
        #     tool_line = reply_tool_line
        #     tool_src = "reply"
        # elif prompt_tool_line:
        #     tool_line = prompt_tool_line
        #     tool_src = "prompt"
        # elif raw_tool_line:
        #     tool_line = raw_tool_line
        #     tool_src = "raw"

        # # Debug: show first lines scanned
        # self._elog(
        #     "tool_scan_lines_debug",
        #     reply_first_lines=reply_lines[:6],
        #     prompt_first_lines=prompt_lines[:6],
        #     raw_first_lines=raw_lines[:6],
        # )

        # # Single authoritative snapshot
        # self._elog(
        #     "tool_scan_snapshot",
        #     reply_has=bool(reply_tool_line),
        #     prompt_has=bool(prompt_tool_line),
        #     raw_has=bool(raw_tool_line),
        #     reply_tool_line=reply_tool_line,
        #     prompt_tool_line=prompt_tool_line,
        #     raw_tool_line=raw_tool_line,
        #     reply_preview=(reply or "")[:80] + ("..." if len(reply or "") > 80 else ""),
        #     prompt_preview=(prompt_for_model or "")[:80] + ("..." if len(prompt_for_model or "") > 80 else ""),
        #     raw_preview=(user_prompt or "")[:80] + ("..." if len(user_prompt or "") > 80 else ""),
        #     tool_enabled=tool_enabled,
        # )

        # if tool_line:
        #     self._elog("tool_gate_reached", source=tool_src, tool_line=tool_line)

        # defense_trace["tool_allowlist"] = {
        #     "enabled": tool_enabled,
        #     "attempted": bool(tool_line),
        #     "executed": False,
        #     "allowed": False,
        #     "blocked": False,
        #     "source": tool_src,
        # }

        # # Tiny fallback parser for logging (unchanged)
        # def _fallback_parse(line: str):
        #     name, args = "(unknown)", {}
        #     try:
        #         if line:
        #             tail = line.split(":", 1)[1].strip()
        #             if tail.lower().startswith("tool:"):
        #                 tail = tail.split(":", 1)[1].strip()

        #             parts = tail.split()
        #             if parts:
        #                 name = parts[0]
        #                 if name.lower() == "tool:":
        #                     parts = parts[1:]
        #                     name = parts[0] if parts else "(unknown)"
        #                 for p in parts[1:]:
        #                     if "=" in p:
        #                         k, v = p.split("=", 1)
        #                         args[k] = v.strip().strip('"')
        #     except Exception:
        #         pass
        #     return name, args

        # self._elog("tool_decision_debug", tool_line=tool_line, tool_src=tool_src, tool_enabled=tool_enabled)

        # tool_text = ""
        # if tool_line:
        #     n0, a0 = _fallback_parse(tool_line)
        #     self._elog("tool_attempt", name=n0, args=a0, role=role, source=tool_src or "unknown", line=tool_line)

        #     if not tool_enabled:
        #         defense_trace["tool_allowlist"]["blocked"] = True
        #         self._elog(
        #             "tool_blocked",
        #             name=n0, args=a0, role=role, source=tool_src or "unknown",
        #             reason="allowlist_disabled",
        #             attempted=True, executed=False, allowed=False, blocked=True,
        #             line=tool_line,
        #         )
        #     else:
        #         try:
        #             tool_res = maybe_execute_tool(tool_line, role, True)
        #             attempted    = bool(tool_res.get("attempted", True))
        #             allowed      = bool(tool_res.get("allowed", False))
        #             blocked_flag = bool(tool_res.get("blocked", not allowed))
        #             executed     = attempted

        #             defense_trace["tool_allowlist"].update({
        #                 "attempted": attempted,
        #                 "executed": executed,
        #                 "allowed": allowed,
        #                 "blocked": blocked_flag,
        #             })

        #             if tool_res.get("effect"):
        #                 tool_text = str(tool_res["effect"])

        #             payload = {
        #                 "name": tool_res.get("name", n0),
        #                 "args": tool_res.get("args", a0),
        #                 "role": role,
        #                 "source": tool_src or "unknown",
        #                 "attempted": attempted,
        #                 "executed": executed,
        #                 "allowed": allowed,
        #                 "blocked": blocked_flag,
        #                 "line": tool_line,
        #             }
        #             if blocked_flag:
        #                 payload["reason"] = tool_res.get("reason", "blocked")
        #                 self._elog("tool_blocked", **payload)
        #             else:
        #                 self._elog("tool_result", **payload)
        #         except Exception as e:
        #             defense_trace["tool_allowlist"]["blocked"] = True
        #             self._elog(
        #                 "tool_blocked",
        #                 name=n0, args=a0, role=role, source=tool_src or "unknown",
        #                 reason=str(e),
        #                 attempted=True, executed=False, allowed=False, blocked=True,
        #                 line=tool_line,
        #             )

