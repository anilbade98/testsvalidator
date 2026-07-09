"""HTTP-based question runner for the Analytics Agent Validator.

Sends questions to the service's SSE streaming endpoint (POST /api/v1/chat/stream),
captures all events, builds per-question records, and saves results incrementally.

Auth:
  1. EVAL_BEARER_TOKEN env var (override, no auto-refresh)
  2. get_auth(env) via Secrets Manager — same path used by integration tests
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from tests.validator.trace_capture import build_record


VALID_FLOWS = {"fdl", "od", "ao", "rca", "advisory", "mixed", "context", "refresh", "conversational"}

# Flows that call at least one data tool — used for CI smoke runs
TOOL_FLOWS = {"fdl", "od", "ao", "rca", "advisory", "mixed"}

MAX_CONVERSATION_TURNS = 5
BATCH_CLARIFICATION_FALLBACK = (
    "No further information will be given. Do not ask any further questions. "
    "Use your best judgement and proceed with reasonable defaults."
)
_AUTH_MARKERS = ("401", "Unauthorized", "Forbidden", "ExpiredToken", "NoCredentials")


# ── Auth ──────────────────────────────────────────────────────────────────────

def _decode_tenant_from_jwt(token: str) -> str:
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        return ""
    for key in ("tenantId", "tenant_id", "tid"):
        if payload.get(key):
            return str(payload[key])
    return ""


def resolve_auth(env: str | None = None) -> dict:
    """Resolve auth credentials. EVAL_BEARER_TOKEN overrides Secrets Manager."""
    bearer = os.environ.get("EVAL_BEARER_TOKEN", "").strip()
    if bearer:
        tenant_id = os.environ.get("EVAL_TENANT_ID", "").strip() or _decode_tenant_from_jwt(bearer)
        return {"token": bearer, "tenant_id": tenant_id, "_bearer_override": True}

    from tests.utils.auth_helpers import get_auth
    if env:
        os.environ["AWS_PROFILE"] = f"{env}-illum"
    return get_auth()


def _refresh_auth(auth: dict, env: str | None) -> bool:
    """Re-authenticate and mutate auth in-place.

    Tries three paths in order:
    1. EVAL_AUTH_EMAIL / EVAL_AUTH_PASSWORD env vars (set by the Streamlit UI
       when the user logged in with email+password — enables refresh for long runs).
    2. Secrets Manager via resolve_auth() when --env is provided.
    3. Cannot refresh (bearer-only mode).
    """
    email = os.environ.get("EVAL_AUTH_EMAIL", "").strip()
    password = os.environ.get("EVAL_AUTH_PASSWORD", "").strip()
    auth_env = os.environ.get("EVAL_AUTH_ENV", "").strip()

    if email and password and auth_env:
        try:
            url = f"https://na1.{auth_env}.nice-incontact.com/public/authentication/v1/login"
            resp = requests.post(url, json={"email": email, "password": password}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            token = data.get("token", "")
            if not token:
                raise ValueError("Empty token from auth API")
            auth["token"] = token
            auth["_bearer_override"] = False  # allow future refreshes
            if data.get("tenantId"):
                auth["tenant_id"] = data["tenantId"]
            os.environ["EVAL_BEARER_TOKEN"] = token
            return True
        except Exception as e:
            print(f"  Token refresh via email+password failed: {e}")
            return False

    if auth.get("_bearer_override"):
        return False

    try:
        fresh = resolve_auth(env=env)
        auth["token"] = fresh["token"]
        if fresh.get("tenant_id"):
            auth["tenant_id"] = fresh["tenant_id"]
        return True
    except Exception as e:
        print(f"  Token refresh failed: {e}")
        return False


# ── SSE streaming ─────────────────────────────────────────────────────────────

def _stream_request(
    base_url: str,
    auth: dict,
    *,
    text: str,
    session_id: str,
    refresh: bool = False,
    context: dict | None = None,
    reference_date: str | None = None,
) -> list[dict]:
    """POST to /api/v1/chat/stream and collect all SSE events."""
    url = f"{base_url.rstrip('/')}/api/v1/chat/stream"
    payload: dict[str, Any] = {"text": text, "sessionId": session_id}
    if refresh:
        payload["refresh"] = True
    if context:
        payload["context"] = context
    if reference_date:
        payload["configuration"] = {"REFERENCE_DATE": reference_date}

    headers = {
        "Authorization": f"Bearer {auth['token']}",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }

    events: list[dict] = []
    t0 = time.time()
    with requests.post(url, json=payload, headers=headers, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        event_type: str | None = None
        for raw_line in resp.iter_lines(decode_unicode=True):
            if raw_line.startswith("event:"):
                event_type = raw_line[len("event:"):].strip()
            elif raw_line.startswith("data:"):
                data_str = raw_line[len("data:"):].strip()
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    data = {"raw": data_str}
                events.append({
                    "t_ms": int((time.time() - t0) * 1000),
                    "event": event_type or "unknown",
                    "data": data,
                })
                event_type = None
            elif not raw_line:
                event_type = None

    return events


def _has_auth_error(events: list[dict], error: str | None) -> bool:
    if error and any(m in error for m in _AUTH_MARKERS):
        return True
    for ev in events:
        if any(m in str(ev) for m in _AUTH_MARKERS):
            return True
    return False


def _extract_text(events: list[dict]) -> str:
    for ev in reversed(events):
        if ev.get("event") == "result":
            return (ev.get("data") or {}).get("text", "") or ""
    chunks = [
        (ev.get("data") or {}).get("text", "")
        for ev in events
        if ev.get("event") == "text"
    ]
    return "".join(chunks).strip()


# ── Single-question runner ────────────────────────────────────────────────────

def _run_one(
    question_meta: dict,
    auth: dict,
    base_url: str,
    *,
    run_id: str,
    env: str | None = None,
    tunnel_pool=None,
    reference_date: str | None = None,
) -> dict:
    qid = question_meta["id"]
    session_id = f"validator-{run_id}-{qid}"
    flow = question_meta.get("flow", "")

    refresh = flow == "refresh"
    context = question_meta.get("context_fixture") if flow in ("refresh", "context") else None

    # Ensure required tunnels are alive just before the HTTP call.
    # A tunnel setup failure (e.g. unconfigured hostname) records an error for
    # this question and returns early — it must not crash the whole run.
    if tunnel_pool is not None:
        expected_tools = question_meta.get("expected_tools") or []
        if expected_tools:
            try:
                tunnel_pool.ensure_for(expected_tools)
            except Exception as tunnel_err:
                return build_record(
                    events=[],
                    question_meta=question_meta,
                    duration_ms=0,
                    error=f"Tunnel setup failed: {tunnel_err}",
                )

    all_events: list[dict] = []
    clarification_questions: list[str] = []
    turn_count = 0
    current_text = question_meta["question"]
    t0 = time.time()
    error: str | None = None

    # Flows that complete in a single round-trip — break after the first response
    # regardless of whether tool calls were detected in the stream.
    # refresh: handled server-side by refresh_handler.py which yields a single
    #   `result` event directly (no tool_start/sub_tool_start emitted).
    # conversational: simple text exchange, no data tools expected.
    _SINGLE_TURN_FLOWS = {"conversational", "refresh"}

    try:
        for turn_idx in range(MAX_CONVERSATION_TURNS):
            turn_events = _stream_request(
                base_url, auth,
                text=current_text,
                session_id=session_id,
                refresh=refresh and turn_idx == 0,
                context=context if turn_idx == 0 else None,
                reference_date=reference_date,
            )
            enriched = [{"turn_idx": turn_idx, **ev} for ev in turn_events]
            all_events.extend(enriched)
            turn_count = turn_idx + 1

            made_tool_calls = any(
                ev.get("event") in ("tool_start", "sub_tool_start")
                for ev in enriched
            )
            if made_tool_calls or flow in _SINGLE_TURN_FLOWS:
                break

            clarification_questions.append(_extract_text(enriched))
            if turn_idx == MAX_CONVERSATION_TURNS - 1:
                break
            current_text = BATCH_CLARIFICATION_FALLBACK
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    duration_ms = int((time.time() - t0) * 1000)

    return build_record(
        events=all_events,
        question_meta=question_meta,
        duration_ms=duration_ms,
        error=error,
        clarification_questions=clarification_questions,
        turn_count=turn_count,
    )


# ── Question selection ────────────────────────────────────────────────────────

def select_questions(
    questions: list[dict],
    *,
    flow: str | None = None,
    tags: list[str] | None = None,
    complexity: str | None = None,
    limit: int | None = None,
    limit_per_flow: int | None = None,
    ids: list[int] | None = None,
    priority_only: bool = False,
    tool_flows_only: bool = False,
) -> list[dict]:
    # --ids bypasses flow / tags / complexity — returns exactly the requested questions.
    if ids:
        id_set = set(ids)
        pool = [q for q in questions if q.get("id") in id_set]
        if limit is not None and limit > 0:
            pool = pool[:limit]
        return pool

    pool = questions
    if priority_only:
        pool = [q for q in pool if q.get("priority")]
    if tool_flows_only:
        pool = [q for q in pool if q.get("flow") in TOOL_FLOWS]
    if flow:
        pool = [q for q in pool if q.get("flow") == flow]
    if tags:
        tag_set = set(tags)
        pool = [q for q in pool if tag_set.intersection(q.get("tags") or [])]
    if complexity:
        pool = [q for q in pool if q.get("complexity") == complexity]
    if limit_per_flow is not None and limit_per_flow > 0:
        seen: dict[str, int] = {}
        filtered = []
        for q in pool:
            f = q.get("flow", "")
            if seen.get(f, 0) < limit_per_flow:
                filtered.append(q)
                seen[f] = seen.get(f, 0) + 1
        pool = filtered
    if limit is not None and limit > 0:
        pool = pool[:limit]
    return pool


# ── Persistence helpers ───────────────────────────────────────────────────────

def _build_run_metadata(*, run_id: str, questions: list[dict], filter_config: dict,
                        base_url: str, env: str | None) -> dict:
    try:
        from app.config import config as settings
        model_id = settings.bedrock_model_id
    except Exception:
        model_id = None
    return {
        "run_id": run_id,
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "env": env,
        "base_url": base_url,
        "model_id": model_id,
        "total": len(questions),
        "filter_config": filter_config,
    }


def _save(out_path: Path, metadata: dict, results: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({**metadata, "results": results}, indent=2, default=str))


def _summarize(r: dict) -> str:
    od_fdl = r.get("od_fdl_results") or []
    usage = r.get("usage") or {}
    return (
        f"  qid={r['question_id']:3d} | flow={r.get('flow', '?'):12s}"
        f" | {r['duration_ms']/1000:.1f}s"
        f" | turns={r.get('turn_count', 1)}"
        f" | sql_results={len(od_fdl)}"
        f" | tokens={usage.get('total_tokens', 0)}"
        f" (in={usage.get('input_tokens', 0)} out={usage.get('output_tokens', 0)}"
        f" cr={usage.get('cache_read_tokens', 0)} cw={usage.get('cache_creation_tokens', 0)})"
        f" | err={r.get('error')}"
    )


# ── Batch runner ──────────────────────────────────────────────────────────────

async def run_all(
    questions: list[dict],
    auth: dict,
    *,
    base_url: str,
    out_path: Path,
    filter_config: dict,
    env: str | None = None,
    tunnel_pool=None,
    workers: int = 1,
    reference_date: str | None = None,
) -> dict:
    run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    metadata = _build_run_metadata(
        run_id=run_id, questions=questions, filter_config=filter_config,
        base_url=base_url, env=env,
    )
    total = len(questions)

    if workers <= 1:
        # Sequential path — incremental saves, clear progress output.
        results: list[dict] = []
        for i, q in enumerate(questions, 1):
            print(f"\n[{i}/{total}] q{q['id']} [{q.get('flow', '?')}] ({q['complexity']}): {q['question']}")
            record = _run_one(q, auth, base_url, run_id=run_id, env=env, tunnel_pool=tunnel_pool, reference_date=reference_date)
            if _has_auth_error(record.get("od_fdl_results") or [], record.get("error")):
                print("  Auth error — refreshing token and retrying ...")
                if _refresh_auth(auth, env):
                    print(f"  Refreshed (tenant_id={auth.get('tenant_id')!r}), retrying ...")
                    record = _run_one(q, auth, base_url, run_id=run_id, env=env, tunnel_pool=tunnel_pool, reference_date=reference_date)
                else:
                    print("  Cannot refresh. Continuing with stale token.")
            results.append(record)
            print(_summarize(record))
            _save(out_path, metadata, results)
        return {**metadata, "results": results}

    # Parallel path — run up to `workers` questions concurrently.
    # Results are kept in insertion order; the file is saved after every
    # completion so partial results are always on disk.
    sem = asyncio.Semaphore(workers)
    auth_lock = asyncio.Lock()   # serialise token refresh across workers
    save_lock = asyncio.Lock()   # serialise file writes
    completed: list[dict] = []
    done_count = 0

    async def _run_async(idx: int, q: dict) -> None:
        nonlocal done_count
        async with sem:
            record = await asyncio.to_thread(
                _run_one, q, auth, base_url,
                run_id=run_id, env=env, tunnel_pool=tunnel_pool, reference_date=reference_date,
            )
            if _has_auth_error(record.get("od_fdl_results") or [], record.get("error")):
                async with auth_lock:
                    refreshed = _refresh_auth(auth, env)
                if refreshed:
                    record = await asyncio.to_thread(
                        _run_one, q, auth, base_url,
                        run_id=run_id, env=env, tunnel_pool=tunnel_pool, reference_date=reference_date,
                    )
            async with save_lock:
                done_count += 1
                print(f"\n[{done_count}/{total}] q{q['id']} [{q.get('flow', '?')}] ({q['complexity']}): {q['question']}")
                print(_summarize(record))
                completed.append(record)
                _save(out_path, metadata, completed)

    await asyncio.gather(*[_run_async(i, q) for i, q in enumerate(questions)])
    return {**metadata, "results": completed}


def load_questions(path: Path) -> list[dict]:
    return json.loads(path.read_text())
