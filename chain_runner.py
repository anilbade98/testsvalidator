"""Chain runner for multi-turn conversation validation.

Loads test_chains.json, executes each chain by sending turns sequentially
within a shared session, and captures per-turn records.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from tests.validator.runner import _stream_request, _extract_text, _has_auth_error, _refresh_auth
from tests.validator.trace_capture import _extract_cortex_traces, _aggregate_usage, _streaming_events_summary, _tools_called, _extract_tool_results


HARNESS_DIR = Path(__file__).resolve().parent
CHAINS_PATH = HARNESS_DIR / "test_chains.json"


# ── Record building ───────────────────────────────────────────────────────────

def _build_turn_record(
    turn_meta: dict,
    turn_index: int,
    events: list[dict],
    duration_ms: int,
    error: str | None,
) -> dict:
    cortex_traces = _extract_cortex_traces(events)
    tool_results = _extract_tool_results(events)
    record: dict = {
        "turn_index": turn_index,
        "question": turn_meta.get("question", ""),
        "expected_tools": turn_meta.get("expected_tools", []),
        "tools_called": _tools_called(events),
        "cortex_traces": cortex_traces,
        "streaming_events": _streaming_events_summary(events),
        "agent_response": _extract_text(events),
        "duration_ms": duration_ms,
        "usage": _aggregate_usage(events),
        "error": error,
        "eval": None,
    }
    if tool_results:
        record["tool_results"] = tool_results
    return record


def _build_chain_record(
    chain_meta: dict,
    run_id: str,
    turn_records: list[dict],
) -> dict:
    return {
        "chain_id": chain_meta.get("chain_id"),
        "name": chain_meta.get("name", ""),
        "flow": chain_meta.get("flow", ""),
        "priority": bool(chain_meta.get("priority", False)),
        "run_id": run_id,
        "turns": turn_records,
        "chain_eval": None,
    }


# ── Execution ─────────────────────────────────────────────────────────────────

def _run_chain(
    chain: dict,
    auth: dict,
    base_url: str,
    *,
    run_id: str,
    env: str | None = None,
    tunnel_pool=None,
) -> dict:
    """Execute all turns of a single chain within one shared session."""
    import time

    chain_id = chain.get("chain_id")
    session_id = f"validator-chain-{run_id}-{chain_id}"
    turns = chain.get("turns") or []
    turn_records: list[dict] = []

    for turn_index, turn_meta in enumerate(turns):
        # Start tunnels for this turn if needed
        if tunnel_pool is not None:
            expected_tools = turn_meta.get("expected_tools") or []
            if expected_tools:
                tunnel_pool.ensure_for(expected_tools)

        t0 = time.time()
        error: str | None = None
        events: list[dict] = []

        try:
            events = _stream_request(
                base_url, auth,
                text=turn_meta["question"],
                session_id=session_id,
            )
        except Exception as e:
            error = f"{type(e).__name__}: {e}"

        duration_ms = int((time.time() - t0) * 1000)
        record = _build_turn_record(turn_meta, turn_index, events, duration_ms, error)
        turn_records.append(record)

    return _build_chain_record(chain, run_id, turn_records)


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_all_chains(
    chains: list[dict],
    auth: dict,
    *,
    base_url: str,
    out_path: Path,
    env: str | None = None,
    tunnel_pool=None,
) -> dict:
    run_id = f"chains_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    results: list[dict] = []
    total = len(chains)

    for i, chain in enumerate(chains, 1):
        cid = chain.get("chain_id")
        name = chain.get("name", "")
        n_turns = len(chain.get("turns") or [])
        print(f"\n[{i}/{total}] chain{cid} [{chain.get('flow','?')}] {n_turns} turns: {name}")

        record = _run_chain(chain, auth, base_url, run_id=run_id, env=env, tunnel_pool=tunnel_pool)

        if any(_has_auth_error([], t.get("error")) for t in record["turns"]):
            print("  Auth error - refreshing token and retrying ...")
            if _refresh_auth(auth, env):
                record = _run_chain(chain, auth, base_url, run_id=run_id, env=env, tunnel_pool=tunnel_pool)

        results.append(record)
        _summarize_chain(record)
        _save_chains(out_path, run_id, results)

    return {"run_id": run_id, "total": len(results), "results": results}


def _summarize_chain(record: dict) -> None:
    for t in record["turns"]:
        traces = t.get("cortex_traces") or []
        usage = t.get("usage") or {}
        print(
            f"  turn {t['turn_index']}"
            f" | {t['duration_ms']/1000:.1f}s"
            f" | sql_calls={len(traces)}"
            f" | tokens={usage.get('total_tokens', 0)}"
            f" | err={t.get('error')}"
        )


def _save_chains(out_path: Path, run_id: str, results: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "run_id": run_id,
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "total": len(results),
        "results": results,
    }, indent=2, default=str))


# ── Catalog loading ───────────────────────────────────────────────────────────

def load_chains(path: Path) -> list[dict]:
    return json.loads(path.read_text())


# ── Pure chain-eval helpers (no deepeval dependency) ─────────────────────────
# Kept here so unit tests can import them without pulling in deepeval.

def format_prior_turns(turns: list[dict], up_to_index: int) -> str:
    """Build a context string from all completed turns before `up_to_index`."""
    parts = []
    for t in turns[:up_to_index]:
        q = t.get("question", "")
        r = t.get("agent_response", "") or ""
        parts.append(f"Turn {t['turn_index'] + 1} - Q: {q}\nA: {r[:500]}")
    return "\n\n---\n\n".join(parts)


def turn_fully_scored(turn: dict) -> bool:
    """True when the turn already has a complete turn_response eval block."""
    ev = turn.get("eval") or {}
    return bool(ev) and ev.get("turn_response") is not None


def aggregate_chain(turns: list[dict]) -> dict:
    """Compute mean + pass-rate aggregates across a chain's turns."""
    tr_scores = [
        t["eval"]["turn_response"]["score"]
        for t in turns
        if (t.get("eval") or {}).get("turn_response", {}) and
        t["eval"]["turn_response"].get("score") is not None
    ]
    cr_scores = [
        t["eval"]["context_retention"]["score"]
        for t in turns
        if (t.get("eval") or {}).get("context_retention") and
        t["eval"]["context_retention"].get("score") is not None
    ]

    def _mean(vals):
        return sum(vals) / len(vals) if vals else None

    def _pass_rate(key):
        scored = [t for t in turns if (t.get("eval") or {}).get(key)]
        if not scored:
            return None
        return sum(1 for t in scored if t["eval"][key].get("passed")) / len(scored)

    return {
        "turn_response_mean": _mean(tr_scores),
        "turn_response_pass_rate": _pass_rate("turn_response"),
        "context_retention_mean": _mean(cr_scores),
        "context_retention_pass_rate": _pass_rate("context_retention"),
    }


def select_chains(
    chains: list[dict],
    *,
    chain_ids: list[int] | None = None,
    flow: str | None = None,
    priority_only: bool = False,
    limit: int | None = None,
) -> list[dict]:
    pool = chains
    if chain_ids:
        id_set = set(chain_ids)
        return [c for c in pool if c.get("chain_id") in id_set]
    if priority_only:
        pool = [c for c in pool if c.get("priority")]
    if flow:
        pool = [c for c in pool if c.get("flow") == flow]
    if limit is not None and limit > 0:
        pool = pool[:limit]
    return pool
