"""Build a per-question result record from collected SSE events.

Extracts:
  - cortex_traces: one entry per SQL call (FDL or Athena)
  - tools_called: ordered list of orchestrator-level tool names
  - agent_response: final text from the result event
  - usage: token counts aggregated across all tool results
  - streaming_events: compact event-type summary

The validator receives raw SSE events from the HTTP stream, not orchestrator
messages. Tool result payloads arrive as JSON inside the `result` event's
`data` field or inside sub-agent `tool_result` events, depending on how the
service is configured.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

TERSE_ROW_LIMIT = 50


def _slice_rows(rows: list) -> tuple[list, int]:
    n = len(rows)
    if n <= TERSE_ROW_LIMIT:
        return rows, n
    half = TERSE_ROW_LIMIT // 2
    return rows[:half] + rows[-half:], n


def _extract_text(events: list[dict]) -> str:
    """Pull final agent response text from events."""
    for ev in reversed(events):
        if ev.get("event") == "result":
            return (ev.get("data") or {}).get("text", "") or ""
    chunks = [
        (ev.get("data") or {}).get("text", "")
        for ev in events
        if ev.get("event") == "text"
    ]
    return "".join(chunks).strip()


def _tools_called(events: list[dict]) -> list[str]:
    """Ordered unique list of tool names from tool_start events."""
    seen: set[str] = set()
    ordered: list[str] = []
    for ev in events:
        if ev.get("event") == "tool_start":
            name = (ev.get("data") or {}).get("name", "")
            if name and name not in seen:
                seen.add(name)
                ordered.append(name)
    return ordered


def _extract_cortex_traces(events: list[dict]) -> list[dict]:
    """Build cortex_traces from the `trace` SSE event emitted by the orchestrator.

    The orchestrator emits a single `trace` event (before the `result` event)
    containing a `cortex_traces` list built from the FDL/Athena agents'
    `sql_queries` payloads. Each entry has: tool_name, sql, schemas, title,
    is_discovery, status, row_count, error.
    """
    traces: list[dict] = []
    for ev in events:
        if ev.get("event") != "trace":
            continue
        for sq in (ev.get("data") or {}).get("cortex_traces") or []:
            rows = sq.get("rows") or []
            full_count = sq.get("row_count")
            sample, _ = _slice_rows(rows)
            traces.append({
                "tool_name": sq.get("tool_name", ""),
                "semantic_model": sq.get("schemas") or [],
                "interpretation": sq.get("title") or "",
                "generated_sql": sq.get("sql") or "",
                "sql_results": sample,
                "sql_row_count_full": full_count,
                "sql_status": sq.get("status") or "UNKNOWN",
                "is_discovery": sq.get("is_discovery", False),
                "error": sq.get("error"),
            })
    return traces


_USAGE_MAP = (
    ("inputTokens", "input_tokens"),
    ("outputTokens", "output_tokens"),
    ("cacheReadInputTokens", "cache_read_tokens"),
    ("cacheWriteInputTokens", "cache_creation_tokens"),
    # totalTokens intentionally omitted — computed from components after aggregation
    # so sub-agent tokens are always included in the final total.
)


def _normalize_usage(raw: dict | None) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for camel, snake in _USAGE_MAP:
        v = raw.get(camel)
        if isinstance(v, int):
            out[snake] = v
    return out


def _aggregate_usage(events: list[dict]) -> dict:
    """Sum token usage from all tool_result events."""
    totals: dict[str, int] = defaultdict(int)
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for ev in events:
        if ev.get("event") not in ("tool_result", "sub_tool_result", "result"):
            continue
        data = ev.get("data") or {}
        usage_raw = data.get("usage")
        if not usage_raw:
            continue
        source = data.get("source") or ev.get("event")
        normalized = _normalize_usage(usage_raw)
        for k, v in normalized.items():
            totals[k] += v
            by_source[source][k] += v

    # Compute total from all components (input + output + cache read + cache write)
    # so it correctly reflects the full cost including sub-agent tokens.
    totals["total_tokens"] = (
        totals.get("input_tokens", 0)
        + totals.get("output_tokens", 0)
        + totals.get("cache_read_tokens", 0)
        + totals.get("cache_creation_tokens", 0)
    )
    return {
        **dict(totals),
        "by_source": {k: dict(v) for k, v in by_source.items()},
    }


def _extract_tool_results(events: list[dict]) -> dict:
    """Extract non-SQL tool results (RCA, AO) from the result SSE event.

    The orchestrator embeds rca/ao payloads under result.data.data so the
    validator can pass them as judge context — these tools produce no SQL
    traces but return real structured data the agent cites in its response.
    """
    out: dict = {}
    for ev in reversed(events):
        if ev.get("event") != "result":
            continue
        data = (ev.get("data") or {}).get("data") or {}
        if isinstance(data, dict):
            if data.get("rca"):
                out["rca"] = data["rca"]
            if data.get("ao"):
                out["ao"] = data["ao"]
        if out:
            break
    return out


def _streaming_events_summary(events: list[dict]) -> list[dict]:
    """Compact summary: count of each event type."""
    counts: dict[str, int] = defaultdict(int)
    for ev in events:
        counts[ev.get("event", "unknown")] += 1
    return [{"event": k, "count": v} for k, v in sorted(counts.items())]


def build_record(
    *,
    events: list[dict],
    question_meta: dict[str, Any],
    duration_ms: int,
    error: str | None,
    clarification_questions: list[str] | None = None,
    turn_count: int = 1,
) -> dict:
    """Build the per-question result record from collected SSE events."""
    cortex_traces = _extract_cortex_traces(events)
    tool_results = _extract_tool_results(events)
    clarification_questions = clarification_questions or []

    record: dict = {
        "question_id": question_meta.get("id"),
        "question": question_meta.get("question"),
        "flow": question_meta.get("flow"),
        "complexity": question_meta.get("complexity"),
        "priority": bool(question_meta.get("priority", False)),
        "tags": question_meta.get("tags", []),
        "expected_tools": question_meta.get("expected_tools", []),
        "tools_called": _tools_called(events),
        "turn_count": turn_count,
        "clarification_requested": bool(clarification_questions),
        "clarification_questions": clarification_questions,
        "cortex_traces": cortex_traces,
        "streaming_events": _streaming_events_summary(events),
        "agent_response": _extract_text(events),
        "duration_ms": duration_ms,
        "usage": _aggregate_usage(events),
        "error": error,
        "eval": None,
    }
    if question_meta.get("context_fixture"):
        record["context_fixture"] = question_meta["context_fixture"]
    if tool_results:
        record["tool_results"] = tool_results
    return record
