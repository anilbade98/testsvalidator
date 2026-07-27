"""Build a per-question result record from collected SSE events.

Extracts:
  - sub_reasoning: FDL/Athena sub-agent reasoning text keyed by source name
  - od_fdl_results: structured query results from the result event's data payload
  - tools_called: ordered list of orchestrator-level tool names
  - agent_response: final text from the result event
  - usage: token counts aggregated across all tool results
  - streaming_events: compact event-type summary

The validator receives raw SSE events from the HTTP stream. SQL tracing uses
the sub-agent reasoning text (extended thinking) and the result event's od/fdl
data — no orchestrator modification required.
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


# Maps a returned data source to the orchestrator tool that produces it. The
# refresh flow calls these tools directly server-side (bypassing the
# orchestrator), so it emits no tool_start events — we recover tool usage from
# the data that actually came back instead.
_SOURCE_TO_TOOL = {
    "fdl": "get_fdl_data",
    "od": "get_athena_data",
    "ao": "ask_automation_opportunity",
    "rca": "query_anomalies",
}


def _tools_from_data(od_fdl_results: list[dict], tool_results: dict) -> list[str]:
    """Infer tool names from the data returned (source → tool), for flows that
    bypass the orchestrator and emit no tool_start events (e.g. refresh)."""
    tools: list[str] = []
    for r in od_fdl_results or []:
        t = _SOURCE_TO_TOOL.get(r.get("tool_name") or "")
        if t and t not in tools:
            tools.append(t)
    for src in (tool_results or {}):
        t = _SOURCE_TO_TOOL.get(src)
        if t and t not in tools:
            tools.append(t)
    return tools


def _extract_sub_reasoning(events: list[dict]) -> dict[str, str]:
    """Collect sub-agent reasoning text per source from `reasoning` SSE events.

    Returns a dict keyed by source name (e.g. "fdl", "athena"), with the full
    concatenated reasoning text for each. The orchestrator passes FDL/Athena
    extended-thinking chunks through as `{"source": "<name>", "text": "..."}`.
    """
    parts: dict[str, list[str]] = {}
    for ev in events:
        if ev.get("event") != "reasoning":
            continue
        src = (ev.get("data") or {}).get("source", "")
        text = (ev.get("data") or {}).get("text", "")
        if src and src != "orchestrator" and text:
            parts.setdefault(src, []).append(text)
    return {src: "\n\n".join(chunks) for src, chunks in parts.items()}


def _extract_od_fdl_results(events: list[dict]) -> list[dict]:
    """Extract od/fdl query result records from the `result` SSE event.

    The orchestrator embeds od/fdl results under result.data.data.{od,fdl}.
    Each item has columnMetaData, resultList, title (= agent interpretation),
    and schema (= schema name used). Returns a flat list across both buckets.
    """
    results: list[dict] = []
    for ev in reversed(events):
        if ev.get("event") != "result":
            continue
        data = (ev.get("data") or {}).get("data") or {}
        if not isinstance(data, dict):
            break
        for src in ("od", "fdl"):
            for item in data.get(src) or []:
                rows = item.get("resultList") or []
                sample, _ = _slice_rows(rows)
                results.append({
                    "tool_name": src,
                    "interpretation": item.get("title") or "",
                    "schema": item.get("schema") or "",
                    "sql": item.get("sql") or "",
                    "column_meta": item.get("columnMetaData") or [],
                    "rows": sample,
                })
        break
    return results


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
    sub_reasoning = _extract_sub_reasoning(events)
    od_fdl_results = _extract_od_fdl_results(events)
    tool_results = _extract_tool_results(events)
    clarification_questions = clarification_questions or []

    # Union of tools seen in the stream (orchestrator flows) and tools inferred
    # from the returned data (server-side-direct flows like refresh, which emit
    # no tool_start events). Keeps tool-compliance accurate for both.
    _stream_tools = _tools_called(events)
    tools_called = _stream_tools + [
        t for t in _tools_from_data(od_fdl_results, tool_results)
        if t not in _stream_tools
    ]

    record: dict = {
        "question_id": question_meta.get("id"),
        "question": question_meta.get("question"),
        "flow": question_meta.get("flow"),
        "complexity": question_meta.get("complexity"),
        "priority": bool(question_meta.get("priority", False)),
        "tags": question_meta.get("tags", []),
        "expected_tools": question_meta.get("expected_tools", []),
        "tools_called": tools_called,
        "turn_count": turn_count,
        "clarification_requested": bool(clarification_questions),
        "clarification_questions": clarification_questions,
        "sub_reasoning": sub_reasoning,
        "od_fdl_results": od_fdl_results,
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
