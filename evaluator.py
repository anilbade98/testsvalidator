"""Score collected records with DeepEval GEval metrics.

Reads a run file produced by runner.py, mutates each record's `eval` field,
and saves back to disk. Re-running on the same file is safe (scores overwrite).

Metrics applied depend on the question's flow:
  - interpretation_coverage: fdl, od, ao, rca, advisory, mixed
  - sql_correctness:          fdl, od, ao, rca, advisory (if SQL), mixed, refresh
  - agent_response:           all flows
  - turn_response / context_retention: chains only (separate pass)
"""

from __future__ import annotations

import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_IN", "NO")

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from deepeval.test_case import LLMTestCase  # noqa: E402

from tests.validator.bedrock_judge import BedrockJudge  # noqa: E402
from tests.validator.metrics import (  # noqa: E402
    PASS_THRESHOLD,
    INTERPRETATION_FLOWS,
    SQL_FLOWS,
    FDL_SQL_CONVENTIONS,
    ATHENA_SQL_CONVENTIONS,
    all_schema_docs,
    build_metrics,
    load_schema_docs,
    schema_docs_for_schemas,
)


_CREDENTIAL_MARKERS = ("ExpiredToken", "NoCredentialsError", "NoCredentials")
EVAL_INTER_RECORD_PAUSE = 3


def _safe_measure(metric, tc: LLMTestCase) -> tuple[float | None, bool | None, str | None]:
    try:
        metric.measure(tc)
        score = float(getattr(metric, "score", 0.0) or 0.0)
        reason = getattr(metric, "reason", "") or ""
        return score, score >= metric.threshold, reason
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


def _is_credential_error(eval_block: dict) -> bool:
    for key in ("interpretation_coverage", "agent_response"):
        reason = str((eval_block.get(key) or {}).get("reason") or "")
        if any(m in reason for m in _CREDENTIAL_MARKERS):
            return True
    for pt in eval_block.get("per_trace") or []:
        if any(m in str(pt.get("sql_reason") or "") for m in _CREDENTIAL_MARKERS):
            return True
    return False


def _check_tool_compliance(record: dict) -> dict | None:
    """Compare expected_tools from the catalog against tools_called in the result.

    Returns a compliance dict, or None if expected_tools is empty (no constraint).
    """
    expected = set(record.get("expected_tools") or [])
    if not expected:
        return None
    called = set(record.get("tools_called") or [])
    missing = sorted(expected - called)
    extra = sorted(called - expected)
    return {
        "expected": sorted(expected),
        "called": sorted(called),
        "missing": missing,
        "extra": extra,
        "compliant": len(missing) == 0,
    }


def _eval_record(record: dict, metrics: dict, schema_cache: dict[str, str]) -> dict:
    question = record.get("question") or ""
    response = record.get("agent_response") or ""
    traces = record.get("cortex_traces") or []
    flow = record.get("flow") or ""
    error = record.get("error")

    eval_out: dict = {
        "tool_compliance": _check_tool_compliance(record),
        "interpretation_coverage": None,
        "per_trace": [],
        "agent_response": None,
    }

    if error:
        eval_out["skipped_reason"] = f"upstream error: {error}"
        return eval_out

    # Collect schemas actually touched
    touched_schemas: list[str] = []
    seen_schemas: set[str] = set()
    for t in traces:
        for s in t.get("semantic_model") or []:
            if s not in seen_schemas:
                seen_schemas.add(s)
                touched_schemas.append(s)

    context_docs = (
        schema_docs_for_schemas(touched_schemas, schema_cache)
        if touched_schemas
        else all_schema_docs(schema_cache)
    )

    # interpretation_coverage
    if flow in INTERPRETATION_FLOWS:
        interpretations = [t.get("interpretation", "") for t in traces if t.get("interpretation")]
        if interpretations:
            joined = "\n\n---\n\n".join(interpretations)
            score, passed, reason = _safe_measure(
                metrics["interpretation_coverage"],
                LLMTestCase(input=question, actual_output=joined, context=[context_docs]),
            )
            eval_out["interpretation_coverage"] = {"score": score, "passed": passed, "reason": reason}

    # sql_correctness (per trace)
    # Conventions from the agent's own SQL guidelines are included alongside schema docs
    # so the judge can detect violations beyond just wrong column names.
    fdl_conventions = FDL_SQL_CONVENTIONS if flow in {"fdl", "advisory", "mixed", "refresh"} else ""
    athena_conventions = ATHENA_SQL_CONVENTIONS if flow in {"od", "advisory", "mixed"} else ""
    sql_convention_ctx = "\n".join(filter(None, [fdl_conventions, athena_conventions]))

    if flow in SQL_FLOWS:
        for t in traces:
            sql = t.get("generated_sql") or ""
            interp = t.get("interpretation") or ""
            if not sql or not interp:
                eval_out["per_trace"].append({
                    "sql_score": None,
                    "sql_passed": None,
                    "sql_reason": "skipped — missing SQL or interpretation",
                })
                continue
            schemas = t.get("semantic_model") or []
            schema_ctx = schema_docs_for_schemas(schemas, schema_cache) if schemas else context_docs
            ctx_parts = [schema_ctx]
            if sql_convention_ctx:
                ctx_parts.append(sql_convention_ctx)
            score, passed, reason = _safe_measure(
                metrics["sql_correctness"],
                LLMTestCase(input=interp, actual_output=sql, context=ctx_parts),
            )
            eval_out["per_trace"].append({"sql_score": score, "sql_passed": passed, "sql_reason": reason})

    # agent_response
    if response:
        results_ctx = "\n---\n".join(
            json.dumps({
                "interpretation": t.get("interpretation"),
                "generated_sql": t.get("generated_sql"),
                "sql_status": t.get("sql_status"),
                "total_row_count": t.get("sql_row_count_full"),
                "results": t.get("sql_results"),
            }, default=str)
            for t in traces
        )
        # For context/refresh flows the agent reads a UI fixture, not SQL.
        fixture = record.get("context_fixture")
        fixture_ctx = json.dumps(fixture, default=str) if fixture else None

        # For RCA/AO flows the agent reads structured REST responses, not SQL.
        # Include the raw tool payloads so the judge can verify cited values.
        tool_results = record.get("tool_results") or {}
        tool_ctx = json.dumps(tool_results, default=str) if tool_results else None

        context_parts = []
        if results_ctx:
            context_parts.append(results_ctx)
        if fixture_ctx:
            context_parts.append(f"UI context fixture provided to agent:\n{fixture_ctx}")
        if tool_ctx:
            context_parts.append(f"Non-SQL tool response data (RCA/AO):\n{tool_ctx}")
        context_parts.append(context_docs)

        score, passed, reason = _safe_measure(
            metrics["agent_response"],
            LLMTestCase(
                input=question,
                actual_output=response,
                context=context_parts,
            ),
        )
        eval_out["agent_response"] = {"score": score, "passed": passed, "reason": reason}

    return eval_out


def _needs_scoring(record: dict) -> bool:
    ev = record.get("eval") or {}
    if not ev:
        return True
    markers = ("TimeoutError", "ExpiredToken", "NoCredentials")
    for key in ("interpretation_coverage", "agent_response"):
        entry = ev.get(key) or {}
        if entry.get("score") is None:
            return True
        if any(m in str(entry.get("reason") or "") for m in markers):
            return True
    for pt in ev.get("per_trace") or []:
        if pt.get("sql_score") is None:
            return True
        if any(m in str(pt.get("sql_reason") or "") for m in markers):
            return True
    return False


def _print_record_summary(record: dict, ev: dict) -> None:
    qid = record.get("question_id")
    q = (record.get("question") or "")[:90]
    flow = record.get("flow", "?")
    dur = (record.get("duration_ms") or 0) / 1000
    usage = record.get("usage") or {}

    def _s(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "  -  "

    print()
    print(f"┌── q{qid} [{flow}]  {q!r}")
    print(f"│  {dur:.1f}s | tokens={usage.get('total_tokens', 0):,}")

    ic = ev.get("interpretation_coverage") or {}
    if ic:
        print(f"│  interpretation_coverage  {_s(ic.get('score'))}  passed={ic.get('passed')}")
        if ic.get("reason"):
            print(f"│    {(ic['reason'])[:200]}")

    for i, pt in enumerate(ev.get("per_trace") or []):
        print(f"│  sql_correctness[{i}]       {_s(pt.get('sql_score'))}  passed={pt.get('sql_passed')}")
        if pt.get("sql_reason"):
            print(f"│    {(pt['sql_reason'])[:200]}")

    ar = ev.get("agent_response") or {}
    if ar:
        print(f"│  agent_response           {_s(ar.get('score'))}  passed={ar.get('passed')}")
        if ar.get("reason"):
            print(f"│    {(ar['reason'])[:200]}")

    tc = ev.get("tool_compliance")
    if tc is not None:
        status = "✓" if tc.get("compliant") else "✗"
        missing = tc.get("missing") or []
        extra = tc.get("extra") or []
        detail = ""
        if missing:
            detail += f" missing={missing}"
        if extra:
            detail += f" extra={extra}"
        print(f"│  tool_compliance          {status}{detail}")

    if ev.get("skipped_reason"):
        print(f"│  SKIPPED: {ev['skipped_reason']}")
    print("└──")


def evaluate_run_file(run_path: Path, *, skip_scored: bool = False) -> dict:
    """Score all records in a run file in place. Returns the updated run dict."""
    run = json.loads(run_path.read_text())
    schema_cache = load_schema_docs()
    judge = BedrockJudge()
    metrics = build_metrics(judge, threshold=PASS_THRESHOLD, verbose=True)

    total = len(run.get("results") or [])
    for i, record in enumerate(run["results"], 1):
        if skip_scored and not _needs_scoring(record):
            print(f"\n[eval {i}/{total}] q{record['question_id']} — already scored, skipping.")
            continue
        if i > 1:
            time.sleep(EVAL_INTER_RECORD_PAUSE)
        print(f"\n[eval {i}/{total}] q{record['question_id']} [{record.get('flow', '?')}] — scoring ...")
        eval_block = _eval_record(record, metrics, schema_cache)
        if _is_credential_error(eval_block):
            print("  Credential error in judge — sleeping 60s for refresh, retrying ...")
            time.sleep(60)
            eval_block = _eval_record(record, metrics, schema_cache)
        record["eval"] = eval_block
        _print_record_summary(record, record["eval"])
        run_path.write_text(json.dumps(run, indent=2, default=str))

    return run
