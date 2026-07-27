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
import threading  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from deepeval.test_case import LLMTestCase  # noqa: E402

from tests.validator.bedrock_judge import BedrockJudge  # noqa: E402
from tests.validator.records import is_chain_record, is_question_record  # noqa: E402
from tests.validator.trace_capture import _tools_from_data  # noqa: E402
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
    for pr in eval_block.get("per_trace") or []:
        if any(m in str(pr.get("sql_reason") or "") for m in _CREDENTIAL_MARKERS):
            return True
    return False


def _check_tool_compliance(record: dict) -> dict | None:
    """Compare expected_tools from the catalog against tools_called in the result.

    Returns a compliance dict, or None if expected_tools is empty (no constraint).

    Back-fills tools inferred from the returned data (source → tool) so that
    server-side-direct flows (refresh) — and older run files captured before that
    inference existed — report accurate compliance when re-evaluated in place.
    """
    expected = set(record.get("expected_tools") or [])
    if not expected:
        return None
    called = set(record.get("tools_called") or [])
    called |= set(_tools_from_data(record.get("od_fdl_results"), record.get("tool_results")))
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
    sub_reasoning = record.get("sub_reasoning") or {}
    od_fdl_results = record.get("od_fdl_results") or []
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

    # Build per-result schema context keyed by tool_name.
    # FDL results carry explicit schema names (e.g. "CALL_SEGMENT").
    # OD/Athena results have schema="" because the Athena tool call input uses
    # athena_sql_query, not fdl_schema_names — fall back to all Athena schema docs.
    def _schema_ctx_for(r: dict) -> str:
        schema_name = r.get("schema") or ""
        if schema_name:
            return schema_docs_for_schemas([schema_name], schema_cache)
        tool_src = r.get("tool_name") or ""
        # Load all docs for the tool's source (fdl → fdl/, od → athena/)
        src_prefix = "fdl/" if tool_src == "fdl" else "athena/"
        src_docs = {k: v for k, v in schema_cache.items() if k.startswith(src_prefix)}
        if src_docs:
            return "\n\n---\n\n".join(src_docs.values())
        return all_schema_docs(schema_cache)

    touched_schemas: list[str] = []
    seen_schemas: set[str] = set()
    for r in od_fdl_results:
        s = r.get("schema") or ""
        if s and s not in seen_schemas:
            seen_schemas.add(s)
            touched_schemas.append(s)

    context_docs = (
        schema_docs_for_schemas(touched_schemas, schema_cache)
        if touched_schemas
        else all_schema_docs(schema_cache)
    )

    # Build reasoning context from sub_reasoning (pre-tool-call agent thinking).
    # Even though short (~200 chars), it reveals which schema/tables the agent
    # identified before fetching docs — useful supplementary signal for the judge.
    reasoning_ctx = "\n\n".join(
        f"## Agent reasoning ({src})\n{text.strip()}"
        for src, text in sub_reasoning.items()
        if text.strip()
    )

    # interpretation_coverage — use interpretation titles from od/fdl result records
    _DATA_TOOLS = {"get_fdl_data", "get_od_data", "get_athena_data"}
    if flow in INTERPRETATION_FLOWS:
        interpretations = [r.get("interpretation", "") for r in od_fdl_results if r.get("interpretation")]
        if interpretations:
            joined = "\n\n---\n\n".join(interpretations)
            ctx_parts = [context_docs]
            if reasoning_ctx:
                ctx_parts.append(reasoning_ctx)
            score, passed, reason = _safe_measure(
                metrics["interpretation_coverage"],
                LLMTestCase(input=question, actual_output=joined, context=ctx_parts),
            )
            eval_out["interpretation_coverage"] = {"score": score, "passed": passed, "reason": reason}
        elif not od_fdl_results:
            tools_called = set(record.get("tools_called") or [])
            if tools_called & _DATA_TOOLS:
                # Agent called a data tool but the service returned an empty result set
                # (0 rows) — no interpretation to evaluate, skip rather than penalise.
                pass
            else:
                # Agent truly made no data tool calls for a data-flow question.
                eval_out["interpretation_coverage"] = {
                    "score": 0.0,
                    "passed": False,
                    "reason": "Agent made no data tool calls for a data-flow question.",
                }

    # sql_correctness — validate that the agent referenced correct columns and tables.
    # SQL comes from od_fdl_results[*].sql, embedded by _merge_tool_result via title matching.
    # When a query failed entirely (data=null, od_fdl_results empty), scoring is skipped.
    fdl_conventions = FDL_SQL_CONVENTIONS if flow in {"fdl", "advisory", "mixed", "refresh"} else ""
    athena_conventions = ATHENA_SQL_CONVENTIONS if flow in {"od", "advisory", "mixed"} else ""
    sql_convention_ctx = "\n".join(filter(None, [fdl_conventions, athena_conventions]))

    if flow in SQL_FLOWS:
        for r in od_fdl_results:
            sql = r.get("sql") or ""
            interp = r.get("interpretation") or ""
            if not sql or not interp:
                eval_out["per_trace"].append({
                    "sql_score": None,
                    "sql_passed": None,
                    "sql_reason": "skipped — missing SQL or interpretation",
                })
                continue
            schema_ctx = _schema_ctx_for(r)
            ctx_parts = [schema_ctx]
            if sql_convention_ctx:
                ctx_parts.append(sql_convention_ctx)
            score, passed, reason = _safe_measure(
                metrics["sql_correctness"],
                LLMTestCase(input=interp, actual_output=sql, context=ctx_parts),
            )
            eval_out["per_trace"].append({
                "sql_score": score,
                "sql_passed": passed,
                "sql_reason": reason,
            })

    # agent_response — use result rows from od/fdl records for factuality check
    if response:
        results_ctx = "\n---\n".join(
            json.dumps({
                "interpretation": r.get("interpretation"),
                "schema": r.get("schema"),
                "columns": [c.get("columnName") for c in (r.get("column_meta") or [])],
                "results": r.get("rows"),
            }, default=str)
            for r in od_fdl_results
        )
        # For context/refresh flows the agent reads a UI fixture, not SQL.
        fixture = record.get("context_fixture")
        fixture_ctx = json.dumps(fixture, default=str) if fixture else None

        # For RCA/AO flows the agent reads structured REST responses, not SQL.
        tool_results = record.get("tool_results") or {}
        tool_ctx = json.dumps(tool_results, default=str) if tool_results else None

        context_parts = []
        if results_ctx:
            context_parts.append(results_ctx)
        if fixture_ctx:
            context_parts.append(f"UI context fixture provided to agent:\n{fixture_ctx}")
        if tool_ctx:
            context_parts.append(f"Non-SQL tool response data (RCA/AO):\n{tool_ctx}")
        if reasoning_ctx:
            context_parts.append(reasoning_ctx)
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

    # refresh flow with no prose answer (e.g. AO refresh returns data, not text):
    # score the refreshed data against the original on-screen data instead, and
    # store it under agent_response so it flows through aggregation/comparison
    # like any quality score.
    elif flow == "refresh" and (record.get("tool_results") or od_fdl_results):
        refreshed = {
            "tool_results": record.get("tool_results"),
            "od_fdl_results": [
                {
                    "schema": r.get("schema"),
                    "columns": [c.get("columnName") for c in (r.get("column_meta") or [])],
                    "rows": (r.get("rows") or [])[:5],
                }
                for r in od_fdl_results
            ],
        }
        refreshed_ctx = json.dumps(refreshed, default=str)[:4000]
        original_ctx = json.dumps(record.get("context_fixture") or {}, default=str)[:4000]
        score, passed, reason = _safe_measure(
            metrics["refresh_correctness"],
            LLMTestCase(
                input=question,
                actual_output=refreshed_ctx,
                context=[f"Original on-screen data before refresh:\n{original_ctx}"],
            ),
        )
        eval_out["agent_response"] = {
            "score": score, "passed": passed, "reason": reason,
            "kind": "refresh_correctness",
        }

    return eval_out


def _needs_scoring(record: dict) -> bool:
    ev = record.get("eval") or {}
    if not ev:
        return True
    markers = ("TimeoutError", "ExpiredToken", "NoCredentials", "UnauthorizedSSO")
    for key in ("interpretation_coverage", "agent_response"):
        entry = ev.get(key) or {}
        if entry.get("score") is None:
            return True
        if any(m in str(entry.get("reason") or "") for m in markers):
            return True
    for pr in ev.get("per_trace") or []:
        if pr.get("sql_score") is None:
            return True
        if any(m in str(pr.get("sql_reason") or "") for m in markers):
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

    for i, pr in enumerate(ev.get("per_trace") or []):
        print(f"│  sql_correctness[{i}]       {_s(pr.get('sql_score'))}  passed={pr.get('sql_passed')}")
        if pr.get("sql_reason"):
            print(f"│    {(pr['sql_reason'])[:200]}")

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


_CRED_REFRESH_LOCK = threading.Lock()


def _make_judge_and_metrics() -> tuple:
    """Create a fresh BedrockJudge + metrics dict. Call per-thread (not thread-safe to share)."""
    j = BedrockJudge()
    return j, build_metrics(j, threshold=PASS_THRESHOLD, verbose=False)


def _wait_for_cred_refresh(qid: int | None) -> None:
    """Coordinate credential refresh across parallel workers.

    The first worker to hit an expired-token error acquires the lock and sleeps
    30s. All other workers wait on the lock instead of sleeping independently.
    """
    if _CRED_REFRESH_LOCK.acquire(blocking=False):
        try:
            print(f"  q{qid} — credential error, sleeping 30s for token refresh ...")
            time.sleep(30)
        finally:
            _CRED_REFRESH_LOCK.release()
    else:
        print(f"  q{qid} — credential error, waiting for token refresh ...")
        with _CRED_REFRESH_LOCK:
            pass


def _score_one(record: dict, schema_cache: dict) -> dict:
    """Score a single record with its own judge. Retries once on credential error."""
    _, metrics = _make_judge_and_metrics()
    eval_block = _eval_record(record, metrics, schema_cache)
    if _is_credential_error(eval_block):
        _wait_for_cred_refresh(record.get("question_id"))
        _, metrics = _make_judge_and_metrics()
        eval_block = _eval_record(record, metrics, schema_cache)
    return eval_block


def evaluate_run_file(
    run_path: Path,
    *,
    skip_scored: bool = False,
    workers: int = 1,
    only_flows: set[str] | None = None,
    only_ids: set[int] | None = None,
) -> dict:
    """Score all records in a run file in place. Returns the updated run dict.

    workers: number of parallel scoring threads (default 1 = sequential).
             Each thread creates its own BedrockJudge so metric state is isolated.
    only_flows / only_ids: when set, restrict (re)scoring to matching question
             records — used to re-score just a subset (e.g. only `refresh`) of a
             completed report cheaply, leaving other records' scores untouched.
    """
    run = json.loads(run_path.read_text())
    schema_cache = load_schema_docs()
    results = run.get("results") or []
    write_lock = threading.Lock()
    counter = [0]

    chain_count = sum(1 for r in results if is_chain_record(r))
    to_score = [
        (i, r) for i, r in enumerate(results)
        if is_question_record(r)  # chain records are scored separately by evaluate_chain_file
        and not (skip_scored and not _needs_scoring(r))
        and (only_flows is None or r.get("flow") in only_flows)
        and (only_ids is None or r.get("question_id") in only_ids)
    ]
    already_scored = len(results) - chain_count - len(to_score)
    if chain_count:
        print(f"[eval] {chain_count} chain record(s) will be scored separately.")
    if already_scored:
        print(f"[eval] Skipping {already_scored} already-scored question record(s).")

    def _work(item: tuple) -> None:
        idx, record = item
        qid = record.get("question_id")
        flow = record.get("flow", "?")
        eval_block = _score_one(record, schema_cache)
        record["eval"] = eval_block
        with write_lock:
            counter[0] += 1
            n = counter[0]
            print(f"\n[eval {n}/{len(to_score)}] q{qid} [{flow}]")
            _print_record_summary(record, eval_block)
            run_path.write_text(json.dumps(run, indent=2, default=str))

    if workers <= 1:
        for item in to_score:
            _work(item)
    else:
        print(f"[eval] Scoring {len(to_score)} records with {workers} parallel workers ...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_work, item): item for item in to_score}
            for fut in as_completed(futures):
                fut.result()

    return run
