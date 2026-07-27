"""Score collected chain records with DeepEval GEval metrics.

Applies per-turn metrics (turn_response, sql_correctness) and chain-level
metrics (context_retention) to results produced by chain_runner.py.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from deepeval.test_case import LLMTestCase

from tests.validator.records import chain_records

from tests.validator.bedrock_judge import BedrockJudge
from tests.validator.metrics import (
    PASS_THRESHOLD,
    SQL_FLOWS,
    FDL_SQL_CONVENTIONS,
    ATHENA_SQL_CONVENTIONS,
    build_chain_metrics,
    build_metrics,
    load_schema_docs,
    schema_docs_for_schemas,
    all_schema_docs,
)

from tests.validator.chain_runner import aggregate_chain, format_prior_turns, turn_fully_scored

_CREDENTIAL_MARKERS = ("ExpiredToken", "NoCredentialsError", "NoCredentials")
EVAL_INTER_TURN_PAUSE = 2


def _safe_measure(metric, tc: LLMTestCase):
    try:
        metric.measure(tc)
        score = float(getattr(metric, "score", 0.0) or 0.0)
        reason = getattr(metric, "reason", "") or ""
        return score, score >= metric.threshold, reason
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


def _eval_turn(
    turn: dict,
    prior_turns: list[dict],
    flow: str,
    single_metrics: dict,
    chain_metrics: dict,
    schema_cache: dict,
) -> dict:
    question = turn.get("question") or ""
    response = turn.get("agent_response") or ""
    sub_reasoning = turn.get("sub_reasoning") or {}
    od_fdl_results = turn.get("od_fdl_results") or []
    error = turn.get("error")
    turn_index = turn.get("turn_index", 0)

    eval_out: dict = {
        "turn_response": None,
        "per_trace": [],
        "context_retention": None,
    }

    if error:
        eval_out["skipped_reason"] = f"upstream error: {error}"
        return eval_out

    # Schema docs — same per-result logic as evaluator.py
    def _schema_ctx_for(r: dict) -> str:
        schema_name = r.get("schema") or ""
        if schema_name:
            return schema_docs_for_schemas([schema_name], schema_cache)
        src_prefix = "fdl/" if r.get("tool_name") == "fdl" else "athena/"
        src_docs = {k: v for k, v in schema_cache.items() if k.startswith(src_prefix)}
        return "\n\n---\n\n".join(src_docs.values()) if src_docs else all_schema_docs(schema_cache)

    touched_schemas = list({r.get("schema") or "" for r in od_fdl_results} - {""})
    context_docs = (
        schema_docs_for_schemas(touched_schemas, schema_cache)
        if touched_schemas else all_schema_docs(schema_cache)
    )

    # SQL conventions
    fdl_conv = FDL_SQL_CONVENTIONS if flow in {"fdl", "advisory", "mixed", "refresh"} else ""
    ath_conv = ATHENA_SQL_CONVENTIONS if flow in {"od", "advisory", "mixed"} else ""
    sql_conv = "\n".join(filter(None, [fdl_conv, ath_conv]))

    if flow in SQL_FLOWS:
        for r in od_fdl_results:
            sql = r.get("sql") or ""
            interp = r.get("interpretation") or ""
            if not sql or not interp:
                eval_out["per_trace"].append({
                    "sql_score": None, "sql_passed": None,
                    "sql_reason": "skipped — missing SQL or interpretation",
                })
                continue
            schema_ctx = _schema_ctx_for(r)
            ctx_parts = [schema_ctx]
            if sql_conv:
                ctx_parts.append(sql_conv)
            score, passed, reason = _safe_measure(
                single_metrics["sql_correctness"],
                LLMTestCase(input=interp, actual_output=sql, context=ctx_parts),
            )
            eval_out["per_trace"].append({
                "sql_score": score, "sql_passed": passed, "sql_reason": reason,
            })

    # turn_response
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
        tool_results = turn.get("tool_results") or {}
        tool_ctx = json.dumps(tool_results, default=str) if tool_results else None

        reasoning_ctx = "\n\n".join(
            f"## Agent reasoning ({src})\n{text.strip()}"
            for src, text in sub_reasoning.items()
            if text.strip()
        )
        ctx_parts = []
        if results_ctx:
            ctx_parts.append(results_ctx)
        if tool_ctx:
            ctx_parts.append(f"Non-SQL tool response data:\n{tool_ctx}")
        if reasoning_ctx:
            ctx_parts.append(reasoning_ctx)
        ctx_parts.append(context_docs)

        score, passed, reason = _safe_measure(
            chain_metrics["turn_response"],
            LLMTestCase(input=question, actual_output=response, context=ctx_parts),
        )
        eval_out["turn_response"] = {"score": score, "passed": passed, "reason": reason}

    # context_retention - only for turns after the first
    if turn_index > 0 and response and prior_turns:
        prior_ctx = format_prior_turns(prior_turns, turn_index)
        score, passed, reason = _safe_measure(
            chain_metrics["context_retention"],
            LLMTestCase(
                input=question,
                actual_output=response,
                context=[prior_ctx, context_docs],
            ),
        )
        eval_out["context_retention"] = {"score": score, "passed": passed, "reason": reason}

    return eval_out


def evaluate_chain_file(run_path: Path, *, skip_scored: bool = False, workers: int = 1) -> dict:
    """Score all turns in all chains in-place. Returns updated run dict.

    workers: parallel chains (each chain gets its own BedrockJudge). Turns
    within a chain are always scored sequentially — context_retention requires
    prior turns to be evaluated first.
    """
    run = json.loads(run_path.read_text())
    schema_cache = load_schema_docs()
    chain_results = chain_records(run.get("results"))
    total_chains = len(chain_results)
    write_lock = threading.Lock()
    counter = [0]

    def _score_chain(chain: dict) -> None:
        # Fresh judge per chain so parallel threads don't share metric state.
        judge = BedrockJudge()
        single_metrics = build_metrics(judge, threshold=PASS_THRESHOLD, verbose=False)
        chain_metrics_map = build_chain_metrics(judge, threshold=PASS_THRESHOLD, verbose=False)
        flow = chain.get("flow") or ""
        turns = chain.get("turns") or []

        for ti, turn in enumerate(turns):
            if skip_scored and turn_fully_scored(turn):
                continue
            if ti > 0:
                time.sleep(EVAL_INTER_TURN_PAUSE)
            eval_block = _eval_turn(
                turn,
                prior_turns=turns,
                flow=flow,
                single_metrics=single_metrics,
                chain_metrics=chain_metrics_map,
                schema_cache=schema_cache,
            )
            turn["eval"] = eval_block

        chain["chain_eval"] = aggregate_chain(turns)

        with write_lock:
            counter[0] += 1
            n = counter[0]
            print(f"\n[chain {n}/{total_chains}] {chain.get('name','?')} [{flow}] "
                  f"- {len(turns)} turns")
            for turn in turns:
                _print_turn_summary(turn, turn.get("eval") or {})
            run_path.write_text(json.dumps(run, indent=2, default=str))

    if workers <= 1:
        for chain in chain_results:
            _score_chain(chain)
    else:
        print(f"[chain-eval] Scoring {total_chains} chains with {workers} parallel workers ...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_score_chain, c): c for c in chain_results}
            for fut in as_completed(futures):
                fut.result()

    run_path.write_text(json.dumps(run, indent=2, default=str))
    return run


def _print_turn_summary(turn: dict, ev: dict) -> None:
    def _s(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "  -  "

    tr = ev.get("turn_response") or {}
    cr = ev.get("context_retention") or {}
    print(f"    turn_response:    {_s(tr.get('score'))}  passed={tr.get('passed')}")
    if cr:
        print(f"    context_retention:{_s(cr.get('score'))}  passed={cr.get('passed')}")
