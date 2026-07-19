"""Compute and print aggregated scores from a validator run file."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean as _mean


def _safe_mean(values: list[float]) -> float | None:
    return _mean(values) if values else None


def aggregate(run: dict) -> dict:
    # Only aggregate single-turn question records; chain records (chain_id/turns) are excluded.
    results = [r for r in (run.get("results") or []) if "question_id" in r]
    total = len(results)
    if total == 0:
        return {"total": 0}

    errors = sum(1 for r in results if r.get("error"))
    durations = [r.get("duration_ms", 0) for r in results]
    tokens_total = [(r.get("usage") or {}).get("total_tokens", 0) for r in results]
    tokens_in = [(r.get("usage") or {}).get("input_tokens", 0) for r in results]
    tokens_out = [(r.get("usage") or {}).get("output_tokens", 0) for r in results]
    sql_calls = [len(r.get("od_fdl_results") or []) for r in results]
    tool_calls = [len(r.get("tools_called") or []) for r in results]

    # Per-metric accumulators
    interp_scores: list[float] = []
    interp_passed = 0

    sql_scores: list[float] = []
    sql_passed = 0

    resp_scores: list[float] = []
    resp_passed = 0
    resp_failed = 0

    tool_compliance_total = 0
    tool_compliance_passed = 0

    by_flow: dict[str, dict] = defaultdict(lambda: {"total": 0, "resp_scores": [], "resp_passed": 0})

    for r in results:
        flow = r.get("flow") or "unknown"
        ev = r.get("eval") or {}
        by_flow[flow]["total"] += 1

        tc = ev.get("tool_compliance")
        if tc is not None:
            tool_compliance_total += 1
            if tc.get("compliant"):
                tool_compliance_passed += 1

        ic = ev.get("interpretation_coverage")
        if ic and ic.get("score") is not None:
            interp_scores.append(ic["score"])
            if ic.get("passed"):
                interp_passed += 1

        ar = ev.get("agent_response")
        if ar and ar.get("score") is not None:
            resp_scores.append(ar["score"])
            by_flow[flow]["resp_scores"].append(ar["score"])
            if ar.get("passed"):
                resp_passed += 1
                by_flow[flow]["resp_passed"] += 1
            else:
                resp_failed += 1

        for pr in ev.get("per_trace") or []:
            if pr.get("sql_score") is not None:
                sql_scores.append(pr["sql_score"])
                if pr.get("sql_passed"):
                    sql_passed += 1

    by_flow_summary = {
        flow: {
            "total": d["total"],
            "agent_response_mean": _safe_mean(d["resp_scores"]),
            "agent_response_pass_rate": (
                d["resp_passed"] / len(d["resp_scores"]) if d["resp_scores"] else None
            ),
        }
        for flow, d in sorted(by_flow.items())
    }

    return {
        "total": total,
        "errors": errors,
        "avg_duration_ms": _safe_mean(durations),
        "avg_total_tokens": _safe_mean(tokens_total),
        "avg_input_tokens": _safe_mean(tokens_in),
        "avg_output_tokens": _safe_mean(tokens_out),
        "avg_sql_call_count": _safe_mean(sql_calls),
        "avg_tool_call_count": _safe_mean(tool_calls),
        "tool_compliance": {
            "n": tool_compliance_total,
            "passed": tool_compliance_passed,
            "pass_rate": (
                tool_compliance_passed / tool_compliance_total
                if tool_compliance_total else None
            ),
        },
        "agent_response": {
            "n": len(resp_scores),
            "mean": _safe_mean(resp_scores),
            "pass_rate": (resp_passed / len(resp_scores)) if resp_scores else None,
            "passed": resp_passed,
            "failed": resp_failed,
        },
        "interpretation_coverage": {
            "n": len(interp_scores),
            "mean": _safe_mean(interp_scores),
            "pass_rate": (interp_passed / len(interp_scores)) if interp_scores else None,
        },
        "sql_correctness": {
            "n": len(sql_scores),
            "mean": _safe_mean(sql_scores),
            "pass_rate": (sql_passed / len(sql_scores)) if sql_scores else None,
        },
        "by_flow": by_flow_summary,
    }


def write_aggregated(run_path: Path) -> dict:
    run = json.loads(run_path.read_text())
    agg = aggregate(run)
    run["aggregated_scores"] = agg
    run_path.write_text(json.dumps(run, indent=2, default=str))
    return agg


def print_summary(agg: dict) -> None:
    def _f(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "  -  "

    def _pct(v):
        return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "  -  "

    def _n(v):
        return f"{v:,.0f}" if isinstance(v, (int, float)) else "  -  "

    print()
    print("=" * 60)
    print("VALIDATOR RUN SUMMARY")
    print("=" * 60)
    print(f"  total questions:  {agg.get('total', 0)}")
    print(f"  errors:           {agg.get('errors', 0)}")
    print(f"  avg latency:      {(agg.get('avg_duration_ms') or 0) / 1000:.1f}s")
    print(f"  avg tokens/q:     {_n(agg.get('avg_total_tokens'))}")
    print()

    ar = agg.get("agent_response") or {}
    ic = agg.get("interpretation_coverage") or {}
    sql = agg.get("sql_correctness") or {}
    tc = agg.get("tool_compliance") or {}

    for label, m in [
        ("agent_response         ", ar),
        ("interpretation_coverage", ic),
        ("sql_correctness        ", sql),
    ]:
        print(f"  {label}  n={m.get('n', 0):3d}  mean={_f(m.get('mean'))}  pass={_pct(m.get('pass_rate'))}")

    if tc.get("n", 0) > 0:
        print(f"  tool_compliance          n={tc['n']:3d}  pass={_pct(tc.get('pass_rate'))}"
              f"  ({tc.get('passed', 0)}/{tc['n']} compliant)")

    by_flow = agg.get("by_flow") or {}
    if by_flow:
        print()
        print("  by flow:")
        for flow, d in sorted(by_flow.items()):
            print(
                f"    {flow:<15}  n={d['total']:3d}"
                f"  agent_mean={_f(d.get('agent_response_mean'))}"
                f"  agent_pass={_pct(d.get('agent_response_pass_rate'))}"
            )
    print("=" * 60)
