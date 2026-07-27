"""CLI entry point for the Analytics Agent Validator.

Usage:
    # Full run against staging — all flows (collect + score + aggregate)
    python -m tests.validator.run --env staging --base-url https://your-service.example.com

    # Priority questions only (smoke run, fast)
    python -m tests.validator.run --env staging --base-url https://... --priority-only

    # Single flow
    python -m tests.validator.run --env staging --base-url https://... --flow fdl

    # Local development (service running on port 80)
    python -m tests.validator.run --env dev --base-url http://localhost:80

    # Collect only (no LLM judge — faster, no Bedrock cost)
    python -m tests.validator.run --env staging --base-url https://... --skip-eval

    # Re-score an existing result file without re-running questions
    python -m tests.validator.run --eval-only results/run_20260617_120000.json

    # Re-score the latest result file
    python -m tests.validator.run --eval-latest

    # Hand-pick questions by id
    python -m tests.validator.run --env staging --base-url https://... --ids 1,7,9
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_IN", "NO")

# Windows consoles default to cp1252, which can't encode the box-drawing
# characters the summary printers use. Force UTF-8 (replace on failure) so a
# cosmetic print can never crash a collection/eval/persist run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse  # noqa: E402
import asyncio  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

from tests.validator.aggregator import print_summary, write_aggregated  # noqa: E402
from tests.validator.comparator import DEFAULT_DELTA, DEFAULT_N  # noqa: E402
from tests.validator.records import has_chains, is_question_record  # noqa: E402
from tests.validator.runner import (  # noqa: E402
    VALID_FLOWS,
    load_questions,
    resolve_auth,
    run_all,
    select_questions,
    TOOL_FLOWS,
)


HARNESS_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = HARNESS_DIR / "test_questions.json"
CHAINS_PATH = HARNESS_DIR / "test_chains.json"
DEFAULT_OUT_DIR = HARNESS_DIR / "results"


def _persist_to_dynamodb(run_path: Path) -> None:
    """Write the run's records to DynamoDB. Best-effort: log + continue on failure
    so a transient AWS hiccup doesn't lose the local JSON result the user already has."""
    try:
        from tests.validator.persistence import persist_run_file
        if persist_run_file(run_path):
            print(f"Persisted to DynamoDB: {run_path.name}")
        else:
            print(f"Not persisted (run incomplete/FAILED — excluded from baseline): {run_path.name}")
    except Exception as exc:
        print(f"WARNING: DynamoDB persist failed: {type(exc).__name__}: {exc}")
        print(f"   Local result file is intact at {run_path}.")


def _is_localhost_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    return "localhost" in base_url or "127.0.0.1" in base_url


def _should_refuse_local_persist(base_url: str | None, allow_local_persist: bool) -> bool:
    """Refuse `--persist` against a localhost service unless CI=true or the user
    explicitly opts in. Shared between the full-run path (where base_url comes
    from argparse) and the re-score path (where it comes from the run JSON's
    metadata)."""
    return (
        _is_localhost_base_url(base_url)
        and not os.environ.get("CI")
        and not allow_local_persist
    )


def _read_run_meta_base_url(run_path: Path) -> str | None:
    try:
        import json
        with run_path.open("r", encoding="utf-8") as fh:
            # Run metadata is stored at the top level (see runner._save), not under "meta".
            return json.load(fh).get("base_url")
    except Exception:
        return None


def _compare_run(
    run_path: Path,
    *,
    n: int = DEFAULT_N,
    delta: float = DEFAULT_DELTA,
    confirm: dict | None = None,
    persist: bool = True,
) -> int:
    """Compare against the baseline, optionally re-run to confirm, then persist if PASSED.

    Returns 1 if a (confirmed) regression is detected, else 0. Best-effort: on any
    failure, logs a warning and returns 0 so a transient AWS error doesn't fail the
    run when the local JSON result is intact.

    confirm: when provided (full-run path with a live service), a dict of
      {auth, base_url, env, tunnel_pool, reference_date, k} used to re-run the
      targeted-flagged questions and confirm only regressions that reproduce.
    """
    try:
        from tests.validator.comparator import compare_run, persist_if_passed
        comparison = compare_run(run_path, n=n, delta=delta)
        if (confirm and confirm.get("k", 0) > 0
                and comparison.get("overall_status") == "REGRESSION"
                and "targeted" in (comparison.get("regression_kinds") or [])):
            comparison = _confirm_targeted_regressions(
                run_path, comparison, n=n, delta=delta, **confirm)
        _print_regression_summary(comparison)
        if persist:
            persist_if_passed(run_path, comparison)
        return 1 if comparison.get("overall_status") == "REGRESSION" else 0
    except Exception as exc:
        print(f"WARNING: Comparison failed: {type(exc).__name__}: {exc}")
        print(f"   Local result file is intact at {run_path}.")
        return 0


def _metric_score(eval_block: dict, name: str):
    """Return the scalar score for a metric from an eval block, or None.

    sql_correctness has no single field — it's the mean of per-trace sql_score.
    """
    if name == "sql_correctness":
        pts = [pr.get("sql_score") for pr in (eval_block.get("per_trace") or [])
               if pr.get("sql_score") is not None]
        return (sum(pts) / len(pts)) if pts else None
    entry = eval_block.get(name) or {}
    return entry.get("score")


def _heal_flagged_records(results: list, flagged_by_qid: dict, samples_by_qid: dict,
                          hard_delta: float) -> tuple[list, list]:
    """Replace each recovered question's stored record with a representative re-run.

    A question is 'recovered' when, for EVERY flagged metric, the median score across
    its re-runs clears the hard-drop threshold (baseline_mean - hard_delta). Recovered
    questions are healed in place (so a transient failure doesn't taint the baseline);
    questions that stay bad keep their original record and remain flagged.

    Returns (healed_qids, still_bad_qids). Mutates `results` in place.
    """
    from statistics import median as _median

    by_qid = {r.get("question_id"): i for i, r in enumerate(results)
              if is_question_record(r)}
    healed, still_bad = [], []
    for qid, metrics in flagged_by_qid.items():
        samples = samples_by_qid.get(qid) or []
        recovered = bool(samples)
        for metric, baseline_mean in metrics.items():
            scores = [s for s in (_metric_score(smp.get("eval") or {}, metric)
                                  for smp in samples) if s is not None]
            if not scores or _median(scores) <= baseline_mean - hard_delta:
                recovered = False
                break
        if not recovered:
            still_bad.append(qid)
            continue
        # Heal with the median-quality sample (by agent_response, else first flagged metric).
        primary = "agent_response" if "agent_response" in metrics else next(iter(metrics))
        ordered = sorted(samples, key=lambda s: _metric_score(s.get("eval") or {}, primary) or 0.0)
        rep = ordered[len(ordered) // 2]
        if qid in by_qid:
            rep["run_id"] = results[by_qid[qid]].get("run_id")  # keep run identity
            results[by_qid[qid]] = rep
            healed.append(qid)
        else:
            still_bad.append(qid)
    return healed, still_bad


def _confirm_targeted_regressions(
    run_path: Path, comparison: dict, *,
    auth: dict, base_url: str, env: str | None, tunnel_pool,
    reference_date: str | None, k: int,
    n: int = DEFAULT_N, delta: float = DEFAULT_DELTA,
) -> dict:
    """Re-run each targeted-flagged question k times, HEAL the ones that recover, then
    re-aggregate + re-compare so the verdict reflects the healed data.

    Recovered questions (transient failures / one-off bad answers) get their stored
    record replaced by a representative re-run, so a cleared run enters the baseline
    with good data instead of the transient failure. Questions that reproduce their
    drop keep their original record and remain a real regression.
    """
    import json
    from collections import defaultdict
    from tests.validator.comparator import DEFAULT_QUESTION_HARD_DELTA, compare_run
    from tests.validator.aggregator import write_aggregated
    from tests.validator.runner import run_one_with_auth_retry, load_questions as _load_q
    from tests.validator.evaluator import _score_one, load_schema_docs

    targeted = comparison.get("targeted") or []
    if not targeted:
        return comparison

    flagged_by_qid: dict = defaultdict(dict)
    for t in targeted:
        flagged_by_qid[t["question_id"]][t["metric"]] = t["baseline_mean"]

    qmeta = {q["id"]: q for q in _load_q(QUESTIONS_PATH)}
    schema_cache = load_schema_docs()
    flagged_qids = [q for q in flagged_by_qid if qmeta.get(q)]
    print(f"\n[confirm] Re-running {len(flagged_qids)} flagged question(s) x{k} "
          "to confirm/heal (a regression is kept only if it reproduces) ...")

    samples_by_qid: dict = {}
    for pos, qid in enumerate(flagged_qids, 1):
        meta = qmeta[qid]
        flow = meta.get("flow", "?")
        flagged = flagged_by_qid[qid]
        flagged_desc = ", ".join(f"{m} (baseline {b:.2f})" for m, b in flagged.items())
        print(f"\n[confirm {pos}/{len(flagged_qids)}] q{qid} [{flow}] — flagged on {flagged_desc}")
        recs = []
        for i in range(k):
            try:
                rec = run_one_with_auth_retry(
                    meta, auth, base_url, run_id=f"confirm-{qid}-{i}",
                    env=env, tunnel_pool=tunnel_pool, reference_date=reference_date)
                # A re-run that errored (401/timeout/etc.) is NOT a valid sample —
                # scoring it as a low result would falsely "confirm" a regression.
                if rec.get("error"):
                    print(f"  attempt {i + 1}/{k}: run error, skipped — {str(rec['error'])[:80]}")
                    continue
                rec["eval"] = _score_one(rec, schema_cache)
                recs.append(rec)
                scores = ", ".join(
                    f"{m}={s:.2f}" for m in flagged
                    if (s := _metric_score(rec.get("eval") or {}, m)) is not None
                )
                print(f"  attempt {i + 1}/{k}: {scores or '(no score)'}")
            except Exception as e:
                print(f"  attempt {i + 1}/{k}: FAILED — {type(e).__name__}: {e}")
        samples_by_qid[qid] = recs

    run = json.loads(Path(run_path).read_text())
    results = run.get("results") or []
    healed, still_bad = _heal_flagged_records(
        results, dict(flagged_by_qid), samples_by_qid, DEFAULT_QUESTION_HARD_DELTA)
    run["results"] = results
    Path(run_path).write_text(json.dumps(run, indent=2, default=str))
    print()
    for qid in flagged_qids:
        verdict = "HEALED (noise / transient — recovered on re-run)" if qid in healed \
            else "CONFIRMED regression (reproduced on re-run)"
        print(f"[confirm] q{qid}: {verdict}")
    print(f"[confirm] {len(healed)} healed {healed}; {len(still_bad)} confirmed {still_bad}")

    # Re-aggregate + re-compare on the healed data → the verdict now reflects reality.
    write_aggregated(Path(run_path))
    fresh = compare_run(Path(run_path), n=n, delta=delta)
    fresh["confirmed"] = True
    fresh["healed_questions"] = healed
    run = json.loads(Path(run_path).read_text())
    run["comparison"] = fresh
    Path(run_path).write_text(json.dumps(run, indent=2, default=str))
    return fresh


def _setup_tunnels(ns):
    """Set up the SSM tunnel pool when --tunnel is requested. Returns (pool, ok)."""
    if not ns.tunnel:
        return None, True
    if not ns.env:
        print("--tunnel requires --env (e.g. --env dev)")
        return None, False
    from tests.validator.tunnel import TunnelPool, install_sigint_handler
    tunnel_pool = TunnelPool.for_env(
        ns.env,
        fdl_port=ns.tunnel_fdl_port, od_port=ns.tunnel_od_port,
        ao_port=ns.tunnel_ao_port, rca_port=ns.tunnel_rca_port,
    )
    install_sigint_handler(tunnel_pool)
    print(f"\nSSM tunnel pool ready for env={ns.env!r} "
          f"(fdl:{ns.tunnel_fdl_port} od:{ns.tunnel_od_port} "
          f"ao:{ns.tunnel_ao_port} rca:{ns.tunnel_rca_port})")
    print("  Tunnels will start on demand just before each question that needs them.")
    print("  AO tunnels direct-to-task (bypasses HTTPS ALB, container speaks HTTP).")
    return tunnel_pool, True


def _run_confirm_file(ns) -> int:
    """--confirm-file: compare an existing (already collected + scored) run file against
    the baseline; if it flags a targeted per-question regression, re-run those questions
    against a live service to confirm/heal, then persist only if it resolves to PASSED.

    Reference date + base URL default to what the run was collected with, so the re-runs
    stay date-consistent. This is what the UI "Persist to DB" button drives.
    """
    import json as _json
    run_path = Path(ns.confirm_file)
    if not run_path.exists():
        print(f"File not found: {run_path}")
        return 1
    run = _json.loads(run_path.read_text())
    env = ns.env or run.get("env")
    base_url = ns.base_url or run.get("base_url")
    reference_date = (run.get("reference_date") or ns.reference_date
                      or os.environ.get("VALIDATOR_REFERENCE_DATE"))

    # A persisted confirm-file run mixes its original records (collected against the
    # STORED base_url) with any healed re-runs (which hit the EFFECTIVE base_url — a
    # --base-url override can point elsewhere than the run was collected). Either being
    # localhost risks polluting the shared baseline with dev-service data, so guard both.
    if ns.persist:
        for provenance, url in (("was collected against", run.get("base_url")),
                                ("would be re-run against", base_url)):
            if _should_refuse_local_persist(url, ns.allow_local_persist):
                print(
                    f"--persist refused: run {provenance} a localhost base_url ({url!r}) "
                    "and CI=true is not set. Pass --allow-local-persist if you really mean it."
                )
                return 1

    if ns.no_compare:
        if ns.persist:
            _persist_to_dynamodb(run_path)
        return 0

    tunnel_pool, ok = _setup_tunnels(ns)
    if not ok:
        return 1
    try:
        confirm = None
        if ns.confirm_runs > 0:
            if not base_url:
                print("[confirm-file] no base URL available — cannot re-run to confirm; "
                      "comparison will stand as-is.")
            else:
                confirm = {
                    "auth": resolve_auth(env=env), "base_url": base_url, "env": env,
                    "tunnel_pool": tunnel_pool, "reference_date": reference_date,
                    "k": ns.confirm_runs,
                }
        exit_code = _compare_run(
            run_path, n=ns.baseline_n, delta=ns.regression_delta,
            confirm=confirm, persist=ns.persist,
        )
        _write_html(run_path)
        return exit_code
    finally:
        if tunnel_pool:
            tunnel_pool.stop_all()


def _print_regression_summary(comparison: dict) -> None:
    status = comparison.get("overall_status", "UNKNOWN")
    baseline_runs = comparison.get("baseline_runs", 0)
    regressions = comparison.get("regressions") or []
    improvements = comparison.get("improvements") or []

    print()
    print("=" * 60)
    print("REGRESSION REPORT")
    print("=" * 60)
    print(f"  Baseline: {baseline_runs} run(s)")
    print(f"  Status:   {status}")
    if comparison.get("baseline_sufficient") is False and status != "FAILED":
        print(f"  (regression check skipped — need {comparison.get('required_baseline_runs')} runs)")
    kinds = comparison.get("regression_kinds") or []
    if kinds:
        print(f"  Signals:  {', '.join(kinds)}")

    # Aggregate signal (the primary verdict driver).
    metrics = comparison.get("metrics") or {}
    if metrics:
        print("\n  AGGREGATE vs BASELINE (drop = baseline - this run):")
        for name, d in metrics.items():
            print(
                f"    {name:<26s} baseline={d.get('baseline_mean', 0):.3f}"
                f" sigma={d.get('sigma', 0):.3f}"
                f" current={d.get('current_mean', 0):.3f}"
                f" drop={d.get('drop', 0):+.3f}  [{d.get('status', '')}]"
            )

    # Per-flow agent_response signal.
    flows = comparison.get("flows") or {}
    if flows:
        print("\n  PER-FLOW agent_response vs BASELINE:")
        for fl, d in sorted(flows.items()):
            mark = "  <== REGRESSION" if d.get("status") == "regression" else ""
            print(
                f"    {fl:<26s} baseline={d.get('baseline_mean', 0):.3f}"
                f" sigma={d.get('sigma', 0):.3f}"
                f" current={d.get('current_mean', 0):.3f}"
                f" drop={d.get('drop', 0):+.3f}  [{d.get('status', '')}]{mark}"
            )
    targeted = comparison.get("targeted") or []
    if targeted:
        print(f"\n  TARGETED BREAKS ({len(targeted)}): "
              + ", ".join(f"q{t['question_id']}/{t['metric']}" for t in targeted))
    breadth = comparison.get("breadth") or {}
    if breadth.get("triggered"):
        print(f"  BREADTH: {breadth.get('regressed_questions')}/{breadth.get('scored_questions')} "
              "questions regressed (broad drop)")

    if regressions:
        print(f"\n  Per-question drops, informational ({len(regressions)}):")
        for r in regressions:
            print(
                f"    q{r['question_id']:<4d} {r['metric']:<26s}"
                f" baseline={r['baseline_mean']:.3f}"
                f" current={r['current_score']:.3f}"
                f" delta={r['delta']:+.3f}"
            )
    else:
        print("\n  No regressions detected.")

    if improvements:
        print(f"\n  IMPROVEMENTS ({len(improvements)}):")
        for r in improvements:
            print(
                f"    q{r['question_id']:<4d} {r['metric']:<26s}"
                f" baseline={r['baseline_mean']:.3f}"
                f" current={r['current_score']:.3f}"
                f" delta={r['delta']:+.3f}"
            )

    performance = comparison.get("performance") or {}
    if performance:
        print("\n  PERFORMANCE vs BASELINE:")
        for entry in performance.values():
            if "delta_pct" in entry:
                delta_str = f"{entry['delta_pct']:+.1f}%"
            else:
                delta_str = f"{entry['delta']:+.2f}"
            arrow = "▲" if (entry.get("delta_pct") or entry.get("delta") or 0) > 0 else "▼"
            flag = "  <== REGRESSION" if entry.get("regressed") else ""
            print(
                f"    {entry['label']:<30s}"
                f" baseline={entry['baseline_mean']:,.1f}"
                f" current={entry['current']:,.1f}"
                f" {arrow} {delta_str}{flag}"
            )
    print("=" * 60)


def _write_html(run_path: Path) -> None:
    try:
        from tests.validator.reporter import generate_html_report
        html_path = generate_html_report(run_path)
        print(f"Report:  {html_path}")
    except Exception as e:
        print(f"HTML report skipped: {e}")


def _parse_ids(s: str | None) -> list[int] | None:
    if not s:
        return None
    return [int(x) for x in s.split(",") if x.strip()]


def _make_filter_config(ns: argparse.Namespace) -> dict:
    return {
        "flow": ns.flow,
        "tags": ns.tags,
        "complexity": ns.complexity,
        "limit": ns.limit,
        "limit_per_flow": ns.limit_per_flow,
        "ids": ns.ids,
        "priority_only": ns.priority_only,
        "tool_flows_only": ns.tool_flows_only,
        "skip_eval": ns.skip_eval,
        "persist": ns.persist,
        "workers": ns.workers,
    }


async def _async_main(ns: argparse.Namespace) -> int:
    # ── Chain evaluation mode ─────────────────────────────────────────────────
    if ns.chains:
        return await _async_chains(ns)

    # ── Re-score only (no agent invocation) ──────────────────────────────────
    if ns.eval_only or ns.eval_latest:
        from tests.validator.evaluator import evaluate_run_file
        if ns.eval_latest:
            run_files = sorted(DEFAULT_OUT_DIR.glob("run_*.json"))
            if not run_files:
                print("No run files found in results/.")
                return 1
            run_path = run_files[-1]
            print(f"Re-scoring latest: {run_path}")
        else:
            run_path = Path(ns.eval_only)
            if not run_path.exists():
                print(f"File not found: {run_path}")
                return 1
            print(f"Re-scoring {run_path} ...")

        # Optional subset filters — re-score only matching question records
        # (e.g. --flow refresh) of a completed report, cheaply, leaving the rest.
        only_flows = {ns.flow} if ns.flow else None
        only_ids = set(_parse_ids(ns.ids)) if ns.ids else None
        if only_flows:
            print(f"[eval] Restricting re-scoring to flow(s): {sorted(only_flows)}")
        if only_ids:
            print(f"[eval] Restricting re-scoring to ids: {sorted(only_ids)}")
        evaluate_run_file(
            run_path, skip_scored=ns.skip_scored, workers=ns.workers,
            only_flows=only_flows, only_ids=only_ids,
        )

        import json as _json2
        run_data = _json2.loads(run_path.read_text())
        if has_chains(run_data.get("results")):
            print("\nRunning chain eval pass ...")
            from tests.validator.chain_evaluator import evaluate_chain_file
            evaluate_chain_file(run_path, skip_scored=ns.skip_scored, workers=getattr(ns, "workers", 1))

        agg = write_aggregated(run_path)
        print_summary(agg)
        _write_html(run_path)
        exit_code = 0
        if ns.persist:
            stored_base_url = _read_run_meta_base_url(run_path)
            if _should_refuse_local_persist(stored_base_url, ns.allow_local_persist):
                print(
                    "--persist refused: run was collected against a localhost base_url "
                    f"({stored_base_url!r}) and CI=true is not set. Persisting a local run "
                    "would pollute the dev-illum validator tables. Pass --allow-local-persist "
                    "if you really mean it."
                )
                return 1
            # When comparing, _compare_run persists only if the verdict is PASSED
            # (FAILED/REGRESSION are kept out of the baseline). Only persist
            # unconditionally when comparison is disabled.
            if ns.no_compare:
                _persist_to_dynamodb(run_path)
            else:
                exit_code = _compare_run(
                    run_path,
                    n=ns.baseline_n,
                    delta=ns.regression_delta,
                )
        return exit_code

    # ── Confirm/heal + persist an existing run file (against a live service) ──
    if ns.confirm_file:
        return _run_confirm_file(ns)

    # ── SSM tunnels (optional, local runs only) ───────────────────────────────
    tunnel_pool, ok = _setup_tunnels(ns)
    if not ok:
        return 1

    # ── Validate flag combinations ───────────────────────────────────────────
    if ns.persist and ns.skip_eval:
        print("--persist requires a full run with eval. Remove --skip-eval or remove --persist.")
        return 1
    # Guard: local-developer runs (service on localhost) must not silently
    # write to the shared dev-illum DynamoDB tables. CI sets CI=true so its
    # nightly runs sail through; a developer who genuinely wants to write to
    # dev-illum from their laptop has to pass --allow-local-persist.
    if ns.persist and _should_refuse_local_persist(ns.base_url, ns.allow_local_persist):
        print(
            "--persist refused: target base_url is localhost and CI=true is not set. "
            "Persisting from a local machine would pollute the dev-illum validator tables. "
            "Pass --allow-local-persist if you really mean it."
        )
        return 1

    # ── Auth + URL required for collection ───────────────────────────────────
    if not ns.base_url:
        print("--base-url is required (e.g. http://localhost:80 when running the service locally, "
              "or http://localhost:8080 when tunneling to the deployed service).")
        return 1

    print("Authenticating ...")
    auth = resolve_auth(env=ns.env)
    if not auth.get("token"):
        print("Authentication failed. Check --env and Secrets Manager access, or set EVAL_BEARER_TOKEN.")
        return 1
    print(f"  tenant_id={auth.get('tenant_id')!r}")

    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Single-turn question run ─────────────────────────────────────────────
    chain_ids_requested = _parse_ids(ns.chain_ids) if ns.chain_ids else None
    question_ids_requested = _parse_ids(ns.ids)

    all_questions = load_questions(QUESTIONS_PATH)
    selected = select_questions(
        all_questions,
        flow=ns.flow,
        tags=[t.strip() for t in ns.tags.split(",") if t.strip()] if ns.tags else None,
        complexity=ns.complexity,
        limit=ns.limit,
        limit_per_flow=ns.limit_per_flow,
        ids=question_ids_requested,
        priority_only=ns.priority_only,
        tool_flows_only=ns.tool_flows_only,
    )
    if not selected and not chain_ids_requested:
        print("No questions matched the given filters. Adjust --flow / --tags / --complexity / --ids.")
        return 1

    total_tests = len(selected) + (len(chain_ids_requested) if chain_ids_requested else 0)
    print(f"Selected {len(selected)} question(s)"
          + (f", {len(chain_ids_requested)} chain(s)" if chain_ids_requested else "")
          + f"  [total {total_tests}]")
    print(f"Workers: {ns.workers} (collection={'parallel' if ns.workers > 1 else 'sequential'}, "
          f"scoring={'parallel' if ns.workers > 1 else 'sequential'})")

    reference_date = ns.reference_date or os.environ.get("VALIDATOR_REFERENCE_DATE")
    if reference_date:
        print(f"Reference date: {reference_date}")
        os.environ["VALIDATOR_REFERENCE_DATE"] = reference_date

    out_path = (
        Path(ns.output)
        if ns.output
        else DEFAULT_OUT_DIR / f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    )
    print(f"Output: {out_path}")

    run: dict = {"results": []}
    if selected:
        run = await run_all(
            selected,
            auth,
            base_url=ns.base_url,
            out_path=out_path,
            filter_config=_make_filter_config(ns),
            env=ns.env,
            tunnel_pool=tunnel_pool,
            workers=ns.workers,
            reference_date=reference_date,
        )

    # ── Chain run (inline, same output file) ─────────────────────────────────
    if chain_ids_requested and CHAINS_PATH.exists():
        from tests.validator.chain_runner import load_chains, select_chains, run_all_chains
        all_chains = load_chains(CHAINS_PATH)
        selected_chains = select_chains(all_chains, chain_ids=chain_ids_requested)
        if selected_chains:
            print(f"\nRunning {len(selected_chains)} chain(s) ...")
            chain_out = out_path.with_suffix(".chains.json")
            chain_run = run_all_chains(
                selected_chains, auth,
                base_url=ns.base_url,
                out_path=chain_out,
                env=ns.env,
                tunnel_pool=tunnel_pool,
                parallel=ns.workers,
                reference_date=reference_date,
            )
            import json as _json
            run.setdefault("results", []).extend(chain_run.get("results", []))
            out_path.write_text(_json.dumps(run, indent=2, default=str))
            chain_out.unlink(missing_ok=True)

    print(f"\nCollected {len(run['results'])} record(s) -> {out_path}")

    if ns.skip_eval:
        print("--skip-eval set; skipping LLM scoring pass.")
        agg = write_aggregated(out_path)
        print_summary(agg)
        _write_html(out_path)
        if tunnel_pool:
            tunnel_pool.stop_all()
        return 0

    eval_workers = ns.workers
    print(f"\nRunning eval pass (workers={eval_workers}) ...")
    from tests.validator.evaluator import evaluate_run_file
    evaluate_run_file(out_path, workers=eval_workers)

    if has_chains(run.get("results")):
        print("\nRunning chain eval pass ...")
        from tests.validator.chain_evaluator import evaluate_chain_file
        evaluate_chain_file(out_path, workers=ns.workers)

    agg = write_aggregated(out_path)
    print_summary(agg)
    _write_html(out_path)

    exit_code = 0
    if ns.persist:
        # _compare_run persists only PASSED runs; persist unconditionally only
        # when comparison is off.
        if ns.no_compare:
            _persist_to_dynamodb(out_path)
        else:
            # Live service is still up here (tunnels stopped below) → enable
            # re-run confirmation of targeted regressions.
            confirm = None
            if ns.confirm_runs > 0:
                confirm = {
                    "auth": auth, "base_url": ns.base_url, "env": ns.env,
                    "tunnel_pool": tunnel_pool, "reference_date": reference_date,
                    "k": ns.confirm_runs,
                }
            exit_code = _compare_run(
                out_path,
                n=ns.baseline_n,
                delta=ns.regression_delta,
                confirm=confirm,
            )

    if tunnel_pool:
        tunnel_pool.stop_all()
    return exit_code




async def _async_chains(ns: argparse.Namespace) -> int:
    """Entry point for --chains mode."""
    from tests.validator.chain_runner import (
        load_chains, select_chains, run_all_chains,
    )

    if not CHAINS_PATH.exists():
        print(f"Chain catalog not found: {CHAINS_PATH}")
        return 1

    # ── SSM tunnels ───────────────────────────────────────────────────────────
    tunnel_pool, ok = _setup_tunnels(ns)
    if not ok:
        return 1

    if not ns.base_url:
        print("--base-url is required for chain runs.")
        return 1

    print("Authenticating ...")
    auth = resolve_auth(env=ns.env)
    if not auth.get("token"):
        print("Authentication failed.")
        return 1
    print(f"  tenant_id={auth.get('tenant_id')!r}")

    all_chains = load_chains(CHAINS_PATH)
    chain_ids = [int(x) for x in ns.chain_ids.split(",") if x.strip()] if ns.chain_ids else None
    selected = select_chains(
        all_chains,
        chain_ids=chain_ids,
        flow=ns.flow,
        priority_only=ns.priority_only,
        limit=ns.limit,
    )
    if not selected:
        print("No chains matched the given filters.")
        return 1

    print(f"Selected {len(selected)}/{len(all_chains)} chains.")
    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(ns.output) if ns.output
        else DEFAULT_OUT_DIR / f"chains_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    )
    print(f"Output: {out_path}")

    run = run_all_chains(
        selected, auth,
        base_url=ns.base_url,
        out_path=out_path,
        env=ns.env,
        tunnel_pool=tunnel_pool,
    )
    print(f"\nCollected {len(run['results'])} chains -> {out_path}")

    if ns.skip_eval:
        print("--skip-eval set; skipping LLM scoring pass.")
        if tunnel_pool:
            tunnel_pool.stop_all()
        return 0

    print("\nRunning chain eval pass ...")
    from tests.validator.chain_evaluator import evaluate_chain_file
    evaluate_chain_file(out_path, workers=ns.workers)
    print(f"\nChain scoring complete -> {out_path}")

    if tunnel_pool:
        tunnel_pool.stop_all()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Analytics Agent Validator CLI")

    p.add_argument("--env", type=str, default=None,
                   help="Environment for auth (dev|staging|production). "
                        "Reads credentials from Secrets Manager at "
                        "{env}-illum/analytics-agentic/secrets. "
                        "Overridden by EVAL_BEARER_TOKEN if set.")
    p.add_argument("--base-url", dest="base_url", type=str, default=None,
                   help="Base URL of the service (e.g. http://localhost:80). "
                        "Required unless using --eval-only / --eval-latest.")
    p.add_argument("--tunnel", action="store_true",
                   help="Enable on-demand SSM tunnels for local runs. Each tunnel is started "
                        "just before the first question that needs it and health-checked before "
                        "each subsequent question. Requires --env and --base-url http://localhost:80.")
    p.add_argument("--tunnel-fdl-port", dest="tunnel_fdl_port", type=int, default=9007,
                   help="Local port for the FDL SSM tunnel (default: 9007).")
    p.add_argument("--tunnel-od-port", dest="tunnel_od_port", type=int, default=9009,
                   help="Local port for the OD SSM tunnel (default: 9009).")
    p.add_argument("--tunnel-ao-port", dest="tunnel_ao_port", type=int, default=8081,
                   help="Local port for the AO SSM tunnel (default: 8081).")
    p.add_argument("--tunnel-rca-port", dest="tunnel_rca_port", type=int, default=9010,
                   help="Local port for the RCA SSM tunnel (default: 9010).")

    p.add_argument("--chains", action="store_true",
                   help="Run multi-turn chain validation (uses test_chains.json) instead "
                        "of single-turn questions.")
    p.add_argument("--chain-ids", dest="chain_ids", type=str, default=None,
                   help="Comma-separated chain IDs to run (e.g. 1,3,5). Chains mode only.")

    p.add_argument("--flow", type=str, default=None,
                   choices=sorted(VALID_FLOWS),
                   help="Filter to a specific flow.")
    p.add_argument("--tags", type=str, default=None,
                   help="Comma-separated tags to filter questions (e.g. csat,acd).")
    p.add_argument("--complexity", type=str, default=None,
                   choices=["basic", "intermediate", "advanced"],
                   help="Filter by complexity level.")
    p.add_argument("--limit", type=int, default=None,
                   help="Max questions to run after other filters.")
    p.add_argument("--limit-per-flow", dest="limit_per_flow", type=int, default=None,
                   help="Max questions per flow (e.g. 1 runs one question from each flow).")
    p.add_argument("--ids", type=str, default=None,
                   help="Comma-separated question ids (overrides other filters).")
    p.add_argument("--priority-only", action="store_true",
                   help="Only run questions tagged priority=true.")
    p.add_argument("--tool-flows-only", dest="tool_flows_only", action="store_true",
                   help=f"Restrict to tool-calling flows only ({', '.join(sorted(TOOL_FLOWS))}). "
                        "Excludes conversational, context, refresh. Intended for CI smoke runs.")

    p.add_argument("--skip-eval", action="store_true",
                   help="Collect traces only; skip LLM scoring pass.")
    p.add_argument("--eval-only", type=str, default=None, metavar="PATH",
                   help="Re-score an existing result JSON without re-running questions.")
    p.add_argument("--eval-latest", action="store_true",
                   help="Re-score the most recent run file in results/.")
    p.add_argument("--confirm-file", dest="confirm_file", type=str, default=None, metavar="PATH",
                   help="Compare an existing scored run file against the baseline and, if it "
                        "flags a targeted per-question regression, re-run those questions "
                        "against the live service to confirm/heal (see --confirm-runs), then "
                        "persist if it resolves to PASSED. Needs --base-url (or the run's "
                        "stored base_url) reachable. Drives the UI 'Persist to DB' button.")
    p.add_argument("--skip-scored", action="store_true",
                   help="Skip records that already have complete scores (for resuming).")
    p.add_argument("--persist", action="store_true",
                   help="Write results and scores to DynamoDB after the run. "
                        "Requires a full scored run — incompatible with --skip-eval. "
                        "Off by default; use only for nightly/scheduled runs. "
                        "Refused when --base-url points at localhost unless CI=true is "
                        "set in the environment or --allow-local-persist is passed.")
    p.add_argument("--allow-local-persist", dest="allow_local_persist", action="store_true",
                   help="Escape hatch: permit --persist even when running against a "
                        "localhost --base-url. Required because that combination usually "
                        "means a developer is testing on their laptop and would otherwise "
                        "pollute the dev-illum validator tables.")
    p.add_argument("--no-compare", dest="no_compare", action="store_true",
                   help="Skip regression comparison after --persist. Useful when there "
                        "is no baseline yet or when comparison is not desired.")
    p.add_argument("--baseline-n", dest="baseline_n", type=int, default=DEFAULT_N,
                   help=f"Baseline size: number of prior runs to compare against, and "
                        f"the minimum required before any regression is assessed "
                        f"(default: {DEFAULT_N}). With fewer than this many eligible "
                        "prior runs on record, the regression check is skipped.")
    p.add_argument("--regression-delta", dest="regression_delta", type=float,
                   default=DEFAULT_DELTA,
                   help=f"Per-question score-drop threshold for informational "
                        f"findings + breadth counting (default: {DEFAULT_DELTA}). "
                        "The pass/fail verdict is driven by aggregate signals.")
    p.add_argument("--confirm-runs", dest="confirm_runs", type=int, default=2,
                   help="When a targeted per-question regression is flagged, re-run each "
                        "flagged question this many times and keep only regressions that "
                        "reproduce (filters agent-variance noise). 0 disables confirmation. "
                        "Full-run path only — needs the live service. Default: 2.")

    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel workers for both collection and scoring "
                        "(default: 1 = sequential). 3-5 is a good balance; higher values "
                        "may hit Bedrock rate limits.")
    p.add_argument("--reference-date", dest="reference_date", type=str, default=None,
                   metavar="YYYY-MM-DD",
                   help="Pin 'today' to a fixed date in all agent prompts. "
                        "Use when test data is static so date-relative queries "
                        "('this month', 'last 30 days') stay in range. "
                        "Example: --reference-date 2025-11-30")

    p.add_argument("--output", type=str, default=None,
                   help="Explicit output path (default: results/run_<timestamp>.json).")

    ns = p.parse_args()

    try:
        return asyncio.run(_async_main(ns))
    except RuntimeError as e:
        if "Timeout should be used inside a task" in str(e):
            return 0
        raise


if __name__ == "__main__":
    raise SystemExit(main())
