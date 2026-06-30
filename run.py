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

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_IN", "NO")

import argparse  # noqa: E402
import asyncio  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

from tests.validator.aggregator import print_summary, write_aggregated  # noqa: E402
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
        persist_run_file(run_path)
        print(f"Persisted to DynamoDB: {run_path.name}")
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
            return (json.load(fh).get("meta") or {}).get("base_url")
    except Exception:
        return None


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
        evaluate_run_file(run_path, skip_scored=ns.skip_scored)
        agg = write_aggregated(run_path)
        print_summary(agg)
        _write_html(run_path)
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
            _persist_to_dynamodb(run_path)
        return 0

    # ── SSM tunnels (optional, local runs only) ───────────────────────────────
    tunnel_pool = None
    if ns.tunnel:
        if not ns.env:
            print("--tunnel requires --env (e.g. --env dev)")
            return 1
        from tests.validator.tunnel import TunnelPool, install_sigint_handler
        tunnel_pool = TunnelPool.for_env(
            ns.env,
            fdl_port=ns.tunnel_fdl_port,
            od_port=ns.tunnel_od_port,
            ao_port=ns.tunnel_ao_port,
            rca_port=ns.tunnel_rca_port,
        )
        install_sigint_handler(tunnel_pool)
        print(f"\nSSM tunnel pool ready for env={ns.env!r} "
              f"(fdl:{ns.tunnel_fdl_port} od:{ns.tunnel_od_port} "
              f"ao:{ns.tunnel_ao_port} rca:{ns.tunnel_rca_port})")
        print("  Tunnels will start on demand just before each question that needs them.")
        print("  AO tunnels direct-to-task (bypasses HTTPS ALB, container speaks HTTP).")

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
    all_questions = load_questions(QUESTIONS_PATH)
    selected = select_questions(
        all_questions,
        flow=ns.flow,
        tags=[t.strip() for t in ns.tags.split(",") if t.strip()] if ns.tags else None,
        complexity=ns.complexity,
        limit=ns.limit,
        limit_per_flow=ns.limit_per_flow,
        ids=_parse_ids(ns.ids),
        priority_only=ns.priority_only,
        tool_flows_only=ns.tool_flows_only,
    )
    if not selected:
        print("No questions matched the given filters. Adjust --flow / --tags / --complexity / --ids.")
        return 1

    print(f"Selected {len(selected)}/{len(all_questions)} questions.")
    out_path = (
        Path(ns.output)
        if ns.output
        else DEFAULT_OUT_DIR / f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    )
    print(f"Output: {out_path}")

    run = await run_all(
        selected,
        auth,
        base_url=ns.base_url,
        out_path=out_path,
        filter_config=_make_filter_config(ns),
        env=ns.env,
        tunnel_pool=tunnel_pool,
    )
    print(f"\nCollected {len(run['results'])} records -> {out_path}")

    if ns.skip_eval:
        print("--skip-eval set; skipping LLM scoring pass.")
        agg = write_aggregated(out_path)
        print_summary(agg)
        _write_html(out_path)
        if tunnel_pool:
            tunnel_pool.stop_all()
        return 0

    print("\nRunning eval pass ...")
    from tests.validator.evaluator import evaluate_run_file
    evaluate_run_file(out_path)
    agg = write_aggregated(out_path)
    print_summary(agg)
    _write_html(out_path)

    if ns.persist:
        _persist_to_dynamodb(out_path)

    if tunnel_pool:
        tunnel_pool.stop_all()
    return 0




async def _async_chains(ns: argparse.Namespace) -> int:
    """Entry point for --chains mode."""
    from tests.validator.chain_runner import (
        load_chains, select_chains, run_all_chains,
    )

    if not CHAINS_PATH.exists():
        print(f"Chain catalog not found: {CHAINS_PATH}")
        return 1

    # ── SSM tunnels ───────────────────────────────────────────────────────────
    tunnel_pool = None
    if ns.tunnel:
        if not ns.env:
            print("--tunnel requires --env")
            return 1
        from tests.validator.tunnel import TunnelPool, install_sigint_handler
        tunnel_pool = TunnelPool.for_env(
            ns.env,
            fdl_port=ns.tunnel_fdl_port,
            od_port=ns.tunnel_od_port,
            ao_port=ns.tunnel_ao_port,
            rca_port=ns.tunnel_rca_port,
        )
        install_sigint_handler(tunnel_pool)

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
    evaluate_chain_file(out_path)
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
