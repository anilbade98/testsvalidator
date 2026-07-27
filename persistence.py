"""DynamoDB writer for validator run results.

Persists per-run aggregate records and per-question detail records to two
DynamoDB tables provisioned by `cdk/app_stack.py`:

  - `{env}-validator-runs`              — one item per run, PK: `run_id`.
                                          GSI `env-timestamp-index` (env, run_timestamp)
                                          drives the comparator's "last N runs" query.
  - `{env}-validator-question-records`  — one item per question per run,
                                          composite PK (run_id, question_id).
                                          GSI `question-run-index` (question_id, run_id)
                                          drives per-question score history.

Opt-in path: the runner only instantiates this writer when invoked with the
`--persist` flag. By default, runs against a localhost `--base-url` are
*also* refused — even with `--persist` set — unless the environment has
`CI=true` or the user explicitly passes `--allow-local-persist`. The intent
is that a developer running on their laptop never accidentally pollutes
the shared `dev-illum` validator tables; only nightly CI (and explicit
opt-in) reaches DynamoDB. All other local runs write only to the local
JSON result file.

DynamoDB writes use `PutItem` semantics, which is upsert by PK. Re-running with
`--eval-latest --persist` correctly replaces existing records without creating
duplicates. The writer is therefore safe to call repeatedly on the same run.
"""

from __future__ import annotations

import json
import logging
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, TypedDict

import boto3
from boto3.dynamodb.conditions import Key

from tests.validator.records import failed_question_ids, question_records, run_is_complete

logger = logging.getLogger(__name__)


# ── Record schemas ────────────────────────────────────────────────────────────
#
# DynamoDB doesn't declare non-key attributes upfront — the table schema only
# pins the PK / SK and any attributes used by a GSI. The "columns" listed in
# the CSA-85049 ticket and HLD live as the shape of the dicts we write, and
# these `TypedDict`s are their canonical definition.
#
# Mark anything not strictly required as `total=False`-style optional. Optional
# fields with a None value are dropped at write time by `_drop_empty`.


class RunRecord(TypedDict, total=False):
    """One row in the `{env}-validator-runs` table.

    PK: `run_id`. GSI `env-timestamp-index` keys: (`env`, `run_timestamp`).
    """
    # Identity / context
    run_id: str                    # required — PutItem PK
    run_timestamp: str             # ISO 8601 — GSI SK
    env: str                       # dev-illum | test-illum | staging-illum | production-illum — GSI PK
    base_url: str
    model_id: str                  # Bedrock model id used by the agent under test
    filter_config: dict            # flow, tags, complexity, priority_only, limit, ids, persist

    # Run aggregates
    question_count: int
    avg_duration_ms: float
    avg_total_tokens: float
    avg_sql_call_count: float
    avg_tool_call_count: float

    # Per-metric aggregates (mean + pass rate at threshold 0.7)
    agent_response_mean: float
    agent_response_pass_rate: float
    interpretation_mean: float
    interpretation_pass_rate: float
    sql_correctness_mean: float
    sql_correctness_pass_rate: float

    # Per-flow agent_response aggregates, keyed by flow name:
    #   {flow: {total, agent_response_mean, agent_response_pass_rate}}
    # Drives flow-level regression detection.
    by_flow: dict

    # Verdict + comparator output (populated by CSA-85050)
    overall_status: str            # PASSED | REGRESSION | FAILED
    comparison: dict


class OdFdlResult(TypedDict, total=False):
    """One od/fdl query result inside a question record's `od_fdl_results` list."""
    tool_name: str
    interpretation: str
    schema: str
    column_meta: list
    rows: list


class QuestionRecord(TypedDict, total=False):
    """One row in the `{env}-validator-question-records` table.

    PK: `run_id`, SK: `question_id`. GSI `question-run-index` keys:
    (`question_id`, `run_id`).
    """
    # Identity
    run_id: str                    # required — PutItem PK
    question_id: int               # required — PutItem SK
    question: str
    flow: str                      # fdl | od | ao | rca | advisory | mixed | context | refresh | conversational
    complexity: str                # basic | intermediate | advanced
    priority: bool

    # Agent output + telemetry
    agent_response: str
    duration_ms: int
    total_tokens: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    od_fdl_results: list[OdFdlResult]
    sql_call_count: int
    tool_call_count: int

    # Per-metric scores (pass threshold 0.7)
    agent_response_score: float
    agent_response_passed: bool
    agent_response_reason: str
    interpretation_score: float    # None for context / refresh / conversational
    interpretation_passed: bool
    interpretation_reason: str
    sql_correctness_score: float   # None for context / conversational
    sql_correctness_passed: bool
    sql_per_trace_scores: list  # per-sub-agent-reasoning breakdown

    # Run-time housekeeping
    error: str                     # absent on success
    regression_flag: bool

# Question records get a TTL so the table doesn't grow unbounded. Run-level
# aggregates are tiny (one item per run) and kept indefinitely for trend lines;
# per-question detail is bulky (~77 items × every run) so 6 months is enough
# for the comparator's "last N runs" lookback while bounding storage.
QUESTION_RECORD_TTL_DAYS = 180


def _to_ddb(value: Any) -> Any:
    """Recursively convert a JSON-shaped value to a DynamoDB-safe representation.

    - `float` → `Decimal` (DynamoDB rejects floats).
    - `list` / `dict` are recursed.
    - Everything else is passed through unchanged (`int`, `str`, `bool`, `None`).
    """
    if isinstance(value, float):
        # str() avoids the float-binary roundoff that Decimal(float) introduces.
        return Decimal(str(value))
    if isinstance(value, list):
        return [_to_ddb(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_ddb(v) for k, v in value.items()}
    return value


def _drop_empty(record: dict) -> dict:
    """Drop top-level keys whose value is `None` or `""`.

    DynamoDB attributes have to either be present with a value or absent —
    storing literal nulls inflates the item and confuses consumers. Optional
    fields (e.g. `error` on a successful run) just shouldn't appear.

    Nested `None`s inside lists/dicts are preserved as DynamoDB Null type,
    because there they're load-bearing (e.g. `interpretation_score: null` on
    a `context` flow record means "metric N/A", not "field absent").
    """
    return {k: _to_ddb(v) for k, v in record.items() if v is not None and v != ""}


class DynamoDBWriter:
    """Writes validator run + question records to the two DynamoDB tables.

    The writer no longer takes a `region` constructor arg — anyone wanting to override should
    set the env var instead. We *do* pass `region_name` to boto3 explicitly
    rather than relying on its default-session lookup, because that lookup
    is cached on first use process-wide and would otherwise pin the writer
    to whatever region was active when the first boto3 call ran (a real
    cross-test pollution risk).

    The boto3 resource handle is lazily constructed on first `persist()`
    call so importing this module (or constructing the writer in a test
    that monkeypatches boto3 afterward) never triggers a real boto3 init.
    """

    def __init__(self, env: str):
        if not env:
            raise ValueError("env is required (e.g. 'staging-illum')")
        self.env = env
        self.runs_table_name = f"{env}-validator-runs"
        self.questions_table_name = f"{env}-validator-question-records"
        self._dynamodb = None
        self._runs_table = None
        self._questions_table = None

    def _ensure_clients(self) -> None:
        if self._dynamodb is None:
            self._dynamodb = boto3.resource("dynamodb")
            self._runs_table = self._dynamodb.Table(self.runs_table_name)
            self._questions_table = self._dynamodb.Table(self.questions_table_name)

    def persist(self, run_record: RunRecord, question_records: list[QuestionRecord]) -> None:
        """Write one run record + N question records.

        Idempotent. Re-invoking with the same `run_id` and `question_id` values
        overwrites existing items (PutItem semantics). See the `RunRecord` and
        `QuestionRecord` TypedDicts in this module for the expected shapes.
        """
        if not run_record.get("run_id"):
            raise ValueError("run_record must include run_id")

        self._ensure_clients()
        self._write_run(run_record)
        self._write_question_records(run_record["run_id"], question_records)
        logger.info(
            "persisted run %s: %d question records to DynamoDB (env=%s)",
            run_record["run_id"], len(question_records), self.env,
        )

    def _write_run(self, record: dict) -> None:
        """Write the run-level aggregate record."""
        self._runs_table.put_item(Item=_drop_empty(record))

    def _write_question_records(self, run_id: str, records: list[dict]) -> None:
        """Batch-write question-level records.

        `Table.batch_writer()` auto-handles the 25-item-per-batch DynamoDB
        limit, retries throttled items, and flushes the buffer on exit.
        Each item is also stamped with a TTL `QUESTION_RECORD_TTL_DAYS`
        into the future so DynamoDB auto-deletes stale detail records.
        """
        ttl_epoch = int(time.time()) + QUESTION_RECORD_TTL_DAYS * 86400
        skipped = 0
        with self._questions_table.batch_writer() as batch:
            for r in records:
                if r.get("run_id") != run_id or r.get("question_id") is None:
                    logger.warning(
                        "skipping question record with mismatched/missing keys: %s",
                        {k: r.get(k) for k in ("run_id", "question_id")},
                    )
                    skipped += 1
                    continue
                item = _drop_empty(r)
                item["ttl"] = ttl_epoch
                batch.put_item(Item=item)
        if skipped:
            logger.warning("skipped %d malformed question records during persist", skipped)


# ── Adapter: local run JSON → DynamoDB records ───────────────────────────────
#
# `tests/validator/runner.py` writes a JSON file in this shape (after the
# aggregator pass adds `aggregated_scores`):
#
#   {
#     "run_id":          "run_20260624_120000",
#     "run_timestamp":   "2026-06-24T12:00:00Z",
#     "env":             "staging" | "dev" | "production" | "staging-illum" | ...,
#     "base_url":        "https://...",
#     "model_id":        "us.anthropic.claude-opus-4-6-v1",
#     "filter_config":   { ... },
#     "results": [
#       {
#         "question_id": 1, "question": "...", "flow": "fdl", "complexity": "basic",
#         "priority": true, "agent_response": "...", "duration_ms": 28100,
#         "usage": {"total_tokens": ..., "input_tokens": ..., ...},
#         "od_fdl_results": [...],
#         "eval": {
#           "agent_response":         {"score": 0.91, "passed": true, "reason": "..."},
#           "interpretation_coverage":{"score": 0.84, "passed": true, "reason": "..."},
#           "per_trace": [{"source": "fdl", "sql_score": 0.87, "sql_passed": true, "sql_reason": "..."}],
#           "tool_compliance":        {"compliant": true},
#         },
#         "error": null,
#       },
#       ...
#     ],
#     "aggregated_scores": {
#       "agent_response":         {"mean": 0.83, "pass_rate": 0.92, ...},
#       "interpretation_coverage":{"mean": 0.79, "pass_rate": 0.85, ...},
#       "sql_correctness":        {"mean": 0.81, "pass_rate": 0.88, ...},
#       "avg_duration_ms": 31400, "avg_total_tokens": 25812, "total": 12, ...
#     }
#   }
#
# The adapter flattens that into `RunRecord` / `QuestionRecord` shapes
# (defined above) and hands them to `DynamoDBWriter.persist`.

# Maps the CLI's short env names to the long form that table names use.
_ENV_SHORT_TO_LONG = {
    "dev":        "dev-illum",
    "test":       "test-illum",
    "staging":    "staging-illum",
    "production": "production-illum",
}


def _resolve_env(env: str | None) -> str:
    """Normalise env to the `{env}-illum` form used in table names.

    Accepts the CLI short form (`dev` / `staging` / etc.) and the full form
    (`dev-illum` / `staging-illum`). Anything else is returned as-is — callers
    are responsible for catching bad envs upstream.
    """
    if not env:
        raise ValueError("env is required to resolve DynamoDB table names")
    return _ENV_SHORT_TO_LONG.get(env, env)


def _build_run_record(run: dict) -> RunRecord:
    """Flatten the run-level JSON into a `RunRecord`."""
    agg = run.get("aggregated_scores") or {}
    ar = agg.get("agent_response") or {}
    ic = agg.get("interpretation_coverage") or {}
    sc = agg.get("sql_correctness") or {}

    return {  # type: ignore[return-value]
        "run_id":        run.get("run_id"),
        "run_timestamp": run.get("run_timestamp"),
        "env":           _resolve_env(run.get("env")) if run.get("env") else None,
        "base_url":      run.get("base_url"),
        "model_id":      run.get("model_id"),
        "filter_config": run.get("filter_config"),
        "question_count": agg.get("total"),
        "avg_duration_ms": agg.get("avg_duration_ms"),
        "avg_total_tokens": agg.get("avg_total_tokens"),
        "avg_sql_call_count": agg.get("avg_sql_call_count"),
        "avg_tool_call_count": agg.get("avg_tool_call_count"),
        "agent_response_mean":      ar.get("mean"),
        "agent_response_pass_rate": ar.get("pass_rate"),
        "interpretation_mean":      ic.get("mean"),
        "interpretation_pass_rate": ic.get("pass_rate"),
        "sql_correctness_mean":      sc.get("mean"),
        "sql_correctness_pass_rate": sc.get("pass_rate"),
        # Per-flow agent_response means/pass-rates — enables flow-level regression
        # detection (a drop concentrated in one flow that the overall mean hides).
        "by_flow":        agg.get("by_flow"),
        "overall_status": run.get("overall_status"),
        "comparison":     run.get("comparison"),
    }


def _build_question_record(run_id: str, result: dict) -> QuestionRecord:
    """Flatten one per-question result into a `QuestionRecord`."""
    usage = result.get("usage") or {}
    ev = result.get("eval") or {}
    ar = ev.get("agent_response") or {}
    ic = ev.get("interpretation_coverage") or {}
    per_trace = ev.get("per_trace") or []

    # `sql_correctness` per-question is the mean across that question's sub-agent
    # reasoning blocks. The aggregator computes the run-level mean; here we preserve
    # the per-reasoning breakdown plus a rolled-up score for the comparator.
    sql_scores = [pr.get("sql_score") for pr in per_trace if pr.get("sql_score") is not None]
    sql_mean = sum(sql_scores) / len(sql_scores) if sql_scores else None
    sql_verdicts = [pr.get("sql_passed") for pr in per_trace]
    if not sql_verdicts or any(v is None for v in sql_verdicts):
        sql_passed_all: bool | None = None
    else:
        sql_passed_all = all(sql_verdicts)

    return {  # type: ignore[return-value]
        "run_id":       run_id,
        "question_id":  result.get("question_id"),
        "question":     result.get("question"),
        "flow":         result.get("flow"),
        "complexity":   result.get("complexity"),
        "priority":     result.get("priority"),
        "agent_response": result.get("agent_response"),
        "duration_ms":  result.get("duration_ms"),
        "total_tokens":      usage.get("total_tokens"),
        "input_tokens":      usage.get("input_tokens"),
        "output_tokens":     usage.get("output_tokens"),
        "cache_read_tokens": usage.get("cache_read_tokens"),
        "od_fdl_results":    result.get("od_fdl_results"),
        "sql_call_count":    len(result.get("od_fdl_results") or []),
        "tool_call_count":   len(result.get("tools_called") or []),
        "agent_response_score":  ar.get("score"),
        "agent_response_passed": ar.get("passed"),
        "agent_response_reason": ar.get("reason"),
        "interpretation_score":  ic.get("score"),
        "interpretation_passed": ic.get("passed"),
        "interpretation_reason": ic.get("reason"),
        "sql_correctness_score":  sql_mean,
        "sql_correctness_passed": sql_passed_all,
        "sql_per_trace_scores": per_trace,
        "error":           result.get("error"),
        "regression_flag": result.get("regression_flag"),
    }


def persist_run_file(run_path: Path) -> bool:
    """Adapter entry point: read a validator run JSON and persist to DynamoDB.

    Called by `tests/validator/run.py` after `write_aggregated()` whenever the
    user passed `--persist`. Reads the env, run_id, etc. from the JSON itself
    so the caller only has to supply the file path. Region resolution is
    handled by boto3's own chain (`AWS_REGION`, `AWS_DEFAULT_REGION`, profile,
    SDK default) — see `DynamoDBWriter` for details.

    Returns True if the run was persisted, False if it was skipped because the
    run is incomplete (a question errored / returned no result). Incomplete runs
    are never written — they must not pollute the shared baseline tables.
    """
    run = json.loads(Path(run_path).read_text())
    env = run.get("env")
    if not env:
        raise ValueError(
            f"{run_path}: cannot persist — run JSON is missing 'env'. "
            "Was the run executed with --env?"
        )

    if not run_is_complete(run.get("results")):
        bad = failed_question_ids(run.get("results"))
        logger.warning(
            "Run %s is incomplete (%d failed question(s): %s) — NOT persisting to "
            "DynamoDB.", run.get("run_id"), len(bad), bad,
        )
        return False

    writer = DynamoDBWriter(env=_resolve_env(env))
    run_record = _build_run_record(run)
    run_id = run_record["run_id"]
    if not run_id:
        raise ValueError(f"{run_path}: cannot persist — run JSON is missing 'run_id'.")

    # Chain records live in the same `results` list but are keyed by `chain_id`,
    # not `question_id`, and are scored/persisted separately. Excluding them here
    # keeps them from surfacing as "malformed" question records downstream.
    q_records = [
        _build_question_record(run_id, r) for r in question_records(run.get("results"))
    ]
    writer.persist(run_record, q_records)
    return True


# ── Reader: DynamoDB → local run-JSON shape ──────────────────────────────────
#
# The History tab needs to show persisted runs with the *same* detail as local
# run files. Persistence flattens the nested run JSON into DynamoDB items; the
# reader is the exact inverse — it rebuilds the run dict (run meta + results[]
# with nested `eval`) so the History view and the HTML reporter can consume a
# persisted run identically to a local one.


def _from_ddb(value: Any) -> Any:
    """Inverse of `_to_ddb`: Decimal → int/float, recursing lists/dicts."""
    if isinstance(value, Decimal):
        # Integral decimals (counts, ids) come back as int; the rest as float.
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, list):
        return [_from_ddb(v) for v in value]
    if isinstance(value, dict):
        return {k: _from_ddb(v) for k, v in value.items()}
    return value


def _reconstruct_result(q: dict) -> dict:
    """Rebuild one `results[]` entry (with nested `eval`) from a QuestionRecord."""
    return {
        "question_id": q.get("question_id"),
        "question": q.get("question"),
        "flow": q.get("flow"),
        "complexity": q.get("complexity"),
        "priority": q.get("priority"),
        "agent_response": q.get("agent_response"),
        "duration_ms": q.get("duration_ms"),
        "usage": {
            "total_tokens": q.get("total_tokens"),
            "input_tokens": q.get("input_tokens"),
            "output_tokens": q.get("output_tokens"),
            "cache_read_tokens": q.get("cache_read_tokens"),
        },
        "od_fdl_results": q.get("od_fdl_results") or [],
        # tools_called isn't stored individually — only its count. Rebuild a
        # placeholder list of the right length so aggregate() recovers the count.
        "tools_called": [None] * int(q.get("tool_call_count") or 0),
        "eval": {
            "agent_response": {
                "score": q.get("agent_response_score"),
                "passed": q.get("agent_response_passed"),
                "reason": q.get("agent_response_reason"),
            },
            "interpretation_coverage": {
                "score": q.get("interpretation_score"),
                "passed": q.get("interpretation_passed"),
                "reason": q.get("interpretation_reason"),
            },
            "per_trace": q.get("sql_per_trace_scores") or [],
        },
        "error": q.get("error"),
        "regression_flag": q.get("regression_flag"),
    }


class DynamoDBReader:
    """Reads persisted runs back into the local run-JSON shape.

    ``env`` must already be the long form used in table names (e.g.
    ``staging-illum``); use ``_resolve_env`` on CLI short forms first.
    """

    def __init__(self, env: str):
        if not env:
            raise ValueError("env is required (e.g. 'staging-illum')")
        self.env = env
        self.runs_table_name = f"{env}-validator-runs"
        self.questions_table_name = f"{env}-validator-question-records"
        self._dynamodb = None

    def _tables(self):
        if self._dynamodb is None:
            self._dynamodb = boto3.resource("dynamodb")
        return (
            self._dynamodb.Table(self.runs_table_name),
            self._dynamodb.Table(self.questions_table_name),
        )

    def list_runs(self, limit: int = 50) -> list[dict]:
        """Return persisted run records for this env, newest first."""
        runs_table, _ = self._tables()
        resp = runs_table.query(
            IndexName="env-timestamp-index",
            KeyConditionExpression=Key("env").eq(self.env),
            ScanIndexForward=False,
            Limit=limit,
        )
        return [_from_ddb(item) for item in resp.get("Items", [])]

    def load_run(self, run_id: str) -> dict | None:
        """Reconstruct a full run dict (run meta + results[] + aggregated_scores).

        Returns None if the run_id isn't present. aggregated_scores is recomputed
        from the reconstructed results so it matches the local shape exactly
        (means, pass rates, by-flow); persisted run-level averages that can't be
        recomputed (SQL/tool call counts) are copied over.
        """
        from tests.validator.aggregator import aggregate

        runs_table, questions_table = self._tables()
        run_item = runs_table.get_item(Key={"run_id": run_id}).get("Item")
        if not run_item:
            return None
        run_item = _from_ddb(run_item)

        items: list[dict] = []
        kwargs: dict = {"KeyConditionExpression": Key("run_id").eq(run_id)}
        while True:
            resp = questions_table.query(**kwargs)
            items.extend(resp.get("Items", []))
            last_key = resp.get("LastEvaluatedKey")
            if not last_key:
                break
            kwargs["ExclusiveStartKey"] = last_key
        questions = sorted(
            (_from_ddb(i) for i in items),
            key=lambda q: q.get("question_id") or 0,
        )

        run = {
            "run_id": run_item.get("run_id"),
            "run_timestamp": run_item.get("run_timestamp"),
            "env": run_item.get("env"),
            "base_url": run_item.get("base_url"),
            "model_id": run_item.get("model_id"),
            "filter_config": run_item.get("filter_config"),
            "overall_status": run_item.get("overall_status"),
            "comparison": run_item.get("comparison"),
            "results": [_reconstruct_result(q) for q in questions],
        }
        agg = aggregate(run)
        # These aren't derivable from stored question records — carry the
        # persisted run-level values over so they aren't lost.
        for field in ("avg_tool_call_count", "avg_sql_call_count"):
            if run_item.get(field) is not None:
                agg[field] = run_item[field]
        run["aggregated_scores"] = agg
        return run


def list_persisted_runs(env: str, limit: int = 50) -> list[dict]:
    """Convenience wrapper: list persisted runs for an env (short or long form)."""
    return DynamoDBReader(env=_resolve_env(env)).list_runs(limit=limit)


def load_persisted_run(env: str, run_id: str) -> dict | None:
    """Convenience wrapper: reconstruct one persisted run (env short or long form)."""
    return DynamoDBReader(env=_resolve_env(env)).load_run(run_id)
