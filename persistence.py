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
import os
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, TypedDict

import boto3

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

    # Per-metric aggregates (mean + pass rate at threshold 0.7)
    agent_response_mean: float
    agent_response_pass_rate: float
    interpretation_mean: float
    interpretation_pass_rate: float
    sql_correctness_mean: float
    sql_correctness_pass_rate: float

    # Verdict + comparator output (populated by CSA-85050)
    overall_status: str            # PASSED | REGRESSION | FAILED
    comparison: dict


class CortexTrace(TypedDict, total=False):
    """One SQL call's trace inside a question record's `cortex_traces` list."""
    tool_name: str
    interpretation: str
    generated_sql: str
    sql_results: list              # sampled rows (the writer's input is already truncated)
    sql_row_count_full: int
    status: str                    # SUCCEEDED | FAILED | REJECTED
    error: str


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
    cortex_traces: list[CortexTrace]

    # Per-metric scores (pass threshold 0.7)
    agent_response_score: float
    agent_response_passed: bool
    agent_response_reason: str
    interpretation_score: float    # None for context / refresh / conversational
    interpretation_passed: bool
    interpretation_reason: str
    sql_correctness_score: float   # None for context / conversational
    sql_correctness_passed: bool
    sql_per_trace_scores: list     # per-SQL-call breakdown for multi-query questions

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

    Region resolution: `AWS_REGION` > `AWS_DEFAULT_REGION` > `us-west-2`
    (where the analytics-agentic stack is deployed). The writer no longer
    takes a `region` constructor arg — anyone wanting to override should
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
            region = (
                os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION")
                or "us-west-2"
            )
            self._dynamodb = boto3.resource("dynamodb", region_name=region)
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
#         "cortex_traces": [...],
#         "eval": {
#           "agent_response":         {"score": 0.91, "passed": true, "reason": "..."},
#           "interpretation_coverage":{"score": 0.84, "passed": true, "reason": "..."},
#           "per_trace": [{"sql_score": 0.87, "sql_passed": true, "sql_reason": "..."}],
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
        "agent_response_mean":      ar.get("mean"),
        "agent_response_pass_rate": ar.get("pass_rate"),
        "interpretation_mean":      ic.get("mean"),
        "interpretation_pass_rate": ic.get("pass_rate"),
        "sql_correctness_mean":      sc.get("mean"),
        "sql_correctness_pass_rate": sc.get("pass_rate"),
        # `overall_status` and `comparison` are populated by CSA-85050. Absent here.
    }


def _build_question_record(run_id: str, result: dict) -> QuestionRecord:
    """Flatten one per-question result into a `QuestionRecord`."""
    usage = result.get("usage") or {}
    ev = result.get("eval") or {}
    ar = ev.get("agent_response") or {}
    ic = ev.get("interpretation_coverage") or {}
    per_trace = ev.get("per_trace") or []

    # `sql_correctness` per-question is the mean across that question's SQL
    # calls (each per-trace entry has its own score). The aggregator computes
    # the run-level mean; here we just preserve the per-trace breakdown plus
    # a rolled-up score so the comparator has both granularities.
    sql_scores = [pt.get("sql_score") for pt in per_trace if pt.get("sql_score") is not None]
    sql_mean = sum(sql_scores) / len(sql_scores) if sql_scores else None
    # `sql_correctness_passed`: True iff every trace has an explicit pass verdict.
    # A trace with `sql_passed=None` (evaluator skipped — missing SQL/interpretation)
    # is N/A, not a failure — so we surface `None` for the whole question rather
    # than letting one missing verdict mark the question as failed.
    sql_verdicts = [pt.get("sql_passed") for pt in per_trace]
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
        "cortex_traces":     result.get("cortex_traces"),
        "agent_response_score":  ar.get("score"),
        "agent_response_passed": ar.get("passed"),
        "agent_response_reason": ar.get("reason"),
        "interpretation_score":  ic.get("score"),
        "interpretation_passed": ic.get("passed"),
        "interpretation_reason": ic.get("reason"),
        "sql_correctness_score":  sql_mean,
        "sql_correctness_passed": sql_passed_all,
        "sql_per_trace_scores":   per_trace,
        "error":           result.get("error"),
        # `regression_flag` is populated by CSA-85050. Absent here.
    }


def persist_run_file(run_path: Path) -> None:
    """Adapter entry point: read a validator run JSON and persist to DynamoDB.

    Called by `tests/validator/run.py` after `write_aggregated()` whenever the
    user passed `--persist`. Reads the env, run_id, etc. from the JSON itself
    so the caller only has to supply the file path. Region resolution is
    handled by boto3's own chain (`AWS_REGION`, `AWS_DEFAULT_REGION`, profile,
    SDK default) — see `DynamoDBWriter` for details.
    """
    run = json.loads(Path(run_path).read_text())
    env = run.get("env")
    if not env:
        raise ValueError(
            f"{run_path}: cannot persist — run JSON is missing 'env'. "
            "Was the run executed with --env?"
        )

    writer = DynamoDBWriter(env=_resolve_env(env))
    run_record = _build_run_record(run)
    run_id = run_record["run_id"]
    if not run_id:
        raise ValueError(f"{run_path}: cannot persist — run JSON is missing 'run_id'.")

    question_records = [
        _build_question_record(run_id, r) for r in (run.get("results") or [])
    ]
    writer.persist(run_record, question_records)
