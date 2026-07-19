"""Regression comparator for the Analytics Agent Validator.

After each scored and persisted run, compare the current run's scores against
the rolling mean of the last N runs and flag regressions.

Called automatically after --persist in tests/validator/run.py unless
--no-compare is set. Results are written back to the local run JSON and to
DynamoDB via a second persist() pass that populates `overall_status`,
`comparison`, and per-question `regression_flag`.
"""

from __future__ import annotations

import json
import logging
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key

logger = logging.getLogger(__name__)

# Metrics tracked per question: (score_field, passed_field, display_name)
_METRICS = [
    ("agent_response_score", "agent_response_passed", "agent_response"),
    ("interpretation_score", "interpretation_passed", "interpretation_coverage"),
    ("sql_correctness_score", "sql_correctness_passed", "sql_correctness"),
]

DEFAULT_N = 10
DEFAULT_DELTA = 0.1
DEFAULT_MIN_BASELINE_N = 2  # minimum prior occurrences before a regression is flagged


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, Decimal):
        return float(v)
    return float(v)


def _safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


class Comparator:
    """Fetches prior run data from DynamoDB and detects regressions.

    The boto3 resource handle is lazily constructed on first compare() call.
    """

    def __init__(
        self,
        env: str,
        *,
        n: int = DEFAULT_N,
        delta: float = DEFAULT_DELTA,
        min_baseline_n: int = DEFAULT_MIN_BASELINE_N,
    ):
        if not env:
            raise ValueError("env is required (e.g. 'staging-illum')")
        self.env = env
        self.n = n
        self.delta = delta
        self.min_baseline_n = min_baseline_n
        self._dynamodb = None
        self._runs_table = None
        self._questions_table = None

    def _ensure_clients(self) -> None:
        if self._dynamodb is None:
            self._dynamodb = boto3.resource("dynamodb")
            self._runs_table = self._dynamodb.Table(f"{self.env}-validator-runs")
            self._questions_table = self._dynamodb.Table(
                f"{self.env}-validator-question-records"
            )

    def _fetch_prior_runs(self, exclude_run_id: str) -> list[dict]:
        """Query env-timestamp-index GSI for the last N runs, excluding current."""
        self._ensure_clients()
        resp = self._runs_table.query(
            IndexName="env-timestamp-index",
            KeyConditionExpression=Key("env").eq(self.env),
            ScanIndexForward=False,  # newest first
            Limit=self.n + 1,  # +1 so we can exclude the current run if it appears
        )
        runs = [
            item
            for item in resp.get("Items", [])
            if item.get("run_id") != exclude_run_id
        ]
        return runs[: self.n]

    def _fetch_question_history(
        self,
        question_ids: list[int],
        prior_run_ids: set[str],
    ) -> dict[int, list[dict]]:
        """Fetch per-question records from prior runs via the question-run-index GSI.

        For each question_id queries the GSI (PK=question_id) and filters to
        items whose run_id is in prior_run_ids.  Handles DynamoDB pagination.
        """
        self._ensure_clients()
        history: dict[int, list[dict]] = {qid: [] for qid in question_ids}
        for qid in question_ids:
            items: list[dict] = []
            kwargs: dict = {
                "IndexName": "question-run-index",
                "KeyConditionExpression": Key("question_id").eq(qid),
            }
            while True:
                resp = self._questions_table.query(**kwargs)
                items.extend(resp.get("Items", []))
                last_key = resp.get("LastEvaluatedKey")
                if not last_key:
                    break
                kwargs["ExclusiveStartKey"] = last_key
            history[qid] = [
                item for item in items if item.get("run_id") in prior_run_ids
            ]
        return history

    def compare(
        self,
        current_run: dict,
        current_questions: list[dict],
    ) -> tuple[dict, list[dict]]:
        """Compare the current run against the N most recent prior runs.

        Returns:
            (comparison_block, updated_question_records_with_regression_flag)

        If no prior runs exist (first ever run for this env), returns overall_status
        "PASSED" with baseline_runs=0 and no findings — the caller still writes this
        back so the run record is fully populated.
        """
        run_id = current_run.get("run_id") or ""
        prior_runs = self._fetch_prior_runs(exclude_run_id=run_id)

        if not prior_runs:
            logger.info(
                "No prior runs found for env=%s — regression check skipped (no baseline).",
                self.env,
            )
            n_q = len(current_questions)
            comparison = {
                "overall_status": "PASSED",
                "baseline_runs": 0,
                "questions_with_baseline": 0,
                "questions_without_baseline": n_q,
                "regressions": [],
                "improvements": [],
                "performance": {},
            }
            updated = [{**q, "regression_flag": False} for q in current_questions]
            return comparison, updated

        prior_run_ids = {str(r["run_id"]) for r in prior_runs}
        question_ids = [
            q["question_id"]
            for q in current_questions
            if q.get("question_id") is not None
        ]
        history = self._fetch_question_history(question_ids, prior_run_ids)

        regressions: list[dict] = []
        improvements: list[dict] = []
        updated_questions: list[dict] = []
        questions_with_baseline = 0
        questions_without_baseline = 0

        for q in current_questions:
            qid = q.get("question_id")
            prior_records = history.get(qid, []) if qid is not None else []

            # A question has a usable baseline only when it appeared in enough prior runs.
            if len(prior_records) >= self.min_baseline_n:
                questions_with_baseline += 1
            else:
                questions_without_baseline += 1

            q_regressions, q_improvements = _compare_question(
                current=q,
                prior_records=prior_records,
                delta=self.delta,
                min_baseline_n=self.min_baseline_n,
            )
            regressions.extend(q_regressions)
            improvements.extend(q_improvements)
            updated_questions.append({**q, "regression_flag": bool(q_regressions)})

        overall_status = "REGRESSION" if regressions else "PASSED"
        comparison = {
            "overall_status": overall_status,
            "baseline_runs": len(prior_runs),
            "questions_with_baseline": questions_with_baseline,
            "questions_without_baseline": questions_without_baseline,
            "regressions": regressions,
            "improvements": improvements,
            "performance": _compare_performance(current_run, prior_runs),
        }
        return comparison, updated_questions


# Performance metrics compared at run level (aggregate), no thresholds — informational only.
# Each entry: (run_record_field, display_label, use_pct)
# use_pct=True  → delta shown as % change   (latency, tokens — large absolute values)
# use_pct=False → delta shown as absolute   (call counts — small values, % misleading)
_PERF_METRICS = [
    ("avg_duration_ms",    "Avg latency (ms)",       True),
    ("avg_total_tokens",   "Avg tokens/question",    True),
    ("avg_sql_call_count", "Avg SQL calls/question", False),
    ("avg_tool_call_count","Avg tool calls/question",False),
]


def _compare_performance(current_run: dict, prior_runs: list[dict]) -> dict:
    """Compare run-level performance aggregates against the baseline.

    Returns a dict keyed by field name.  Values are informational only — no
    pass/fail verdict, no effect on overall_status.  Fields absent from prior
    runs (e.g. avg_sql_call_count on older persisted records) are omitted.
    """
    result: dict = {}
    for field, label, use_pct in _PERF_METRICS:
        current_val = _to_float(current_run.get(field))
        prior_vals = [
            v for r in prior_runs
            if (v := _to_float(r.get(field))) is not None
        ]
        if current_val is None or not prior_vals:
            continue
        baseline = _safe_mean(prior_vals)
        delta = current_val - baseline
        entry: dict = {
            "label": label,
            "baseline_mean": round(baseline, 2),
            "current": round(current_val, 2),
        }
        if use_pct and baseline > 0:
            entry["delta_pct"] = round(delta / baseline * 100, 1)
        else:
            entry["delta"] = round(delta, 2)
        result[field] = entry
    return result


def _compare_question(
    current: dict,
    prior_records: list[dict],
    delta: float,
    min_baseline_n: int = DEFAULT_MIN_BASELINE_N,
) -> tuple[list[dict], list[dict]]:
    """Detect regressions and improvements for one question across all metrics.

    A regression is flagged when either:
    - current_score < baseline_mean - delta  (score dropped beyond threshold), or
    - current_passed is False AND baseline_pass_rate >= 0.5  (previously mostly passing, now failing)

    An improvement is flagged when current_score > baseline_mean + delta.

    Returns (regressions, improvements), each a list of finding dicts.
    Metrics with a None current score (e.g. interpretation on a conversational flow)
    are silently skipped. Metrics with fewer than min_baseline_n prior data points
    are also skipped — too few samples to establish a reliable baseline.
    """
    regressions: list[dict] = []
    improvements: list[dict] = []
    qid = current.get("question_id")

    for score_field, passed_field, metric_name in _METRICS:
        current_score = _to_float(current.get(score_field))
        if current_score is None:
            continue  # metric N/A for this flow

        prior_scores = [
            s
            for r in prior_records
            if (s := _to_float(r.get(score_field))) is not None
        ]
        if len(prior_scores) < min_baseline_n:
            continue  # insufficient baseline — skip to avoid noise from thin data

        baseline_mean = _safe_mean(prior_scores)
        prior_passed = [
            r.get(passed_field)
            for r in prior_records
            if r.get(passed_field) is not None
        ]
        baseline_pass_rate = (
            sum(1 for p in prior_passed if p) / len(prior_passed)
            if prior_passed
            else None
        )
        current_passed = current.get(passed_field)

        finding = {
            "question_id": qid,
            "metric": metric_name,
            "baseline_mean": round(baseline_mean, 4),
            "current_score": round(current_score, 4),
            "delta": round(current_score - baseline_mean, 4),
        }

        score_regressed = current_score < baseline_mean - delta
        pass_regressed = (
            current_passed is False
            and baseline_pass_rate is not None
            and baseline_pass_rate >= 0.5
        )

        if score_regressed or pass_regressed:
            regressions.append(finding)
        elif current_score > baseline_mean + delta:
            improvements.append(finding)

    return regressions, improvements


# ── End-to-end adapter ────────────────────────────────────────────────────────


def compare_and_persist(
    run_path: Path,
    *,
    n: int = DEFAULT_N,
    delta: float = DEFAULT_DELTA,
    min_baseline_n: int = DEFAULT_MIN_BASELINE_N,
) -> dict:
    """Adapter: read local run JSON, run comparison, write results back to JSON and DynamoDB.

    Called by run.py after _persist_to_dynamodb(). Returns the comparison block.

    Flow:
      1. Read the local run JSON (already has aggregated_scores from write_aggregated).
      2. Build QuestionRecord / RunRecord shapes from it (same as the initial persist).
      3. Run Comparator.compare() → comparison block + updated question records.
      4. Stamp overall_status, comparison, and per-question regression_flag back to
         the local JSON so the file stays the authoritative record.
      5. Re-persist the enriched records to DynamoDB (PutItem overwrites existing items).
    """
    from tests.validator.persistence import (
        DynamoDBWriter,
        _build_question_record,
        _build_run_record,
        _resolve_env,
    )

    run_path = Path(run_path)
    run = json.loads(run_path.read_text())
    env_raw = run.get("env")
    if not env_raw:
        raise ValueError(
            f"{run_path}: cannot compare — run JSON is missing 'env'."
        )

    env = _resolve_env(env_raw)
    run_id = run.get("run_id") or ""
    results = run.get("results") or []

    # Build record shapes as they'd look in DynamoDB (without comparison fields yet).
    current_questions = [_build_question_record(run_id, r) for r in results]
    current_run = _build_run_record(run)

    comparator = Comparator(env=env, n=n, delta=delta, min_baseline_n=min_baseline_n)
    comparison, updated_questions = comparator.compare(current_run, current_questions)

    # Promote comparison results to the top-level run record.
    current_run["overall_status"] = comparison["overall_status"]
    current_run["comparison"] = comparison

    # Stamp results back to the local JSON.
    run["overall_status"] = comparison["overall_status"]
    run["comparison"] = comparison
    q_flags = {q["question_id"]: q.get("regression_flag", False) for q in updated_questions}
    for r in results:
        if r.get("question_id") is not None:
            r["regression_flag"] = q_flags.get(r["question_id"], False)
    run_path.write_text(json.dumps(run, indent=2, default=str))

    # Re-persist enriched records (PutItem overwrites; idempotent).
    writer = DynamoDBWriter(env=env)
    writer.persist(current_run, updated_questions)

    return comparison
