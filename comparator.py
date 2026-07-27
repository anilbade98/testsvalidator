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
from decimal import Decimal
from pathlib import Path
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key

from tests.validator.records import (
    failed_question_ids,
    is_question_record,
    question_records,
    run_is_complete,
)

logger = logging.getLogger(__name__)

# Metrics tracked per question: (score_field, passed_field, display_name)
_METRICS = [
    ("agent_response_score", "agent_response_passed", "agent_response"),
    ("interpretation_score", "interpretation_passed", "interpretation_coverage"),
    ("sql_correctness_score", "sql_correctness_passed", "sql_correctness"),
]

# Run-level (aggregate) metrics: (display_name, mean_field, pass_rate_field).
# These are the low-variance signals the verdict is primarily based on.
_AGG_METRICS = [
    ("agent_response", "agent_response_mean", "agent_response_pass_rate"),
    ("interpretation_coverage", "interpretation_mean", "interpretation_pass_rate"),
    ("sql_correctness", "sql_correctness_mean", "sql_correctness_pass_rate"),
]

DEFAULT_N = 10
# Per-question score-drop threshold (informational findings + breadth counting).
# Raised above the observed per-question noise floor (~0.11) and onto the score
# grid; single-run per-question deltas below this are treated as noise.
DEFAULT_DELTA = 0.2

# Aggregate-level verdict knobs (the low-variance signals drive PASS/REGRESSION).
DEFAULT_WARN_SIGMA = 1.0          # >= this many σ below baseline mean → gray zone
DEFAULT_REGRESSION_SIGMA = 2.0    # >= this many σ below → clear REGRESSION
DEFAULT_REGRESSION_FLOOR = 0.03   # min absolute mean-drop to matter (guards σ≈0)
# The internal diagnostic sub-metrics (interpretation_coverage, sql_correctness) are
# noisier than the agent_response answer score even in aggregate, so their per-question
# swings can net a marginal ~0.03-0.05 drift with no real change. They keep their place
# in the aggregate σ-band (a broad collapse still fails), but need a bigger absolute
# drop than agent_response before it counts as a regression.
DEFAULT_SUBMETRIC_REGRESSION_FLOOR = 0.06
# Metrics whose aggregate drop uses the primary (tight) floor; all others use the
# sub-metric (loose) floor above.
_PRIMARY_AGG_METRICS = frozenset({"agent_response"})
DEFAULT_TREND_DELTA = 0.05        # newest-half vs oldest-half mean decline → trend
DEFAULT_MIN_PASS_RATE = 0.7       # absolute pass-rate floor (deterministic backstop)

# Per-question corroboration knobs.
DEFAULT_QUESTION_HARD_DELTA = 0.3  # a single question dropping this far (and failing)
DEFAULT_BREADTH_FRAC = 0.15        # gray zone → REGRESSION if >= this frac regressed
DEFAULT_BREADTH_MIN = 5            #   ...or at least this many questions regressed

# The per-question `targeted` verdict signal only considers these metrics. The
# internal sub-metrics (interpretation_coverage, sql_correctness) have very high
# per-question run-to-run variance (±0.4 swings) and would otherwise fire on noise
# every run; they still contribute to the low-variance AGGREGATE σ-band, which
# catches broad regressions in them robustly.
_TARGETED_METRICS = frozenset({"agent_response"})

# Per-flow σ-band knobs. A flow mean over k questions has run-to-run σ ≈ 0.11/√k,
# so small flows are far noisier than the 77-question aggregate. Use a higher floor
# than the aggregate, skip flows with too few questions to judge, and require the
# drop to be corroborated by more than one individually-regressed question (so a
# single low outlier in a small flow doesn't read as a flow-wide regression).
DEFAULT_FLOW_FLOOR = 0.08
DEFAULT_FLOW_MIN_QUESTIONS = 5
DEFAULT_FLOW_MIN_CORROBORATING = 2

# Performance (cost) regression knobs — one-sided: only *increases* are bad.
# Latency/tokens are noisier than scores, so require BOTH a σ-multiple AND a
# minimum percentage jump before flagging, to ignore normal run-to-run jitter.
DEFAULT_PERF_SIGMA = 2.0
DEFAULT_PERF_MIN_PCT = 0.25
# Run-level fields that drive a performance verdict: (field, regression_kind).
_PERF_VERDICT_FIELDS = [
    ("avg_duration_ms", "perf_latency"),
    ("avg_total_tokens", "perf_tokens"),
]


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, Decimal):
        return float(v)
    return float(v)


def _safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _pstdev(values: list[float]) -> float:
    """Population standard deviation; 0.0 for fewer than 2 points."""
    if len(values) < 2:
        return 0.0
    mu = sum(values) / len(values)
    return (sum((v - mu) ** 2 for v in values) / len(values)) ** 0.5


def _skipped_comparison(*, overall_status: str, baseline_runs: int,
                        required_baseline_runs: int, questions_without_baseline: int,
                        **extra) -> dict:
    """Comparison dict for the paths where no regression check ran — an insufficient
    baseline (PASSED) or an incomplete run (FAILED). One skeleton so the two paths
    can't drift apart. `extra` carries path-specific keys (e.g. complete /
    failed_question_ids for the FAILED path)."""
    return {
        "overall_status": overall_status,
        "baseline_runs": baseline_runs,
        "required_baseline_runs": required_baseline_runs,
        "baseline_sufficient": False,
        "regression_kinds": [],
        "questions_with_baseline": 0,
        "questions_without_baseline": questions_without_baseline,
        "regressions": [],
        "improvements": [],
        "performance": {},
        **extra,
    }


class Comparator:
    """Fetches prior run data from DynamoDB and detects regressions.

    The boto3 resource handle is lazily constructed on first compare() call.
    """

    def __init__(self, env: str, *, n: int = DEFAULT_N, delta: float = DEFAULT_DELTA):
        if not env:
            raise ValueError("env is required (e.g. 'staging-illum')")
        self.env = env
        self.n = n          # baseline size AND minimum prior-run count required to compare
        self.delta = delta  # per-question score-drop threshold (informational + breadth)
        # Detection thresholds. The DEFAULT_* module constants are the single source of
        # truth — tune there. They are not constructor params because nothing varies
        # them per run; only n/delta are exposed (via --baseline-n / --regression-delta).
        self.warn_sigma = DEFAULT_WARN_SIGMA
        self.regression_sigma = DEFAULT_REGRESSION_SIGMA
        self.regression_floor = DEFAULT_REGRESSION_FLOOR
        self.submetric_regression_floor = DEFAULT_SUBMETRIC_REGRESSION_FLOOR
        self.trend_delta = DEFAULT_TREND_DELTA
        self.min_pass_rate = DEFAULT_MIN_PASS_RATE
        self.question_hard_delta = DEFAULT_QUESTION_HARD_DELTA
        self.breadth_frac = DEFAULT_BREADTH_FRAC
        self.breadth_min = DEFAULT_BREADTH_MIN
        self.perf_sigma = DEFAULT_PERF_SIGMA
        self.perf_min_pct = DEFAULT_PERF_MIN_PCT
        self.flow_floor = DEFAULT_FLOW_FLOOR
        self.flow_min_questions = DEFAULT_FLOW_MIN_QUESTIONS
        self.flow_min_corroborating = DEFAULT_FLOW_MIN_CORROBORATING
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
        """Query env-timestamp-index GSI for the latest N eligible prior runs.

        Newest first, excluding the current run and keeping only runs marked
        PASSED — the baseline is known-good runs only. Requiring PASSED (rather
        than merely excluding FAILED) also filters out legacy REGRESSION rows
        written by the pre-PR `compare_and_persist`, which persisted every run
        unconditionally, and any older rows with no `overall_status`. Paginates
        so that skipped/current runs consumed early don't cause us to return
        fewer than N when N eligible runs actually exist.
        """
        self._ensure_clients()
        runs: list[dict] = []
        kwargs: dict = {
            "IndexName": "env-timestamp-index",
            "KeyConditionExpression": Key("env").eq(self.env),
            "ScanIndexForward": False,  # newest first
        }
        while len(runs) < self.n:
            resp = self._runs_table.query(**kwargs)
            for item in resp.get("Items", []):
                if item.get("run_id") == exclude_run_id:
                    continue
                if item.get("overall_status") != "PASSED":
                    continue  # baseline = PASSED only (excludes FAILED / REGRESSION / legacy)
                runs.append(item)
                if len(runs) >= self.n:
                    break
            last_key = resp.get("LastEvaluatedKey")
            if not last_key:
                break
            kwargs["ExclusiveStartKey"] = last_key
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

        Regression is only assessed when at least ``n`` eligible prior runs
        exist. With fewer than ``n`` on record the baseline is too thin, so the
        check is skipped: overall_status is "PASSED", ``baseline_sufficient`` is
        False, and no question is flagged. The caller still writes this back so
        the run record is fully populated.
        """
        run_id = current_run.get("run_id") or ""
        prior_runs = self._fetch_prior_runs(exclude_run_id=run_id)

        if len(prior_runs) < self.n:
            logger.info(
                "Only %d eligible prior run(s) for env=%s (need %d) — "
                "regression check skipped (insufficient baseline).",
                len(prior_runs), self.env, self.n,
            )
            comparison = _skipped_comparison(
                overall_status="PASSED",
                baseline_runs=len(prior_runs),
                required_baseline_runs=self.n,
                questions_without_baseline=len(current_questions),
            )
            updated = [{**q, "regression_flag": False} for q in current_questions]
            return comparison, updated

        prior_run_ids = {str(r["run_id"]) for r in prior_runs}
        question_ids = [
            q["question_id"]
            for q in current_questions
            if q.get("question_id") is not None
        ]
        history = self._fetch_question_history(question_ids, prior_run_ids)

        # Per-question comparison, then each independent detector. Every detector
        # returns its detail block plus the regression `kinds` it fires; compare()
        # just orchestrates and assembles.
        q = self._score_questions(current_questions, history)

        kinds: set[str] = set()
        metric_details, agg_kinds, gray_zone = self._aggregate_sigma(prior_runs, current_run)
        kinds |= agg_kinds
        flow_details, flow_kinds = self._flow_sigma(
            prior_runs, current_run, q["flow_ar_regressed"])
        kinds |= flow_kinds
        trend_detail, trend_kinds = self._trend(prior_runs, current_run)
        kinds |= trend_kinds
        floor_detail, floor_kinds = self._pass_rate_floor(current_run)
        kinds |= floor_kinds
        if q["targeted"]:
            kinds.add("targeted")
        # Breadth reads `kinds` (needs sigma/trend already decided above).
        breadth_detail, breadth_kinds = self._breadth(
            n_regressed=q["n_regressed"], n_scored=q["questions_with_baseline"],
            gray_zone=gray_zone, kinds=kinds)
        kinds |= breadth_kinds
        performance, perf_kinds = self._performance_verdict(current_run, prior_runs)
        kinds |= perf_kinds

        comparison = {
            "overall_status": "REGRESSION" if kinds else "PASSED",
            "baseline_runs": len(prior_runs),
            "required_baseline_runs": self.n,
            "baseline_sufficient": True,
            "regression_kinds": sorted(kinds),
            "metrics": metric_details,
            "flows": flow_details,
            "trend": trend_detail,
            "floor": floor_detail,
            "breadth": breadth_detail,
            "targeted": q["targeted"],
            "questions_with_baseline": q["questions_with_baseline"],
            "questions_without_baseline": q["questions_without_baseline"],
            "regressions": q["regressions"],
            "improvements": q["improvements"],
            "performance": performance,
        }
        return comparison, q["updated_questions"]

    # ── Detectors ────────────────────────────────────────────────────────────────

    def _score_questions(self, current_questions: list[dict], history: dict) -> dict:
        """Per-question comparison: informational regressions/improvements, the
        targeted single-question signal, and per-flow corroboration counts."""
        regressions: list[dict] = []
        improvements: list[dict] = []
        updated_questions: list[dict] = []
        questions_with_baseline = 0
        questions_without_baseline = 0
        n_regressed = 0
        targeted: list[dict] = []
        # Per-flow count of questions whose agent_response regressed (>= self.delta),
        # used to corroborate a per-flow σ-band drop (so one outlier doesn't flag a flow).
        flow_ar_regressed: dict = {}

        for q in current_questions:
            qid = q.get("question_id")
            prior_records = history.get(qid, []) if qid is not None else []
            if prior_records:
                questions_with_baseline += 1
            else:
                questions_without_baseline += 1

            q_regressions, q_improvements = _compare_question(
                current=q, prior_records=prior_records, delta=self.delta,
            )
            regressions.extend(q_regressions)
            improvements.extend(q_improvements)
            if q_regressions:
                n_regressed += 1
            q_flow = q.get("flow")
            for f in q_regressions:
                if f.get("metric") == "agent_response" and f["delta"] <= -self.delta:
                    flow_ar_regressed[q_flow] = flow_ar_regressed.get(q_flow, 0) + 1
                # Targeted: a single question's ANSWER dropping hard AND now failing —
                # a real break the aggregate can miss. Restricted to agent_response;
                # the noisy internal sub-metrics live in the aggregate σ-band only.
                if (f.get("metric") in _TARGETED_METRICS
                        and f["delta"] <= -self.question_hard_delta
                        and f.get("current_passed") is False):
                    targeted.append(f)
            updated_questions.append({**q, "regression_flag": bool(q_regressions)})

        return {
            "regressions": regressions,
            "improvements": improvements,
            "updated_questions": updated_questions,
            "questions_with_baseline": questions_with_baseline,
            "questions_without_baseline": questions_without_baseline,
            "n_regressed": n_regressed,
            "targeted": targeted,
            "flow_ar_regressed": flow_ar_regressed,
        }

    def _aggregate_sigma(self, prior_runs: list[dict], current_run: dict) -> tuple[dict, set, bool]:
        """Aggregate σ-band per metric (primary, low-variance signal). Returns
        (metric_details, kinds, gray_zone)."""
        metric_details: dict = {}
        kinds: set[str] = set()
        gray_zone = False
        for name, mean_field, _pr in _AGG_METRICS:
            baseline_means = [
                m for r in prior_runs
                if (m := _to_float(r.get(mean_field))) is not None
            ]
            current_mean = _to_float(current_run.get(mean_field))
            if current_mean is None or len(baseline_means) < 2:
                continue
            mu = _safe_mean(baseline_means)
            sigma = _pstdev(baseline_means)
            drop = mu - current_mean
            # Diagnostic sub-metrics need a bigger absolute drop than agent_response.
            floor = (self.regression_floor if name in _PRIMARY_AGG_METRICS
                     else self.submetric_regression_floor)
            regress_thr = max(self.regression_sigma * sigma, floor)
            warn_thr = max(self.warn_sigma * sigma, floor)
            status = "ok"
            if drop >= regress_thr:
                status = "regression"
                kinds.add("sigma")
            elif drop >= warn_thr:
                status = "warn"
                gray_zone = True
            metric_details[name] = {
                "baseline_mean": round(mu, 4),
                "current_mean": round(current_mean, 4),
                "drop": round(drop, 4),
                "sigma": round(sigma, 4),
                "regression_threshold": round(regress_thr, 4),
                "status": status,
            }
        return metric_details, kinds, gray_zone

    def _flow_sigma(self, prior_runs: list[dict], current_run: dict,
                    flow_ar_regressed: dict) -> tuple[dict, set]:
        """Per-flow agent_response σ-band — catches a regression concentrated in one
        flow that the overall aggregate mean dilutes. Returns (flow_details, kinds)."""
        flow_details: dict = {}
        kinds: set[str] = set()
        cur_by_flow = current_run.get("by_flow") or {}
        all_flows = sorted({
            fl for r in prior_runs for fl in (r.get("by_flow") or {})
        } | set(cur_by_flow))
        for fl in all_flows:
            current_flow_mean = _to_float((cur_by_flow.get(fl) or {}).get("agent_response_mean"))
            baseline_flow_means = [
                m for r in prior_runs
                if (m := _to_float((r.get("by_flow") or {}).get(fl, {}).get("agent_response_mean"))) is not None
            ]
            if current_flow_mean is None or len(baseline_flow_means) < 2:
                continue
            flow_total = int((cur_by_flow.get(fl) or {}).get("total") or 0)
            mu_f = _safe_mean(baseline_flow_means)
            sigma_f = _pstdev(baseline_flow_means)
            drop_f = mu_f - current_flow_mean
            # Higher floor than the aggregate — small flows are inherently noisier.
            regress_thr_f = max(self.regression_sigma * sigma_f, self.flow_floor)
            corroborating = flow_ar_regressed.get(fl, 0)
            flow_status = "ok"
            if flow_total < self.flow_min_questions:
                # Too few questions to tell a flow-wide regression from question noise.
                flow_status = "skipped"
            elif drop_f >= regress_thr_f and corroborating >= self.flow_min_corroborating:
                flow_status = "regression"
                kinds.add("flow")
            elif drop_f >= regress_thr_f:
                # Mean dropped but only one question moved it — an outlier, not a
                # flow-wide regression. Surface as a warning, don't fail on it.
                flow_status = "warn"
            flow_details[fl] = {
                "baseline_mean": round(mu_f, 4),
                "current_mean": round(current_flow_mean, 4),
                "drop": round(drop_f, 4),
                "sigma": round(sigma_f, 4),
                "regression_threshold": round(regress_thr_f, 4),
                "questions": flow_total,
                "corroborating_regressions": corroborating,
                "status": flow_status,
            }
        return flow_details, kinds

    def _trend(self, prior_runs: list[dict], current_run: dict) -> tuple[dict, set]:
        """Slow drift: newest-half vs oldest-half mean over baseline + current."""
        trend_detail: dict = {}
        kinds: set[str] = set()
        prior_sorted = sorted(prior_runs, key=lambda r: str(r.get("run_timestamp") or ""))
        for name, mean_field, _pr in _AGG_METRICS:
            series = [
                m for r in prior_sorted
                if (m := _to_float(r.get(mean_field))) is not None
            ]
            cur = _to_float(current_run.get(mean_field))
            if cur is not None:
                series.append(cur)
            if len(series) < 4:
                continue
            half = len(series) // 2
            old_mean = _safe_mean(series[:half])
            new_mean = _safe_mean(series[half:])
            decline = old_mean - new_mean
            if decline >= self.trend_delta:
                kinds.add("trend")
            trend_detail[name] = {
                "old_half_mean": round(old_mean, 4),
                "new_half_mean": round(new_mean, 4),
                "decline": round(decline, 4),
            }
        return trend_detail, kinds

    def _pass_rate_floor(self, current_run: dict) -> tuple[dict, set]:
        """Absolute agent_response pass-rate floor (deterministic backstop)."""
        floor_detail: dict = {}
        kinds: set[str] = set()
        cur_pr = _to_float(current_run.get("agent_response_pass_rate"))
        if cur_pr is not None:
            floor_detail = {
                "agent_response_pass_rate": round(cur_pr, 4),
                "min_pass_rate": self.min_pass_rate,
            }
            if cur_pr < self.min_pass_rate:
                kinds.add("floor")
        return floor_detail, kinds

    def _breadth(self, *, n_regressed: int, n_scored: int, gray_zone: bool,
                 kinds: set) -> tuple[dict, set]:
        """Gray-zone tie-breaker: fire only when a metric is in the gray zone and no
        conclusive aggregate signal already fired. Broad = real; few = noise."""
        breadth_frac_val = (n_regressed / n_scored) if n_scored else 0.0
        triggered = False
        add: set[str] = set()
        if gray_zone and "sigma" not in kinds and "trend" not in kinds:
            if n_regressed >= self.breadth_min or breadth_frac_val >= self.breadth_frac:
                triggered = True
                add.add("breadth")
        return {
            "regressed_questions": n_regressed,
            "scored_questions": n_scored,
            "frac": round(breadth_frac_val, 4),
            "triggered": triggered,
            "gray_zone": gray_zone,
        }, add

    def _performance_verdict(self, current_run: dict, prior_runs: list[dict]) -> tuple[dict, set]:
        """Performance (cost) regression: latency/tokens increasing beyond both a
        σ-multiple and a minimum % (one-sided — only increases matter)."""
        kinds: set[str] = set()
        performance = _compare_performance(current_run, prior_runs)
        for field, kind in _PERF_VERDICT_FIELDS:
            entry = performance.get(field)
            baseline_vals = [
                v for r in prior_runs
                if (v := _to_float(r.get(field))) is not None
            ]
            current_val = _to_float(current_run.get(field))
            if entry is None or current_val is None or len(baseline_vals) < 2:
                continue
            mu = _safe_mean(baseline_vals)
            sigma = _pstdev(baseline_vals)
            threshold = max(self.perf_sigma * sigma, self.perf_min_pct * mu)
            regressed = (current_val - mu) > threshold
            entry["regressed"] = regressed
            entry["threshold"] = round(threshold, 2)
            if regressed:
                kinds.add(kind)
        return performance, kinds


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
) -> tuple[list[dict], list[dict]]:
    """Detect regressions and improvements for one question across all metrics.

    A regression is flagged when either:
    - current_score < baseline_mean - delta  (score dropped beyond threshold), or
    - current_passed is False AND baseline_pass_rate >= 0.5  (previously mostly passing, now failing)

    An improvement is flagged when current_score > baseline_mean + delta.

    Returns (regressions, improvements), each a list of finding dicts.
    Metrics with a None current score (e.g. interpretation on a conversational flow)
    are silently skipped, as are metrics with no prior data points for this
    question (nothing to average). The sufficiency of the *run* baseline is
    enforced upstream in ``Comparator.compare``.
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
        if not prior_scores:
            continue  # no prior data points for this metric — nothing to compare

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
            "current_passed": current_passed,
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


def compare_run(
    run_path: Path,
    *,
    n: int = DEFAULT_N,
    delta: float = DEFAULT_DELTA,
) -> dict:
    """Compute the comparison for a run and stamp it into the local JSON — NO persist.

    Returns the comparison block. Splitting compute from persist lets the caller
    insert a re-run confirmation step (which needs the live service) between
    computing the verdict and deciding whether to persist.
    """
    from tests.validator.persistence import (
        _build_question_record,
        _build_run_record,
        _resolve_env,
    )

    run_path = Path(run_path)
    run = json.loads(run_path.read_text())
    env_raw = run.get("env")
    if not env_raw:
        raise ValueError(f"{run_path}: cannot compare — run JSON is missing 'env'.")

    env = _resolve_env(env_raw)
    run_id = run.get("run_id") or ""
    results = run.get("results") or []

    current_questions = [_build_question_record(run_id, r) for r in question_records(results)]
    current_run = _build_run_record(run)

    # A run only counts if every question executed and returned a result. An
    # incomplete run is marked FAILED (no comparison) and excluded from baselines.
    if not run_is_complete(results):
        bad_ids = failed_question_ids(results)
        logger.warning(
            "Run %s is incomplete (%d question(s) errored/skipped) — marking FAILED, "
            "no regression comparison.", run_id, len(bad_ids),
        )
        comparison = _skipped_comparison(
            overall_status="FAILED",
            baseline_runs=0,
            required_baseline_runs=n,
            questions_without_baseline=len(current_questions),
            complete=False,
            failed_question_ids=bad_ids,
        )
        updated_questions = [{**q, "regression_flag": False} for q in current_questions]
    else:
        comparator = Comparator(env=env, n=n, delta=delta)
        comparison, updated_questions = comparator.compare(current_run, current_questions)
        comparison["complete"] = True

    # Stamp verdict + per-question flags back to the local JSON.
    run["overall_status"] = comparison["overall_status"]
    run["comparison"] = comparison
    q_flags = {q["question_id"]: q.get("regression_flag", False) for q in updated_questions}
    for r in results:
        if is_question_record(r):
            r["regression_flag"] = q_flags.get(r["question_id"], False)
    run_path.write_text(json.dumps(run, indent=2, default=str))
    return comparison


def persist_if_passed(run_path: Path, comparison: dict) -> bool:
    """Persist the run to DynamoDB iff its verdict is PASSED. Returns True if written."""
    from tests.validator.persistence import persist_run_file
    run_id = json.loads(Path(run_path).read_text()).get("run_id") or ""
    if comparison.get("overall_status") != "PASSED":
        logger.warning(
            "Run %s is %s — not persisting to DynamoDB (baseline = PASSED runs only).",
            run_id, comparison.get("overall_status"),
        )
        return False
    return persist_run_file(run_path)
