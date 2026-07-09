"""Unit tests for tests/validator/persistence.py — DynamoDBWriter.

Uses `moto` to spin up an in-process mock DynamoDB. The mock tables are
created with the same schema (PK / SK / GSIs / TTL attribute) as the real
CDK-provisioned tables so the tests exercise the actual put/query paths
the runner will hit in CI.
"""
from __future__ import annotations

import json
import time
from decimal import Decimal
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from tests.validator.persistence import (
    QUESTION_RECORD_TTL_DAYS,
    DynamoDBWriter,
    _build_question_record,
    _build_run_record,
    _drop_empty,
    _resolve_env,
    _to_ddb,
    persist_run_file,
)

ENV = "dev-illum"
REGION = "us-west-2"


@pytest.fixture(autouse=True)
def _aws_credentials(monkeypatch):
    """Mask any real AWS credentials so moto's `@mock_aws` intercepts cleanly.

    boto3 falls back to env vars / `~/.aws/credentials` / instance metadata in
    that order. Without this fixture, a developer with `AWS_PROFILE` set in
    their shell will see moto silently bypassed and the tests hit real
    DynamoDB (or fail with `ResourceNotFoundException` if the prod tables
    don't exist in their account).
    """
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    # DynamoDBWriter resolves region via AWS_REGION first, then AWS_DEFAULT_REGION.
    # Set both — a developer shell with AWS_REGION set would otherwise leak through
    # and point the writer at a different region than the mock tables.
    monkeypatch.setenv("AWS_REGION", REGION)
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    monkeypatch.delenv("AWS_PROFILE", raising=False)


def _create_tables(client):
    """Create the two validator tables in mock DynamoDB with the schemas
    matching cdk/app_stack.py."""
    client.create_table(
        TableName=f"{ENV}-validator-runs",
        BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[
            {"AttributeName": "run_id", "AttributeType": "S"},
            {"AttributeName": "env", "AttributeType": "S"},
            {"AttributeName": "run_timestamp", "AttributeType": "S"},
        ],
        KeySchema=[{"AttributeName": "run_id", "KeyType": "HASH"}],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "env-timestamp-index",
                "KeySchema": [
                    {"AttributeName": "env", "KeyType": "HASH"},
                    {"AttributeName": "run_timestamp", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
    )
    client.create_table(
        TableName=f"{ENV}-validator-question-records",
        BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[
            {"AttributeName": "run_id", "AttributeType": "S"},
            {"AttributeName": "question_id", "AttributeType": "N"},
        ],
        KeySchema=[
            {"AttributeName": "run_id", "KeyType": "HASH"},
            {"AttributeName": "question_id", "KeyType": "RANGE"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "question-run-index",
                "KeySchema": [
                    {"AttributeName": "question_id", "KeyType": "HASH"},
                    {"AttributeName": "run_id", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
    )


def _run_record(run_id="run-001", env=ENV, ts="2026-06-18T10:00:00Z", **overrides) -> dict:
    base = {
        "run_id": run_id,
        "env": env,
        "run_timestamp": ts,
        "base_url": "https://staging.example.com",
        "model_id": "us.anthropic.claude-opus-4-6-v1",
        "filter_config": {"priority_only": True, "tunnel": False},
        "question_count": 12,
        "avg_duration_ms": 31_400,
        "avg_total_tokens": 25_812,
        "agent_response_mean": 0.83,
        "agent_response_pass_rate": 0.92,
        "interpretation_mean": 0.79,
        "interpretation_pass_rate": 0.85,
        "sql_correctness_mean": 0.81,
        "sql_correctness_pass_rate": 0.88,
        "overall_status": "PASSED",
        "comparison": {},
    }
    base.update(overrides)
    return base


def _question_record(run_id="run-001", question_id=1, **overrides) -> dict:
    base = {
        "run_id": run_id,
        "question_id": question_id,
        "question": "What's our average CSAT this month?",
        "flow": "fdl",
        "complexity": "basic",
        "priority": True,
        "agent_response": "Average CSAT is 4.3 out of 5.",
        "duration_ms": 28_100,
        "total_tokens": 24_500,
        "input_tokens": 21_200,
        "output_tokens": 3_300,
        "cache_read_tokens": 0,
        "od_fdl_results": [
            {
                "tool_name": "fdl",
                "interpretation": "average CSAT, current month",
                "schema": "CALL_SEGMENT",
                "column_meta": [{"columnName": "csat"}],
                "rows": [[4.3]],
            }
        ],
        "agent_response_score": 0.91,
        "agent_response_passed": True,
        "agent_response_reason": "Direct, correct, no fabrication.",
        "interpretation_score": 0.84,
        "interpretation_passed": True,
        "interpretation_reason": "Interpretation matches intent.",
        "sql_correctness_score": 0.87,
        "sql_correctness_passed": True,
        "sql_per_trace_scores": [{"source": "fdl", "sql_score": 0.87, "sql_passed": True}],
        "regression_flag": False,
    }
    base.update(overrides)
    return base


# ── helpers ──────────────────────────────────────────────────────────────


class TestHelpers:
    def test_to_ddb_converts_floats_to_decimal(self):
        out = _to_ddb({"score": 0.91, "tokens": 24_500})
        assert isinstance(out["score"], Decimal)
        assert out["score"] == Decimal("0.91")
        assert out["tokens"] == 24_500  # int stays int

    def test_to_ddb_recurses_into_lists_and_dicts(self):
        out = _to_ddb({"nested": [{"a": 0.5}, {"b": [0.25, 0.75]}]})
        assert out["nested"][0]["a"] == Decimal("0.5")
        assert out["nested"][1]["b"] == [Decimal("0.25"), Decimal("0.75")]

    def test_drop_empty_drops_none_and_empty_string(self):
        out = _drop_empty({"a": 1, "b": None, "c": "", "d": "kept", "e": 0})
        assert out == {"a": 1, "d": "kept", "e": 0}  # zero is kept

    def test_drop_empty_preserves_nested_none(self):
        """Nested None inside a list/dict is meaningful (metric N/A) and stays."""
        out = _drop_empty({"per_trace": [{"score": None}]})
        assert out == {"per_trace": [{"score": None}]}


# ── constructor ──────────────────────────────────────────────────────────


class TestInit:
    def test_requires_env(self):
        with pytest.raises(ValueError, match="env is required"):
            DynamoDBWriter(env="")

    @mock_aws
    def test_constructs_table_names_from_env(self):
        w = DynamoDBWriter(env="dev-illum")
        assert w.runs_table_name == "dev-illum-validator-runs"
        assert w.questions_table_name == "dev-illum-validator-question-records"

    @mock_aws
    def test_does_not_call_boto3_until_persist(self):
        """boto3 resource construction is deferred to first persist() call so
        importing this module or constructing the writer in a test that hasn't
        yet set up mock credentials never touches a real boto3 client."""
        w = DynamoDBWriter(env=ENV)
        assert w._dynamodb is None
        assert w._runs_table is None
        assert w._questions_table is None


# ── persist ──────────────────────────────────────────────────────────────


class TestPersist:
    @mock_aws
    def test_writes_run_record_with_correct_shape(self):
        client = boto3.client("dynamodb", region_name=REGION)
        _create_tables(client)
        writer = DynamoDBWriter(env=ENV)

        writer.persist(_run_record(), [_question_record()])

        runs = boto3.resource("dynamodb", region_name=REGION).Table(f"{ENV}-validator-runs")
        item = runs.get_item(Key={"run_id": "run-001"})["Item"]
        assert item["env"] == ENV
        assert item["overall_status"] == "PASSED"
        assert item["agent_response_mean"] == Decimal("0.83")
        assert item["filter_config"] == {"priority_only": True, "tunnel": False}

    @mock_aws
    def test_writes_all_question_records(self):
        client = boto3.client("dynamodb", region_name=REGION)
        _create_tables(client)
        writer = DynamoDBWriter(env=ENV)

        questions = [_question_record(question_id=qid) for qid in range(1, 11)]
        writer.persist(_run_record(), questions)

        table = boto3.resource("dynamodb", region_name=REGION).Table(
            f"{ENV}-validator-question-records"
        )
        resp = table.query(KeyConditionExpression="run_id = :r", ExpressionAttributeValues={":r": "run-001"})
        assert resp["Count"] == 10

    @mock_aws
    def test_upsert_overwrites_no_duplicates(self):
        client = boto3.client("dynamodb", region_name=REGION)
        _create_tables(client)
        writer = DynamoDBWriter(env=ENV)

        # First write.
        writer.persist(_run_record(), [_question_record(question_id=7, agent_response_score=0.50)])
        # Re-score, re-persist with updated score.
        writer.persist(_run_record(), [_question_record(question_id=7, agent_response_score=0.95)])

        table = boto3.resource("dynamodb", region_name=REGION).Table(
            f"{ENV}-validator-question-records"
        )
        resp = table.query(KeyConditionExpression="run_id = :r", ExpressionAttributeValues={":r": "run-001"})
        assert resp["Count"] == 1  # NOT 2 — the second write overwrote the first
        assert resp["Items"][0]["agent_response_score"] == Decimal("0.95")

    @mock_aws
    def test_batch_writer_handles_more_than_25_items(self):
        """boto3 batch_writer chunks at 25 — verify all N items end up in the table."""
        client = boto3.client("dynamodb", region_name=REGION)
        _create_tables(client)
        writer = DynamoDBWriter(env=ENV)

        questions = [_question_record(question_id=qid) for qid in range(1, 78)]  # 77 questions
        writer.persist(_run_record(), questions)

        table = boto3.resource("dynamodb", region_name=REGION).Table(
            f"{ENV}-validator-question-records"
        )
        count = 0
        resp = table.query(KeyConditionExpression="run_id = :r", ExpressionAttributeValues={":r": "run-001"})
        count += resp["Count"]
        while "LastEvaluatedKey" in resp:
            resp = table.query(
                KeyConditionExpression="run_id = :r",
                ExpressionAttributeValues={":r": "run-001"},
                ExclusiveStartKey=resp["LastEvaluatedKey"],
            )
            count += resp["Count"]
        assert count == 77

    @mock_aws
    def test_question_records_get_ttl_around_180_days(self):
        client = boto3.client("dynamodb", region_name=REGION)
        _create_tables(client)
        writer = DynamoDBWriter(env=ENV)

        now = int(time.time())
        writer.persist(_run_record(), [_question_record()])

        table = boto3.resource("dynamodb", region_name=REGION).Table(
            f"{ENV}-validator-question-records"
        )
        item = table.get_item(Key={"run_id": "run-001", "question_id": 1})["Item"]
        expected = now + QUESTION_RECORD_TTL_DAYS * 86400
        # Wide tolerance: on a slow / contended CI runner the table-create + persist +
        # get_item round-trip can take tens of seconds. We only care that the writer's
        # stamp is the expected horizon — not that it's stamped within a second of `now`.
        assert abs(int(item["ttl"]) - expected) < 120

    @mock_aws
    def test_run_record_has_no_ttl(self):
        """Run-level aggregates are kept indefinitely (small + needed for trend lines)."""
        client = boto3.client("dynamodb", region_name=REGION)
        _create_tables(client)
        writer = DynamoDBWriter(env=ENV)

        writer.persist(_run_record(), [_question_record()])

        runs = boto3.resource("dynamodb", region_name=REGION).Table(f"{ENV}-validator-runs")
        item = runs.get_item(Key={"run_id": "run-001"})["Item"]
        assert "ttl" not in item

    @mock_aws
    def test_persist_requires_run_id_in_run_record(self):
        client = boto3.client("dynamodb", region_name=REGION)
        _create_tables(client)
        writer = DynamoDBWriter(env=ENV)

        with pytest.raises(ValueError, match="run_id"):
            writer.persist(_run_record(run_id=""), [_question_record()])

    @mock_aws
    def test_drops_optional_none_fields_at_top_level(self):
        """An `error` of None on a successful run shouldn't write the attribute."""
        client = boto3.client("dynamodb", region_name=REGION)
        _create_tables(client)
        writer = DynamoDBWriter(env=ENV)

        writer.persist(_run_record(), [_question_record(error=None)])

        table = boto3.resource("dynamodb", region_name=REGION).Table(
            f"{ENV}-validator-question-records"
        )
        item = table.get_item(Key={"run_id": "run-001", "question_id": 1})["Item"]
        assert "error" not in item

    @mock_aws
    def test_skips_mismatched_question_records(self):
        """A record with the wrong run_id or missing question_id is skipped, not written."""
        client = boto3.client("dynamodb", region_name=REGION)
        _create_tables(client)
        writer = DynamoDBWriter(env=ENV)

        writer.persist(
            _run_record(run_id="run-A"),
            [
                _question_record(run_id="run-A", question_id=1),
                _question_record(run_id="run-B", question_id=2),  # mismatched run_id — skipped
                _question_record(run_id="run-A", question_id=None),  # missing SK — skipped
            ],
        )

        table = boto3.resource("dynamodb", region_name=REGION).Table(
            f"{ENV}-validator-question-records"
        )
        # Only the well-formed record landed in run-A's partition.
        resp = table.query(KeyConditionExpression="run_id = :r", ExpressionAttributeValues={":r": "run-A"})
        assert resp["Count"] == 1
        # And the mismatched-run_id record didn't accidentally land under run-B either.
        resp_b = table.query(KeyConditionExpression="run_id = :r", ExpressionAttributeValues={":r": "run-B"})
        assert resp_b["Count"] == 0


# ── GSI behavior ─────────────────────────────────────────────────────────


class TestGSIs:
    @mock_aws
    def test_env_timestamp_gsi_returns_runs_ordered_for_env(self):
        """Comparator's 'last N runs for env=staging' query path."""
        client = boto3.client("dynamodb", region_name=REGION)
        _create_tables(client)
        writer = DynamoDBWriter(env=ENV)

        for i, ts in enumerate(["2026-06-15T10:00:00Z", "2026-06-16T10:00:00Z", "2026-06-17T10:00:00Z"]):
            writer.persist(_run_record(run_id=f"run-{i}", ts=ts), [])

        # Different env — should not appear in env=dev-illum query.
        writer.persist(_run_record(run_id="run-other-env", env="staging-illum"), [])

        runs = boto3.resource("dynamodb", region_name=REGION).Table(f"{ENV}-validator-runs")
        resp = runs.query(
            IndexName="env-timestamp-index",
            KeyConditionExpression="env = :e",
            ExpressionAttributeValues={":e": ENV},
            ScanIndexForward=False,  # newest first
            Limit=2,
        )
        assert resp["Count"] == 2
        assert resp["Items"][0]["run_timestamp"] == "2026-06-17T10:00:00Z"
        assert resp["Items"][1]["run_timestamp"] == "2026-06-16T10:00:00Z"

    @mock_aws
    def test_question_run_gsi_returns_question_history(self):
        """Comparator's per-question score history query path."""
        client = boto3.client("dynamodb", region_name=REGION)
        _create_tables(client)
        writer = DynamoDBWriter(env=ENV)

        for i in range(3):
            writer.persist(
                _run_record(run_id=f"run-{i}"),
                [
                    _question_record(run_id=f"run-{i}", question_id=42, agent_response_score=0.5 + 0.1 * i),
                    _question_record(run_id=f"run-{i}", question_id=99),  # noise: different question
                ],
            )

        table = boto3.resource("dynamodb", region_name=REGION).Table(
            f"{ENV}-validator-question-records"
        )
        resp = table.query(
            IndexName="question-run-index",
            KeyConditionExpression="question_id = :q",
            ExpressionAttributeValues={":q": 42},
        )
        assert resp["Count"] == 3
        scores = sorted(float(item["agent_response_score"]) for item in resp["Items"])
        assert scores == [0.5, 0.6, 0.7]


# ── Adapter: local run JSON → DynamoDB records ────────────────────────────


def _local_run_json(run_id="run_20260624_120000", env="dev", **overrides) -> dict:
    """A realistic run JSON in the shape produced by runner.run_all() + write_aggregated()."""
    base = {
        "run_id": run_id,
        "run_timestamp": "2026-06-24T12:00:00Z",
        "env": env,
        "base_url": "https://staging.example.com",
        "model_id": "us.anthropic.claude-opus-4-6-v1",
        "total": 2,
        "filter_config": {"priority_only": True, "tunnel": False},
        "results": [
            {
                "question_id": 1,
                "question": "What's our average CSAT this month?",
                "flow": "fdl",
                "complexity": "basic",
                "priority": True,
                "agent_response": "CSAT is 4.3/5.",
                "duration_ms": 28_100,
                "turn_count": 1,
                "usage": {
                    "total_tokens": 24_500,
                    "input_tokens": 21_200,
                    "output_tokens": 3_300,
                    "cache_read_tokens": 0,
                },
                "od_fdl_results": [
                    {
                        "tool_name": "fdl",
                        "interpretation": "average CSAT, current month",
                        "schema": "CALL_SEGMENT",
                        "column_meta": [{"columnName": "csat"}],
                        "rows": [[4.3]],
                    }
                ],
                "eval": {
                    "agent_response": {"score": 0.91, "passed": True, "reason": "Direct."},
                    "interpretation_coverage": {"score": 0.84, "passed": True, "reason": "OK."},
                    "per_trace": [{"source": "fdl", "sql_score": 0.87, "sql_passed": True, "sql_reason": "OK."}],
                    "tool_compliance": {"compliant": True},
                },
                "error": None,
            },
            {
                "question_id": 2,
                "question": "Hi",
                "flow": "conversational",
                "complexity": "basic",
                "priority": False,
                "agent_response": "Hello!",
                "duration_ms": 1_500,
                "turn_count": 1,
                "usage": {"total_tokens": 500, "input_tokens": 400, "output_tokens": 100},
                "od_fdl_results": [],
                "eval": {
                    "agent_response": {"score": 0.95, "passed": True, "reason": "Greeting OK."},
                    # conversational flow: no interpretation / per_trace
                },
                "error": None,
            },
        ],
        "aggregated_scores": {
            "total": 2,
            "errors": 0,
            "avg_duration_ms": 14_800.0,
            "avg_total_tokens": 12_500.0,
            "agent_response": {"mean": 0.93, "pass_rate": 1.0, "passed": 2, "failed": 0, "n": 2},
            "interpretation_coverage": {"mean": 0.84, "pass_rate": 1.0, "n": 1},
            "sql_correctness": {"mean": 0.87, "pass_rate": 1.0, "n": 1},
            "tool_compliance": {"n": 2, "passed": 2, "pass_rate": 1.0},
        },
    }
    base.update(overrides)
    return base


class TestResolveEnv:
    @pytest.mark.parametrize("short,long_form", [
        ("dev", "dev-illum"),
        ("test", "test-illum"),
        ("staging", "staging-illum"),
        ("production", "production-illum"),
    ])
    def test_short_env_expands_to_illum_suffix(self, short, long_form):
        assert _resolve_env(short) == long_form

    def test_already_long_env_passes_through(self):
        assert _resolve_env("staging-illum") == "staging-illum"

    def test_unknown_env_passes_through_unchanged(self):
        # Defensive: unknown values are returned as-is so callers see a clear
        # ResourceNotFoundException downstream instead of a silent rename.
        assert _resolve_env("custom-illum") == "custom-illum"

    def test_empty_env_raises(self):
        with pytest.raises(ValueError, match="env is required"):
            _resolve_env("")

    def test_none_env_raises(self):
        with pytest.raises(ValueError, match="env is required"):
            _resolve_env(None)


class TestBuildRunRecord:
    def test_flattens_aggregated_scores_into_top_level_attrs(self):
        run = _local_run_json()
        rec = _build_run_record(run)
        assert rec["run_id"] == "run_20260624_120000"
        assert rec["env"] == "dev-illum"  # short "dev" → "dev-illum"
        assert rec["model_id"] == "us.anthropic.claude-opus-4-6-v1"
        assert rec["question_count"] == 2
        assert rec["agent_response_mean"] == 0.93
        assert rec["agent_response_pass_rate"] == 1.0
        assert rec["interpretation_mean"] == 0.84
        assert rec["sql_correctness_mean"] == 0.87
        assert rec["filter_config"] == {"priority_only": True, "tunnel": False}

    def test_missing_aggregated_scores_leaves_metric_fields_none(self):
        """An eval-skipped run has no aggregated_scores yet."""
        run = _local_run_json()
        run.pop("aggregated_scores")
        rec = _build_run_record(run)
        assert rec["agent_response_mean"] is None
        assert rec["agent_response_pass_rate"] is None
        assert rec["interpretation_mean"] is None
        # Run-level identity still populated.
        assert rec["run_id"] == "run_20260624_120000"

    def test_long_env_passes_through(self):
        run = _local_run_json(env="staging-illum")
        rec = _build_run_record(run)
        assert rec["env"] == "staging-illum"


class TestBuildQuestionRecord:
    def test_flattens_usage_and_eval_blocks(self):
        run = _local_run_json()
        q = run["results"][0]
        rec = _build_question_record("run_20260624_120000", q)
        assert rec["run_id"] == "run_20260624_120000"
        assert rec["question_id"] == 1
        assert rec["flow"] == "fdl"
        assert rec["total_tokens"] == 24_500
        assert rec["input_tokens"] == 21_200
        assert rec["agent_response_score"] == 0.91
        assert rec["agent_response_passed"] is True
        assert rec["interpretation_score"] == 0.84
        assert rec["sql_correctness_score"] == 0.87
        assert rec["sql_correctness_passed"] is True
        assert len(rec["od_fdl_results"]) == 1

    def test_conversational_flow_has_no_interpretation_or_sql(self):
        """Per-trace + interpretation are None for a conversational flow."""
        run = _local_run_json()
        q = run["results"][1]
        rec = _build_question_record("run_20260624_120000", q)
        assert rec["flow"] == "conversational"
        assert rec["interpretation_score"] is None
        assert rec["sql_correctness_score"] is None  # no per_trace at all
        assert rec["sql_correctness_passed"] is None
        assert rec["sql_per_trace_scores"] == []

    def test_multi_sql_question_averages_per_trace_scores(self):
        """A question with multiple SQL calls — sql_correctness_score is the mean,
        sql_correctness_passed is the AND of all passes."""
        run = _local_run_json()
        q = run["results"][0]
        q["eval"]["per_trace"] = [
            {"source": "fdl", "sql_score": 0.9, "sql_passed": True, "sql_reason": "ok"},
            {"source": "athena", "sql_score": 0.6, "sql_passed": False, "sql_reason": "bad"},
        ]
        rec = _build_question_record("run_20260624_120000", q)
        assert rec["sql_correctness_score"] == pytest.approx(0.75)
        assert rec["sql_correctness_passed"] is False
        assert len(rec["sql_per_trace_scores"]) == 2


class TestPersistRunFile:
    @mock_aws
    def test_end_to_end_persist_from_local_json(self, tmp_path):
        """Full round-trip: write a local run JSON, call persist_run_file,
        verify both DynamoDB tables ended up with the right items."""
        client = boto3.client("dynamodb", region_name=REGION)
        _create_tables(client)

        run = _local_run_json()
        run_path = tmp_path / "run_20260624_120000.json"
        run_path.write_text(json.dumps(run, indent=2))

        persist_run_file(run_path)

        runs = boto3.resource("dynamodb", region_name=REGION).Table(
            f"{ENV}-validator-runs"
        )
        run_item = runs.get_item(Key={"run_id": "run_20260624_120000"})["Item"]
        assert run_item["env"] == ENV
        assert run_item["agent_response_mean"] == Decimal("0.93")
        assert run_item["question_count"] == 2

        questions = boto3.resource("dynamodb", region_name=REGION).Table(
            f"{ENV}-validator-question-records"
        )
        resp = questions.query(
            KeyConditionExpression="run_id = :r",
            ExpressionAttributeValues={":r": "run_20260624_120000"},
        )
        assert resp["Count"] == 2
        items = {int(item["question_id"]): item for item in resp["Items"]}
        assert items[1]["flow"] == "fdl"
        assert items[1]["agent_response_score"] == Decimal("0.91")
        assert items[2]["flow"] == "conversational"

    @mock_aws
    def test_missing_env_in_json_raises(self, tmp_path):
        run = _local_run_json()
        run.pop("env")
        run_path = tmp_path / "run.json"
        run_path.write_text(json.dumps(run))
        with pytest.raises(ValueError, match="missing 'env'"):
            persist_run_file(run_path)

    @mock_aws
    def test_missing_run_id_raises(self, tmp_path):
        client = boto3.client("dynamodb", region_name=REGION)
        _create_tables(client)
        run = _local_run_json()
        run.pop("run_id")
        run_path = tmp_path / "run.json"
        run_path.write_text(json.dumps(run))
        with pytest.raises(ValueError, match="missing 'run_id'"):
            persist_run_file(run_path)

    @mock_aws
    def test_re_persist_is_idempotent(self, tmp_path):
        """Calling persist_run_file twice on the same file produces no duplicates —
        same path the runner uses for --eval-latest --persist re-scoring."""
        client = boto3.client("dynamodb", region_name=REGION)
        _create_tables(client)

        run_path = tmp_path / "run_20260624_120000.json"
        run_path.write_text(json.dumps(_local_run_json()))

        persist_run_file(run_path)
        # Mutate a score and re-persist (simulating --eval-latest --persist).
        run = _local_run_json()
        run["results"][0]["eval"]["agent_response"]["score"] = 0.50
        run_path.write_text(json.dumps(run))
        persist_run_file(run_path)

        questions = boto3.resource("dynamodb", region_name=REGION).Table(
            f"{ENV}-validator-question-records"
        )
        resp = questions.query(
            KeyConditionExpression="run_id = :r",
            ExpressionAttributeValues={":r": "run_20260624_120000"},
        )
        # Still 2 records — re-write overwrote, didn't duplicate.
        assert resp["Count"] == 2
        items = {int(item["question_id"]): item for item in resp["Items"]}
        assert items[1]["agent_response_score"] == Decimal("0.50")  # new score won
