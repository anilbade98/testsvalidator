# Analytics Agent Validator

Automated quality framework for the analytics-agentic service. Sends curated questions to the service's SSE streaming endpoint, captures agent responses and SQL traces, and scores them with an LLM judge (DeepEval GEval backed by Claude Sonnet on Bedrock).

---

## Directory layout

```
tests/validator/
  run.py              — CLI entry point
  runner.py           — HTTP SSE client, question filtering, batch runner
  tunnel.py           — SSM port-forwarding tunnels for local runs
  evaluator.py        — DeepEval scoring pass
  metrics.py          — GEval metric definitions (3 metrics)
  bedrock_judge.py    — DeepEval LLM wrapper for Claude on Bedrock
  aggregator.py       — Per-run score aggregation and summary printer
  trace_capture.py    — Parses SSE events into structured records
  validate_catalog.py — JSON schema validator for test_questions.json
  test_questions.json — 77 curated questions across 9 flows
  results/            — Run output files (JSON, git-ignored)
```

---

## Known issues

| Symptom | Cause | Fix |
|---|---|---|
| `ExpiredTokenException when calling StartQueryExecution` | Staging temp credentials expired mid-run — Athena uses boto3 directly with your local AWS creds | Re-run your temp-creds script, then retry |
| AO/RCA questions always fail with "technical difficulties" | `appsettings.Local.json` was pointing to `localhost:808x` without a tunnel | Already fixed — local config now uses the public staging HTTPS URLs directly |
| `agent_response` scores all 0.0–0.3 despite correct answers | `sql_results` was empty — judge penalised "fabricated" numbers it couldn't verify | Already fixed — result rows are now captured in traces |
| `conversational` question triggers data queries | Multi-turn loop sent `BATCH_CLARIFICATION_FALLBACK` after greeting | Already fixed — conversational flow breaks after turn 0 |

---

## Prerequisites

```powershell
pip install -r requirements.txt

# AWS credentials for the target environment
# Run your temp-creds script, or ensure a named profile is active:
#   dev-illum | staging-illum | production-illum | ...
```

The eval pass (LLM judge) uses Bedrock. Ensure your active credentials have `bedrock:InvokeModel` access.

---

## Authentication

Authentication is resolved in order:

1. **`EVAL_BEARER_TOKEN`** environment variable — used as-is, no refresh.
2. **Secrets Manager** at `{env}-illum/analytics-agentic/secrets` — same path used by integration tests. Requires `--env` and a working AWS profile named `{env}-illum`.

```powershell
# Option 1 — bearer token override (simplest for local testing)
$env:EVAL_BEARER_TOKEN = "<token>"
$env:EVAL_TENANT_ID   = "<tenant-id>"   # optional — decoded from JWT if omitted

# Option 2 — Secrets Manager (used in CI)
python -m tests.validator.run --env staging ...
```

---

## Running the validator

All commands are run from the repo root:

```powershell
python -m tests.validator.run [options]
```

### Common flags

| Flag | Description |
|---|---|
| `--env` | Environment: `dev`, `staging`, `production`, etc. Sets the AWS profile and Secrets Manager path. |
| `--base-url` | Base URL of the analytics-agentic service, e.g. `http://localhost:80`. Required unless using `--eval-only` / `--eval-latest`. |
| `--flow` | Run only questions for a specific flow (see [Flows](#flows) below). |
| `--limit` | Max questions after all other filters. |
| `--limit-per-flow` | Pick at most N questions from each flow (e.g. `--limit-per-flow 1` for a smoke run). |
| `--priority-only` | Only run questions tagged `priority: true`. |
| `--tool-flows-only` | Restrict to tool-calling flows (`fdl`, `od`, `ao`, `rca`, `advisory`, `mixed`). Excludes `conversational`, `context`, `refresh`. Intended for CI smoke runs. |
| `--complexity` | Filter by `basic`, `intermediate`, or `advanced`. |
| `--tags` | Comma-separated tag filter, e.g. `--tags csat,acd`. |
| `--ids` | Comma-separated question IDs — overrides all other filters. |
| `--skip-eval` | Collect traces only; skip LLM scoring. Faster, no Bedrock cost. |
| `--persist` | Write results to DynamoDB (CI only — off by default). |
| `--output` | Custom output file path (default: `results/run_<timestamp>.json`). |

### Examples

```powershell
# Smoke run — one question from each flow, no LLM scoring
python -m tests.validator.run `
    --env staging --base-url http://localhost:80 `
    --tunnel --limit-per-flow 1 --skip-eval

# All priority questions, full scoring
python -m tests.validator.run `
    --env staging --base-url http://localhost:80 `
    --tunnel --priority-only

# Single flow
python -m tests.validator.run `
    --env staging --base-url http://localhost:80 `
    --tunnel --flow fdl --skip-eval

# Re-score an existing run file (no agent calls, Bedrock only)
python -m tests.validator.run --eval-only results/run_20260619_120000.json

# Re-score the latest run file
python -m tests.validator.run --eval-latest

# Resume interrupted scoring (skip already-scored records)
python -m tests.validator.run --eval-latest --skip-scored

# Specific questions by ID
python -m tests.validator.run `
    --env staging --base-url http://localhost:80 `
    --tunnel --ids 1,7,14 --skip-eval
```

---

## Local tunnels (`--tunnel`)

When running the service locally (`python main.py`), it cannot reach FDL, OD, or RCA directly. The `--tunnel` flag opens on-demand SSM port-forwarding sessions just before the first question that needs each service.

**Requires:** `--env` and AWS credentials with SSM access to the target environment.

```powershell
python -m tests.validator.run `
    --env staging `
    --base-url http://localhost:80 `
    --tunnel `
    --flow fdl --limit 1 --skip-eval
```

### Tunnel topology

| Service | Local port | Remote host | Remote port | Relay | Notes |
|---|---|---|---|---|---|
| FDL | 9007 | `fdl-svc-alb.{env}-illum.internal` | 9007 | jump server | internal HTTP |
| OD | 9009 | `od-analytics-dq-alb.{env}-illum.internal` | 80 | jump server | internal HTTP |
| AO | 8081 | ECS task private IP (dynamic) | 80 | any SSM instance | direct-to-task; bypasses HTTPS ALB — container speaks plain HTTP |
| RCA | 9010 | `eai-anomaly-detection.illuminate.service.nicecxone-staging.com` | 80 | jump server | public hostname, HTTP port 80 |
| Athena | — | boto3 direct | — | — | — |

### Custom tunnel ports

```powershell
--tunnel-fdl-port 9007    # default
--tunnel-od-port  9009    # default
--tunnel-ao-port  8081    # default
--tunnel-rca-port 9010    # default
```

### Running against the deployed staging service (no local service)

If you want to run the validator against the already-deployed staging service rather than a local one, use the `tunnel` skill to open the analytics-agentic ALB tunnel instead:

```powershell
# In a separate terminal — opens localhost:8080 → analytics-agentic-alb.staging-illum.internal:80
# (Use the tunnel skill: /tunnel staging)

python -m tests.validator.run `
    --env staging `
    --base-url http://localhost:8080 `
    --limit-per-flow 1 --skip-eval
# No --tunnel flag needed — the service handles FDL/AO/RCA calls internally
```

---

## Flows

Questions are classified into 9 flows:

| Flow | Description | Data sources |
|---|---|---|
| `fdl` | Agent performance, calls, queues, CSAT, evaluations | FDL / Snowflake |
| `od` | Knowledge base queries, article effectiveness, contact summaries | Athena (OD) |
| `ao` | Automation opportunity analysis | AO MCP service |
| `rca` | Anomaly detection and root-cause analysis | RCA service |
| `advisory` | Recommendations, coaching, quality insights | FDL + Athena |
| `mixed` | Questions requiring both FDL and Athena data | FDL + Athena |
| `context` | Questions that reference prior context injected via `context_fixture` | Any |
| `refresh` | Questions that trigger a session refresh | Any |
| `conversational` | Questions with no data query (greetings, help, out-of-scope) | None |

---

## Metrics

Three GEval metrics are applied. All use Claude Sonnet 4.6 on Bedrock as the judge. Pass threshold: **0.7**.

| Metric | Applies to | What it measures |
|---|---|---|
| `interpretation_coverage` | `fdl`, `od`, `ao`, `rca`, `advisory`, `mixed` | Whether the agent's set of query interpretations covers the user's intent given available schema |
| `sql_correctness` | Same + `refresh` | Per-SQL: whether the generated query correctly implements its stated interpretation |
| `agent_response` | All flows | Response relevance + factual accuracy against SQL results and schema docs |

Each scoring pass makes 2 Bedrock calls per metric (DeepEval GEval: generate evaluation steps → produce score). A full run against all 77 questions costs roughly 450–550 Bedrock calls.

### Skipping eval

`--skip-eval` collects traces only. You can score later:

```powershell
python -m tests.validator.run --eval-latest
```

---

## Output format

Each run produces a single JSON file in `results/`:

```json
{
  "run_id": "run_20260619_120130",
  "run_timestamp": "2026-06-19T12:01:30Z",
  "env": "staging",
  "base_url": "http://localhost:80",
  "total": 9,
  "filter_config": { ... },
  "results": [
    {
      "question_id": 1,
      "question": "What is the average handle time last week?",
      "flow": "fdl",
      "complexity": "basic",
      "duration_ms": 8420,
      "agent_response": "The average handle time last week was 4m 32s ...",
      "cortex_traces": [
        {
          "tool_name": "get_fdl_data",
          "semantic_model": ["SEGMENT"],
          "interpretation": "Average handle time for last week",
          "generated_sql": "SELECT AVG(...) FROM ...",
          "sql_status": "success",
          "row_count": 1
        }
      ],
      "usage": { "input_tokens": 4200, "output_tokens": 310, "total_tokens": 4510 },
      "eval": {
        "interpretation_coverage": { "score": 0.85, "passed": true, "reason": "..." },
        "per_trace": [{ "sql_score": 0.90, "sql_passed": true, "sql_reason": "..." }],
        "agent_response": { "score": 0.88, "passed": true, "reason": "..." }
      }
    }
  ],
  "aggregated_scores": {
    "total": 9,
    "errors": 0,
    "agent_response": { "mean": 0.84, "pass_rate": 0.89 },
    "interpretation_coverage": { "mean": 0.81, "pass_rate": 0.83 },
    "sql_correctness": { "mean": 0.87, "pass_rate": 0.91 },
    "by_flow": { "fdl": { "agent_response_mean": 0.86, "agent_response_pass_rate": 1.0 } }
  }
}
```

---

## Validating the question catalog

```powershell
python -m tests.validator.validate_catalog
```

Checks every question in `test_questions.json` for required fields, valid flow names, valid complexity values, and unique IDs. Run this before committing changes to the catalog.

---

## Adding questions

Edit `test_questions.json`. Each entry must have:

```json
{
  "id": 60,
  "question": "...",
  "flow": "fdl",
  "complexity": "basic",
  "priority": false,
  "expected_tools": ["get_fdl_data"],
  "golden_answer": null,
  "tags": ["aht"],
  "multi_turn": false
}
```

Required fields: `id` (unique int), `question`, `flow`, `complexity`, `priority`, `expected_tools`, `golden_answer`, `tags`, `multi_turn`.

Valid flows: `fdl`, `od`, `ao`, `rca`, `advisory`, `mixed`, `context`, `refresh`, `conversational`.

Valid complexity: `basic`, `intermediate`, `advanced`.

Run `validate_catalog` after editing.

---

## Run strategy

Three modes, each with a different scope:

| Mode | Trigger | Command | Questions | Eval | Persist |
|---|---|---|---|---|---|
| **CI smoke** | Every PR | `--tool-flows-only --limit-per-flow 1 --skip-eval` | ~6 (one per tool flow) | No | No |
| **Nightly** | Scheduled | `--priority-only --persist` | ~19 priority q's | Yes | Yes |
| **Manual** | Before deploy / ad-hoc | any filters | Developer's choice | Optional | No |

`--persist` requires a full scored run — it is incompatible with `--skip-eval` and the CLI will reject that combination. Persisting unscored records would produce an invalid baseline for regression comparison.

### CI smoke run

Fast gate (~2 min) — catches service crashes, auth failures, wrong tool calls. No LLM scoring.

```bash
python -m tests.validator.run \
    --env staging \
    --base-url https://analytics-agentic-alb.staging-illum.internal \
    --tool-flows-only \
    --limit-per-flow 1 \
    --skip-eval
```

### Nightly run

Full scoring pass + baseline comparison. Writes to DynamoDB with `--persist` for regression tracking.

```bash
python -m tests.validator.run \
    --env staging \
    --base-url https://analytics-agentic-alb.staging-illum.internal \
    --priority-only \
    --persist
```

The `--persist` flag is intentionally off by default to prevent local test runs from polluting the baseline.
