# Analytics Agent Validator

Automated quality framework for the analytics-agentic service. Sends curated questions to the service's SSE streaming endpoint, captures agent responses and SQL traces, and scores them with an LLM judge (DeepEval GEval backed by Claude Sonnet on Bedrock).

---

## Directory layout

```
tests/validator/
  run.py              — CLI entry point (run / eval-only / confirm-file modes)
  runner.py           — HTTP SSE client, question filtering, batch runner
  tunnel.py           — SSM port-forwarding tunnels (data services + deployed ALB)
  evaluator.py        — DeepEval scoring pass
  metrics.py          — GEval metric definitions (3 metrics)
  bedrock_judge.py    — DeepEval LLM wrapper for Claude on Bedrock (temperature=0)
  aggregator.py       — Per-run score aggregation and summary printer
  comparator.py       — Regression detection vs the DynamoDB baseline
  persistence.py      — DynamoDB writer/reader (runs + question records)
  records.py          — Record classification (question vs chain) + completeness
  reporter.py         — HTML report generation
  trace_capture.py    — Parses SSE events into structured records
  chain_runner.py     — Multi-turn chain execution
  chain_evaluator.py  — Multi-turn chain scoring
  ui.py               — Streamlit UI (Run / History / Catalog tabs)
  validate_catalog.py — JSON schema validator for test_questions.json
  test_questions.json — 77 curated questions across 9 flows
  test_chains.json    — Multi-turn conversation chains
  results/            — Run output files (JSON + HTML, git-ignored)
```

---

## Known issues

| Symptom | Cause | Fix |
|---|---|---|
| `ExpiredTokenException when calling StartQueryExecution` | Staging temp credentials expired mid-run — Athena uses boto3 directly with your local AWS creds | Re-run your temp-creds script, then retry |
| `ExpiredToken`/`ProfileNotFound` **only** in the eval or Persist-to-DB step (run pass succeeds) | Process resolved the stale/absent `default` profile — the run pass is pure HTTP, but Bedrock (eval) and DynamoDB (persist) call boto3 directly | Restart with `AWS_PROFILE=<env>-illum` set (see [Streamlit UI](#streamlit-ui)); ensure `~/.aws/config` has that SSO profile |
| `UnauthorizedSSOTokenError` in eval while service/HTTP calls work | The **SSO session** expired. The bearer token (service auth) and the AWS SSO session are independent — refreshing the token does nothing for SSO. Role creds auto-refresh only while the SSO session is alive | `aws sso login --profile <env>-illum`, then retry. A long batch can outlive the SSO session — re-login between chunks |
| `404` on `/api/v1/chat/stream` against a deployed service (`/health` works) | The ALB tunnel reached a **different service** — this one isn't deployed in that env | Use Local mode, or verify the deployment. The UI now verifies the tunnel target and refuses a wrong-service tunnel |
| AO/RCA questions always fail with "technical difficulties" | `appsettings.Local.json` was pointing to `localhost:808x` without a tunnel | Already fixed — local config now uses the public staging HTTPS URLs directly |
| `agent_response` scores all 0.0–0.3 despite correct answers | `sql_results` was empty — judge penalised "fabricated" numbers it couldn't verify | Already fixed — result rows are now captured in traces |
| `conversational` question triggers data queries | Multi-turn loop sent `BATCH_CLARIFICATION_FALLBACK` after greeting | Already fixed — conversational flow breaks after turn 0 |

---

## Prerequisites

```powershell
pip install -r requirements.txt

# AWS credentials for the target environment.
# Set AWS_PROFILE to the env's SSO profile at launch (see "Streamlit UI" below) —
# this binds all boto3 calls to a refreshable SSO profile instead of the
# `default` profile, which SSO-only credential setups do not keep refreshed.
#   dev-illum | staging-illum | production-illum | ...
#
# The profile must exist in ~/.aws/config as an SSO profile with a live session:
#   aws sso login --profile staging-illum
```

The eval pass (LLM judge) uses Bedrock. Ensure the profile has `bedrock:InvokeModel` access.

---

## Streamlit UI

Launch the interactive validator UI with the SSO profile and environment set on
the command line. `AWS_PROFILE` is set at launch so the Streamlit process — and
the run/eval subprocesses it spawns, plus the in-process Persist-to-DB path —
all inherit a refreshable SSO profile:

```powershell
$env:AWS_PROFILE = "staging-illum"; $env:ENV = "staging"; $env:APP_ENV = "local"; streamlit run tests/validator/ui.py
```

To run the service under test locally in a second terminal, use the same prefix:

```powershell
$env:AWS_PROFILE = "staging-illum"; $env:ENV = "staging"; $env:APP_ENV = "local"; python main.py
```

> **Why set `AWS_PROFILE` at launch (not mid-run)?** boto3 caches its default
> session, and a long-running Streamlit process freezes whatever credentials it
> resolved on first use. Setting the profile *before* launch means every AWS
> call (Bedrock judge, DynamoDB persist) resolves the SSO profile and picks up
> renewed tokens across the run. If you see `ExpiredToken` only during the eval
> or persist step while the run pass succeeds, the process resolved the stale
> `default` profile — restart with `AWS_PROFILE` set.

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
| `--persist` | Write results to DynamoDB after a full scored run (off by default). Only PASSED runs enter the baseline. |
| `--allow-local-persist` | Allow `--persist` when the base URL is localhost (the UI sets this; needed for the deployed-via-tunnel case too). Otherwise localhost persists are refused unless `CI=true`. |
| `--reference-date` | `YYYY-MM-DD` anchor date. Passed to the service as `REFERENCE_DATE` (SQL date filtering) and, for non-SQL flows, prepended to the question text as "Today is …". Stored in the run so a later confirm/heal re-runs with the same date. |
| `--baseline-n` | Baseline size / minimum prior PASSED runs required before regression is assessed (default 10). Fewer → check skipped. |
| `--regression-delta` | Per-question informational drop threshold (default 0.2). |
| `--confirm-runs` | On a `targeted` per-question regression, re-run each flagged question this many times to confirm/heal (default 2; `0` disables). See [Regression detection](#regression-detection). |
| `--confirm-file PATH` | Compare an existing scored run against the baseline, confirm/heal targeted regressions against the live service, then persist if it resolves to PASSED. Drives the UI "Persist to DB" button. |
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

### Running against a deployed service (no local service)

To run against an already-deployed service instead of a local one, open an SSM
port-forward to its ALB (`analytics-agentic-alb.{env}-illum.internal:80` → `localhost:8080`)
and point `--base-url` at it. No `--tunnel` flag is needed — the deployed service reaches
FDL/OD/AO/RCA itself.

In the **Streamlit UI**, selecting **Deployed** opens this ALB tunnel automatically and
**verifies the target** actually serves this service (it reads `/openapi.json` and checks
for `/api/v1/chat/stream`) — a `/health` 200 alone is not trusted, since any service
answers `/health`. If the tunnel lands on the wrong service, it tears down and tells you.

From the CLI, open it via the `tunnel` skill (`/tunnel <env>`), then:

```powershell
python -m tests.validator.run `
    --env staging `
    --base-url http://localhost:8080 `
    --limit-per-flow 1 --skip-eval
```

> **The service must actually be deployed in that environment.** If the ECS cluster
> (`analytics-agentic-fargate`) doesn't exist there, the ALB DNS may resolve to an
> unrelated service and requests will 404 on `/api/v1/chat/stream`. Verify the deployment
> or use **Local mode** (`python main.py`), which is the proven path for baseline runs.

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

## Regression detection

When a run is persisted with comparison enabled (the default under `--persist`), it is
compared against the **baseline** — the last N **PASSED** runs for the environment in
DynamoDB. FAILED and REGRESSION runs never enter the baseline, so a developer's bad run
can't poison the reference the next run compares against.

- The check is **skipped** until at least `--baseline-n` (default 10) eligible prior runs
  exist — comparing against a thin, noisy baseline is worse than not comparing.
- A run is **FAILED** (and excluded) if any question errored or returned no result.
- Only a **PASSED** run is persisted. A REGRESSION run keeps its verdict in the local
  JSON but is not written to the baseline.

### The signals

The verdict is `regression_kinds` — the set of signals that fired. It is designed so that
run-to-run **agent non-determinism** (per-question score σ ≈ 0.11, ±0.4 swings) does not
read as a regression, while a real broad or localised drop still does. The signals are
tuned to their variance: the whole-run aggregate is very stable (σ ≈ 0.013), per-flow
means less so, and single-question scores least of all.

| Kind | Signal | Fires when | Noise guard |
|---|---|---|---|
| `sigma` | Aggregate σ-band per metric | metric mean drops ≥ `max(2σ, floor)` below baseline | `agent_response` floor **0.03**; diagnostic sub-metrics (`interpretation_coverage`, `sql_correctness`) floor **0.06** — noisier, so they need a bigger drop |
| `flow` | Per-flow `agent_response` σ-band | a flow's mean drops ≥ `max(2σ, 0.08)` **and** ≥ 2 of its questions individually regressed | flows with < 5 questions are `skipped`; a drop with < 2 corroborating questions is `warn`, not a failure |
| `targeted` | Single-question answer drop | a question's **`agent_response`** drops ≥ 0.3 **and** is now failing | restricted to `agent_response` (sub-metrics excluded); then **re-run confirmed** (below) |
| `trend` | Slow drift | newest-half vs oldest-half mean decline ≥ 0.05 over baseline+current | uses the stable aggregate means |
| `floor` | Absolute pass-rate floor | `agent_response` pass-rate < 0.7 | deterministic backstop |
| `breadth` | Gray-zone tie-breaker | a metric is in the 1–2σ gray zone **and** ≥ 15% / ≥ 5 questions regressed | only in the gray zone |
| `perf_latency` / `perf_tokens` | Cost regression | avg latency/tokens up ≥ `max(2σ, +25%)` | one-sided (only increases), needs both σ and % |

**Three-layer philosophy:** `agent_response` (the answer-quality metric the judge scores)
is the primary gate with the tight floor. The internal diagnostic sub-metrics stay in the
aggregate σ-band as a safety net for a *broad* SQL/interpretation collapse, but with a
higher floor so their marginal noise doesn't fail a run — especially one whose answers
actually improved. Per-question and per-flow signals add localised sensitivity, hardened
by corroboration and re-run confirmation.

### Re-run confirmation + healing

The `targeted` per-question signal is the most noise-prone, so it is **confirmed** before
it counts. On a `targeted` regression the runner re-runs each flagged question
`--confirm-runs` times (default 2) against the live service and re-scores:

- **Recovered** (median re-run clears the drop threshold): the question was a transient
  failure or one-off bad answer. Its stored record is **healed** — replaced with a
  representative re-run — so a cleared run enters the baseline with good data instead of
  the transient 0.0, and the run re-aggregates + re-compares to PASSED.
- **Reproduces** the drop: a real regression. The record is left as-is and the run stays
  REGRESSION.

Confirmation needs a live service, so it runs on the full-run path and via
`--confirm-file` (the UI "Persist to DB" button). It only touches `targeted` — the stable
`sigma`/`flow`/`trend`/`floor` signals stand on their own. This means a freshly-collected
report you only *view* still shows its raw (pre-confirmation) comparison; the verdict
resolves when you confirm/persist.

### Tuning

All thresholds are constructor params on `Comparator` (see the `DEFAULT_*` constants in
`comparator.py`) and the key ones are exposed as CLI flags (`--baseline-n`,
`--regression-delta`, `--confirm-runs`). Lowering false positives raises the bar for real
regressions — after retuning, sanity-check sensitivity by comparing a known-bad build
against the baseline and confirming it still flags.

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
