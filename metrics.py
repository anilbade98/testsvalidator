"""GEval metric definitions for the Analytics Agent Validator.

Single-turn metrics:
  - interpretation_coverage: one score per question
  - sql_correctness:          one score per SQL call
  - agent_response:           one score per question

Chain metrics (multi-turn):
  - turn_response:        per-turn quality (same criteria as agent_response)
  - context_retention:    whether later turns correctly build on prior context

Each metric uses evaluation_steps (3-4 sub-criteria) instead of a single
criteria string. This produces finer-grained decimal scores (e.g. 0.83, 0.67)
because DeepEval averages the individual step scores rather than relying on
one integer 0-10 rating.

Pass threshold: 0.7.
"""

from __future__ import annotations

from pathlib import Path

from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCaseParams


PASS_THRESHOLD = 0.7

_SCHEMA_DOCS_DIRS = {
    "fdl": Path(__file__).resolve().parents[2] / "app" / "schema-docs" / "fdl",
    "athena": Path(__file__).resolve().parents[2] / "app" / "schema-docs" / "athena",
}

SQL_FLOWS = {"fdl", "od", "ao", "rca", "advisory", "mixed", "refresh"}
INTERPRETATION_FLOWS = {"fdl", "od", "ao", "rca", "advisory", "mixed"}


# ── SQL conventions extracted from agent prompts ──────────────────────────────
# These supplement schema column docs — they capture the behavioral rules the
# agent is supposed to follow when writing SQL, so the judge can detect
# violations of conventions beyond just wrong column names.

FDL_SQL_CONVENTIONS = """
## FDL (Snowflake SQL) Conventions the agent must follow

### Tenant scoping — NO tenantid filter needed
FDL queries run against the Snowflake DataLake. Data is automatically scoped to the tenant
by the FDL service API authentication. There is NO TenantId column to filter on and it is
CORRECT to omit any tenantid / TenantId WHERE condition. Do NOT penalise FDL SQL for
a missing tenantid filter.

### Date syntax (Snowflake)
FDL uses Snowflake SQL. Valid date expressions include DATEADD(), CURRENT_DATE(),
CURRENT_TIMESTAMP(), and DATEDIFF(). Do NOT apply Athena/Presto date conventions
(BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD') to FDL queries — those are Athena-only.

### Counting segments
- Always use COUNT(DISTINCT CallSegmentId) for segment counts — never COUNT(*) or COUNT(column) without DISTINCT.

### Handled contacts
- Add HandleTime > 0 when aggregating handled contacts from AgentContacts.
- Exclude ContactTypeName = 'Consult' when calculating HoldTime metrics.

### Behavioral scores
- InappropriateActionIndexScore is INVERSE (higher = worse) — never average it with other behavioral scores.
- Valid behavioral scores to average: BeEmpatheticIndexScore, ActiveListeningIndexScore,
  BuildRapportIndexScore, DemonstrateOwnershipIndexScore, EffectiveQuestioningIndexScore.

### Cross-schema JOIN HAVING rule
Apply HAVING count(<key>) = 1 ONLY for these specific cross-schema joins to prevent row duplication:
  - CALL_SEGMENT + CONTACT: JOIN ON ContactNo, add HAVING count(cs.ContactNo) = 1
  - AGENT_CONTACT + CONTACT: JOIN ON ContactNo, add HAVING count(ac.ContactNo) = 1
  - AGENT_CONTACT + CALL_SEGMENT: JOIN ON ContactNo, add HAVING count(ac.ContactNo) = 1
  - CALL_SEGMENT + QUALITY_MANAGEMENT: JOIN ON CallSegmentId = SegmentId — NO HAVING clause
Do NOT add HAVING count = 1 to single-table queries or joins without row-duplication risk.

### Other
- Always include both AgentId and AgentName for agent-level questions.
- Add NOT NULL check for CallReason, OutcomeName, PointOfContactNo, PointOfContactName.
- Weeks start on Sunday: DATE_TRUNC('week', col + INTERVAL '1 day') - INTERVAL '1 day'.
"""

ATHENA_SQL_CONVENTIONS = """
## Athena (Presto SQL) Conventions the agent must follow

### Required filters (every query)
- Must include a static tenantid equality filter: `tenantid = '<uuid>'`.
  If the SQL shows `tenantid = 'auto-injected'`, treat it as correct — it is a placeholder
  the service replaces with the real tenant UUID before execution.
  Subquery form (`tenantid = (SELECT DISTINCT tenantid …)`) is WRONG — Athena partition
  projection requires a static equality condition.
- Must include: date range filter (date BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD')
- The 'date' column is VARCHAR 'YYYY-MM-DD' — do NOT cast to DATE type.

### Table usage
- Do NOT prefix table names with the database name.
- Use lowercase column names.
- Apply LIMIT unless question asks for all results.
"""


def load_schema_docs() -> dict[str, str]:
    """Load all schema markdown files from fdl + athena docs dirs."""
    cache: dict[str, str] = {}
    for source, docs_dir in _SCHEMA_DOCS_DIRS.items():
        if docs_dir.exists():
            for md_file in docs_dir.glob("*.md"):
                cache[f"{source}/{md_file.name}"] = md_file.read_text(encoding="utf-8")
    return cache


def schema_docs_for_schemas(schemas: list[str], cache: dict[str, str]) -> str:
    parts: list[str] = []
    for schema in schemas:
        slug = schema.lower()
        for key, body in cache.items():
            if slug in key:
                parts.append(f"# {schema}\n\n{body}")
                break
    return "\n\n---\n\n".join(parts) if parts else "\n\n---\n\n".join(cache.values())


def all_schema_docs(cache: dict[str, str]) -> str:
    return "\n\n---\n\n".join(cache.values())


def build_metrics(
    llm: DeepEvalBaseLLM,
    threshold: float = PASS_THRESHOLD,
    *,
    verbose: bool = True,
) -> dict[str, GEval]:
    def _g(name: str, steps: list[str]) -> GEval:
        return GEval(
            name=name,
            evaluation_steps=steps,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.CONTEXT,
            ],
            model=llm,
            threshold=threshold,
            verbose_mode=verbose,
            async_mode=False,
        )

    return {
        "interpretation_coverage": _g(
            "Interpretation Coverage",
            [
                "The query interpretations correctly identify the relevant schema tables "
                "and columns for the user's question, using appropriate domain terminology "
                "and field synonyms documented in the schema context.",

                "Taken together, the interpretations cover all distinct analytical aspects "
                "of the user's question — no major dimension or sub-question is left unaddressed. "
                "Evaluate the set as a whole, not each interpretation in isolation.",

                "Each interpretation is feasible given the available data model — no impossible, "
                "non-existent, or unsupported columns or tables are referenced. When the user's "
                "terminology does not map directly to a schema field, exploring reasonable "
                "alternatives from the schema is correct behavior.",
            ],
        ),

        "sql_correctness": _g(
            "SQL Technical Correctness",
            [
                "The SQL references only documented table and column names from the schema context, "
                "using the correct aliases and naming conventions (e.g. FDL alias names, not "
                "physical Snowflake identifiers). Column synonyms in the schema docs are authoritative.",

                "The SQL includes all required filters (date range, tenant ID, channel where needed) "
                "and applies the correct row-level guards (e.g. HandleTime > 0 for handled contacts, "
                "ContactTypeName != 'Consult' for hold time, NOT NULL for optional columns).",

                "The SQL uses correct aggregations: COUNT(DISTINCT CallSegmentId) for segment counts, "
                "appropriate GROUP BY, and HAVING only where cross-schema row duplication is a risk. "
                "Behavioral score InappropriateActionIndexScore is not averaged with other scores.",

                "The SQL logic correctly implements the stated interpretation — joins, calculations, "
                "filters, and output columns match what the interpretation describes. Only evaluate "
                "whether the SQL implements its interpretation, not whether the interpretation "
                "itself was the right question to ask.",
            ],
        ),

        "agent_response": _g(
            "Agent Response Quality",
            [
                "The response directly and relevantly answers the user's question with "
                "specific, data-driven insights. It cites concrete values, identifies "
                "meaningful patterns, and offers actionable context — not vague generalities. "
                "Domain terminology may differ from the user's wording; accept valid associations.",

                "Every specific numerical claim is traceable to the SQL results in the context "
                "or is a valid arithmetic derivation (sum, average, percentage, ratio, delta) "
                "from those results. Verify the arithmetic rather than rejecting derived values. "
                "A factual error is a claim not traceable to any result or valid computation — "
                "fabricated values, wrong scales, or misattributed data.",

                "The response correctly interprets domain context: metric scales and thresholds "
                "(e.g. CSAT 0-10 where ~5=neutral), business meaning of dimensions, and "
                "limitations of the data (e.g. partial week, small sample size, missing data). "
                "Do not penalize claims plausible from unseen rows — only penalize claims that "
                "contradict visible SQL rows or are clearly fabricated.",
            ],
        ),
    }


def build_chain_metrics(
    llm: DeepEvalBaseLLM,
    threshold: float = PASS_THRESHOLD,
    *,
    verbose: bool = True,
) -> dict[str, GEval]:
    """GEval metrics specific to multi-turn chain evaluation."""

    def _g(name: str, steps: list[str]) -> GEval:
        return GEval(
            name=name,
            evaluation_steps=steps,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.CONTEXT,
            ],
            model=llm,
            threshold=threshold,
            verbose_mode=verbose,
            async_mode=False,
        )

    return {
        "turn_response": _g(
            "Turn Response Quality",
            [
                "The response directly and relevantly answers the current turn's question "
                "with specific, data-driven insights. It cites concrete values, identifies "
                "meaningful patterns, and offers actionable context — not vague generalities.",

                "Every specific numerical claim is traceable to the SQL results or prior "
                "turn context provided, or is a valid arithmetic derivation from those. "
                "Verify derivations rather than rejecting them. Only flag claims not "
                "traceable to any visible data.",

                "The response correctly applies domain knowledge — metric scales, threshold "
                "meanings, and business context — consistent with the schema documentation "
                "and prior conversation context.",
            ],
        ),

        "context_retention": _g(
            "Context Retention",
            [
                "The response correctly references or builds on specific information "
                "established in earlier turns — figures, entities, time ranges, or conclusions "
                "that were already discussed. It does not re-derive from scratch what was "
                "already known from prior turns.",

                "The response does not contradict or ignore relevant findings from previous "
                "turns. If earlier turns surfaced specific data points (e.g. a team name, "
                "a metric value, an anomaly), the response either references them correctly "
                "or has a clear reason for not doing so.",
            ],
        ),
    }
