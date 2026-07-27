"""Classification helpers for validator run `results` records.

A run's `results` list mixes two record kinds that share the list but have
different shapes:

  - **question records** — single-turn, keyed by ``question_id``.
  - **chain records**    — multi-turn conversations, keyed by ``chain_id``.

Every consumer (aggregator, evaluator, chain_evaluator, persistence, reporter,
comparator, run) has to separate the two before processing. Historically each
did it inline with either ``"question_id" in r`` or ``r.get("question_id") is
not None`` — two subtly different tests that could drift apart. When persistence
forgot the filter entirely, chain records were mis-handled as "malformed"
question records.

Centralise the taxonomy here so there is one definition of "what is a question
vs a chain", and every call site stays in agreement.
"""

from __future__ import annotations

from typing import Iterable

QUESTION_KEY = "question_id"
CHAIN_KEY = "chain_id"


def is_question_record(record: dict) -> bool:
    """True if ``record`` is a single-turn question record (has ``question_id``)."""
    return QUESTION_KEY in record


def is_chain_record(record: dict) -> bool:
    """True if ``record`` is a multi-turn chain record (has ``chain_id``)."""
    return CHAIN_KEY in record


def question_records(results: Iterable[dict] | None) -> list[dict]:
    """Return only the question records from a run's ``results`` list."""
    return [r for r in (results or []) if is_question_record(r)]


def chain_records(results: Iterable[dict] | None) -> list[dict]:
    """Return only the chain records from a run's ``results`` list."""
    return [r for r in (results or []) if is_chain_record(r)]


def has_chains(results: Iterable[dict] | None) -> bool:
    """True if any chain record is present in ``results``."""
    return any(is_chain_record(r) for r in (results or []))


def question_failed(record: dict) -> bool:
    """True if a question did not execute successfully.

    "Executed successfully" means: no ``error``, and the agent produced *some*
    result — either a text answer (``agent_response``) or structured tool output
    (``od_fdl_results`` / ``tool_results``). Some flows legitimately return data
    without prose (e.g. an automation-opportunity refresh yields rows, not a
    summary), so an empty ``agent_response`` alone is NOT a failure when data
    came back. Only an explicit error, or a truly empty record (no answer and no
    data), counts as failed.
    """
    if record.get("error"):
        return True
    if (record.get("agent_response") or "").strip():
        return False
    # No text answer — but structured tool output means it still executed.
    if record.get("od_fdl_results") or record.get("tool_results"):
        return False
    return True


def failed_question_ids(results: Iterable[dict] | None) -> list:
    """Question ids (from question records) that errored or returned no result."""
    return [
        r.get("question_id")
        for r in question_records(results)
        if question_failed(r)
    ]


def run_is_complete(results: Iterable[dict] | None) -> bool:
    """True if the run has question records and every one of them succeeded.

    An empty run (no question records) is not complete — there is nothing to
    attest to.
    """
    qs = question_records(results)
    return bool(qs) and not any(question_failed(r) for r in qs)
