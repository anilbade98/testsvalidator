"""Schema validation for test_questions.json.

Run:
    python -m tests.validator.validate_catalog

Exits 0 on success, 1 on any validation error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = HARNESS_DIR / "test_questions.json"

VALID_FLOWS = {"fdl", "od", "ao", "rca", "advisory", "mixed", "context", "refresh", "conversational"}
VALID_COMPLEXITY = {"basic", "intermediate", "advanced"}


def _err(path: str, msg: str) -> str:
    return f"  [{path}] {msg}"


def validate_questions(questions: list[dict]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[int] = set()

    for i, q in enumerate(questions):
        loc = f"question[{i}]"

        qid = q.get("id")
        if not isinstance(qid, int):
            errors.append(_err(loc, f"'id' must be an int, got {type(qid).__name__}"))
        elif qid in seen_ids:
            errors.append(_err(loc, f"duplicate id={qid}"))
        else:
            seen_ids.add(qid)

        if not isinstance(q.get("question"), str) or not q["question"].strip():
            errors.append(_err(loc, "'question' must be a non-empty string"))

        flow = q.get("flow")
        if flow not in VALID_FLOWS:
            errors.append(_err(loc, f"'flow' must be one of {sorted(VALID_FLOWS)}, got {flow!r}"))

        complexity = q.get("complexity")
        if complexity not in VALID_COMPLEXITY:
            errors.append(_err(loc, f"'complexity' must be one of {sorted(VALID_COMPLEXITY)}, got {complexity!r}"))

        if not isinstance(q.get("priority"), bool):
            errors.append(_err(loc, f"'priority' must be a bool, got {type(q.get('priority')).__name__}"))

        if not isinstance(q.get("expected_tools"), list):
            errors.append(_err(loc, "'expected_tools' must be a list"))
        elif not all(isinstance(t, str) for t in q["expected_tools"]):
            errors.append(_err(loc, "'expected_tools' must be a list of strings"))

        if "golden_answer" not in q:
            errors.append(_err(loc, "'golden_answer' field is required (use null if not set)"))

        if not isinstance(q.get("tags"), list):
            errors.append(_err(loc, "'tags' must be a list"))
        elif not all(isinstance(t, str) for t in q["tags"]):
            errors.append(_err(loc, "'tags' must be a list of strings"))

        if not isinstance(q.get("multi_turn"), bool):
            errors.append(_err(loc, f"'multi_turn' must be a bool, got {type(q.get('multi_turn')).__name__}"))

    return errors


def main() -> int:
    all_errors: list[str] = []

    if not QUESTIONS_PATH.exists():
        print(f"ERROR: {QUESTIONS_PATH} not found")
        return 1

    try:
        questions = json.loads(QUESTIONS_PATH.read_text())
    except json.JSONDecodeError as exc:
        print(f"ERROR: {QUESTIONS_PATH.name} is not valid JSON — {exc}")
        return 1

    if not isinstance(questions, list):
        print(f"ERROR: {QUESTIONS_PATH.name} must be a JSON array, got {type(questions).__name__}")
        return 1

    q_errors = validate_questions(questions)

    if q_errors:
        print(f"\ntest_questions.json — {len(q_errors)} error(s):")
        for e in q_errors:
            print(e)
        all_errors.extend(q_errors)

    if not all_errors:
        q_by_flow: dict[str, int] = {}
        for q in questions:
            q_by_flow[q["flow"]] = q_by_flow.get(q["flow"], 0) + 1
        priority_count = sum(1 for q in questions if q.get("priority"))

        print(f"test_questions.json  OK — {len(questions)} questions, {priority_count} priority")
        for flow, count in sorted(q_by_flow.items()):
            print(f"  {flow:<15} {count}")
        return 0

    print(f"\n{len(all_errors)} total error(s). Fix and re-run.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
