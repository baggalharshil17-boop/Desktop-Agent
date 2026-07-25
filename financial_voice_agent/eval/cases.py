from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    name: str
    transcript: str
    expected_tools: list[str]
    forbidden_tools: list[str]
    mocked_screen_result: dict | None = None
    skip_reason: str | None = None


def load_eval_cases(path: str) -> list[EvalCase]:
    with open(path, "r", encoding="utf-8") as f:
        raw_cases = json.load(f)
    return [
        EvalCase(
            name=c["name"],
            transcript=c["transcript"],
            expected_tools=c.get("expected_tools", []),
            forbidden_tools=c.get("forbidden_tools", []),
            mocked_screen_result=c.get("mocked_screen_result"),
            skip_reason=c.get("skip_reason"),
        )
        for c in raw_cases
    ]
