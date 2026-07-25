from __future__ import annotations

import json
import os


class FixtureNotFoundError(Exception):
    pass


def load_fixture(name: str, fixtures_dir: str = "fixtures") -> dict:
    path = os.path.join(fixtures_dir, f"{name}.json")
    if not os.path.exists(path):
        raise FixtureNotFoundError(f"Fixture '{name}' not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
