from __future__ import annotations

import os

from dotenv import dotenv_values


def read_env_file(path: str) -> dict[str, str]:
    if not os.path.exists(path):
        return {}
    return dict(dotenv_values(path))


def merge_env_values(
    existing: dict[str, str], new_values: dict[str, str], *, overwrite: bool
) -> dict[str, str]:
    if overwrite:
        return {**existing, **new_values}
    return {**new_values, **existing}


def write_env_file(path: str, values: dict[str, str]) -> None:
    lines = [f"{key}={values[key]}" for key in sorted(values)]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n" if lines else "")
