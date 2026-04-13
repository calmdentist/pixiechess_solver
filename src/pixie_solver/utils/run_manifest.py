from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

from pixie_solver.utils.serialization import JsonValue, canonical_json

PIXIE_RULESET_ID = "pixie-lite-v0"


def build_run_manifest(
    *,
    command: str,
    args: dict[str, JsonValue],
    artifacts: dict[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    return {
        "ruleset": PIXIE_RULESET_ID,
        "command": command,
        "args": dict(args),
        "artifacts": dict(artifacts or {}),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_commit": _git_commit(),
        },
    }


def write_run_manifest(
    path: str | Path,
    *,
    command: str,
    args: dict[str, JsonValue],
    artifacts: dict[str, JsonValue] | None = None,
) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        canonical_json(
            build_run_manifest(command=command, args=args, artifacts=artifacts),
            indent=2,
        ),
        encoding="utf-8",
    )


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
