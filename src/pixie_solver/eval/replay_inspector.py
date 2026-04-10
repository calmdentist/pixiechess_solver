from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ReplaySummary:
    positions: int = 0
    outcomes: dict[str, int] | None = None


def summarize_replay(path: str) -> ReplaySummary:
    raise NotImplementedError(
        f"Replay inspection lands in milestone M7. Requested path={path!r}."
    )
