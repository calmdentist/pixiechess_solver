# PixieChess Solver

PixieChess is a chess variant with frequently changing magical pieces. This repository builds a solver around one core bet:

1. special-piece rules should compile into one crisp executable DSL,
2. the simulator should stay exact and deterministic,
3. the learned model should focus on strategy under the current rules rather than legality.

The current foundation covers:

- a Python package layout aligned to the implementation plan,
- frozen core runtime types for pieces, moves, events, and game state,
- a minimal DSL parser/validator/compiler path,
- starter hand-authored piece programs,
- a deterministic simulator with orthodox movement, legal-move filtering, hooks, and replay traces,
- a CLI boundary for compile and verification flows,
- importable interfaces for simulator, search, model, training, and evaluation work.

## Layout

- `implementation_plan.md`: execution plan provided for this workspace
- `pixiechess_solver_design.md`: architecture and modeling rationale
- `docs/`: working reference docs for the initial implementation
- `data/pieces/handauthored/`: hand-authored starter piece programs
- `src/pixie_solver/`: package source
- `tests/`: foundation tests

## Quick Start

```bash
python3 -m unittest discover -s tests
PYTHONPATH=src python3 -m pixie_solver compile-piece --file data/pieces/handauthored/phasing_rook.json --pretty
PYTHONPATH=src python3 -m pixie_solver verify-piece --file data/pieces/handauthored/war_automaton.json
```

## Current Milestone

This repo is at the M2 boundary:

- repository skeleton and core interfaces are in place,
- DSL compilation works for hand-authored starter pieces,
- the simulator covers orthodox movement plus the minimal v1 magical modifiers/effects,
- deterministic replay is available for move/event verification,
- search, model, training, and evaluation are still scaffolded for later milestones.
