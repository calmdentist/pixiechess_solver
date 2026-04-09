# implementation_plan.md

# PixieChess Solver — Implementation Plan

## 1. Purpose

This document breaks the PixieChess solver into clear implementation domains, interfaces, and milestones so one or more coding agents can build it in parallel without stepping on each other.

The guiding product decision is:

> keep runtime rules crisp, keep rule repair external, and keep the learned model focused on strategic evaluation under the current executable simulator.

So the system has three primary pillars:

- **Rule system**: English description → DSL → simulator → mismatch-driven patching
- **Search/engine system**: exact transitions, legal moves, self-play, tactical solving
- **Learning system**: rule-conditioned policy/value model that scores legal candidate moves

---

## 2. Non-Negotiable Design Constraints

These constraints should be treated as architectural law unless intentionally revised.

### 2.1 Single current executable DSL
At runtime there is exactly one current executable DSL program per piece.

### 2.2 Simulator is source of truth
The policy/value model never decides legality or next-state transitions.

### 2.3 LLM is external to the hot path
Use the LLM API for:
- initial English → DSL compilation
- mismatch-driven DSL patching
- optional test generation

Do not put the LLM into move evaluation or MCTS expansion in v1.

### 2.4 Policy head scores legal candidates
Do not use a fixed AlphaZero chess action tensor.

### 2.5 Keep v1 simple
Prefer:
- exactness
- reproducibility
- modularity
- good tests
over cleverness.

---

## 3. Top-Level Domain Separation

Split implementation into six domains:

1. **Core types and state**
2. **DSL + rule compiler/repair loop**
3. **Simulator**
4. **Search + self-play**
5. **Model + training**
6. **Evaluation + tooling**

These domains are separable enough that multiple coding agents can work in parallel after interfaces are frozen.

---

## 4. Recommended Repository Layout

```text
pixie-solver/
├── README.md
├── docs/
│   ├── pixiechess_solver_design.md
│   ├── implementation_plan.md
│   ├── dsl_reference.md
│   ├── simulator_invariants.md
│   └── eval_protocol.md
├── configs/
│   ├── engine/
│   ├── parser/
│   ├── model/
│   └── training/
├── data/
│   ├── pieces/
│   │   ├── handauthored/
│   │   ├── generated/
│   │   └── repaired/
│   ├── tactical_positions/
│   ├── selfplay/
│   └── logs/
├── scripts/
│   ├── compile_piece.py
│   ├── repair_piece.py
│   ├── verify_piece.py
│   ├── run_match.py
│   ├── generate_selfplay.py
│   ├── train_model.py
│   └── evaluate.py
├── src/pixie_solver/
│   ├── core/
│   ├── dsl/
│   ├── llm/
│   ├── simulator/
│   ├── search/
│   ├── model/
│   ├── training/
│   ├── eval/
│   ├── cli/
│   └── utils/
└── tests/
```

---

## 5. Domain 1 — Core Types and State

## Goal
Define the canonical runtime objects used everywhere else.

## Deliverables
- `GameState`
- `PieceClass`
- `PieceInstance`
- `Move`
- `Event`
- `StateDelta`
- serialization / hashing utilities

## Required files
- `src/pixie_solver/core/state.py`
- `src/pixie_solver/core/piece.py`
- `src/pixie_solver/core/move.py`
- `src/pixie_solver/core/event.py`
- `src/pixie_solver/core/hash.py`

## Required interfaces

### GameState
Must contain:
- standard board state
- active piece instances
- piece instance state
- side to move
- move/repetition metadata
- pending events

### PieceClass
Must contain:
- class id
- base piece type
- compiled movement/capture modifiers
- compiled hooks
- instance state schema

### PieceInstance
Must contain:
- piece class id
- color
- square
- instance variables

### Event
Must contain:
- event type
- actor/target ids if applicable
- payload
- source cause

## Exit criteria
- core types are importable and stable
- state can be serialized/deserialized
- state hash is deterministic
- no downstream module defines its own incompatible shadow types

---

## 6. Domain 2 — DSL + Rule Compiler / Repair Loop

## Goal
Represent special pieces as executable programs and maintain them using an external LLM patch loop.

## Deliverables
- DSL schema
- parser
- validator
- compiler from YAML/JSON representation to runtime piece class
- LLM compile API wrapper
- mismatch-driven repair API wrapper
- regression test harness for repaired pieces

## Required files
- `src/pixie_solver/dsl/schema.py`
- `src/pixie_solver/dsl/parser.py`
- `src/pixie_solver/dsl/validator.py`
- `src/pixie_solver/dsl/compiler.py`
- `src/pixie_solver/llm/compile_piece.py`
- `src/pixie_solver/llm/repair_piece.py`

## DSL requirements
Each piece must support:
- base piece type
- movement modifiers
- capture modifiers
- hooks
- conditions
- effects
- instance state schema

## LLM compile interface
Input:
- English piece description

Output:
- candidate DSL program
- optional explanation
- optional generated tests

## LLM repair interface
Input:
- English piece description
- current DSL
- current board state
- move
- predicted next state
- actual next state
- structured diff

Output:
- patched DSL
- explanation
- optional tests

## Important constraint
The runtime DSL must always resolve to one current executable piece program.

No probabilistic semantics in the simulator.

## Exit criteria
- 5–10 hand-selected pieces compile into valid DSL
- repair loop can patch at least one known mismatch example
- repaired DSL passes validation and regression replay

---

## 7. Domain 3 — Simulator

## Goal
Execute moves exactly under current DSL piece programs.

## Deliverables
- legal move generation
- move application
- hook/event execution
- fixed-point resolution
- invariant checks
- debug trace mode

## Required files
- `src/pixie_solver/simulator/engine.py`
- `src/pixie_solver/simulator/movegen.py`
- `src/pixie_solver/simulator/hooks.py`
- `src/pixie_solver/simulator/effects.py`
- `src/pixie_solver/simulator/fixed_point.py`
- `src/pixie_solver/simulator/invariants.py`

## Functional requirements

### Legal move generation
Must combine:
- base chess geometry
- movement/capture modifiers
- current piece instance state

### Move application
Must:
- apply the move
- emit base events
- resolve hooks
- update piece instance state
- produce the final next state

### Fixed-point resolution
Must:
- execute hook/event cascades deterministically
- terminate cleanly or fail loudly on non-termination risk

### Invariants
Must guarantee:
- no duplicate occupancy
- piece instance mapping consistent
- kings valid
- no impossible final state
- deterministic replay

## Exit criteria
- standard-chess parity works when no special pieces are active
- War Automaton-style triggered updates work
- at least 20 simulator tests pass
- trace logs are readable enough to diagnose mismatches

---

## 8. Domain 4 — Search + Self-Play

## Goal
Provide a strong exact-planning backbone and generate training data.

## Deliverables
- PUCT/MCTS engine
- optional simple value fallback
- self-play loop
- replay writer
- tactical suite runner

## Required files
- `src/pixie_solver/search/mcts.py`
- `src/pixie_solver/search/puct.py`
- `src/pixie_solver/search/node.py`
- `src/pixie_solver/training/selfplay.py`

## Key design choice
Keep search algorithm simple initially:
- exact simulator
- legal candidate moves
- model priors/value when available

## Self-play requirements
Each example must store:
- board state
- piece programs
- piece instance states
- legal move list
- search visit distribution
- final outcome

## Tactical suite
Create a curated set of special-piece tactical positions.
This is required for honest iteration.

## Exit criteria
- MCTS returns legal moves on all tested states
- self-play generates reproducible games
- tactical suite runner works
- search-only baseline is available before model integration

---

## 9. Domain 5 — Model + Training

## Goal
Train a simple rule-conditioned policy/value model that learns strategic consequences of piece programs.

## Deliverables
- structured normalized DSL encoder
- piece-instance / board token encoder
- candidate move scorer
- value head
- training loop
- checkpointing
- inference wrapper for search

## Required files
- `src/pixie_solver/model/dsl_encoder.py`
- `src/pixie_solver/model/board_encoder.py`
- `src/pixie_solver/model/move_encoder.py`
- `src/pixie_solver/model/policy_value.py`
- `src/pixie_solver/training/dataset.py`
- `src/pixie_solver/training/train.py`

## Key modeling stance

### Do
- encode DSL in a structured normalized form
- keep field boundaries explicit
- embed piece instance state
- score legal candidate moves
- predict position value

### Do not
- try to learn legality
- try to replace the simulator
- use a fixed chess move tensor
- build a graph/AST encoder in v1 unless needed by experiments

## Suggested model decomposition

### DSL encoder
Input fields:
- base piece type
- movement modifiers
- capture modifiers
- hooks
- effect/condition templates
- instance state schema

Output:
- piece class embedding

### Board encoder
Input per occupied square:
- square embedding
- color embedding
- base piece type embedding
- piece class embedding
- instance state embedding

Use:
- transformer over piece tokens + global tokens

### Move scorer
For each legal move token:
- source square
- destination square
- moving piece embedding
- target embedding if present
- whether hooks trigger
- one-step simulator consequence summary

Output:
- score over legal candidates

### Value head
Input:
- pooled board context
- piece-program summary
- optional volatility features

Output:
- scalar value or W/D/L logits

## Losses
- policy cross-entropy to MCTS visit distribution
- value loss to final game outcome
- optional auxiliaries only if necessary later

## Exit criteria
- model forward pass works on real simulator states
- legal-candidate policy head functions
- training loop over replay data runs stably
- model improves search over search-only baseline or at least reduces node count

---

## 10. Domain 6 — Evaluation + Tooling

## Goal
Make the project measurable, debuggable, and runnable by agents without ambiguity.

## Deliverables
- tactical evaluation suite
- replay inspector
- piece compile/repair CLI
- self-play runner
- train/eval CLI
- metrics dashboard/logging hooks

## Required files
- `src/pixie_solver/eval/tactical.py`
- `src/pixie_solver/eval/engine_matches.py`
- `src/pixie_solver/eval/replay_inspector.py`
- `src/pixie_solver/cli/main.py`

## Must-have commands
```bash
pixie compile-piece --text "..."
pixie verify-piece --file ...
pixie repair-piece --piece ... --trace ...
pixie run-match --white ... --black ...
pixie selfplay --config ...
pixie train --config ...
pixie eval-tactics --suite ...
```

## Exit criteria
- all critical workflows are one-command runnable
- failures produce actionable traces
- tactical and training metrics can be compared across runs

---

## 11. Milestone Plan

## M0 — Project skeleton and interfaces
### Goals
- repository bootstrapped
- module layout fixed
- core interfaces documented

### Output
- empty module files
- typed interface stubs
- basic CLI
- docs checked in

### Exit criteria
- all modules import
- no unresolved interface ambiguity remains

---

## M1 — Core types + DSL + hand-authored pieces
### Goals
- implement core runtime types
- implement DSL parser/validator/compiler
- add 5–10 initial hand-authored pieces

### Suggested starter pieces
- War Automaton
- one phasing rook
- one knockback rook
- one timer/counter piece
- one zone-control piece
- one reactive trigger piece

### Exit criteria
- all starter pieces compile successfully
- tests cover invalid and valid DSL cases

---

## M2 — Simulator MVP
### Goals
- move generation
- move application
- hook execution
- invariant checking

### Exit criteria
- standard chess parity without special pieces
- basic trigger pieces execute correctly
- deterministic replay works

---

## M3 — LLM compile + repair loop
### Goals
- English → DSL compile path
- mismatch-driven repair path
- DSL regression tests from repaired examples

### Exit criteria
- at least one known ambiguity case can be repaired successfully
- patched DSL can be reloaded and replayed

---

## M4 — Search-only baseline
### Goals
- PUCT/MCTS with simulator
- tactical suite
- self-play generation without neural priors

### Exit criteria
- search-only engine plays complete games
- tactical suite baseline recorded
- replay data emitted cleanly

---

## M5 — Model MVP
### Goals
- structured normalized DSL encoder
- board/piece-state encoder
- candidate move scorer
- value head

### Exit criteria
- model can train on self-play replay
- inference returns candidate policy + value
- integrated with search wrapper

---

## M6 — Training loop + iterative self-play
### Goals
- policy/value training
- checkpointing
- iterative self-play refresh

### Exit criteria
- model-guided search outperforms or accelerates search-only baseline
- training metrics stable over multiple cycles

---

## M7 — Hardening + evaluation
### Goals
- better tactical coverage
- regression suite for DSL patches
- replay inspector
- CLI polish

### Exit criteria
- coding agents can run compile/repair/selfplay/train/eval end-to-end
- failures are diagnosable with logs and traces

---

## 12. Parallelization Plan for Multiple Coding Agents

## Agent A — Core + DSL
Owns:
- core runtime objects
- DSL schema/parser/compiler/validator

Blocked by:
- none initially

Produces artifacts for:
- simulator
- model
- CLI

---

## Agent B — Simulator
Owns:
- move generation
- hook/event engine
- invariants
- tracing

Blocked by:
- core types
- DSL compiler output contract

Produces artifacts for:
- search
- self-play
- mismatch detector

---

## Agent C — LLM integration
Owns:
- compile API wrapper
- repair API wrapper
- mismatch diff formatting
- piece regression harness

Blocked by:
- DSL schema
- simulator traces/state serializer

Produces artifacts for:
- rule loop
- production updates
- CLI

---

## Agent D — Search + self-play
Owns:
- PUCT/MCTS
- self-play loop
- replay format

Blocked by:
- simulator API
- core move/state contracts

Produces artifacts for:
- training
- tactical eval
- baseline engine

---

## Agent E — Model + training
Owns:
- DSL encoder
- board encoder
- move scorer
- value head
- training loop

Blocked by:
- replay format
- simulator state serialization contract
- search-generated targets

Produces artifacts for:
- model-guided search
- evaluation

---

## Agent F — Eval + tooling
Owns:
- tactical suite
- engine match harness
- replay inspector
- CLI aggregation
- docs enforcement

Blocked by:
- partial outputs from most other domains

Produces artifacts for:
- project usability
- regression control
- performance visibility

---

## 13. Interface Freeze Points

These should be frozen as early as possible to reduce agent churn.

### Freeze 1
Core types:
- `GameState`
- `PieceClass`
- `PieceInstance`
- `Move`
- `Event`

### Freeze 2
Simulator API:
- `legal_moves(state)`
- `apply_move(state, move)`
- `is_terminal(state)`
- `result(state)`

### Freeze 3
Replay format:
- state serialization
- legal move list
- visit distribution
- outcome label

### Freeze 4
Model input contract:
- serialized piece programs
- piece instance state format
- candidate move feature format

---

## 14. Suggested Build Order

If one agent is doing almost everything:

1. M0 project skeleton
2. M1 core + DSL
3. M2 simulator
4. M4 search-only baseline
5. M3 LLM compile/repair loop
6. M5 model MVP
7. M6 training loop
8. M7 hardening

If multiple agents are available:

1. Freeze core types
2. Parallelize:
   - DSL
   - simulator
   - LLM wrappers
3. After simulator stabilizes:
   - search
   - tactical suite
4. After replay format stabilizes:
   - model/training
5. Finish with:
   - tooling
   - regression
   - hardening

---

## 15. Definition of Success

A successful v1 should be able to:

- compile a new piece description into DSL
- execute that piece correctly in the simulator
- patch the piece when observed transitions disagree
- generate self-play under the current simulator
- train a rule-conditioned policy/value model
- use that model to guide MCTS
- solve curated tactical special-piece positions better over time

That is enough to validate the architecture.

---

## 16. Final Note to Coding Agents

Do not overcomplicate v1.

Specifically:
- do not add uncertainty into the runtime DSL
- do not add AST/graph encoders unless experiments clearly justify them
- do not put the LLM into the hot path of move selection
- do not try to replace exact search with neural rollout magic

The project wins if it is:
- exact,
- modular,
- testable,
- and strategically strong under changing rules.

That is the bar.
