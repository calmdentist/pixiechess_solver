# Generalized Piece Program Implementation Plan

## 1. Purpose

This document turns the generalized piece-program runtime design into a concrete
implementation plan for this repository.

The goal is not a greenfield rewrite. The goal is to migrate from the current
minimal DSL and move-centric simulator to:

- canonical executable piece programs,
- explicit action intents, effects, events, and traces,
- a stronger repair loop,
- and a model/search stack that can condition on richer semantics.

The plan is staged to preserve current functionality while upgrading the core
contracts.

## 2. Guiding Decisions

### 2.1 No big-bang rewrite

The current engine, search loop, and training loop already work well enough to
serve as a migration harness.

We should:

- introduce new contracts in parallel,
- lower the current DSL into the new runtime,
- prove semantic equivalence on existing pieces,
- and only then cut over the engine.

### 2.2 Keep the simulator authoritative

The model still does not decide:

- legal actions,
- transitions,
- event resolution,
- or terminal conditions.

### 2.3 The current DSL becomes a frontend subset

The existing JSON/YAML DSL should be treated as a legacy frontend that compiles
to the new canonical `ProgramIR`.

This preserves:

- current hand-authored pieces,
- current tests,
- and current curriculum fixtures.

### 2.4 Keep action identity stable

The search/training pipeline depends on stable candidate identities.

So even after the generalized runtime lands, the engine must still expose:

- a stable action id,
- a canonical serialized action,
- and deterministic replay keyed by that action.

### 2.5 Defer authoring-language questions

A Python-like piece-program surface is optional and deferred.

The first implementation target is:

- canonical IR,
- interpreter/executor,
- lowerer from the current DSL,
- and repair against that IR.

## 3. Repository Strategy

The least disruptive path is to add the new runtime alongside current modules.

## 3.1 New or expanded core contracts

Recommended files:

- `src/pixie_solver/core/action.py`
- `src/pixie_solver/core/effect.py`
- `src/pixie_solver/core/trace.py`
- expand `src/pixie_solver/core/event.py`
- expand `src/pixie_solver/core/state.py`
- keep `src/pixie_solver/core/move.py` as a compatibility layer initially

## 3.2 New program package

Recommended new package:

- `src/pixie_solver/program/`

Suggested contents:

- `schema.py`
- `canonicalize.py`
- `validator.py`
- `compiler.py`
- `lower_legacy_dsl.py`
- `stdlib.py`
- `contexts.py`

This package owns the canonical piece-program representation.

## 3.3 Simulator refactor

Keep the `simulator/` package, but split responsibilities more explicitly.

Recommended additions:

- `src/pixie_solver/simulator/actiongen.py`
- `src/pixie_solver/simulator/legality.py`
- `src/pixie_solver/simulator/commit.py`
- `src/pixie_solver/simulator/resolution.py`
- `src/pixie_solver/simulator/trace.py`

The existing `engine.py`, `movegen.py`, `hooks.py`, and `effects.py` can remain
as compatibility and migration surfaces until cutover.

## 3.4 Rules/repair upgrades

The rules package remains the outer-loop home, but it should move from
final-state diffs toward trace-aware mismatch repair.

Key files to extend:

- `src/pixie_solver/rules/mismatch.py`
- `src/pixie_solver/rules/repair.py`
- `src/pixie_solver/curriculum/pipeline.py`
- `src/pixie_solver/rules/registry.py`

## 4. Migration Principle

Every phase should satisfy one rule:

> existing hand-authored pieces and search/training workflows continue to run
> unless the phase explicitly changes the contract.

That means we need explicit compatibility layers rather than an all-at-once API
rename.

## 5. Phase Plan

## Phase 0 — Contract Scaffolding

### Objective

Add the new runtime objects without changing behavior.

### Deliverables

- `ActionIntent`
- `Effect`
- richer `Event`
- `TraceFrame`
- richer `StateDelta`
- stable action id helper

### Proposed implementation

1. Add `core/action.py` with:
   - `ActionIntent`
   - `stable_action_id`
   - canonical `to_dict()` / `from_dict()`
2. Add `core/effect.py` with typed effect objects.
3. Add `core/trace.py` with:
   - `TraceFrame`
   - `TransitionTrace`
4. Expand `core/event.py` so events can carry:
   - source action id
   - source frame id
   - richer payload metadata
5. Expand `StateDelta` to optionally hold:
   - `action`
   - `effects`
   - `trace`
   - created / removed piece ids

### Compatibility stance

- `Move` remains the public search/self-play action type for now.
- Add conversion helpers between `Move` and `ActionIntent`.
- No engine behavior should change in this phase.

### Exit criteria

- serialization/hashing tests pass for all new types
- no current tests regress
- `Move`-based engine still behaves identically

## Phase 1 — Canonical Program IR

### Objective

Introduce `ProgramIR` and make the existing DSL compile into it.

### Deliverables

- program schema
- canonicalization
- validation
- lowering from current DSL
- compile path from DSL -> `ProgramIR`

### Proposed implementation

1. Add `program/schema.py` for:
   - program structure
   - action block structure
   - reaction block structure
   - local variable / expression / query forms
2. Add `program/canonicalize.py`.
3. Add `program/validator.py`.
4. Add `program/lower_legacy_dsl.py`:
   - `inherit_base`
   - `phase_through_allies`
   - range modifiers
   - push-capture
   - current hook/condition/effect subset
5. Update `dsl/compiler.py` so it can emit both:
   - current `PieceClass`
   - canonical `ProgramIR`

### Important decision

Do not remove the current DSL package.
Treat it as a compatibility frontend.

### Tests

- every hand-authored piece lowers successfully
- canonicalization is deterministic
- invalid IR is rejected
- DSL -> IR snapshots are stable

### Exit criteria

- all current pieces compile into valid `ProgramIR`
- IR digests are stable
- no simulator cutover yet

## Phase 2 — Runtime Interpreter And Query Library

### Objective

Build the executor for `ProgramIR` while keeping the old engine alive.

### Deliverables

- read-only action and event contexts
- standard query library
- effect application helpers
- reaction rule execution

### Proposed implementation

1. Add `program/contexts.py` with:
   - `ActionContext`
   - `EventContext`
2. Add `program/stdlib.py` with bounded query operations:
   - geometry transforms
   - occupancy queries
   - ray/path tracing
   - piece filters
   - local/global state reads
3. Add simulator modules:
   - `actiongen.py`
   - `commit.py`
   - `resolution.py`
4. Implement typed effect application against `GameState`.
5. Implement deterministic event ordering and fixed-point resolution.

### Compatibility stance

- the interpreter should initially run only in tests and shadow mode
- the production engine still uses the legacy path

### Tests

- interpreter executes lowered current pieces
- trace output is deterministic
- fixed-point limits fail loudly
- invariants still hold after each transition

### Exit criteria

- interpreter can enumerate actions and resolve reactions for current pieces
- legacy and interpreter transitions can be compared in tests

## Phase 3 — Semantic Equivalence Harness

### Objective

Prove that the new runtime reproduces current piece behavior before cutover.

### Deliverables

- dual-engine comparison harness
- replay equivalence fixtures
- property tests over randomized legal states

### Proposed implementation

1. Add a test helper that runs:
   - legacy action enumeration / apply
   - IR action enumeration / apply
   - state and trace comparison
2. Add equivalence tests for:
   - orthodox chess states
   - current magical pieces
   - hook-heavy edge cases
3. Add randomized regression:
   - generate legal states
   - compare legal action sets by stable id
   - compare successor states for matched actions

### Exit criteria

- current hand-authored pieces are equivalent on curated fixtures
- randomized equivalence holds at acceptable coverage
- mismatch cases are explainable and intentionally resolved

## Phase 4 — Engine Cutover

### Objective

Switch the simulator/search/self-play stack to the new runtime behind the same
high-level API.

### Deliverables

- `legal_moves()` backed by `ActionIntent` generation
- `apply_move()` backed by action commit + fixed-point resolution
- stable traces emitted everywhere

### Proposed implementation

1. Keep public functions:
   - `legal_moves(state)`
   - `apply_move(state, move)`
2. Internally:
   - convert `Move` to `ActionIntent`
   - execute through the new runtime
3. Update `movegen.py` and `transition.py` to become:
   - compatibility wrappers
   - or remove them after confidence is high
4. Stamp every replay and self-play example with:
   - engine version
   - registry digest
   - program digest set

### Compatibility stance

- search and self-play should not need a breaking API change in this phase
- `Move` can remain the external type until the model/search refactor

### Exit criteria

- full unit suite passes on the new engine path
- self-play artifacts remain reproducible
- search-only behavior does not regress on baseline fixtures

## Phase 5 — Repair Loop Upgrade

### Objective

Make the outer loop operate on executable traces and regression cases instead of
thin final-state diffs.

### Deliverables

- trace-aware mismatch objects
- regression-case support as a first-class path
- richer curriculum probes
- registry admission at cycle boundaries

### Proposed implementation

1. Extend `rules/mismatch.py` to compare:
   - predicted action
   - predicted trace frames
   - predicted effects/events
   - predicted next state
2. Extend `rules/repair.py` request payload to include:
   - trace
   - implicated frames
   - prior regression cases
3. Make curriculum use regression cases by default.
4. Upgrade registry records to include:
   - program digest
   - engine version
   - runtime library version
5. Integrate verified program admission into `train-loop` at cycle boundaries.

### Important constraint

Do not hot-swap piece programs mid-game or mid-search.

### Exit criteria

- compile -> mismatch -> repair -> verify -> admit can run end-to-end
- repaired programs survive prior regression fixtures
- `train-loop` can consume newly admitted pieces on later cycles without restart

## Phase 6 — First New Semantic Family

### Objective

Use the new runtime to support a mechanic the current DSL cannot express.

### Recommended target

Reflection / edge-bounce movement.

### Deliverables

- path tracing with reflection
- one new piece family, for example a reflecting bishop
- probes and replay fixtures for the new family
- compile and repair examples using that family

### Why this phase matters

Without this phase, the new runtime is only a refactor.
This phase is where it starts supporting the actual research claim.

### Exit criteria

- the runtime can express and execute a reflecting piece correctly
- repair fixtures exist for at least one ambiguous reflection rule
- tactical tests include the new family

## Phase 7 — Model/Search Interface Upgrade

### Objective

Upgrade the model inputs to consume richer semantics from the new runtime.

### Deliverables

- explicit action-parameter encoding
- structured program encoding over `ProgramIR`
- semantic probe features from the engine
- candidate-to-context interaction, likely cross-attention

### Proposed implementation

1. Replace the lossy DSL-only encoding path with:
   - structured program blocks
   - argument tokens
   - state-schema tokens
2. Replace move-only encoding with action encoding that includes:
   - action kind
   - params
   - target refs
   - engine-derived one-ply trace summaries
3. Add engine-derived semantic features such as:
   - reachable squares
   - query-derived control relations
   - volatility summaries
   - event-trigger summaries
4. Upgrade policy/value architecture to support action-to-board/program
   interaction.

### Important sequencing

Do not start here before engine cutover and repair-loop upgrades are stable.

### Exit criteria

- model forward pass consumes `ProgramIR` and richer action semantics
- guided search beats or outperforms the previous model baseline on held-out
  states or node-efficiency benchmarks

## Phase 8 — Proof Harness

### Objective

Build the experimental harness needed to support the thesis.

### Deliverables

- held-out piece-family splits
- compile-only and repair-after-observation benchmarks
- tactical suites for unseen families
- arena adaptation curves across rule changes

### Required metrics

- compile success rate on held-out pieces
- repair success after `k` mismatches
- regression survival after repair
- arena score versus prior checkpoint under changed rules
- search-only versus guided-search node efficiency

### Exit criteria

- the repo can produce evidence, not just artifacts
- the evaluation distinguishes:
  - known primitives, new compositions
  - and genuinely new primitives

## 6. Suggested PR Sequence

The safest implementation sequence is:

1. PR1: core runtime contracts only
2. PR2: `ProgramIR` schema and lowering from current DSL
3. PR3: interpreter + standard library in shadow mode
4. PR4: equivalence harness
5. PR5: engine cutover behind compatibility wrappers
6. PR6: trace-aware repair loop and registry versioning
7. PR7: reflection mechanic and first genuinely new piece family
8. PR8: model/search interface upgrade
9. PR9: held-out proof harness

Each PR should land with tests and a narrow scope.

## 7. Parallelization Plan

Once Phase 1 contracts are stable, work can split cleanly.

### Track A — Core And Program IR

Owns:

- core runtime contracts
- `program/` package
- legacy DSL lowering

### Track B — Simulator Runtime

Owns:

- interpreter
- commit/resolution
- legality and invariants
- equivalence harness

### Track C — Outer Loop

Owns:

- mismatch/repair payloads
- curriculum probes
- registry/versioning
- train-loop integration

### Track D — Model And Search

Owns:

- action semantics encoding
- richer program encoder
- search integration
- evaluation harness

## 8. Testing Strategy

Each phase needs explicit test coverage.

## 8.1 Contract tests

- serialization round-trips
- stable hashes and action ids
- canonicalization snapshots

## 8.2 Equivalence tests

- legal action set parity
- successor-state parity
- trace parity where semantics match exactly

## 8.3 Stress tests

- fixed-point termination safety
- randomized state transitions
- replay round-trip on large self-play batches

## 8.4 Outer-loop tests

- compile success fixtures
- repair success fixtures
- regression replay survival
- registry admission and reload behavior

## 8.5 Model/search tests

- richer action encoding determinism
- guided search agreement and node-efficiency comparisons
- held-out piece-family evaluation harness

## 9. Major Risks

### 9.1 Engine rewrite risk

The engine is the highest-risk change.
That is why semantic equivalence and shadow execution are required before
cutover.

### 9.2 IR overreach risk

If the first IR is too expressive, canonicalization and repair will become
fragile. Start with bounded constructs and extend only when forced by a real
piece family.

### 9.3 Model-before-runtime risk

If we redesign the model before the runtime semantics are stable, we will waste
time on encoders for a moving target.

### 9.4 Evaluation ambiguity risk

If we do not separate:

- known primitives,
- new compositions,
- and new runtime primitives,

we will make claims the experiments do not support.

## 10. Immediate Next Steps

The best next implementation step is not the model.

It is:

1. add `ActionIntent`, `Effect`, `TraceFrame`, and richer `StateDelta`,
2. add `ProgramIR` plus lowering from the current DSL,
3. build an interpreter in shadow mode,
4. and add semantic equivalence tests against the current engine.

That is the shortest path to a safe migration and the cleanest foundation for
the actual research experiments.
