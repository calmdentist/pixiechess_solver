# Strategy-Conditioned Search Implementation Plan

This document turns
[docs/strategy_conditioned_search_design.md](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/docs/strategy_conditioned_search_design.md)
into a concrete implementation plan against the current codebase.

The goal is not "add LLM prose to the model."
The goal is:

1. let the LLM propose sparse high-level strategy hypotheses,
2. make the policy/value model condition on those hypotheses,
3. keep the simulator and MCTS authoritative,
4. and create a clean A/B path against the current `world_conditioned_v2` model.

## 1. Current Baseline

The current relevant stack is:

- [src/pixie_solver/model/policy_value_v2.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/model/policy_value_v2.py)
- [src/pixie_solver/model/board_encoder.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/model/board_encoder.py)
- [src/pixie_solver/model/program_encoder.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/model/program_encoder.py)
- [src/pixie_solver/model/action_encoder_v2.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/model/action_encoder_v2.py)
- [src/pixie_solver/search/mcts.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/search/mcts.py)
- [src/pixie_solver/training/selfplay.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/training/selfplay.py)
- [src/pixie_solver/training/dataset.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/training/dataset.py)
- [src/pixie_solver/training/train.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/training/train.py)

Today:

- the executable world model is already part of the model context
- the action encoder is already structured
- the policy/value model already supports context-to-action cross-attention
- MCTS still treats the learned model as the only learned source of priors/values
- self-play examples do not yet carry strategy context

That means the right next move is not a rewrite.
It is a targeted extension of the current `world_conditioned_v2` path.

## 2. Locked Decisions

### 2.1 Keep search exact

Do not let the LLM override legality, transitions, or terminal logic.

The simulator remains authoritative for:

- legal moves
- state transitions
- query evaluation
- event resolution
- terminal checks

The strategist is advisory only.

### 2.2 Strategy is structured, not raw text

The LLM should emit a canonical `StrategyHypothesis` object that compiles to
deterministic strategy tokens.

Do not feed raw paragraphs into the model path in v1.

### 2.3 Strategy should be sparse

Do not call the LLM every move.

Recommended v1 cadence:

- once per game at the start
- optional refresh on explicit failure triggers later

### 2.4 Search-derived targets remain the truth

The policy/value net should train on:

- MCTS visit distributions
- root/final values

It should not train to imitate LLM suggestions directly.

### 2.5 Ship via compatibility

Keep:

- `baseline_v1`
- `world_conditioned_v2`

Add:

- `strategy_conditioned_v3`

The new path should be A/B testable without disturbing the current working
world-model stack.

## 3. Proposed Module Boundaries

Recommended new files:

- `src/pixie_solver/strategy/schema.py`
- `src/pixie_solver/strategy/canonicalize.py`
- `src/pixie_solver/strategy/compiler.py`
- `src/pixie_solver/strategy/encoder.py`
- `src/pixie_solver/strategy/provider.py`
- `src/pixie_solver/model/policy_value_v3.py`

Recommended touched files:

- `src/pixie_solver/model/board_encoder.py`
- `src/pixie_solver/model/__init__.py`
- `src/pixie_solver/model/policy_value.py`
- `src/pixie_solver/search/mcts.py`
- `src/pixie_solver/search/__init__.py`
- `src/pixie_solver/training/dataset.py`
- `src/pixie_solver/training/selfplay.py`
- `src/pixie_solver/training/train.py`
- `src/pixie_solver/training/checkpoint.py`
- `src/pixie_solver/cli/main.py`

Optional later file:

- `src/pixie_solver/strategy/memory.py`

That should wait until the basic strategy-conditioned path proves useful.

## 4. Phase 0: Contracts And Compatibility

This phase adds no new model behavior.

### 4.1 Add strategy architecture name

Extend `PolicyValueConfig` with:

- `architecture = "strategy_conditioned_v3"`

Keep old checkpoint loading behavior unchanged.

### 4.2 Add strategy metadata fields to replay contracts

Extend `SelfPlayExample.metadata` usage in
[src/pixie_solver/training/dataset.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/training/dataset.py)
with optional fields:

- `strategy_id`
- `strategy_digest`
- `strategy_scope`
- `strategy_confidence`
- `strategy_provider`

No loader rewrite is required because metadata is already open-ended.

### 4.3 Add no-op provider wiring

Introduce a `StrategyProvider` interface that can return:

- zero strategies
- one strategy
- or multiple strategies

The initial implementation should include:

- `NullStrategyProvider`
- `StaticStrategyProvider`

This lets the training/self-play loop accept strategy plumbing before any LLM
integration lands.

## 5. Phase 1: StrategyHypothesis Schema

Build a canonical strategy object and deterministic serialization path.

### 5.1 Schema

Implement `StrategyHypothesis` from the design doc with v1-required fields:

- `strategy_id`
- `summary`
- `confidence`
- `subgoals`
- `action_biases`
- `avoid_biases`
- `success_predicates`
- `failure_triggers`
- `metadata`

### 5.2 Canonicalization

Add:

- stable sorting
- normalized numeric fields
- canonical JSON serialization
- stable digest helper

This should mirror the `ProgramIR` discipline.

### 5.3 Validation

Validator rules should include:

- non-empty `strategy_id`
- confidence in `[0, 1]`
- no duplicate subgoal/action-bias ids if ids are present
- bounded list lengths in v1

The point is to keep strategy context small and learnable.

## 6. Phase 2: Strategy Token Encoder

This is the first real model-facing phase.

### 6.1 Add encoder

Create `src/pixie_solver/strategy/encoder.py` with:

- `StrategyTokenSpec`
- `EncodedStrategyBatch`
- `StrategyTokenEncoder`

The encoder should mirror `ProgramIRTokenEncoder`, but for strategy records.

### 6.2 Token groups

Tokenize at least:

- strategy root
- summary
- confidence bucket
- one token per subgoal
- one token per action bias
- one token per avoid bias
- one token per success predicate
- one token per failure trigger

### 6.3 Keep it deterministic

Preserve canonical field order.
Sort optional maps.
Do not depend on raw text tokenization in v1.

## 7. Phase 3: Board/Context Integration

This phase extends the current context stream.

### 7.1 Extend `EncodedBoard`

Update [src/pixie_solver/model/board_encoder.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/model/board_encoder.py)
so `EncodedBoard` can optionally carry:

- `strategy_tokens`
- `strategy_token_specs`
- `context_strategy_token_span`

### 7.2 Keep world/program context intact

Do not replace any current board/program/probe context.
Strategy tokens should be additive.

### 7.3 Strategy-aware state encoding entrypoint

Add a new encoding API:

```python
encode_state_with_strategy(
    state: GameState,
    *,
    strategy_hypotheses: Sequence[Mapping[str, Any]],
) -> EncodedBoard
```

The existing strategy-free encoding API should remain available and route to
empty strategy context.

## 8. Phase 4: Strategy-Conditioned Model

This is the main architecture phase.

### 8.1 Add `PolicyValueModelV3`

Build on [src/pixie_solver/model/policy_value_v2.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/model/policy_value_v2.py),
not from scratch.

Recommended changes:

- reuse current context transformer shape
- include strategy tokens in the context stream
- keep action cross-attention exactly as the core scoring path

### 8.2 Auxiliary heads

Add optional heads guarded by config:

- `strategy_alignment_head`
- `strategy_progress_head`

These should be optional in v1 and can default off if needed.

### 8.3 Architecture config

Add config knobs for:

- `max_strategy_tokens`
- `max_strategies_per_state`
- optional strategy loss weights

Do not over-expose dozens of new knobs in the first pass.

## 9. Phase 5: Self-Play And Search Integration

This phase wires strategy into actual gameplay.

### 9.1 Add provider hook to self-play

Update [src/pixie_solver/training/selfplay.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/training/selfplay.py)
so each game can receive a `StrategyProvider`.

Recommended v1 behavior:

- call provider once at game start
- attach returned strategy metadata to all examples from that game

### 9.2 Pass strategy into search/model calls

Extend `run_mcts(...)` in
[src/pixie_solver/search/mcts.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/search/mcts.py)
to accept:

- `strategy_hypotheses: Sequence[Mapping[str, Any]] | None = None`

This should only affect the learned-model path.
Search behavior without a model should remain unchanged.

### 9.3 Root-only optional bias

If a second search-side hook is needed, add:

- a small root-only strategy bonus or prior reweighting

This should be:

- off by default at first
- bounded
- never the only driver of move selection

The first useful version may skip explicit root bonus entirely and rely on
strategy-conditioned policy priors only.

## 10. Phase 6: Training Integration

### 10.1 Carry strategy metadata through training

`train_from_replays(...)` does not need a new dataset format, but it does need
the model-construction and forward path to read strategy metadata from each
example.

Recommended collation behavior:

- batch remains a list of `SelfPlayExample`
- each example may provide zero or more strategy hypotheses via metadata

### 10.2 Do not require strategy at training time

The model should support:

- strategy-free examples
- strategy-conditioned examples

That allows:

- partial rollouts
- ablations
- curriculum mixing

### 10.3 Auxiliary losses

If auxiliary strategy heads are enabled, compute losses from:

- strategy-alignment labels
- strategy-progress labels

These labels can initially be heuristic or omitted entirely.
They should not block the first strategy-conditioned training pass.

## 11. Phase 7: First Strategist Providers

### 11.1 Static provider

Create a small fixture-backed provider for deterministic testing.

This is required before any LLM integration.

### 11.2 Oracle provider

For the first real experiment, use a hand-authored or oracle strategy provider
that emits strategy hypotheses for known mechanic families.

This is the cheapest way to test whether strategy conditioning helps before
putting a live LLM in the loop.

### 11.3 LLM provider

Only after the static/oracle path works should we add:

- prompt builder
- structured output parsing
- retry/validation logic
- caching by `(world_digest, opening_digest, objective_digest)`

This provider should be sparse and cache-heavy.

## 12. Phase 8: Evaluation Harness

The first valid experiment is not "turn it on and hope."

### 12.1 A/B comparison

Compare:

- `baseline_v1`
- `world_conditioned_v2`
- `strategy_conditioned_v3`

### 12.2 Fixed world-change setting

Run on:

- one curriculum family already seen
- one held-out family if possible

### 12.3 Adaptation-speed metric

Measure:

- value/policy quality after rule introduction
- self-play strength after rule introduction
- games or cycles needed to recover performance

### 12.4 Bad-strategy robustness

Inject low-quality strategies and confirm:

- search can reject them
- the model does not collapse

This is a mandatory control.

## 13. Recommended PR Sequence

Recommended landing order:

1. strategy schema, canonicalization, and encoder
2. replay/checkpoint/config compatibility
3. board/context integration
4. `PolicyValueModelV3`
5. self-play/search strategy plumbing
6. static/oracle provider and tests
7. first A/B experiment harness
8. live LLM provider

This keeps risk low and preserves a working baseline after every step.

## 14. Recommended V1 Scope

The smallest serious version is:

- one strategy per game
- strategy chosen at game start only
- no mid-game refresh
- no multi-strategy arbitration
- no explicit search bonus beyond learned priors
- no live LLM yet; use static/oracle strategies first

That is enough to answer the first real question:

> does strategy conditioning improve adaptation under changing executable rules?

## 15. Bottom Line

The right implementation path is:

- treat strategy as a new canonical context object
- add it to the current world-conditioned model
- keep search exact
- learn from search-filtered outcomes
- prove value with static/oracle strategies before adding a live LLM

If that works, the LLM moves from "world-model maintainer" to
"world-model maintainer plus strategist," which is a much stronger claim.
