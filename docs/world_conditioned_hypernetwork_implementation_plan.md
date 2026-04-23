# World-Conditioned Hypernetwork Implementation Plan

This document turns
[docs/world_conditioned_hypernetwork_design.md](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/docs/world_conditioned_hypernetwork_design.md)
into a concrete implementation plan against the current codebase.

The goal is not "replace the model with generated weights."
The goal is:

1. keep a shared policy/value backbone,
2. compile the current `ProgramIR + StrategyIR` into a small
   `AdapterBundle`,
3. specialize the executor for the current world without retraining from
   scratch,
4. keep exact search authoritative,
5. and create a clean A/B path against the current
   `world_conditioned_v2` baseline and the planned strategy-conditioned path.

This plan assumes the architectural framing from:

- [docs/executable_world_model_design.md](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/docs/executable_world_model_design.md)
- [docs/strategy_conditioned_search_design.md](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/docs/strategy_conditioned_search_design.md)
- [docs/world_conditioned_hypernetwork_design.md](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/docs/world_conditioned_hypernetwork_design.md)

## 1. Current Baseline

The current relevant stack is:

- [src/pixie_solver/model/policy_value_v2.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/model/policy_value_v2.py)
- [src/pixie_solver/model/policy_value.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/model/policy_value.py)
- [src/pixie_solver/model/board_encoder.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/model/board_encoder.py)
- [src/pixie_solver/model/program_encoder.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/model/program_encoder.py)
- [src/pixie_solver/model/action_encoder_v2.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/model/action_encoder_v2.py)
- [src/pixie_solver/search/mcts.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/search/mcts.py)
- [src/pixie_solver/training/selfplay.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/training/selfplay.py)
- [src/pixie_solver/training/dataset.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/training/dataset.py)
- [src/pixie_solver/training/train.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/training/train.py)
- [src/pixie_solver/training/checkpoint.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/training/checkpoint.py)

Today:

- the executable world model is already encoded into the model context,
- the action encoder is already structured,
- the model already supports context-to-action cross-attention,
- exact search is already the authoritative verifier,
- and the missing piece is not basic world conditioning.

The missing piece is compiled specialization.

That means the right next move is not a ground-up rewrite.
It is a targeted extension of the current world-conditioned path.

## 2. Locked Decisions

### 2.1 Compile from the absolute current world, not explicit deltas

The hypernetwork should initially consume:

- canonical `ProgramIR`
- canonical `StrategyIR`

Do not make "world delta" the core input in v1.

World deltas may become an optimization later, but the first implementation
should compile from the current authoritative world directly.

### 2.2 Keep token conditioning

Do not replace program and strategy tokens with only hypernetwork outputs.

The recommended v1 is hybrid:

- explicit world and strategy tokens remain in context,
- and the hypernetwork adds small world-specific adapters.

### 2.3 Generate only small adapters

Do not generate full model weights.

Allowed v1 outputs:

- FiLM-style scale and shift,
- low-rank residual adapters,
- attention bias terms,
- small gating vectors.

### 2.4 Freeze adapters during search

Once a search tree begins, the current `AdapterBundle` is fixed.

The bundle may change only when:

- the authoritative world model changes,
- the strategy changes,
- or the current root state is re-entered under a different compile context.

### 2.5 Search-derived targets remain the truth

The policy/value net and hypernetwork should still train on:

- MCTS visit distributions,
- search-backed values,
- final outcomes,
- optional uncertainty calibration targets.

They should not train to imitate the hypernetwork output directly.

### 2.6 The LLM stays event-driven

The LLM should not be pulled into the inner loop at every move or every node.

The LLM remains responsible for:

- world-model repair,
- strategy revision,
- and high-level failure analysis.

The hypernetwork is the fast specialization path between LLM calls.

### 2.7 Ship via compatibility

Keep:

- `baseline_v1`
- `world_conditioned_v2`

Potentially keep or add:

- `strategy_conditioned_v3`

Add:

- `hypernetwork_conditioned_v4`

The new path must remain A/B testable without disturbing current checkpoints.

## 3. Proposed Module Boundaries

Recommended new files:

- `src/pixie_solver/hypernet/schema.py`
- `src/pixie_solver/hypernet/compiler.py`
- `src/pixie_solver/hypernet/layers.py`
- `src/pixie_solver/hypernet/cache.py`
- `src/pixie_solver/model/policy_value_v4.py`

Recommended touched files:

- `src/pixie_solver/model/board_encoder.py`
- `src/pixie_solver/model/program_encoder.py`
- `src/pixie_solver/model/__init__.py`
- `src/pixie_solver/model/policy_value.py`
- `src/pixie_solver/search/mcts.py`
- `src/pixie_solver/search/__init__.py`
- `src/pixie_solver/training/dataset.py`
- `src/pixie_solver/training/selfplay.py`
- `src/pixie_solver/training/train.py`
- `src/pixie_solver/training/checkpoint.py`
- `src/pixie_solver/training/inference_service.py`
- `src/pixie_solver/cli/main.py`

Optional later files:

- `src/pixie_solver/hypernet/memory.py`
- `src/pixie_solver/hypernet/delta_compiler.py`

These should wait until the basic adapter path proves useful.

## 4. Phase 0: Contracts And Compatibility

This phase adds no new model behavior.

### 4.1 Add architecture name

Extend `PolicyValueConfig` with:

- `architecture = "hypernetwork_conditioned_v4"`

Old checkpoint loading must stay unchanged.

### 4.2 Add adapter metadata to replay contracts

Extend `SelfPlayExample.metadata` usage with optional fields:

- `world_digest`
- `strategy_digest`
- `adapter_bundle_id`
- `adapter_bundle_digest`
- `adapter_compile_source`
- `adapter_provider`

This keeps replay artifacts analyzable without forcing loader rewrites.

### 4.3 Add compile-context plumbing

Introduce a simple runtime object that can carry:

- `ProgramIR`
- `StrategyIR`
- current `AdapterBundle`

The first version may simply be:

```python
CompiledWorldContext = {
    world_digest,
    strategy_digest,
    adapter_bundle,
    metadata,
}
```

This gives self-play, inference, and training a stable handoff object.

## 5. Phase 1: AdapterBundle Schema

Build a canonical adapter object and deterministic serialization path.

### 5.1 Schema

Implement `AdapterBundle` with at least:

- `bundle_id`
- `world_digest`
- `strategy_digest`
- `layer_modulations`
- `attention_biases`
- `gating_values`
- `metadata`

Low-rank adapters may be added immediately or in Phase 3 depending on code
complexity.

### 5.2 Canonicalization

Add:

- stable field ordering,
- canonical numeric formatting,
- canonical JSON serialization,
- stable digest helper.

This should mirror the current `ProgramIR` discipline.

### 5.3 Validation

Validator rules should include:

- non-empty `bundle_id`,
- matching or nullable `strategy_digest`,
- bounded tensor/list sizes in v1,
- shape consistency for each layer target.

The point is to keep the specialization small and safe.

## 6. Phase 2: Hypernetwork Compiler V1

This is the first real model-facing phase.

### 6.1 Add compiler

Create `src/pixie_solver/hypernet/compiler.py` with:

- `WorldCompilerHypernetwork`
- `compile_adapter_bundle(...)`
- `CompiledAdapterMetrics`

### 6.2 Inputs

The compiler should consume:

- encoded `ProgramIR`
- encoded `StrategyIR`
- optional small compile metadata features

It should not consume:

- per-node board state,
- legal moves,
- or raw text in v1.

### 6.3 Outputs

Start with the smallest useful outputs:

- per-layer FiLM scale and shift
- small residual gating vectors

Low-rank adapters should be the next addition, not necessarily the first one.

### 6.4 Determinism

Given the same canonical world and strategy inputs, compilation should be
deterministic at inference time.

## 7. Phase 3: Adapter-Capable Backbone Blocks

This phase modifies the executor internals so adapter bundles can actually be
applied.

### 7.1 Add adapter-aware layers

Create `src/pixie_solver/hypernet/layers.py` with wrappers for:

- adapter-aware MLP blocks
- adapter-aware attention blocks
- feature gating utilities
- FiLM application helpers

### 7.2 Keep the old path intact

Every adapter-aware block should have a zero-effect path when no bundle is
provided.

That makes ablations and checkpoint fallback straightforward.

### 7.3 Minimize surface area

Do not rewrite the whole model stack in v1.

Only the high-value blocks in the current backbone need adapter hooks.

## 8. Phase 4: `PolicyValueModelV4`

This is the first full model architecture.

### 8.1 Add model

Create `src/pixie_solver/model/policy_value_v4.py` with:

- explicit world/program/state/action context path
- optional strategy token path
- adapter injection points
- policy head
- value head
- uncertainty head

### 8.2 Preserve the current context path

Reuse as much of the current `world_conditioned_v2` input path as possible:

- board tokens
- program tokens
- action tokens
- probe features

The difference is that those computations now run through adapter-aware blocks.

### 8.3 Adapter application

The forward path should accept:

- `EncodedBoard`
- `EncodedActionsV2`
- optional encoded strategy inputs
- optional `AdapterBundle`

If no bundle is provided, the model should still behave sanely and remain
testable.

## 9. Phase 5: Compile Lifecycle And Cache

This phase makes the architecture operational.

### 9.1 Compile once per world/strategy pair

Add a small runtime cache keyed by:

- `world_digest`
- `strategy_digest`

The goal is:

- compile adapters once,
- reuse them for many node evaluations,
- recompile only when the world or strategy changes.

### 9.2 Search-tree invariants

Once a root search begins:

- the `AdapterBundle` is frozen
- and all node evaluations under that root share it

Do not let the search tree mutate its own specialization.

### 9.3 Inference-service compatibility

If batched inference is enabled, the compile cache must be visible to the
inference worker so it does not regenerate bundles for every request.

## 10. Phase 6: Training Integration

This phase teaches the training stack about compiled specialization.

### 10.1 Extend training config

Add config fields for:

- adapter rank or size limits
- whether strategy-conditioned compilation is enabled
- whether uncertainty head is enabled
- compile-cache behavior

### 10.2 Replay loading

Training examples should carry enough metadata to reconstruct or look up the
correct compile context.

### 10.3 Joint training

Train:

- the shared backbone
- the hypernetwork compiler
- and the uncertainty head

jointly on search-derived targets.

### 10.4 Keep mixed-world batches

Do not restrict training batches to one world at a time.

Mixed-world batches are part of how the backbone learns transferable structure
and the hypernetwork learns specialization.

## 11. Phase 7: Adaptive Search Integration

This phase uses the new uncertainty output to spend search budget better.

### 11.1 Add uncertainty-aware scheduling

Extend search config with optional policies such as:

- expand more when uncertainty is high
- expand more when top move margin is small
- expand less when policy is decisive and uncertainty is low

### 11.2 Keep exact semantics

Adaptive search should change only:

- budget,
- node-expansion schedule,
- or root-depth allocation.

It should not change legality or simulator authority.

### 11.3 Start simple

The first implementation should probably vary only root simulations or root
expansion count.

Do not redesign the full tree policy in the first pass.

## 12. Phase 8: Repair And Recompile Hooks

This phase closes the loop with the LLM-maintained world model.

### 12.1 Recompile on world repair

When the world model changes due to an observed contradiction:

- invalidate the current adapter bundle,
- compile a new one from the repaired world and strategy,
- and continue from the new authoritative world.

### 12.2 Recompile on strategy refresh

If the strategy layer is active and the strategy changes:

- generate a new bundle
- but do not mutate any already-running search tree

The new bundle applies only to subsequent planning.

### 12.3 Keep attribution clean

Do not blur:

- world-model repair,
- strategy revision,
- and hypernetwork recompilation.

Each should remain separately observable in logs and metrics.

## 13. Phase 9: Evaluation Harness

This phase decides whether the architecture is actually useful.

### 13.1 Required baselines

Compare:

1. `world_conditioned_v2`
2. `strategy_conditioned_v3` if available
3. `hypernetwork_conditioned_v4` without strategy
4. `hypernetwork_conditioned_v4` with strategy

### 13.2 Required metrics

Track:

- adaptation speed after world changes
- value calibration on changed worlds
- policy quality under held-out mechanic families
- search efficiency at equal strength
- compile-cache hit rate
- repair-to-recompile latency
- uncertainty-vs-error calibration

### 13.3 Required ablations

At minimum:

- token conditioning only
- adapters only
- token conditioning + adapters
- fixed search budget
- adaptive search budget

Without these, it will be impossible to tell whether the hypernetwork is doing
real work.

## 14. Recommended PR Sequence

The clean landing order is:

1. config/checkpoint compatibility scaffolding
2. `AdapterBundle` schema and validation
3. hypernetwork compiler skeleton
4. adapter-aware layers with zero-effect fallback
5. `PolicyValueModelV4`
6. compile-cache runtime integration
7. training and replay metadata integration
8. adaptive-search hooks
9. repair/recompile hooks
10. ablation and experiment harness

This order keeps the risk localized and the checkpoints debuggable.

## 15. Recommended V1 Scope

The narrow but serious first version should be:

- one compiled bundle per world
- optional null or static strategy input
- FiLM and small gating only
- no full low-rank stack yet unless it proves cheap to land
- frozen bundle during search
- one uncertainty head
- root-level adaptive budget only
- no explicit world-delta compiler
- no live LLM strategy generation yet

The point is to prove that compiled specialization helps at all.

## 16. Open Questions

Questions to defer until after v1:

- should strategy generate separate adapters from the world model?
- should there be separate bundles for opening and tactical phases?
- should the hypernetwork consume repair traces directly?
- when is world-delta compilation better than absolute-world compilation?
- does adapter memory across related worlds help or overfit?

These are good research questions, but they should not block the first build.
