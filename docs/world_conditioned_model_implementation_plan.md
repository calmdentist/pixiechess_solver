# World-Conditioned Model Implementation Plan

This document turns the model-upgrade direction into a concrete implementation
plan against the current codebase.

The goal is not merely "use a transformer."
The goal is:

1. make the policy/value model consume executable semantics in a much richer way,
2. keep the simulator authoritative,
3. preserve the current training loop while the new model is introduced,
4. and create a clean A/B path against the existing baseline.

## 1. Current Baseline

The current model stack is:

- [src/pixie_solver/model/board_encoder.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/model/board_encoder.py:34)
- [src/pixie_solver/model/dsl_encoder.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/model/dsl_encoder.py:1)
- [src/pixie_solver/model/move_encoder.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/model/move_encoder.py:1)
- [src/pixie_solver/model/policy_value.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/model/policy_value.py:71)

The critical limitations are:

- `DSLFeatureEncoder` compresses piece semantics into pooled hashed summaries.
- the transformer only contextualizes global and piece tokens
- legal actions are scored afterward by an MLP, not through action-to-semantics attention
- move features rely heavily on one-ply handcrafted consequence features
- the model does not consume `ProgramIR` structure directly

This is sufficient for a rule-aware baseline.
It is not sufficient for the thesis-level claim that the learner understands and
uses executable semantics.

## 2. Locked Decisions

### 2.1 Keep the simulator exact

Do not move rule execution into the model.

The simulator remains the source of truth for:

- legal action enumeration
- state transitions
- event resolution
- query evaluation
- terminal and legality checks

The model only supplies:

- priors over legal actions
- a value estimate

### 2.2 Use structured `ProgramIR`, not raw source

The model should consume canonical `ProgramIR` plus derived query/runtime
features, not raw JSON text or raw Python source.

That keeps:

- semantics canonical
- hashing/replay stable
- training inputs deterministic
- and architecture aligned with the execution kernel

### 2.3 Make actions first-class tokens

Do not keep the current "encode board, then score moves independently" shape.

The new model should build one token per legal action and let those action tokens
cross-attend into contextualized world/program tokens.

### 2.4 Ship via compatibility, not replacement

Do not delete the current model path immediately.

The implementation should support:

- current model as baseline
- new world-conditioned model as candidate
- shared training/self-play/checkpoint plumbing where possible

## 3. Target Architecture

Recommended high-level structure:

```python
WorldConditionedPolicyValueModel = {
    context_encoder,
    action_encoder,
    action_cross_attention,
    policy_head,
    value_head,
}
```

### 3.1 Context tokens

The context stream should contain:

- one global world token
- one entity token per active piece/entity instance
- one or more program tokens per referenced piece class / entity program
- optional query-derived semantic probe tokens

The context encoder should be a transformer encoder.

### 3.2 Program tokens

Program tokens should come from canonical `ProgramIR`, not `PieceClass`.

For the first implementation pass, tokenize:

- `base_archetype`
- `action_blocks`
- `query_blocks`
- `reaction_blocks`
- state schema fields
- block kinds
- operator names
- structured args as typed key/value tokens

This does not need full AST sophistication in v1.
It does need to preserve block boundaries and argument structure.

### 3.3 Action tokens

Every legal action should become one token containing:

- actor entity id / class id
- from / to refs
- action kind
- target refs
- normalized params
- tags

The action encoder should be cheap and deterministic.
It should not depend on applying the move to the simulator for every action by
default.

### 3.4 Cross-attention

After context encoding:

- action tokens cross-attend into contextualized world/program tokens
- optionally add one shallow self-attention block over action tokens afterward

This is the key architectural change.
It lets action scores depend on executable semantics instead of only a pooled
board embedding.

### 3.5 Value head

The value head should read from either:

- a dedicated global token
- or an explicit pooled context token set

Do not value from action tokens.

## 4. Proposed Module Boundaries

Recommended new files:

- `src/pixie_solver/model/program_encoder.py`
- `src/pixie_solver/model/action_encoder_v2.py`
- `src/pixie_solver/model/semantic_features.py`
- `src/pixie_solver/model/policy_value_v2.py`

Recommended touched files:

- `src/pixie_solver/model/__init__.py`
- `src/pixie_solver/training/train.py`
- `src/pixie_solver/training/checkpoint.py`
- `src/pixie_solver/cli/main.py`

Optional shared helper extraction:

- common embedding utilities from `_features.py`
- a reusable token-padding helper

## 5. Data Contracts

The model upgrade should not require a replay format rewrite.

Current `SelfPlayExample` in
[src/pixie_solver/training/dataset.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/training/dataset.py:12)
already includes:

- `state`
- `legal_moves`
- `legal_move_ids`
- visit targets
- root value
- outcome

That is enough for v2.

What should be added to metadata for future experiments:

- world-model digest
- active verified piece digests
- optional program digests per class
- model architecture name

These are useful for later split generation and auditability, but they are not a
hard blocker for the first model upgrade.

## 6. Checkpoint Strategy

The checkpoint layer currently assumes one `PolicyValueModel` class in
[src/pixie_solver/training/checkpoint.py](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/src/pixie_solver/training/checkpoint.py:20).

This needs a small upgrade before model v2 lands.

Recommended change:

- add `architecture: str = "baseline_v1"` to `PolicyValueConfig`
- branch on `architecture` in checkpoint loading
- keep baseline loading behavior unchanged for old checkpoints

Recommended architecture names:

- `baseline_v1`
- `world_conditioned_v2`

Do not create a second completely separate checkpoint format if it can be
avoided.

## 7. Phase Plan

## Phase 0 — Compatibility Scaffold

### Objective

Prepare the training/checkpoint/CLI stack to support more than one model
architecture.

### Deliverables

- `PolicyValueConfig.architecture`
- model factory helper
- checkpoint load/save support for architecture selection
- CLI flag for model architecture in train/train-loop

### Files

- `src/pixie_solver/model/policy_value.py`
- `src/pixie_solver/training/checkpoint.py`
- `src/pixie_solver/training/train.py`
- `src/pixie_solver/cli/main.py`

### Exit criteria

- old checkpoints still load
- new checkpoints record architecture
- CLI can train baseline explicitly with `baseline_v1`

## Phase 1 — `ProgramIR` Token Encoder

### Objective

Replace pooled DSL summaries with structured program tokens.

### Deliverables

- `ProgramIRTokenEncoder`
- tokenization for block kinds, operators, schema fields, args, and constants
- deterministic ordering and padding contract

### Important design choice

Do not require raw source files at inference time.

The encoder should work from:

- `PieceClass` plus lowered `ProgramIR`, or
- directly from canonical `ProgramIR`

### Files

- new: `src/pixie_solver/model/program_encoder.py`
- touch: `src/pixie_solver/model/board_encoder.py`

### Exit criteria

- program tokens are produced deterministically from current hand-authored pieces
- unit tests prove stable token ordering across runs

## Phase 2 — Action Token Encoder

### Objective

Replace the current move-scoring path with structured action tokens.

### Deliverables

- `ActionTokenEncoderV2`
- encoded action token fields:
  - actor id/class
  - from / to square
  - action kind
  - target refs
  - params
  - tags

### Important design choice

Do not make one-ply simulator application the primary action representation.

One-ply consequence features can remain as optional extra features, but not the
core representation.

### Files

- new: `src/pixie_solver/model/action_encoder_v2.py`
- touch: `src/pixie_solver/model/move_encoder.py` only if extracting shared helpers

### Exit criteria

- action tokens align exactly with legal move ids
- action encoding cost is materially lower than the current consequence-heavy path

## Phase 3 — World/Program Context Encoder

### Objective

Build a transformer encoder over:

- global world token
- entity tokens
- program tokens
- optional semantic probe tokens

### Deliverables

- updated board/world encoder that returns:
  - context token tensor
  - token type ids or masks
  - mapping metadata for entities/programs

### Optional semantic probes in v1

Use a small set only:

- capturability/control summaries from the query layer
- counts of action/query/reaction blocks
- piece-local state-schema summaries

Do not overbuild semantic probes in the first pass.

### Files

- `src/pixie_solver/model/board_encoder.py`
- `src/pixie_solver/model/program_encoder.py`
- optional: `src/pixie_solver/model/semantic_features.py`

### Exit criteria

- context encoder runs on current training data without simulator changes
- token counts remain manageable for batch training

## Phase 4 — Action Cross-Attention Model

### Objective

Implement `PolicyValueModelV2`.

### Deliverables

- action tokens cross-attend into context tokens
- policy logits come from contextualized action tokens
- value comes from a dedicated global token or pooled context

### Recommended minimal shape

1. context transformer encoder
2. action projection
3. one or two cross-attention blocks
4. optional shallow action self-attention
5. policy/value heads

### Files

- new: `src/pixie_solver/model/policy_value_v2.py`
- touch: `src/pixie_solver/model/__init__.py`

### Exit criteria

- forward API matches the current training code contract
- batching supports variable numbers of legal actions
- inference works in self-play and search

## Phase 5 — Training Integration

### Objective

Train the new model without disrupting the current loop.

### Deliverables

- architecture-selectable model creation in `train.py`
- architecture-selectable checkpoint loading in self-play inference
- train-loop wiring for `world_conditioned_v2`

### Important rule

Do not remove the baseline model path yet.

All training integration should support:

- `baseline_v1`
- `world_conditioned_v2`

### Exit criteria

- train command works end-to-end with `world_conditioned_v2`
- train-loop can run with either model

## Phase 6 — Evaluation And Ablation

### Objective

Prove that the new architecture is actually using semantics.

### Required comparisons

- `baseline_v1` vs `world_conditioned_v2`
- board-only ablation vs full program-conditioned model
- action cross-attention on vs off
- optional consequence-feature ablation

### Minimum metrics

- policy loss / value loss
- search guidance strength
- node-efficiency comparison in search
- same-board/different-rules sensitivity test

### Exit criteria

- v2 matches or beats baseline on current self-play metrics
- v2 shows stronger sensitivity to changed executable semantics

## 8. First PR Sequence

The first practical sequence should be:

1. checkpoint/config compatibility scaffold
2. `ProgramIRTokenEncoder`
3. `ActionTokenEncoderV2`
4. `PolicyValueModelV2` with cross-attention
5. train-loop / self-play wiring
6. semantic-sensitivity evaluation

This order is important.

If checkpoint/config compatibility is not done first, the model upgrade becomes
harder to test in the loop.

## 9. Risks

### 9.1 Token explosion

Program tokens plus entity tokens plus action tokens can grow quickly.

Mitigation:

- keep v1 program tokenization compact
- share embeddings across block/operator namespaces where reasonable
- cap or bucket large argument payloads

### 9.2 Slow action encoding

If action tokens still depend heavily on simulator rollouts, inference will be too
slow for MCTS.

Mitigation:

- make structural action encoding primary
- treat one-ply consequences as optional small augmentations

### 9.3 Weak semantic use despite richer inputs

The architecture can still ignore the program side.

Mitigation:

- add same-board/different-rules evaluation early
- add board-only and no-cross-attention ablations

## 10. Recommendation

The next code pass should be:

- Phase 0
- then Phase 1
- then Phase 2

Do not jump straight into a full `PolicyValueModelV2` without first adding the
compatibility scaffold and deterministic `ProgramIR` tokenization.

That is the shortest path to a world-conditioned model we can actually train,
evaluate, and compare against the current baseline.
