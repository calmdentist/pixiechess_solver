# Search And Policy/Value Plan

This document turns the existing M4/M5/M6 ideas into an implementation plan that
can be executed directly against the current codebase.

The short version:

1. finish a strong search-only baseline first,
2. stabilize replay and self-play data contracts,
3. add a small rule-conditioned policy/value network,
4. feed that model back into search.

Search comes first on purpose. If search targets are still moving, model work
becomes wasted motion.

## Locked Decisions

### 1. Search-first, model-second

The next implementation pass should be `M4`, not `M5`.

Reasons:

- search is the backbone that generates policy targets,
- self-play depends on search output format,
- tactical evaluation depends on a working engine,
- model integration is much easier once the search/data contracts are fixed.

The correct order is:

1. search-only baseline,
2. self-play and replay fixtures,
3. model MVP,
4. tactical regression harness,
5. model-guided search.

### 2. Candidate-scoring policy head

Do not build a fixed chess move tensor.

The simulator already enumerates legal moves exactly. The policy head should score
legal candidates only.

That means:

- search always requests `legal_moves(state)`,
- each legal move gets a stable move id,
- the model returns logits keyed by move id,
- search normalizes only over those legal candidates.

### 3. The simulator remains the source of truth

The model does not decide:

- legality,
- next-state transitions,
- hook resolution,
- terminal conditions.

The model only supplies:

- priors over legal moves,
- a value estimate for the current state.

### 4. One small PyTorch model, no exotic stack

Use plain PyTorch for v1.

The initial network can be built from standard `Embedding`, `TransformerEncoder`,
`Dataset`, and `DataLoader` primitives in the official docs. There is no need for
Lightning, graph libraries, custom kernels, or a separate program-analysis model in
the first pass.

### 5. Reproducible data is a hard requirement

Every self-play position must be reproducible from:

- canonical serialized state,
- serialized legal moves,
- visit counts / visit probabilities,
- final outcome,
- optional root value and search diagnostics.

Never key policy targets by list index alone. Legal move order is an implementation
detail and will drift as the simulator evolves.

## Phase Plan

## Phase A — Search Contracts

This is the first code pass.

### A.1 Stable move identity

Add one canonical move-id function:

- input: `Move`
- output: stable string id

Recommended form:

- `move_id = stable_digest(move.to_dict())`

Why this should be the canonical key:

- fixed-width,
- deterministic,
- easy to store in visit distributions,
- safe across reordering,
- easy to compare in tests and datasets.

### A.2 Search node structure

Keep the first tree simple.

Recommended runtime shape:

- `SearchNode`
  - `state_hash`
  - `to_play`
  - `visit_count`
  - `value_sum`
  - `is_expanded`
  - `is_terminal`
  - `terminal_value`
  - `legal_moves`
  - `move_ids`
  - `children_by_move_id`

- `SearchEdge`
  - `move`
  - `move_id`
  - `prior`
  - `visit_count`
  - `value_sum`
  - `child_state_hash`
  - cached `StateDelta` or small diagnostics if useful

Do not build a full transposition DAG in v1. A pure tree is easier to reason about
and much easier to debug. A small cache for legal moves or terminal checks is fine,
but avoid multi-parent visit sharing in the first implementation.

### A.3 Search API

The search entry point should become something close to:

```python
def run_mcts(
    state: GameState,
    *,
    simulations: int,
    policy_value_model: PolicyValueModel | None = None,
    evaluator: StateEvaluator | None = None,
) -> SearchResult:
    ...
```

The contract should be:

- if a model is present, use model priors and value,
- otherwise use uniform priors and a deterministic fallback evaluator,
- always return legal moves only,
- always return a visit distribution keyed by stable move ids.

### A.4 Root result contract

`SearchResult` should grow to include:

- `selected_move`
- `selected_move_id`
- `visit_distribution`
- `visit_counts`
- `root_value`
- `legal_moves`
- `policy_logits` if a model was used
- `metadata` for diagnostics

This keeps search output reusable for both gameplay and training.

## Phase B — Search-Only Baseline

This is still `M4`, but now with behavior concrete enough to implement.

### B.1 Selection / expansion / backup

Use standard PUCT:

- selection with `puct_score`,
- expand one leaf at a time,
- evaluate the leaf,
- backpropagate signed value from the current player perspective.

### B.2 Leaf evaluation before the model exists

Do not use random rollouts.

Random rollouts are especially misleading in a rule-rich game with hooks, counters,
and unusual tactics. They add variance without buying clarity.

The initial fallback evaluator should be deterministic and cheap:

- terminal result if terminal,
- otherwise a handcrafted static value.

Recommended first static evaluator:

- material-like score by piece class,
- mobility bonus,
- check / in-check penalty,
- side-to-move perspective,
- optional event-volatility feature later if needed.

The piece-class value should not be hard-coded only by orthodox type. Use a simple
DSL-aware heuristic:

- start from base-piece value,
- add bonuses for range extension,
- add bonuses for phasing or push-capture semantics,
- add bonuses for hooks / stateful behavior conservatively.

This does not need to be perfect. It only needs to be deterministic and better than
zero.

### B.3 Deterministic self-play

The self-play loop should:

1. sample or construct an initial state,
2. run MCTS,
3. record the root training example,
4. apply the selected move,
5. repeat until terminal or move cap.

For reproducibility:

- seed all randomness,
- include temperature schedule explicitly,
- store search settings with each game,
- emit exact replay traces alongside training examples.

### B.4 Tactical suite deferred

The tactical runner is valuable, but it is not part of the immediate next coding
pass.

Defer it until:

- search-only self-play is emitting stable data,
- replay/example formats are no longer moving,
- we are ready to compare search-only and model-guided behavior cleanly.

When it lands, the first suite should include:

- orthodox checkmates and escape problems,
- phase-through-allies tactics,
- push-capture tactics,
- trigger/counter tactics,
- promotion and castling edge cases.

## Phase C — Data Contracts

Search and training should not share loose ad hoc dicts.

### C.1 Training example schema

`SelfPlayExample` should expand into something like:

- `state`
- `legal_moves`
- `legal_move_ids`
- `visit_distribution`
- `visit_counts`
- `selected_move_id`
- `root_value`
- `outcome`
- `metadata`

Important rule:

- `visit_distribution` keys must always be stable move ids,
- `legal_moves` must remain serialized explicitly,
- `outcome` must be from the perspective of `state.side_to_move`.

### C.2 Game-level record

Keep game replay and training examples separate.

Recommended outputs from self-play:

- `ReplayTrace`
  - exact move/event/state replay
- `SelfPlayGame`
  - metadata about one game
  - all `SelfPlayExample`s for that game

This separation matters because replay is for engine verification, while examples are
for training.

### C.3 File format

Use JSONL for early experimentation.

Why:

- trivial to inspect,
- easy to diff,
- easy to append,
- easy to shard later.

Compression can be added later if dataset size becomes annoying.

## Phase D — Model MVP

Only start this once search-only self-play works and the data contract is stable.

## D.1 Model inputs

The model should consume three information sources:

1. active piece-class programs,
2. current board / piece-instance state,
3. legal candidate moves.

### D.1.1 Piece-class program encoder

The DSL encoder should not be a text model.

Use structured canonical features from compiled piece classes:

- base piece type,
- movement modifier ops,
- capture modifier ops,
- hook event types,
- condition ops,
- effect ops,
- instance state schema fields.

Recommended first implementation:

- small categorical embeddings per field,
- pooled summaries over modifier / hook lists,
- small MLP combiner,
- one output embedding per piece class.

This is much cheaper and more stable than trying to teach the model a raw DSL string.

### D.1.2 Board encoder

Represent the board as piece tokens plus a few global tokens.

Per-piece token features:

- square embedding,
- color embedding,
- base piece type embedding,
- piece-class embedding,
- piece-local state embedding.

Global token features:

- side to move,
- castling rights,
- en passant square,
- halfmove / fullmove clocks,
- repetition count summary if and when draw rules are added.

Recommended first model body:

- `d_model = 192`
- `num_layers = 4`
- `num_heads = 8`
- dropout `0.1`

This is intentionally small enough to train locally but expressive enough to start.

### D.1.3 Piece-local state encoding

Current DSL state is scalar and flat. Keep the encoder matched to that.

Encode piece state as:

- declared field-name embedding,
- declared field-type embedding,
- scalar value projection,
- pooled per-piece state summary.

Do not special-case counters, timers, or modes beyond what is already encoded in the
schema. The DSL already flattened those categories.

### D.1.4 Move encoder

Each legal move should become a candidate token or feature bundle containing:

- source square,
- destination square,
- move kind,
- promotion piece type if any,
- moving piece embedding,
- target piece embedding if present,
- whether the move captures / castles / promotes / pushes,
- one-step consequence summary.

### D.1.5 One-step consequence summary

This is worth doing in v1.

For each legal move, run the exact simulator once and extract cheap features such as:

- material delta,
- check status before / after,
- event count,
- changed piece count,
- resulting side to move,
- whether a piece was removed,
- whether piece-local state changed.

This is a very strong shortcut. It lets the network reason about consequences without
forcing it to learn one-ply simulation internally.

## D.2 Model outputs

The first model should output:

- `policy_logits: dict[move_id, float]`
- `value: float`

Use scalar value in `[-1, 1]` first.

W/D/L logits can be added later if scalar value proves too blunt, but scalar value is
the simplest target for initial integration.

## D.3 Losses

Start with only two losses:

- policy cross-entropy to the MCTS visit distribution,
- value MSE to final outcome.

Do not add auxiliary tactical or event-count losses in the first model pass. Keep the
training signal narrow until the baseline works.

## Phase E — Model-Guided Search

Once the model exists:

- priors come from policy logits over legal moves,
- leaf value comes from the value head,
- search fallback evaluator remains available for ablations and failures.

The key comparison should be:

- search-only baseline,
- model-guided search at equal simulation budget,
- model-guided search at reduced simulation budget.

The model is successful if it either:

- improves tactical strength,
- or reaches similar strength with fewer nodes.

## Concrete Implementation Sequence

This is the order I would actually code:

1. `search/node.py`
   - expand node and edge data structures
2. `search/mcts.py`
   - selection, expansion, backup, root result contract
3. `training/dataset.py`
   - expand training example schema
4. `training/selfplay.py`
   - deterministic self-play loop
5. `model/dsl_encoder.py`
   - structured piece-class encoder
6. `model/board_encoder.py`
   - piece/global token encoder
7. `model/move_encoder.py`
   - candidate move features and one-step summaries
8. `model/policy_value.py`
   - small transformer + policy/value heads
9. `training/train.py`
   - dataset loading, batching, optimization, checkpoints
10. `search/mcts.py`
   - plug in model priors/value behind the same API
11. `eval/tactical.py`
    - add the tactical regression harness once the self-play/model path is stable

That keeps each phase verifiable before the next one starts.

## Immediate Next PRs

The next three implementation PRs should be:

### PR 1 — Search-only core

- stable move ids,
- expanded search node / result contracts,
- PUCT MCTS,
- deterministic fallback evaluator,
- tests on legal move selection and backup behavior.

### PR 2 — Self-play and replay pipeline

- deterministic self-play game generation,
- JSONL replay/example writer,
- search reproducibility checks.

### PR 3 — Model MVP

- PyTorch dependency,
- DSL / board / move encoders,
- policy/value model,
- training loop on self-play data,
- inference wrapper wired into search.

### PR 4 — Tactical regression harness

- curated tactical suite,
- baseline metrics runner,
- search-only vs model-guided comparisons.

## Things Explicitly Deferred

Do not do these in the next pass:

- transposition DAG with shared statistics,
- graph neural networks,
- raw-text DSL encoder,
- large decoder-only model,
- end-to-end learned legality,
- policy head over fixed global action tensor,
- auxiliary losses before the base policy/value loss is stable.

## Success Criteria

We should consider this whole track healthy if we can say:

- search-only MCTS plays legal complete games reproducibly,
- self-play emits stable replay + policy targets,
- the model trains on real replay data without schema hacks,
- model-guided search improves strength or reduces node count,
- tactical regressions are measured continuously.
