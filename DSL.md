# Minimal Viable DSL v1

## 1. Purpose

This document defines the first implementable PixieChess piece DSL.

The goal of v1 is not to cover every possible magical mechanic. The goal is to define a small, deterministic, repair-friendly rule language that we can actually build now.

v1 is intentionally narrow:

- every piece starts from one orthodox chess base type,
- rules are expressed as structured data, not code,
- legal move generation stays engine-driven,
- piece-local state is flat and typed,
- reactive behavior happens through a small hook system,
- there is one canonical representation for compile, validate, diff, and repair.

If a mechanic cannot be expressed cleanly in this DSL, that mechanic is out of scope for v1 and should be deferred to DSL v2 rather than forcing complexity into the engine.

---

## 2. Non-Goals

The v1 DSL does not support:

- arbitrary user-defined functions,
- loops or recursion,
- nested boolean expression trees,
- global rule rewrites unrelated to a piece,
- probabilistic effects,
- runtime uncertainty,
- custom free-form executable code,
- multiple equivalent source encodings for the same mechanic.

These are deliberate exclusions. They keep the simulator deterministic and keep LLM compile/repair tractable.

---

## 3. Runtime Contract

The runtime system has four core objects:

### 3.1 GameState

The full game state, including:

- 8x8 board occupancy,
- piece instances,
- side to move,
- move/repetition metadata,
- pending events.

The board grid alone is not enough because PixieChess pieces may have local state and event-driven effects.

### 3.2 PieceClass

A compiled rule object with:

- `piece_id`
- `name`
- `base_piece_type`
- `instance_state_schema`
- `movement_modifiers`
- `capture_modifiers`
- `hooks`

### 3.3 PieceInstance

A runtime instance of a piece class:

```python
PieceInstance = {
    id: str,
    class_id: str,
    color: "white" | "black",
    square: "a1".."h8" | None,
    state: dict[str, JsonScalar],
}
```

### 3.4 Simulator

The simulator is the only component that decides:

- legal moves,
- state transitions,
- event ordering,
- hook execution,
- invariant enforcement.

The DSL does not execute itself. It supplies normalized operators that the engine interprets.

---

## 4. Core Design Rules

These rules are architectural law for v1:

1. There is one canonical source representation.
2. The canonical source representation is also the compiler input.
3. The compiler produces normalized runtime data, not generated code.
4. Movement modifiers alter primary move generation.
5. Capture modifiers alter what happens when a legal move targets an occupied enemy square.
6. Hooks run only in response to engine-emitted events.
7. Conditions inside one hook are an implicit logical `AND`.
8. If you need `OR`, write multiple hooks.
9. Piece-local state is flat, typed, and explicit.
10. Any mechanic that needs a general expression language is out of scope for v1.

---

## 5. Canonical Source Format

The DSL may be authored as JSON or YAML, but after parsing it must normalize to this exact shape:

```yaml
piece_id: phasing_rook
name: Phasing Rook
base_piece_type: rook
instance_state_schema:
  - name: cooldown
    type: int
    default: 0
    description: Turns until this piece can trigger again.
movement_modifiers:
  - op: inherit_base
    args: {}
capture_modifiers:
  - op: inherit_base
    args: {}
hooks:
  - event: piece_captured
    conditions:
      - op: self_on_board
        args: {}
    effects:
      - op: move_piece
        args:
          piece: self
          to:
            relative_to: self
            offset: [0, 1]
    priority: 0
metadata: {}
```

### 5.1 Required Top-Level Fields

- `piece_id`
- `name`
- `base_piece_type`
- `instance_state_schema`
- `movement_modifiers`
- `capture_modifiers`
- `hooks`

### 5.2 Optional Top-Level Fields

- `metadata`

No other top-level fields are part of v1.

---

## 6. Top-Level Field Semantics

### 6.1 `piece_id`

- lowercase snake_case
- unique within the active ruleset
- semantic identifier used in state, tests, and repair diffs

### 6.2 `name`

- human-readable display name
- non-empty string
- not used for simulator semantics

### 6.3 `base_piece_type`

One of:

- `pawn`
- `knight`
- `bishop`
- `rook`
- `queen`
- `king`

Every v1 piece starts from one of these orthodox archetypes.

### 6.4 `instance_state_schema`

A list of flat typed fields. See Section 7.

### 6.5 `movement_modifiers`

An ordered list of engine-known movement operators. See Section 8.

### 6.6 `capture_modifiers`

An ordered list of engine-known capture operators. See Section 9.

### 6.7 `hooks`

An ordered list of reactive rules. See Section 10.

### 6.8 `metadata`

Arbitrary non-semantic data for provenance, notes, or source text.

The simulator must ignore `metadata`.

---

## 7. Piece-Local State

### 7.1 State Model

Piece-local state is a flat mapping from field name to scalar value.

Allowed field types:

- `bool`
- `int`
- `float`
- `str`

There is no nested state object model in v1.

This means the earlier `counters / timers / flags / enums` grouping is not part of the canonical DSL. Those are authoring concepts, not runtime schema categories.

### 7.2 State Field Shape

```yaml
- name: charges
  type: int
  default: 0
  description: Number of stored charges.
```

### 7.3 Rules

- `name` must be lowercase snake_case
- `type` must be one of the four allowed scalar types
- `default` should be present for every field
- field names must be unique per piece class

### 7.4 Naming Conventions

Use naming conventions rather than schema categories:

- timers: `cooldown`, `turns_until_spawn`
- counters: `charges`, `stack_count`
- flags: `armed`, `activated_this_turn`
- modes: `mode` with `type: str`

Restricted enums are not part of v1. If a piece has modes, store the mode as a string.

---

## 8. Movement Modifiers

### 8.1 Design Rule

Movement modifiers do not define an arbitrary move program.

Instead, the engine:

1. starts from the orthodox geometry of `base_piece_type`,
2. applies the ordered `movement_modifiers`,
3. produces legal quiet moves to empty squares.

If a piece needs completely custom geometry unrelated to any orthodox base type, it is out of scope for v1.

### 8.2 Allowed Movement Operators

#### `inherit_base`

Use the orthodox move geometry of the base piece.

Arguments:

```yaml
op: inherit_base
args: {}
```

This should usually be the first movement modifier.

#### `phase_through_allies`

Friendly pieces do not block ray-based movement.

Arguments:

```yaml
op: phase_through_allies
args: {}
```

This only affects blocking behavior. It does not make allied squares legal destinations unless another rule explicitly says so.

#### `extend_range`

Increase the maximum ray length of the base geometry.

Arguments:

```yaml
op: extend_range
args:
  extra_steps: 1
```

#### `limit_range`

Reduce the maximum ray length of the base geometry.

Arguments:

```yaml
op: limit_range
args:
  max_steps: 1
```

### 8.3 Out of Scope for v1

These are intentionally excluded:

- custom ray definitions,
- custom jump tables,
- free-form movement replacement programs,
- boolean `when` clauses inside movement modifiers.

If movement depends on piece-local state in a complex way, that is a DSL v2 problem.

---

## 9. Capture Modifiers

### 9.1 Design Rule

Capture modifiers apply when a primary move reaches an enemy-occupied square that is legal under the current piece geometry.

They change the resolution of that occupied-square interaction. They do not replace the engine's entire move-generation pipeline.

### 9.2 Allowed Capture Operators

#### `inherit_base`

Use normal chess capture semantics for the base piece.

Arguments:

```yaml
op: inherit_base
args: {}
```

#### `replace_capture_with_push`

Instead of capturing the target, push it in the direction of travel.

Arguments:

```yaml
op: replace_capture_with_push
args:
  distance: 1
  edge_behavior: remove_if_pushed_off_board
```

Allowed `edge_behavior` values:

- `block`
- `remove_if_pushed_off_board`

### 9.3 Semantics of `replace_capture_with_push`

When this operator is active:

1. the moving piece reaches an enemy-occupied square,
2. the target is pushed in the move direction by `distance`,
3. if the push path is blocked and `edge_behavior` is `block`, the move is illegal,
4. if the target is pushed off board and `edge_behavior` is `remove_if_pushed_off_board`, the target is removed,
5. the moving piece occupies the target square if the move succeeds.

### 9.4 Out of Scope for v1

These are excluded:

- splash capture,
- swap capture,
- mark/debuff-only captures,
- multiple alternative capture programs for one piece.

Those can be added in later DSL versions if needed.

---

## 10. Hooks

Hooks are the only reactive mechanism in v1.

Each hook has:

- one `event`,
- zero or more `conditions`,
- one or more `effects`,
- one integer `priority`.

Canonical shape:

```yaml
- event: piece_captured
  conditions:
    - op: self_on_board
      args: {}
  effects:
    - op: move_piece
      args:
        piece: self
        to:
          relative_to: self
          offset: [0, 1]
  priority: 0
```

### 10.1 Hook Execution Rules

1. Hooks are selected by exact event match.
2. Their conditions are evaluated in list order.
3. The condition list is an implicit `AND`.
4. Matching hooks execute effects in list order.
5. Lower `priority` runs first.
6. Ties break by declaration order.
7. The simulator owns event queueing and fixed-point safety.

### 10.2 Minimal Event Set

v1 supports only these events:

- `move_committed`
- `piece_captured`
- `turn_start`
- `turn_end`

Anything else is out of scope for now.

This smaller event surface is intentional. Event proliferation is one of the fastest ways to make the simulator brittle.

---

## 11. Conditions

### 11.1 Design Rule

Conditions are flat engine-known predicates.

There is no nested boolean AST in v1. No `all`, `any`, or `not`.

If a mechanic needs an `OR`, write multiple hooks.

### 11.2 Allowed Condition Operators

#### `self_on_board`

True if the hook owner still has a square.

```yaml
op: self_on_board
args: {}
```

#### `square_empty`

True if the referenced square exists and is empty.

```yaml
op: square_empty
args:
  square:
    relative_to: self
    offset: [0, 1]
```

#### `square_occupied`

True if the referenced square exists and contains any piece.

```yaml
op: square_occupied
args:
  square:
    absolute: e4
```

#### `state_field_eq`

True if a piece-local state field equals the given scalar.

```yaml
op: state_field_eq
args:
  name: mode
  value: charging
```

#### `state_field_gte`

True if a numeric state field is greater than or equal to the given value.

```yaml
op: state_field_gte
args:
  name: charges
  value: 2
```

#### `state_field_lte`

True if a numeric state field is less than or equal to the given value.

```yaml
op: state_field_lte
args:
  name: cooldown
  value: 0
```

#### `piece_color_is`

True if the referenced piece has the given color.

```yaml
op: piece_color_is
args:
  piece: self
  color: white
```

### 11.3 Out of Scope for v1

These are excluded:

- nested `all / any / not`,
- target-pattern logic trees,
- custom expressions,
- event payload queries beyond explicit future operators.

---

## 12. Effects

### 12.1 Design Rule

Effects are flat engine-known actions.

They are the only way hooks modify state.

### 12.2 Allowed Effect Operators

#### `move_piece`

Move a referenced piece to a referenced square.

```yaml
op: move_piece
args:
  piece: self
  to:
    relative_to: self
    offset: [0, 1]
```

The simulator must reject the effect if the referenced move is impossible at execution time.

#### `capture_piece`

Capture a referenced piece using normal capture bookkeeping and event emission.

```yaml
op: capture_piece
args:
  piece: event_target
```

#### `set_state`

Set a piece-local state field to a scalar value.

```yaml
op: set_state
args:
  piece: self
  name: armed
  value: true
```

#### `increment_state`

Add a numeric amount to a piece-local numeric field.

```yaml
op: increment_state
args:
  piece: self
  name: charges
  amount: 1
```

Negative values are allowed and act as decrements.

#### `emit_event`

Emit one of the supported engine events.

```yaml
op: emit_event
args:
  event: turn_end
```

This operator exists for controlled event chaining. It should be used sparingly.

### 12.3 Out of Scope for v1

These are excluded:

- silent remove-without-capture semantics,
- swap,
- push as a generic hook effect,
- bulk area effects,
- arbitrary effect scripting.

---

## 13. References

Hooks and conditions often need to point at pieces or squares.

v1 uses a small explicit reference model.

### 13.1 Piece References

Allowed symbolic piece references:

- `self`
- `event_source`
- `event_target`

No other piece selectors are part of v1.

### 13.2 Square References

A square reference is one of:

#### Absolute square

```yaml
absolute: e4
```

#### Relative square

```yaml
relative_to: self
offset: [0, 1]
```

### 13.3 Relative Offset Semantics

`offset` is a pair of integers in mover-relative coordinates:

- first element: horizontal delta
- second element: forward/back delta

Examples:

- `[0, 1]`: one square forward
- `[0, -1]`: one square backward
- `[1, 0]`: one square to the piece's right
- `[-1, 0]`: one square to the piece's left

The meaning of forward/backward/right/left depends on the referenced piece's color.

This is the canonical v1 form. Symbolic words like `forward` are not part of the executable representation.

---

## 14. Canonicalization Rules

The parser/compiler pipeline must normalize all source into a stable canonical form.

Canonicalization rules:

1. Missing optional maps become `{}`.
2. Missing optional lists become `[]`.
3. Hooks always include `conditions`, `effects`, and `priority`.
4. Modifier, condition, and effect objects always use the `{op, args}` shape.
5. Relative offsets normalize to integer pairs.
6. List order is preserved because list order is semantic.
7. Unknown fields are rejected.

There should be one obvious serialized representation for every compiled piece program.

---

## 15. Compile Target

The DSL compiles into normalized runtime data, not generated executable code.

Conceptually:

```python
CompiledPieceClass = {
    "piece_id": ...,
    "name": ...,
    "base_piece_type": ...,
    "instance_state_schema": ...,
    "movement_modifiers": ...,
    "capture_modifiers": ...,
    "hooks": ...,
}
```

The engine then applies generic logic for:

- orthodox movement,
- modifier handling,
- capture resolution,
- event dispatch,
- hook execution,
- invariant checking.

This is the key simplification. The DSL is data. The simulator is behavior.

---

## 16. Examples

### 16.1 War Automaton

English:

`A pawn that moves one square forward whenever a piece is captured, if the square is empty.`

```yaml
piece_id: war_automaton
name: War Automaton
base_piece_type: pawn
instance_state_schema: []
movement_modifiers:
  - op: inherit_base
    args: {}
capture_modifiers:
  - op: inherit_base
    args: {}
hooks:
  - event: piece_captured
    conditions:
      - op: self_on_board
        args: {}
      - op: square_empty
        args:
          square:
            relative_to: self
            offset: [0, 1]
    effects:
      - op: move_piece
        args:
          piece: self
          to:
            relative_to: self
            offset: [0, 1]
    priority: 0
metadata: {}
```

### 16.2 Phasing Rook

English:

`A rook that can move through allied pieces.`

```yaml
piece_id: phasing_rook
name: Phasing Rook
base_piece_type: rook
instance_state_schema: []
movement_modifiers:
  - op: inherit_base
    args: {}
  - op: phase_through_allies
    args: {}
capture_modifiers:
  - op: inherit_base
    args: {}
hooks: []
metadata: {}
```

### 16.3 Sumo Rook

English:

`A rook that knocks pieces back instead of capturing.`

```yaml
piece_id: sumo_rook
name: Sumo Rook
base_piece_type: rook
instance_state_schema: []
movement_modifiers:
  - op: inherit_base
    args: {}
capture_modifiers:
  - op: replace_capture_with_push
    args:
      distance: 1
      edge_behavior: remove_if_pushed_off_board
hooks: []
metadata: {}
```

---

## 17. Explicit Omissions

The following are intentionally deferred:

- custom non-orthodox geometry,
- movement rules with arbitrary condition trees,
- event types beyond the minimal set,
- enum-constrained state fields,
- area-of-effect and zone mechanics,
- global game rules outside a piece,
- uncertainty-aware runtime semantics.

If we need any of these, we should add them deliberately in a future version rather than smuggling them into v1.

---

## 18. Implementation Order

This spec is meant to be implemented in this order:

1. parser for canonical source,
2. validator for top-level shape and allowed operators,
3. canonicalizer,
4. compiler into normalized `PieceClass`,
5. simulator support for orthodox movement plus the minimal modifier/effect set,
6. regression fixtures for the starter pieces.

That is enough to support:

- compile,
- verify,
- patch,
- replay,
- early self-play on supported mechanics.

---

## 19. Bottom Line

The minimal v1 DSL is:

- orthodox-base-piece-centered,
- declarative,
- flat,
- deterministic,
- hook-driven,
- small enough to validate and repair reliably.

The critical boundary is simple:

- the DSL describes piece semantics as data,
- the simulator performs legal move generation and execution,
- the model later learns strategy under those exact rules.

That is the version we should build first.
