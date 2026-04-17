# Generalized Piece Program Runtime Design

## 1. Purpose

This document proposes the next runtime architecture for PixieChess.

The current minimal DSL is good enough to prove out a small closed world of
magical pieces. It is not expressive enough for the larger research goal:

- pieces should behave like executable programs,
- the simulator should reproduce those semantics exactly,
- the LLM should compile and repair those programs from sparse language and
  observations,
- and the RL system should learn strategy under the current executable rules.

The key shift is:

> move from a narrow hand-enumerated DSL to a generalized piece-program runtime,
> while keeping the runtime canonical, typed, deterministic, and learnable.

This is not a proposal to use arbitrary Python as the simulator contract.
Python-like syntax can be an authoring surface, but the executable source of
truth should be a canonical structured IR.

## 2. Core Thesis

To solve PixieChess, we need to replicate operational semantics, not just piece
descriptions.

The correct abstraction is:

- a shared `GameState`,
- a per-piece `PieceProgram`,
- a per-instance local state,
- explicit player-selectable action intents,
- explicit engine-applied effects,
- explicit events and deterministic event resolution.

In other words, a piece is not just a list of modifiers. A piece is a small
program that:

1. reads the current game state,
2. enumerates legal action intents for itself,
3. reacts to engine events,
4. proposes typed effects,
5. and maintains local state through those effects.

That structure is much closer to the actual PixieChess program we need to
reconstruct and search through.

## 3. Non-Negotiable Constraints

The generalized runtime should preserve the design constraints that make the
current system debuggable.

### 3.1 One current executable rule set

At runtime there is exactly one current executable program per piece class.

### 3.2 Deterministic simulator

Given:

- a canonical piece registry,
- a canonical state,
- and a canonical selected action,

the next state and emitted trace must be deterministic.

### 3.3 The engine owns mutation

Piece programs may propose actions and effects.
They may not perform arbitrary mutation directly.

This distinction is critical for:

- stable action IDs,
- replay,
- mismatch localization,
- legality checks,
- training targets,
- and search correctness.

### 3.4 Bounded, sandboxed program space

The executable runtime must forbid:

- imports,
- hidden global state,
- randomness,
- unbounded loops,
- recursion,
- filesystem or network access,
- and order-dependent side effects outside the engine contract.

### 3.5 Canonical serialized form

Every piece program must lower to a canonical structured representation that can
be:

- hashed,
- diffed,
- versioned,
- embedded by the model,
- and patched by the repair loop.

## 4. Runtime Objects

## 4.1 GameState

`GameState` remains the global source of truth.

Suggested shape:

```python
GameState = {
    board,
    side_to_move,
    piece_instances,
    global_state,
    pending_events,
    move_index,
    repetition_info,
    metadata,
}
```

Where:

- `board` tracks occupancy and geometry,
- `piece_instances` maps instance id to per-piece runtime state,
- `global_state` tracks game-wide counters and mode flags,
- `pending_events` is the deterministic event queue,
- `metadata` is non-semantic debug and provenance data.

## 4.2 PieceProgram

Each piece class owns one canonical executable program.

Conceptual interface:

```python
PieceProgram = {
    piece_id,
    base_archetype,
    state_schema,
    constants,
    action_rules,
    reaction_rules,
    metadata,
}
```

Important: `base_archetype` is a convenience, not the whole semantics.
The runtime should not assume all future pieces are just orthodox pieces plus a
small modifier list.

## 4.3 PieceInstance

Each active piece instance contains:

- `instance_id`
- `piece_id`
- `color`
- `square`
- `local_state`
- `status`

`local_state` is where timers, charge, flags, remembered targets, modes, and
other piece-specific memory live.

## 4.4 ActionIntent

An `ActionIntent` is a player-selectable move candidate.

Suggested shape:

```python
ActionIntent = {
    action_kind,
    actor_id,
    from_square,
    to_square,
    params,
    tags,
}
```

Examples:

- standard move
- capture
- push-capture
- swap
- teleport
- promote
- activate ability

The important property is that an action is a typed intent, not an opaque
successor board state.

## 4.5 Effect

An `Effect` is an engine-applied mutation primitive.

Suggested effect families:

- move piece
- remove piece
- create piece
- set local state
- increment local state
- set global state
- emit event
- transform piece class
- relocate piece
- mark status

Effects should be explicit, typed, and traceable.

## 4.6 Event

Events are how pieces react to game evolution.

Suggested event families:

- turn start
- turn end
- action selected
- action committed
- piece moved
- piece removed
- piece entered square
- piece left square
- attack resolved
- local state changed
- custom synthetic event emitted by another rule

## 4.7 StateDelta / Trace

Every transition should emit a canonical trace object.

```python
StateDelta = {
    action_id,
    changed_piece_ids,
    removed_piece_ids,
    created_piece_ids,
    effects,
    emitted_events,
    debug_frames,
}
```

This is essential for:

- mismatch repair,
- replay verification,
- curriculum probes,
- and future model features.

## 5. Engine Phases

The engine should have explicit phases.

### Phase 1: Enumerate candidate actions

For the side to move:

1. ask each active friendly piece program to enumerate candidate `ActionIntent`s,
2. canonicalize and deduplicate them,
3. attach stable action ids,
4. reject malformed or non-canonical outputs.

### Phase 2: Legality filtering

The engine validates candidate intents against:

- board geometry,
- occupancy,
- side-to-move rules,
- king safety or PixieChess-Lite terminal rules,
- and runtime invariants.

Programs may propose invalid intents; the engine decides legality.

### Phase 3: Commit selected action

Once search or a player selects one legal action:

1. apply the base action semantics,
2. produce initial effects,
3. emit initial events,
4. record the first trace frame.

### Phase 4: Resolve event queue to fixed point

While events remain:

1. pop the next event in canonical order,
2. find matching reaction rules,
3. evaluate guards,
4. produce typed effects,
5. apply them through the engine,
6. emit follow-on events,
7. append trace frames.

The loop stops at a deterministic fixed point or fails loudly at a configured
safety limit.

### Phase 5: Finalization

After event resolution:

- update repetition and counters,
- switch side to move if appropriate,
- run invariant checks,
- evaluate terminal conditions,
- return `(next_state, state_delta)`.

## 6. Piece Program Contract

The runtime contract should look like this conceptually:

```python
enumerate_actions(ctx: ActionContext) -> list[ActionIntent]
react(ctx: EventContext, event: Event) -> list[Effect]
```

Where:

- `ActionContext` is read-only access to board, geometry, occupancy, local state,
  global state, and query helpers,
- `EventContext` is read-only access to the same plus the current event and
  transition trace so far.

The program may compute:

- squares,
- paths,
- local variables,
- conditions,
- counters,
- filtered candidate sets,
- and effect parameters.

The program may not:

- mutate `GameState` directly,
- allocate untracked hidden state,
- or skip the engine's typed effect system.

## 7. Canonical Program Space

The runtime should use a structured IR, not raw code text.

## 7.1 Why not arbitrary Python

Raw Python is attractive as an authoring language, but it is the wrong
canonical representation because it is:

- hard to sandbox correctly,
- hard to canonicalize,
- hard to diff semantically,
- hard to embed for the model,
- and too easy for the LLM to exploit with brittle shortcuts.

## 7.2 Recommended structure

Use a typed IR with:

- function-like blocks,
- local variables,
- conditionals,
- bounded loops over engine-provided collections,
- geometry/query library calls,
- typed action constructors,
- typed effect constructors,
- typed event matchers.

Conceptual shape:

```python
ProgramIR = {
    piece_id,
    state_schema,
    constants,
    action_blocks: [ActionBlock],
    reaction_blocks: [ReactionBlock],
}
```

Each block should be canonical JSON-like data or a small bytecode-like form.

## 7.3 Standard library, not full language

To keep the runtime expressive but learnable, expose a standard semantic
library:

- board geometry queries
- occupancy queries
- piece filters
- ray tracing
- neighborhood queries
- relative coordinate transforms
- path transforms
- event field accessors
- state reads
- simple arithmetic and comparison

This gives the LLM and the model a manageable semantic vocabulary.

## 8. Intents Versus Effects

This is the most important runtime distinction.

## 8.1 Action intents are player choices

Examples:

- move from `c1` to `h6`
- activate ability on `e5`
- castle king-side

These must be:

- enumerable,
- comparable,
- and assignable stable ids.

## 8.2 Effects are transition primitives

Examples:

- move actor to target square
- push target one square east
- remove target
- increment charge
- emit `piece_removed`

These happen inside transition resolution and do not appear as player-selectable
actions.

## 8.3 Why this matters

If a piece directly returns mutated successor boards, we lose:

- stable action identity,
- action-level priors,
- clean replay,
- localized mismatch repair,
- and most useful search diagnostics.

The runtime should therefore be:

> player chooses among explicit intents; engine resolves those intents into
> effects and events.

## 9. Example: Reflecting Bishop

The motivating example is a bishop that reflects off board edges.

This should be expressible as a piece program over generic geometry primitives.

Conceptually:

```python
enumerate_actions(ctx):
    for direction in diagonal_directions():
        ray = trace_path(
            start=ctx.self_square,
            direction=direction,
            step_rule="reflect_at_edge",
            stop_rule="stop_on_second_blocker",
            max_steps=14,
        )
        yield from ray_to_move_intents(ray)
```

This does not require the engine to know about "bounce bishop" specifically.
It requires the runtime library to support:

- path tracing,
- edge reflection,
- and conversion from traced paths to typed intents.

This is the right level of generalization.

## 10. Outer Learning Loop

The rule loop should target the canonical IR.

### 10.1 Compile

Input:

- short natural-language description
- current runtime library reference
- examples if available

Output:

- canonical `ProgramIR`
- explanation
- optional generated tests/probes

### 10.2 Verify

Compile output must be verified by:

- schema checks
- engine execution checks
- deterministic replay
- probe-based behavior checks

### 10.3 Repair

Mismatch repair should operate on:

- current `ProgramIR`
- before state
- selected action
- predicted trace
- observed trace
- predicted next state
- observed next state
- implicated piece ids
- prior regression cases

This is much stronger than patching from a shallow final-state diff alone.

### 10.4 Deploy

Accepted repaired programs should be versioned into the registry and become the
current executable rule set at a controlled boundary.

Recommended boundary for v1:

- between self-play/train-loop cycles,
- never mid-search,
- never mid-game.

## 11. Inner Learning Loop

The inner loop should continue to treat the simulator as authoritative.

### 11.1 Search

MCTS remains the correct base algorithm.

Search should consume:

- exact legal action intents,
- exact transition results,
- exact event resolution,
- stable action ids,
- and optional cached trace summaries.

### 11.2 Model inputs

A stronger model should consume:

- board tokens,
- piece-instance-state tokens,
- canonical program tokens from `ProgramIR`,
- semantic probe summaries derived from the engine,
- candidate action tokens with explicit action parameters,
- and one-ply trace summaries.

### 11.3 Why compiled semantics matter

The current model compresses the DSL too aggressively.
A generalized runtime should let us expose richer semantics than:

- just op names,
- just pooled counts,
- or just orthodox material.

Examples of high-value semantic features:

- reachable squares under current rule semantics
- query-derived control relations under current rule semantics
- push / swap / teleport geometry
- event-trigger summaries
- local-state dependency summaries
- volatility and cascade depth estimates

### 11.4 Candidate scoring

The policy head should still score legal candidates only.

This remains true even in the generalized runtime because:

- legality is still engine-defined,
- candidate count is still state-dependent,
- and action semantics still vary by piece program.

### 11.5 Cross-attention recommendation

A strong v2 should likely use:

- board/program context tokens
- candidate action tokens
- cross-attention from candidate actions into contextualized board/program state

That is a more natural fit than compressing all semantics into a single pooled
piece-class embedding.

## 12. Proof Strategy

This runtime enables a cleaner proof plan.

### 12.1 Rule-learning proof

Measure:

- compile-only success on held-out piece families
- repair success after `k` observed mismatches
- regression survival on old probes after repair
- time-to-correctness for newly introduced pieces

### 12.2 Strategic-generalization proof

Measure:

- arena strength on rulesets containing newly introduced pieces
- held-out tactical suites for new piece families
- node efficiency and score rate versus search-only baselines
- adaptation curve after registry updates

### 12.3 Important evaluation boundary

We should distinguish:

- new programs over known primitives
- new compositions of known geometry/control constructs
- genuinely new runtime primitives

These are different scientific claims and should not be conflated.

## 13. Migration Plan

The current repo should evolve in stages.

### Stage A: Introduce explicit runtime abstractions

Add:

- `ActionIntent`
- `Effect`
- `Event`
- `StateDelta`
- fixed-point execution frames

without changing user-visible behavior.

### Stage B: Lower current DSL into the new runtime

Treat the existing DSL as a subset frontend that compiles into `ProgramIR`.

This preserves current pieces and tests while moving the engine to the new
abstractions.

### Stage C: Add richer query primitives

Add geometry/query helpers needed for the next real mechanic family, such as:

- reflection
- zone effects
- directional auras
- conditional relocation

### Stage D: Upgrade repair loop

Repair should operate on:

- traces,
- regression cases,
- generated probes,
- and versioned registry updates between cycles.

### Stage E: Upgrade model conditioning

Replace the current lossy DSL embedding path with:

- structured program tokens,
- explicit action parameter tokens,
- semantic probe features,
- and candidate-to-context cross-attention.

## 14. Open Questions

### 14.1 How expressive should the IR be?

It should be expressive enough for reflection, stateful triggers, and
nontrivial geometry, but not so expressive that canonicalization and repair
become intractable.

### 14.2 How much semantics should be exposed directly to the model?

The model should not re-implement the simulator, but it should likely receive
engine-derived semantic summaries rather than only raw program structure.

### 14.3 When should repaired programs enter training?

The safest answer is:

- admit at cycle boundaries,
- stamp every self-play artifact with the registry version used,
- and evaluate adaptation explicitly after each rule change.

## 15. Recommended Decision

Adopt the following design:

- canonical structured `PieceProgram` IR
- explicit `ActionIntent`, `Effect`, `Event`, and `StateDelta`
- deterministic engine-owned mutation and fixed-point event resolution
- restricted program runtime with a rich geometry/query standard library
- LLM compile/repair targeting the canonical IR
- MCTS over exact simulator semantics
- rule-conditioned model over board state, local state, program structure, and
  candidate action semantics

In short:

> The right generalization is not "let pieces run arbitrary code."
> The right generalization is "treat pieces as canonical executable programs
> over shared game state, with explicit intents, effects, and events."
