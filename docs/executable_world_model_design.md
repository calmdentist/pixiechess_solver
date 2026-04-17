# Executable World Model Design

## 1. Purpose

This document reframes the project around its strongest thesis:

> the core problem is not "LLM writes magical chess pieces."
> the core problem is "LLM maintains a domain-specific executable world model,
> and RL+search optimize behavior inside that inferred world."

PixieChess is the initial domain, not the final abstraction.

The purpose of this design is to define the right top-level architecture before
we overfit the runtime to chess-specific structure.

## 2. Thesis

Humans maintain internal world models.
We revise them from observation, language, and reasoning.
Those world models are not arbitrary raw code, but they are also not fixed
hand-written ontologies.

Language works at a high level of abstraction.
Humans routinely create domain-specific languages on top of general language.
Those DSLs inherit general reasoning structure from language while introducing
their own domain-specific terms and executable logic.

This project aims to test the analogous hypothesis for LLM-guided RL:

- an LLM can create and continuously repair a domain-specific executable world
  model,
- that world model can serve as the simulator for planning and self-play,
- and an AlphaZero-style search/RL system can learn strong behavior inside that
  current inferred world model.

In this framing:

- the LLM is the high-level world-model maintainer,
- the executable world model is the current domain language,
- and the policy/value net plus MCTS optimize behavior under that current
  executable semantics.

## 3. Core Position

The project should not be framed as "better piece modifiers."

It should be framed as:

- an `ExecutableWorldModel`,
- maintained by an LLM,
- executed by a deterministic kernel,
- queried by search,
- and improved through observation-driven repair.

Pieces remain important, but they are not the only semantic objects.
They are one kind of program inside a larger world model.

That matters because some semantics are not piece-local:

- altered victory rules,
- respawn rules,
- turn-order changes,
- board-wide status effects,
- board topology changes,
- and future observation-model changes.

If we define the whole system as only "piece programs," we risk baking in the
wrong abstraction too early.

## 4. Architecture Overview

The system has three distinct layers.

### 4.1 World-Model Maintenance Layer

This is the LLM-facing layer.

Responsibilities:

- interpret natural language descriptions,
- propose executable semantics,
- repair semantics from observed mismatches,
- maintain regression memories,
- and output canonical world-model updates.

This layer should be expressive and language-like.
It should support revision, ambiguity, and partial knowledge.

### 4.2 Execution Kernel

This is the authoritative simulator.

Responsibilities:

- hold the canonical current world model,
- execute it deterministically,
- maintain state and event resolution,
- expose stable action identities,
- emit replayable traces,
- and answer the queries search depends on.

This layer must be strict, bounded, typed, and replayable.

### 4.3 Planning / Learning Layer

This is the policy/value net plus MCTS layer.

Responsibilities:

- search over the current executable world,
- condition on both state and semantics,
- learn values and action priors under the active objective,
- and improve with self-play and evaluation.

This layer should never invent simulator behavior.
The kernel remains authoritative.

## 5. The Right Top-Level Object

The right canonical object is not just `PieceProgram`.

It is:

```python
ExecutableWorldModel = {
    world_schema,
    entity_programs,
    global_rule_programs,
    objective_program,
    query_programs,
    constants,
    metadata,
}
```

This is the executable language for the current domain.

For PixieChess today:

- `entity_programs` mostly correspond to pieces,
- `global_rule_programs` handle turn structure and shared rules,
- `objective_program` remains capture-the-king,
- and `query_programs` answer legality- and planning-relevant questions.

## 6. Minimal Execution Kernel

The kernel should stay general but small.

Its job is not to contain domain semantics.
Its job is to schedule and apply domain semantics.

The minimal kernel needs:

- canonical `WorldState`
- canonical selected `ActionIntent`
- canonical `Effect`
- canonical `Event`
- deterministic scheduler
- bounded execution limits
- stable ids and hashes
- replayable `StateDelta` / trace

Everything else should be pushed into the executable world model where
possible.

## 7. Canonical Runtime Objects

These objects are still needed, but they should now be understood as kernel
objects rather than chess objects.

### 7.1 WorldState

```python
WorldState = {
    topology,
    occupants,
    entity_instances,
    global_state,
    active_objective,
    side_to_act,
    pending_events,
    metadata,
}
```

For current PixieChess, `topology` is an `8x8` board.
That is a current domain choice, not a permanent assumption.

### 7.2 EntityInstance

```python
EntityInstance = {
    instance_id,
    program_id,
    owner,
    location,
    local_state,
    status,
}
```

In current PixieChess, an entity instance is a piece instance.

### 7.3 ActionIntent

An `ActionIntent` is a player-selectable candidate.

It must stay:

- typed,
- canonical,
- stable-id addressable,
- and separate from direct mutation.

Search acts on `ActionIntent`.
It does not act on opaque successor states.

### 7.4 Effect

An `Effect` is a typed delta applied by the kernel.

Examples:

- move entity
- remove entity
- create entity
- relocate entity
- transform program id
- set local state
- increment local state
- set global state
- emit event

### 7.5 Event

An `Event` is how programs react to state evolution.

Examples:

- action_committed
- entity_removed
- entity_entered_location
- turn_start
- turn_end
- custom emitted event

### 7.6 Trace

Every transition must emit a canonical trace.

The trace is essential for:

- replay,
- mismatch localization,
- repair,
- evaluation,
- and eventually model features.

## 8. Programs Inside The World Model

The world model should support multiple program types.

### 8.1 Entity Programs

These are the current descendants of `PieceProgram`.

An entity program defines the local semantics of a class of entity.

For current PixieChess, pieces are entity programs.

Entity programs should own:

- movement logic,
- capture logic,
- derived-query logic,
- reactive event logic,
- and local-state logic.

### 8.2 Global Rule Programs

These define semantics that are not naturally local to one piece.

Examples:

- turn sequencing
- special respawn windows
- weather or board-wide modifiers
- future board-size or topology transforms

### 8.3 Objective Program

This defines what counts as success, failure, and legality-relevant strategic
constraints.

For the current scope:

- objective: capture the opponent king
- legality: your move is illegal if, after full resolution, your king is
  capturable

This lets the model learn "move the king out of check" while keeping the win
condition as capture-the-king.

Later, more complex objectives can be swapped in without forcing the whole
runtime to become piece-centric.

### 8.4 Query Programs

Search depends on queries that should not be hard-coded to orthodox chess.

Examples:

- enumerate selectable actions
- derive legality- and planning-relevant query facts
- test whether a target is capturable
- test terminal state under the current objective

These queries may be implemented by composing entity/global/objective programs,
but they should conceptually belong to the world model rather than legacy chess
helpers.

## 9. The Key Design Principle

The mistake is not "too much structure."
The mistake is "structure at the wrong level."

We should remove domain assumptions, not remove all execution structure.

Good structure:

- deterministic scheduler
- typed effects
- stable action ids
- replayable traces
- bounded execution

Bad structure:

- assuming all semantics are piece-local
- assuming derived relations are orthodox chess geometry
- assuming action generation and query semantics are derivable from hard-coded
  rook/bishop/knight rules
- assuming the objective is permanently fused into piece logic

## 10. Authoring Language Versus Execution IR

The LLM-facing language can be relatively open-ended.

The execution kernel cannot.

So we should explicitly separate:

- an expressive authoring/update layer
- from a canonical executable IR

That gives us:

- flexible reasoning and repair at the language layer,
- deterministic execution at the kernel layer,
- and canonical semantics for hashing, replay, and training.

In other words:

- the LLM should think in a rich domain language,
- but the kernel should run a canonical world-model IR.

## 11. Search And Learning Interface

This framing has direct consequences for the inner loop.

The policy/value model should not just see:

- board state
- plus a tiny DSL summary

It should see:

- current world state
- executable world-model semantics
- candidate actions
- objective semantics
- and possibly trace-derived semantic features

This likely implies:

- structured program encoding
- language-model-like components for semantics
- cross-attention between state, action, and world-model representation
- exact simulator still authoritative underneath

MCTS remains the planner.
The model remains the learned guide.
But the guide must interpret executable semantics, not just board geometry.

## 12. Outer-Loop Learning Story

The outer loop should be described as world-model maintenance.

At a high level:

1. The LLM receives a natural language description, prior programs, and current
   regression history.
2. It proposes or patches the current executable world model.
3. The kernel compiles and runs that world model.
4. The system compares predicted traces against observed traces.
5. Mismatches are localized and fed back to the LLM.
6. The repaired world model becomes the current simulator for later planning and
   training.

This is closer to the intended thesis than "compile a new magical rook."

## 13. Why PixieChess Is The Right Domain

PixieChess is useful because it is constrained enough to be executable and
evaluatable, but rich enough to require genuine semantic generalization.

It sits in the right middle zone:

- not pure language with no exact simulator,
- not fixed chess with no semantic novelty,
- and not a totally open physical world.

That makes it a good proving ground for:

- LLM world-model maintenance
- exact executable semantics
- and AlphaZero-style planning inside an inferred simulator

## 14. Current Domain Instantiation

For the next phase, the world model should still instantiate to:

- `8x8` board topology
- alternating turns
- two players
- capture-the-king objective
- king-safety-aware move legality

Those are current domain settings, not the final abstraction.

This keeps the scope bounded while we generalize the semantics layer.

## 15. Immediate Design Consequences

This reframing changes how follow-up design should proceed.

We should not start from "what is the perfect `PieceProgram` API?"

We should start from:

- what is the minimal execution kernel?
- what belongs in the executable world model?
- what are the first module types inside that world model?
- what generic queries does search require?

In practice, that means:

- `PieceProgram` becomes a subtype of entity program,
- global rules and objective semantics become first-class programs too,
- and legality / query / terminal evaluation must eventually come from the
  world-model query layer rather than legacy chess code.

## 16. Summary

The strongest version of the project is:

- LLM maintains a domain-specific executable world model
- deterministic kernel executes that world model
- RL+MCTS search inside the current executable world
- observations drive world-model repair over time

This is the design we should optimize for.

The important tradeoff is:

- keep the language-facing side flexible enough to support true semantic
  generalization,
- while keeping the execution kernel strict enough for determinism, replay, and
  search.

That is the right balance if the goal is not just to support magical chess
pieces, but to demonstrate LLM-guided RL over continually updated executable
world models.
