# EntityProgram Contract

## 1. Purpose

This document defines the lower-level contract for the current entity-program
layer under the broader executable-world-model framing in
`docs/executable_world_model_design.md`.

In current PixieChess, entity programs are piece programs.
That is a current domain instantiation, not the top-level architecture.

This document therefore answers a narrower question:

> given the broader `ExecutableWorldModel`, what should one entity program
> promise to the execution kernel?

## 2. Position In The Architecture

The project should now be understood as:

- `ExecutableWorldModel` as the top-level executable domain object
- entity programs as one module type inside that world model
- global rule programs, objective logic, and query logic as separate module
  families

So this contract is subordinate to the world-model design.
It is no longer the top-level design of the whole project.

## 3. Current Domain Scope

For the current PixieChess phase, the domain settings remain:

- `8x8` board topology
- alternating turns
- capture-the-king objective
- king-safety-aware legality

Within that domain, an entity program describes the local semantics of one
piece class.

## 4. EntityProgram Responsibilities

An entity program owns the local behavior of an entity class.

In current PixieChess, that means a piece program owns:

- movement semantics
- capture semantics
- query semantics that describe control, capturability, and other derived facts
- local-state semantics
- event-reaction semantics

An entity program does not own:

- turn scheduling
- terminal adjudication
- repetition or clocks
- direct arbitrary board mutation
- replay or hashing

Those remain kernel or higher-level world-model responsibilities.

## 5. Conceptual Contract

Conceptually:

```python
EntityProgram = {
    program_id,
    name,
    state_schema,
    constants,
    action_logic,
    query_logic,
    reaction_logic,
    metadata,
}
```

This is a local module inside the larger `ExecutableWorldModel`, not the
definition of the entire world.

## 6. Kernel-Facing Queries

The current entity-program contract is defined by three kernel-facing queries.

### 6.1 Action Query

```python
enumerate_actions(ctx: ActionContext) -> list[ActionIntent]
```

This answers:

- what selectable actions this entity proposes on its controller's turn
- with stable, typed arguments
- before higher-level legality filtering

Examples for PixieChess:

- move
- capture
- push-capture
- teleport
- promote
- activate ability

### 6.2 Derived Query Fact Query

```python
enumerate_query_facts(ctx: QueryContext) -> list[QueryFact]
```

This answers:

- what derived relations or semantic facts this entity contributes now
- under the current world state
- independent of player action selection

This query is important because legality- and planning-relevant semantics should
not remain hidden inside legacy chess geometry or hard-coded attack helpers.

For current PixieChess, one important query family is threat-like control over
locations. `ThreatContext` and `ThreatMark` remain as compatibility views for
that current domain-specific query family.

### 6.3 Event Query

```python
on_event(ctx: EventContext, event: Event) -> list[Effect]
```

This answers:

- how this entity reacts to state evolution
- how its local state changes
- and what follow-on effects it proposes

Chained triggers live here.

## 7. Context Objects

The entity-program layer currently needs three read-only contexts.

### 7.1 ActionContext

Contains:

- current world state
- current entity instance
- current program
- occupancy / topology views

### 7.2 QueryContext

Contains:

- current world state
- current entity instance
- current program
- optional query-family selector
- occupancy / topology views

This is intentionally distinct from `ActionContext` even if the initial fields
overlap. Derived queries are not just turn actions.

### 7.3 EventContext

Contains:

- current world state
- current entity instances
- owning entity id
- current program
- triggering event
- optional current trace frame

## 8. Output Objects

The entity-program layer interacts with the kernel through typed outputs.

### 8.1 ActionIntent

Player-selectable candidate.

### 8.2 QueryFact

Derived semantic query result, not a selectable action.

It should at least identify:

- query kind
- optional subject ref
- optional target ref

For current PixieChess, `ThreatMark` is a compatibility specialization that maps
to a query fact with:

- `query_kind = threat_kind`
- `subject_ref = source_entity_id`
- `target_ref = target_square`

### 8.3 Effect

Typed delta applied by the kernel.
Effects are the only way an entity program changes the world.

## 9. Current PixieChess Interpretation

Under the current domain settings:

- entity programs are piece programs
- the objective remains capture-the-king
- legality remains king-safety-aware
- terminal evaluation still belongs above the entity-program layer

That means a future king with special survival behavior is still compatible with
this design:

- the king entity program may react during event resolution
- the objective layer decides win/loss only after resolution settles

So special survival mechanics do not force terminal logic into the local piece
contract.

## 10. Why This Contract Exists

The point of this contract is not to freeze the entire project around pieces.

The point is to define the first local semantic module cleanly enough that:

- the execution kernel stays deterministic
- the LLM repair loop has a canonical patch surface
- and the planning model has a structured semantics interface

This is a lower-level contract, not the final research abstraction.

## 11. Summary

Lock the lower-level contract as:

- entity programs are local semantic modules inside a broader executable world
  model
- current piece programs are one instantiation of entity programs
- entity programs expose action, derived-query, and event-reaction queries
- they emit typed outputs, not arbitrary mutation
- the kernel and higher-level world model still own legality, scheduling, and
  terminal semantics

That is the right subordinate contract under the broader executable-world-model
design.
