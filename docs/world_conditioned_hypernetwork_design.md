# World-Conditioned Hypernetwork Design

## 1. Purpose

This document extends the executable-world-model project with a stronger inner
loop claim:

> under rapidly changing executable worlds, it may not be enough to let the
> policy/value model read world semantics as tokens.
> the model may need to compile those semantics into world-specific computation.

The goal is not "generate the whole network from language."
The goal is:

- keep a shared executor backbone,
- compile the current `ProgramIR` and `StrategyIR` into small world-specific
  adapters,
- and let exact search reuse that compiled specialization many times inside the
  current world.

This design sits on top of:

- [docs/executable_world_model_design.md](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/docs/executable_world_model_design.md)
- [docs/strategy_conditioned_search_design.md](/Users/calmdentist/conductor/workspaces/pixiechess_solver/new-york/docs/strategy_conditioned_search_design.md)

## 2. Thesis

In a fluid world, two things change:

- the current rules of the world,
- and the best strategic abstractions for acting under those rules.

Token conditioning helps the model read those changes.
But token conditioning alone may force the model to re-derive the same world
specialization at every evaluated state.

The stronger hypothesis is:

- the LLM maintains and repairs the current executable world model,
- the LLM or another strategist emits sparse `StrategyIR`,
- a small hypernetwork compiles `ProgramIR + StrategyIR` into an
  `AdapterBundle`,
- and the shared policy/value backbone uses those adapters to become a
  world-specialized executor until the world model changes again.

In this framing, the learned executor is not only world-aware.
It is world-specialized.

## 3. Core Position

The correct persistent object is still not the exact search tree.

The correct persistent objects are:

- the executable world model,
- the strategy hypotheses,
- the shared executor backbone,
- and optionally world-conditioned adapter memories.

The exact search tree remains local and ephemeral.

The hypernetwork does not replace:

- the LLM world-model maintainer,
- the strategist,
- the simulator,
- or exact search.

It is a compiler layer between semantics and execution.

## 4. Non-Goals

This design does not assume:

- full-network weight generation from raw text,
- per-node weight synthesis during search,
- direct LLM-to-weights generation at inference time,
- replacing structured token conditioning with only hypernetwork outputs,
- trusting the hypernetwork over the exact simulator,
- or mutating the executor while a search tree is in progress.

These are intentionally excluded in v1 because they are expensive, unstable, or
make attribution unclear.

## 5. The Right Separation

The model should see a joint context, but the ontology should remain separate.

### 5.1 `ProgramIR`

Represents the laws of the current world:

- entity programs,
- query programs,
- reaction logic,
- constants,
- objective logic,
- and world semantics.

### 5.2 `StateSnapshot`

Represents the current facts inside that world:

- occupancy,
- per-entity local state,
- global state,
- side to act,
- pending events,
- and current query-derived facts.

### 5.3 `ActionSet`

Represents the currently instantiated legal actions.

These are not abstract action schemas.
They are concrete bindings of world semantics to the current state.

### 5.4 `StrategyIR`

Represents sparse, high-level preferences for acting inside the current world.

### 5.5 `WorldContext`

The policy/value model conditions on a single joint context:

```python
WorldContext = {
    program_ir,
    state_snapshot,
    action_set,
    strategy_ir,
}
```

But these should not collapse into one canonical IR.

That separation is what keeps:

- repair clean,
- caching tractable,
- and rules distinct from ordinary state evolution.

## 6. Architecture Overview

The full stack becomes:

1. `LLM World-Model Maintainer`
2. `LLM Strategist`
3. `World Compiler Hypernetwork`
4. `World-Specialized Executor`
5. `Adaptive Exact Search`

### 6.1 World-Model Maintainer

Inputs:

- observations,
- mismatch traces,
- repair history,
- natural language descriptions.

Outputs:

- repaired `ProgramIR`,
- repaired world metadata.

This remains the outer loop.

### 6.2 Strategist

Inputs:

- current `ProgramIR`,
- current objective,
- opening state or recent trajectory,
- optional uncertainty diagnostics.

Outputs:

- `StrategyIR`

This remains sparse and advisory.

### 6.3 World Compiler Hypernetwork

Inputs:

- canonical `ProgramIR` tokens,
- canonical `StrategyIR` tokens,
- optional world-family metadata.

Outputs:

- `AdapterBundle`

The `AdapterBundle` contains small world-specific modifications to a shared
executor backbone.

### 6.4 World-Specialized Executor

Inputs:

- current `StateSnapshot`,
- current `ActionSet`,
- optional direct `ProgramIR` and `StrategyIR` tokens,
- compiled `AdapterBundle`.

Outputs:

- policy priors,
- state value,
- uncertainty estimate,
- optional auxiliary diagnostics.

### 6.5 Adaptive Exact Search

Inputs:

- authoritative simulator,
- world-specialized executor outputs.

Outputs:

- visit distribution,
- backed-up values,
- verified tactical corrections.

Search remains exact.
Only the evaluator is specialized.

## 7. Why Hypernetworks Here

The motivation is not novelty for its own sake.

The motivation is that changing worlds may require changing computation, not
just changing context.

Examples:

- in one world, geometry and long-range control matter most,
- in another, event chains and local state counters dominate,
- in another, a specific new piece changes which action patterns deserve early
  attention.

Pure token conditioning tells the model what world it is in.
Hypernetwork conditioning can reshape how the model computes inside that world.

That is why this architecture may be a better fit than a fixed shared backbone
alone.

## 8. Recommended Model Structure

The recommended v1 design is a hybrid:

- keep explicit `ProgramIR` and `StrategyIR` tokens in the context,
- add a hypernetwork that emits small adapter weights,
- and apply those adapters inside a shared backbone.

Do not choose between token conditioning and hypernetwork conditioning in v1.
Use both.

### 8.1 Shared Backbone

The shared executor backbone should still process:

- state/entity tokens,
- action tokens,
- optional program/strategy tokens,
- global summary tokens.

This remains the fast reusable core.

### 8.2 Hypernetwork Compiler

The hypernetwork should consume only canonical structured inputs:

- `ProgramIR`
- `StrategyIR`

It should not consume:

- raw prose,
- per-node board state,
- or raw replay text in v1.

This keeps the specialization stable and cacheable.

### 8.3 Adapter Injection Points

The initial adapter targets should be small and interpretable:

- per-layer FiLM-style scale and shift,
- low-rank residual adapters on attention and MLP blocks,
- attention bias terms,
- gating scalars for specific feature streams.

Avoid full layer replacement or full tensor generation.

### 8.4 Output Heads

The executor should still expose:

- policy head,
- value head,
- uncertainty head.

The uncertainty head matters because adaptive search needs a learned trigger for
where exact planning budget should be spent.

## 9. AdapterBundle Contract

Suggested v1 contract:

```python
AdapterBundle = {
    bundle_id,
    world_digest,
    strategy_digest,
    layer_modulations,
    attention_biases,
    low_rank_adapters,
    gating_values,
    metadata,
}
```

### 9.1 Required Fields

- `bundle_id`
- `world_digest`
- `layer_modulations`

### 9.2 Desiderata

The bundle should be:

- deterministic for the same canonical world and strategy,
- small enough to cache and checkpoint,
- safe to freeze during search,
- and easy to regenerate after repair.

## 10. Lifecycle

The compile-and-execute lifecycle should be:

1. current world model `W_t` is accepted as authoritative
2. strategist produces `S_t`
3. hypernetwork compiles `A_t = compile(W_t, S_t)`
4. search and acting use the shared executor with `A_t` frozen
5. environment transition is observed
6. if predicted and observed outcomes match sufficiently, continue using `A_t`
7. if mismatch is significant, trigger world-model repair
8. repaired world `W_{t+1}` and optional strategy `S_{t+1}` produce new
   `A_{t+1}`

This is the correct timescale.

The adapters should update when the world model changes, not at every search
node.

## 11. Repair Interaction

The repair loop should remain contradiction-driven.

When reality disagrees with the current world model:

- localize the mismatch,
- repair `ProgramIR`,
- optionally revise `StrategyIR`,
- and recompile the `AdapterBundle`.

The hypernetwork should not be asked to "fix" simulator mismatches on its own.
It is downstream of the world model.

That separation is important because it preserves causal clarity:

- world-model errors belong to the world-model maintainer,
- execution errors belong to the learned executor,
- tactical errors belong to planning or evaluation.

## 12. Adaptive Search Implications

This design pairs naturally with adaptive search.

The executor should provide:

- policy priors,
- values,
- uncertainty estimates,
- optional strategy-progress signals.

Search should increase budget when:

- uncertainty is high,
- the top candidate moves are close,
- tactical volatility is high,
- or strategy failure triggers are near.

Search can stay light when:

- uncertainty is low,
- the policy is decisive,
- and the position looks semantically routine under the current world.

This is the right complement to a world-specialized executor.

## 13. Training Story

The executor should still train on search-verified targets, not on hypernetwork
outputs directly.

Primary targets:

- MCTS visit distributions,
- search-backed root values,
- final outcomes.

Optional auxiliary targets:

- uncertainty calibration,
- strategy-progress prediction,
- world-family classification,
- adapter-usage diagnostics.

The hypernetwork and shared backbone should be trained jointly so that:

- the backbone learns general transferable game structure,
- and the hypernetwork learns how to specialize that structure for each world.

## 14. Why Not A Small Generic LLM

This architecture is LM-shaped in one important sense:

- it consumes structured symbolic inputs,
- and it conditions on language-like world and strategy descriptions.

But it should not be treated as a generic next-token model.

The inner loop requires:

- high-throughput inference,
- calibrated values,
- stable action priors,
- and tight integration with exact search.

That favors a specialized control model with LM-like semantics, not a generic
autoregressive language model.

The hypernetwork gives some of the flexibility people want from language models
without paying the full cost of putting an LLM in the inner loop.

## 15. Recommended V1

The narrow but serious first version should be:

- shared transformer backbone,
- explicit `ProgramIR` and `StrategyIR` tokens still present,
- hypernetwork conditioned only on canonical world and strategy inputs,
- low-rank or FiLM-style adapters only,
- one `AdapterBundle` per world,
- adapters frozen for the duration of a search tree,
- recompilation only on world repair or explicit strategy refresh,
- uncertainty head enabled for adaptive search.

This keeps the experiment interpretable.

## 16. Evaluation Criteria

The right comparisons are:

1. `world_conditioned_v2`
2. `strategy_conditioned_v3`
3. `hypernetwork_conditioned_v4`
4. `strategy + hypernetwork`

Key metrics:

- adaptation speed after world change,
- value calibration under new mechanics,
- policy quality under held-out mechanic families,
- search efficiency at equal strength,
- and robustness after repair events.

The key question is:

> does compiling the current world into adapter weights let the system adapt
> faster and plan more efficiently than token conditioning alone?

## 17. Risks

Main risks:

- the hypernetwork overfits to world identity rather than semantics,
- the adapters become too large and erase the shared backbone,
- training becomes unstable because specialization is too strong,
- the strategy input injects noisy or wrong specialization,
- exact search masks the actual contribution of the specialized executor.

These are why v1 should stay small and tightly ablated.

## 18. Success Condition

This design is worth pursuing if it shows all of the following:

- faster adaptation to new worlds than the current world-conditioned baseline,
- better performance at equal search budget,
- cleaner uncertainty signals for adaptive search,
- and gains that persist on held-out mechanic families.

If that happens, the result is stronger than "the model can read the world."

It would suggest the model can compile the world into its own computation.
