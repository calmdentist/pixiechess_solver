# Strategy-Conditioned Search Design

## 1. Purpose

This document extends the executable-world-model project with a new claim:

> the LLM should not only maintain the executable world model.
> it should also act as a sparse high-level strategist that proposes
> abstractions, plans, and search guidance for the learned policy/value system.

The goal is to speed adaptation under changing rules without giving up exact
search or simulator correctness.

## 2. Thesis

Under frequently changing rules, self-play alone may be too slow.

The core missing capability is not only rule understanding.
It is rapid strategic adaptation.

Humans do not respond to a new game mechanic by waiting for millions of games of
experience. They form explicit strategy hypotheses:

- develop a certain piece early
- trade off a dangerous new piece
- avoid opening lines for a reflected attacker
- force event chains that trigger a local advantage

This design aims to test the analogous hypothesis:

- the LLM reads the current executable world model and current game context
- it emits a small number of strategic hypotheses
- the policy/value model conditions on those hypotheses
- MCTS tests them against the exact simulator
- the learned model absorbs successful strategic structure over time

## 3. Core Position

The right hierarchy is:

1. `ExecutableWorldModel`
2. `StrategyHypothesis`
3. `WorldConditionedPolicyValueModel`
4. `ExactSearch`

The LLM does not replace search.
The LLM does not replace the learned policy/value net.
The LLM does not directly mutate network weights online.

Instead:

- the LLM proposes strategic context
- the network becomes the fast executor of that context
- MCTS grounds everything in the exact current simulator

## 4. Non-Goals

This design does not assume:

- direct cross-attention from an LLM into raw model weights
- end-to-end differentiable training through the LLM
- per-move LLM calls
- replacing MCTS with a language planner
- trusting the LLM as the source of truth for tactics or legality

These may become research directions later, but they are not the first system to
build.

## 5. System Overview

### 5.1 World-Model Maintainer

Inputs:

- natural language descriptions
- probe mismatches
- replay traces
- repaired program history

Outputs:

- updated `ExecutableWorldModel`
- world-model metadata

This is the already-established outer loop.

### 5.2 Strategist

Inputs:

- current `ExecutableWorldModel`
- objective specification
- current state or opening state
- recent trajectory summary
- optional search diagnostics and uncertainty summaries

Outputs:

- `StrategyHypothesis` records

The strategist is sparse and high-level.
It may be called:

- once per game
- once per cycle for opening families
- or on explicit trigger conditions

It should not be called every move in the initial design.

### 5.3 Executor

The executor is the learned policy/value model.

Inputs:

- board/world state tokens
- program/world-model tokens
- strategy tokens

Outputs:

- action priors over legal moves
- value estimate
- optional auxiliary strategy-alignment outputs

### 5.4 Verifier

The verifier is exact MCTS over the current authoritative simulator.

Inputs:

- exact current `WorldState`
- legal actions from the simulator
- policy/value outputs from the executor

Outputs:

- visit distribution
- backed-up value
- optional strategy-quality diagnostics

MCTS remains the reality filter.

## 6. StrategyHypothesis Contract

The strategist should output a structured object, not free-form prose.

Suggested v1 contract:

```python
StrategyHypothesis = {
    strategy_id,
    title,
    summary,
    horizon,
    confidence,
    subgoals,
    action_biases,
    avoid_biases,
    target_relations,
    success_predicates,
    failure_triggers,
    refresh_triggers,
    metadata,
}
```

### 6.1 Required Fields

- `strategy_id`
- `summary`
- `confidence`

### 6.2 Structured Fields

#### `subgoals`

A list of intermediate state descriptions.

Examples:

- activate a specific piece
- open a file
- remove a high-threat opposing piece
- preserve a local state counter

#### `action_biases`

Action schemas the strategist wants search to consider earlier.

Examples:

- prioritize rook-file clearing moves
- prioritize forcing captures
- prioritize king evacuation

#### `avoid_biases`

Patterns the strategist considers dangerous.

Examples:

- avoid diagonal exposure against reflected control
- avoid trades that neutralize the only phasing attacker

#### `success_predicates`

Conditions that indicate the strategy is working.

Examples:

- target piece becomes active
- opponent king remains capturable in many lines
- dangerous piece removed

#### `failure_triggers`

Conditions under which the strategy should be abandoned or revised.

Examples:

- key piece captured
- required lane cannot be opened
- opponent’s counter-threat exceeds threshold

### 6.3 Scope

Each strategy should declare its intended scope:

- `opening`
- `middlegame`
- `tactical`
- `endgame`
- `world_specific`
- `generic`

This matters for reuse across changing worlds.

## 7. Strategy Tokens

`StrategyHypothesis` should be compiled into deterministic `StrategyToken`s for
the learned model.

Suggested token groups:

- strategy root token
- summary token
- confidence token
- one token per subgoal
- one token per action bias
- one token per avoid bias
- one token per success predicate
- one token per failure trigger

The initial encoder should stay structured and canonical, like `ProgramIR`.
Do not feed raw long-form LLM text into the model path in v1.

## 8. Model Integration

The learned model should not consume strategy as a post-hoc scalar.
Strategy should be a first-class context stream.

Recommended context streams:

- world tokens
- entity tokens
- program tokens
- semantic probe tokens
- strategy tokens

Recommended attention structure:

1. context transformer over all non-action tokens
2. action tokens cross-attend into the contextualized stream
3. shallow action-side self-attention
4. policy/value heads

This preserves the current `world_conditioned_v2` direction while extending the
context with strategy.

## 9. Search Integration

The first version should bias search only softly.

### 9.1 Root-Prior Bias

Use strategy tokens to influence model policy priors.
MCTS then consumes those priors as usual.

### 9.2 Optional Strategy Bonus

If needed, add a small non-authoritative bonus for strategy-consistent moves at
the root only.

This bonus must stay soft.
Search must remain able to reject bad strategies.

### 9.3 Multi-Hypothesis Support

The strategist may emit `K` candidate strategies.

Recommended v1 options:

- choose the top-confidence strategy only
- or evaluate a small mixture of `K <= 3`

Do not begin with a large strategy set.

## 10. How The Executor Learns Strategies

The executor learns to realize strategies through search-filtered training, not
through direct imitation of LLM outputs.

The correct training target remains:

- MCTS visit distribution
- final outcome / root value

The strategy acts as conditioning context.

That means the network sees examples like:

- world `W`
- state `S`
- strategy `P`
- search-preferred action distribution `pi`
- search/root value `v`

Over time it learns:

- which concrete moves implement strategy `P` in states like `S`
- which strategies correlate with promising values in worlds like `W`

The LLM proposal is not the ground truth.
Search is the ground truth.

## 11. Auxiliary Learning Signals

V1 should support optional auxiliary heads.

### 11.1 Strategy-Alignment Head

Predict whether a legal move is strategy-consistent.

This gives denser signal than policy supervision alone.

### 11.2 Strategy-Progress Head

Predict whether the current state is advancing or harming the active strategy.

### 11.3 Strategy-Selection Head

Given a state and several strategy hypotheses, predict which one is most likely
to lead to strong search outcomes.

This is useful for later multi-strategy arbitration.

## 12. Strategy Refresh

The strategist should not be called every move by default.

Recommended refresh triggers:

- start of game
- large world change
- major tactical collapse
- explicit failure trigger reached
- every `N` plies where `N` is large

For current PixieChess, start-of-game plus failure-trigger refresh is the right
v1 default.

## 13. Reuse Across Worlds

Strategies should be reusable more often than exact search trees.

That is one of the main motivations for this design.

Examples:

- “activate the phase rook early”
- “trade off the dangerous reflected attacker”
- “force capture chains to trigger the surge piece”

These are higher-level than exact state/action caches.

So the persistent memory object should eventually be:

- world-conditioned strategy memory
- not just exact tree memory

## 14. Data Contracts

Self-play examples should eventually carry:

- world-model digest
- active strategy id(s)
- strategy token digest
- optional strategy confidence
- optional strategy-alignment labels
- optional strategy-progress labels

This makes ablations and transfer studies possible.

## 15. Evaluation Plan

The system should be evaluated in stages.

### 15.1 Core Ablation

Compare:

- baseline
- world-conditioned only
- world-conditioned + strategy tokens

### 15.2 Adaptation-Speed Test

Introduce a new piece family every game or every short block.
Measure:

- policy quality after rule change
- value quality after rule change
- self-play strength after rule change
- games needed to recover performance

### 15.3 Held-Out Rule Families

Train on one set of mechanic families.
Evaluate on held-out families with the strategist enabled.

### 15.4 Bad-Advice Robustness

Inject deliberately poor strategies and verify:

- MCTS rejects them
- the network does not collapse
- strategy conditioning remains advisory, not brittle

## 16. Failure Modes

Major risks:

- LLM emits plausible but wrong strategies
- strategy tokens become noisy and hard to learn
- search follows strategy priors too strongly
- the learned model overfits verbal abstractions instead of durable semantics
- strategy generation cost dominates runtime

Mitigations:

- keep strategy sparse
- keep search authoritative
- use search-derived targets, not raw LLM imitation
- support strategy rejection and refresh
- log strategy success/failure explicitly

## 17. Recommended V1

The first implementation should be intentionally narrow.

### 17.1 Build

- `StrategyHypothesis` schema
- deterministic strategy token encoder
- strategy tokens added to `world_conditioned_v2`
- one active strategy per game
- search still exact and mostly unchanged

### 17.2 Train

- condition on strategy tokens
- target remains MCTS visit distribution and final/root value
- optional strategy-alignment auxiliary head

### 17.3 Evaluate

- adaptation speed after new pieces
- A/B against strategy-free model
- held-out family test if possible

## 18. Stronger Future Versions

Possible later upgrades:

- multi-strategy mixture
- search-time strategy arbitration
- strategy-conditioned replay prioritization
- activation summaries from the learned model fed back to the strategist
- cross-world semantic strategy memory
- uncertainty-aware strategist queries

These are later phases.
They should not block the first strategy-conditioned system.

## 19. Bottom Line

The right design is:

- LLM as world-model maintainer
- LLM as sparse strategist
- policy/value net as fast executor
- MCTS as exact verifier

If this works under frequently changing rules, it would be much stronger evidence
for LLM-guided adaptive intelligence than program-conditioned self-play alone.
