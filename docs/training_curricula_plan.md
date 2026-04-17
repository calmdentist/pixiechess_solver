# Training Curricula Plan

## 1. Purpose

This document defines the training curricula needed to support the actual
thesis of the project:

> an LLM maintains and repairs an executable world model, and
> RL + MCTS learn strong behavior inside that changing world model.

The key point is that there are really three curricula, not one:

- an **outer-loop world-model curriculum**
- an **inner-loop policy/value curriculum**
- an **evaluation curriculum** that prevents us from fooling ourselves

These curricula must be coordinated, but they should not be collapsed into a
single metric.

## 2. What "Solid" Means

For this project, a solid training curriculum means:

- newly introduced mechanics enter through a controlled schedule
- the LLM is tested on compile and repair, not only compile-only generation
- admitted world models are versioned and become part of training
- the policy/value net sees a balanced progression from familiar to novel worlds
- search is used as a teacher, not as a crutch that hides failed generalization
- fixed held-out evaluations exist at every stage

If any of those are missing, the system may still run, but it is not yet a
credible proof harness.

## 3. Design Principles

### 3.1 Separate world-model correctness from policy strength

The outer loop should be evaluated on whether the inferred world model matches
the executable teacher world.

The inner loop should be evaluated on whether the agent becomes strong inside
the active world distribution.

These are related, but they are not the same.

### 3.2 Do not expose broken worlds to RL training

A candidate world model should only enter the training world pool after:

- compile / validation success
- probe success or repair success
- regression survival
- admission into the verified registry

The policy/value net should train on verified worlds, not raw failed LLM
outputs.

### 3.3 Hold out mechanics, not just seeds

A serious proof requires held-out semantics:

- held-out parameter settings
- held-out mechanic compositions
- and eventually held-out mechanic families

Changing random seeds inside the same mechanic family is not enough.

### 3.4 Keep a foundation distribution alive

If the world distribution keeps changing, the inner loop will destabilize unless
we keep a stable fraction of training on simpler worlds.

The curriculum should always retain:

- standard/base worlds
- earlier admitted worlds
- and recent worlds

### 3.5 Evaluate transfer under fixed search budgets

If search budget grows whenever the world gets harder, we will not know whether
the learned model improved or whether MCTS simply did more work.

## 4. The Three Curricula

### 4.1 Outer Loop Curriculum

This curriculum trains and evaluates the LLM as an executable world-model
maintainer.

Its unit is a `WorldModelTask`:

```python
WorldModelTask = {
    task_id,
    family_id,
    split,
    description,
    teacher_world_model,
    diagnostic_probes,
    rollout_fixtures,
    regression_fixtures,
    metadata,
}
```

Current synthetic pieces are a narrow special case of this object.

### 4.2 Inner Loop Curriculum

This curriculum trains the policy/value model and search stack under a changing
distribution of verified worlds.

Its unit is a `TrainingWorldBucket`:

```python
TrainingWorldBucket = {
    bucket_id,
    world_model_digests,
    difficulty_tier,
    novelty_tier,
    sampling_weight,
}
```

### 4.3 Evaluation Curriculum

This curriculum defines what is never trained on directly and what gates
progress.

Without this layer, train-loop metrics will overstate success.

## 5. Outer Loop Curriculum

## 5.1 Goal

The outer loop should prove:

- compile from vague description
- detect mismatch from observations
- repair the executable world model
- survive regressions after repair
- version the repaired result into the registry

## 5.2 Curriculum Stages

### Stage O0: Legacy-compatible mechanics

Purpose:

- prove the compile / mismatch / repair / verify / admit pipeline end to end

Families:

- phasing
- push-capture
- event-triggered movement
- local counters / charge

Success criterion:

- high compile success
- high repair success after 1 to 2 mismatches
- near-perfect regression survival on prior probes

### Stage O1: Parametric hidden semantics

Purpose:

- make vague descriptions materially underspecified

Examples:

- hidden push distance
- hidden charge amount
- hidden edge behavior
- hidden trigger scope

Success criterion:

- model must infer hidden parameters from traces, not from text alone

### Stage O2: New query semantics

Purpose:

- introduce the first worlds whose legality / threat structure cannot be reduced
  to current legacy geometry

Examples:

- reflection
- bounce
- wraparound control
- delayed capture zones

Success criterion:

- compile and repair work on mechanics that depend on non-legacy query behavior

### Stage O3: Mechanic composition

Purpose:

- move from single isolated mechanics to worlds with interacting semantics

Examples:

- reflecting + phasing piece
- bounce + push effect
- local charge + triggered reactions

Success criterion:

- repair remains localized and does not regress previously correct behavior

### Stage O4: Held-out family generalization

Purpose:

- reserve some mechanic families for proof-level evaluation

Examples:

- do not train the LLM on one family such as reflection variants
- only evaluate compile and repair there after the rest of the curriculum is
  mature

Success criterion:

- nontrivial compile / repair success on families not seen during training or
  prompt exemplars

## 5.3 Split Design

Each mechanic family should have:

- `train`: families and parameters eligible for admission into the active
  registry
- `val`: families and parameters used for curriculum tuning
- `test_seen_family`: new parameters or compositions within seen families
- `test_heldout_family`: families excluded from training

The hardest mistake here would be to mix these splits inside the registry and
later lose the ability to state what was actually held out.

## 5.4 Admission Gate

A candidate world should only be admitted if it passes all of:

1. canonical compile / validation
2. diagnostic probe success
3. rollout agreement on a small fixed fixture set
4. regression survival on previously passed probes for the same family
5. metadata stamping with family, split, digests, and admission cycle

If repair succeeds on the immediate mismatch but breaks prior fixtures, it
should not be admitted.

## 5.5 Outer-Loop Metrics

Track at least:

- compile success rate
- compile success on first try
- repair success after `k` mismatches
- average mismatches to acceptance
- regression survival after repair
- rollout agreement on fixed fixtures
- time / token cost per accepted world

The primary metric is not "did the LLM produce code."
It is "did the system recover executable semantics reliably."

## 6. Inner Loop Curriculum

## 6.1 Goal

The inner loop should prove:

- strong play on the current verified world distribution
- graceful adaptation as world semantics evolve
- useful semantic conditioning in the learned policy/value model
- transfer to novel worlds under fixed search budgets

## 6.2 Training World Buckets

The world sampler should maintain four buckets:

### Bucket I0: Foundation worlds

- standard/base worlds
- no or minimal magical mechanics

Purpose:

- stabilize value learning
- preserve ordinary king-safety and tactical priors

### Bucket I1: Known-mechanic worlds

- admitted worlds from mechanic families already established in training

Purpose:

- strengthen competence under nontrivial but familiar semantics

### Bucket I2: Recent-admission worlds

- newly admitted worlds from the last `N` cycles

Purpose:

- force adaptation to changing semantics

### Bucket I3: Composition worlds

- worlds containing multiple special pieces or interacting mechanics

Purpose:

- pressure-test semantic compositionality

Held-out worlds do not belong to any training bucket.

## 6.3 Sampling Schedule

Recommended early schedule:

### Phase I0: Bootstrap

- foundation: `80%`
- known-mechanic: `20%`
- recent-admission: `0%`
- composition: `0%`

Use this while the world-conditioned model is still stabilizing.

### Phase I1: Controlled novelty

- foundation: `50%`
- known-mechanic: `35%`
- recent-admission: `15%`
- composition: `0%`

This is the first phase where newly admitted worlds enter RL training.

### Phase I2: Mixed-world competence

- foundation: `30%`
- known-mechanic: `40%`
- recent-admission: `20%`
- composition: `10%`

### Phase I3: Composition pressure

- foundation: `20%`
- known-mechanic: `35%`
- recent-admission: `25%`
- composition: `20%`

This should only happen after O2-level query semantics are stable.

## 6.4 Replay Strategy

The current train loop trains only on the current cycle's examples.
That is too brittle for evolving worlds.

The intended replay structure should keep:

- a rolling foundation buffer
- a rolling active-world buffer
- a recent-world challenge buffer

Recommended rule:

- retain the last `K` cycles for active-world replay
- upsample recent-admission worlds for the first few cycles after admission
- never let foundation worlds drop below a fixed floor

This is the simplest way to reduce catastrophic forgetting while still adapting
to new semantics.

## 6.5 Search Budget Curriculum

When new worlds are admitted, the search budget should temporarily be more
helpful than usual.

Recommended policy:

- newly admitted worlds get a higher self-play simulation budget for a short
  warmup window
- once policy/value metrics stabilize on those worlds, return them to the
  standard budget

This makes the early targets more reliable without permanently hiding model
weakness.

Do not change the evaluation budget when doing this.

## 6.6 Self-Play Progression

Within each cycle:

1. sample worlds from the current bucket mixture
2. generate self-play with the active champion or candidate model
3. write examples with world-model metadata
4. train on a balanced replay window
5. validate on fixed `val` worlds and recent admissions
6. optionally run arena promotion

The major missing feature in the current codebase is that the world pool should
be reloaded between cycles, not only once at startup.

## 6.7 Inner-Loop Metrics

Track at least:

- arena strength on foundation worlds
- arena strength on known-mechanic worlds
- arena strength on recent-admission worlds
- reduced-budget strength relative to search-only baseline
- policy cross-entropy / top-1 agreement per bucket
- value error per bucket
- model-guided search win rate at matched compute

The critical signal is not raw self-play loss.
It is whether the model becomes useful under changed semantics.

## 7. Evaluation Curriculum

## 7.1 Fixed Evaluation Suites

Maintain separate suites for:

- foundation strength
- seen-family generalization
- recent-admission adaptation
- held-out parameter generalization
- held-out family generalization

None of these should be sampled into training by accident.

## 7.2 Required Evaluation Modes

At minimum:

1. world-model compile / repair evaluation
2. deterministic replay verification
3. tactical / query fixture evaluation
4. reduced-budget model-guided search evaluation
5. arena strength evaluation

Today the biggest gap is that tactical and replay-inspector evaluation are still
stubs, so this curriculum must be implemented, not only specified.

## 7.3 Thesis-Level Pass Conditions

The first serious claim should require all of:

- no major regression on foundation strength
- strong repair success on seen mechanic families
- nontrivial transfer on held-out parameter settings
- model-guided search beats the baseline model on novel verified worlds at fixed
  simulation budgets
- some positive transfer on held-out mechanic families

If only search-only MCTS succeeds, that is still a simulator result, not the
full thesis result.

## 8. Combined Cycle Schedule

The intended steady-state cycle is:

```text
1. Select outer-loop tasks from the curriculum frontier
2. Compile / probe / repair / verify candidate worlds
3. Admit accepted worlds into the verified registry
4. Reload the active verified world pool
5. Sample self-play worlds from curriculum buckets
6. Generate self-play and append to replay buffers
7. Train the policy/value model
8. Run validation and arena gates
9. Advance curriculum frontier based on results
```

This gives one coherent experimental loop:

- outer loop expands and repairs the executable world
- inner loop learns inside the expanded executable world

## 9. Data and Metadata Requirements

Every admitted world and every replay row should be stamped with:

- `world_model_digest`
- `family_id`
- `split`
- `admission_cycle`
- `source` (`handauthored`, `synthetic`, `repaired`)
- `novelty_tier`
- `query_block_kinds`
- `action_block_kinds`

Without this, later bucketed evaluation and replay balancing become painful.

## 10. First Serious Experimental Ladder

Recommended order:

### Milestone T1: Closed outer-loop proof

- expand synthetic families modestly
- close compile -> repair -> verify -> admit inside cycle boundaries
- prove newly admitted worlds are reloaded in later cycles

### Milestone T2: Stable inner-loop adaptation

- add replay window and curriculum buckets
- show the new model improves over `baseline_v1` on recent-admission worlds
- hold foundation regressions below a chosen threshold

### Milestone T3: First genuinely new mechanic family

- add one non-legacy query/action family such as reflection
- run the full outer + inner loop on that family

### Milestone T4: Held-out proof run

- freeze the held-out family split
- run the full system end to end
- report compile, repair, and strength results under fixed budgets

This is the first milestone that really tests the thesis.

## 11. Immediate Implementation Priorities

The next implementation steps should be:

1. integrate registry reload into `train-loop` at cycle boundaries
2. add curriculum metadata to registry records and self-play examples
3. replace single-cycle-only training with a replay window and balanced sampler
4. expand the outer-loop curriculum generator beyond the current four recipe
   families
5. build fixed evaluation suites, especially tactical and held-out world suites

That ordering is deliberate.
It improves the proof harness before adding much more mechanic surface area.

## 12. Recommendation

The correct framing is:

- outer loop learns and repairs executable worlds
- inner loop learns to act inside the admitted world distribution
- evaluation proves generalization across world changes

The project should not jump straight to ever-more exotic mechanics until the
curriculum machinery above is in place.

Otherwise we will generate novelty faster than we can measure learning.
