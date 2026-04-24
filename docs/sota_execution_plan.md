# SOTA Execution Plan

## 1. Purpose

This document defines the execution order for turning the current PixieChess
stack into a serious state-of-the-art candidate on the specific problem it is
actually positioned to win:

> rapid adaptation under changing executable game rules, measured under fixed
> search budgets and held-out mechanic families.

This is not a generic "make the model bigger" plan.
It is a benchmark-first, ablation-heavy, claim-disciplined plan.

This document composes and sequences the existing domain plans:

- [implementation_plan.md](/Users/calmdentist/conductor/workspaces/pixiechess_solver/algiers/implementation_plan.md)
- [docs/training_curricula_plan.md](/Users/calmdentist/conductor/workspaces/pixiechess_solver/algiers/docs/training_curricula_plan.md)
- [docs/strategy_conditioned_search_implementation_plan.md](/Users/calmdentist/conductor/workspaces/pixiechess_solver/algiers/docs/strategy_conditioned_search_implementation_plan.md)
- [docs/world_conditioned_hypernetwork_implementation_plan.md](/Users/calmdentist/conductor/workspaces/pixiechess_solver/algiers/docs/world_conditioned_hypernetwork_implementation_plan.md)

## 2. The Claim

The target claim is:

> on verified executable worlds with changing rules, the PixieChess stack
> adapts faster and reaches higher strength at fixed search budget than simpler
> search-only, world-conditioned, and prompt-only baselines.

The first paper-quality claim should be limited to:

- perfect-information turn-based worlds,
- exact legal move generation and transitions,
- held-out mechanic families,
- abrupt rule changes between short blocks or games,
- fixed evaluation search budgets.

Do not claim "general intelligence" or "any world" from this result.

## 3. North-Star Metrics

Primary metrics:

- adaptation AUC after a world change,
- games-to-recover to a fixed strength threshold after a world change,
- win rate at matched search budget,
- win rate at matched wall-clock budget where practical,
- held-out family win rate,
- held-out family policy cross-entropy,
- held-out family value error,
- uncertainty calibration error,
- outer-loop repair success on verified world tasks.

Secondary metrics:

- foundation-world regression,
- recent-admission-world regression,
- search efficiency at equal strength,
- token / time cost per admitted world model,
- strategy refresh rate and strategy usefulness.

## 4. Locked Decisions

### 4.1 Benchmark before extra architecture

No new architecture work counts unless it lands inside a benchmark harness that
can prove whether it helped adaptation under fixed budgets.

### 4.2 Search stays authoritative

The simulator and search remain the source of truth for legality, transitions,
and tactical correction.

### 4.3 Strategy must be explicit before hypernetwork expansion

Implement a real strategy-conditioned `v3` before spending further complexity on
`v4`.

### 4.4 Uncertainty must be trained before it is trusted

The uncertainty head must get explicit training targets before adaptive search
is treated as a meaningful result.

### 4.5 Kill weak branches early

If `v3` does not beat `v2` on adaptation, pause strategy complexity.
If `v4` does not beat `v3` on adaptation or matched-budget strength, pause
hypernetwork complexity.

## 5. Required Baselines

Every serious run should compare against:

1. `search_only`
2. `baseline_v1`
3. `world_conditioned_v2`
4. `strategy_conditioned_v3`
5. `hypernetwork_conditioned_v4`

Optional external baselines:

- prompt-only LLM move selection,
- prompt-to-code world model plus planner,
- search-only with stronger handcrafted evaluator.

The internal baselines are mandatory.

## 6. Phase Plan

## Phase 0: Benchmark Harness And Metadata

Goal:

- define the benchmark tightly enough that "SOTA" can be evaluated honestly.

Deliverables:

- fixed world-family splits,
- abrupt world-change evaluation episodes,
- world metadata stamped into replay and checkpoints,
- one benchmark CLI and one canonical report format.

Required work:

- extend replay metadata with:
  - `world_model_digest`
  - `family_id`
  - `split`
  - `novelty_tier`
  - `admission_cycle`
  - `strategy_digest`
  - `search_budget`
  - `model_architecture`
- add benchmark manifests for:
  - foundation worlds
  - known-mechanic worlds
  - recent-admission worlds
  - composition worlds
  - held-out seen-family parameter splits
  - held-out family splits
- add a benchmark runner that produces:
  - win/loss/draw
  - policy CE
  - top-1 agreement
  - value error
  - calibration
  - adaptation curves

Primary files:

- `src/pixie_solver/training/dataset.py`
- `src/pixie_solver/training/selfplay.py`
- `src/pixie_solver/training/checkpoint.py`
- `src/pixie_solver/cli/main.py`
- new `src/pixie_solver/eval/benchmark.py`
- new `data/benchmarks/`

Exit criteria:

- a single command can evaluate any checkpoint on all benchmark suites,
- no training example can be missing world-family metadata,
- held-out suites are impossible to accidentally sample into training.

## Phase 1: Stronger Learning Targets

Goal:

- make the current learning objectives fit fast adaptation instead of only
  eventual self-play convergence.

Required changes:

- train value on a blended target:
  - search root value,
  - optional deeper reanalysis value,
  - final outcome
- keep policy target as search visit distribution
- add auxiliary targets:
  - trigger count
  - event-chain length bucket
  - material delta
  - legal-reply count
  - king-danger delta
- add optional per-example weights for recent-admission worlds

Primary files:

- `src/pixie_solver/training/train.py`
- `src/pixie_solver/training/dataset.py`
- `src/pixie_solver/training/selfplay.py`
- `src/pixie_solver/model/policy_value.py`
- `src/pixie_solver/model/policy_value_v2.py`

Exit criteria:

- training can use blended value targets without breaking old checkpoints,
- auxiliary targets are logged and ablatable,
- `v2` improves on recent-admission adaptation over the old objective.

## Phase 2: World-Aware Replay And Curriculum

Goal:

- force the model to adapt under world changes without catastrophic forgetting.

Required changes:

- implement the replay buckets from
  [docs/training_curricula_plan.md](/Users/calmdentist/conductor/workspaces/pixiechess_solver/algiers/docs/training_curricula_plan.md):
  - foundation
  - known-mechanic
  - recent-admission
  - composition
- reload the active verified world pool every cycle
- add abrupt world-switch blocks in self-play
- temporarily increase self-play search budget for newly admitted worlds
- keep evaluation budget fixed

Primary files:

- `src/pixie_solver/training/train.py`
- `src/pixie_solver/training/selfplay.py`
- `src/pixie_solver/cli/main.py`
- verified world registry plumbing

Exit criteria:

- train-loop summary reports bucket-level metrics,
- recent-admission worlds are explicitly upsampled,
- world changes happen often enough that adaptation curves are meaningful.

## Phase 3: Strategy-Conditioned `v3`

Goal:

- make strategy a real model input instead of a metadata field.

Required changes:

- implement structured strategy tokens,
- add strategy tokens to the context stream,
- optionally add:
  - strategy-alignment head
  - strategy-progress head
- keep MCTS targets as the only policy/value supervision truth
- allow bad-strategy robustness evaluation

Primary files:

- `src/pixie_solver/strategy/encoder.py`
- `src/pixie_solver/model/policy_value_v3.py`
- `src/pixie_solver/model/board_encoder.py`
- `src/pixie_solver/search/mcts.py`
- `src/pixie_solver/training/selfplay.py`
- `src/pixie_solver/training/train.py`

Exit criteria:

- `v3` beats `v2` on recent-admission and held-out-family adaptation at fixed
  simulations,
- bad-strategy injection does not collapse search,
- strategy usefulness is visible in benchmark reports.

Kill criterion:

- if `v3` does not show adaptation gains over `v2`, stop adding strategy
  complexity and return to objective / benchmark work.

## Phase 4: Uncertainty And Adaptive Search

Goal:

- make adaptive search real instead of cosmetic.

Required changes:

- define uncertainty targets:
  - absolute value error against deeper search,
  - disagreement under reanalysis,
  - or bucketed "needs-more-search" labels
- train the uncertainty head explicitly
- evaluate:
  - calibration error
  - strength at matched average simulation budget
  - strength at matched wall-clock budget

Primary files:

- `src/pixie_solver/model/policy_value_v4.py`
- `src/pixie_solver/training/train.py`
- `src/pixie_solver/search/mcts.py`
- benchmark reporting

Exit criteria:

- uncertainty correlates with deeper-search correction,
- adaptive search improves matched-budget strength,
- fixed-budget strength is not degraded materially.

## Phase 5: Hypernetwork `v4` As An Honest Ablation

Goal:

- prove whether compiled specialization actually helps.

Required changes:

- keep adapters small and frozen per world/strategy digest,
- compare:
  - tokens only
  - adapters only
  - tokens + adapters
- log compile cost, cache hit rate, and bundle reuse

Primary files:

- `src/pixie_solver/hypernet/*`
- `src/pixie_solver/model/policy_value_v4.py`
- `src/pixie_solver/training/checkpoint.py`
- benchmark reporting

Exit criteria:

- `v4` beats `v3` on at least one thesis metric:
  - adaptation speed
  - matched-budget strength
  - held-out-family transfer
  - uncertainty calibration coupled to adaptive search

Kill criterion:

- if `v4` does not beat `v3`, treat the hypernetwork as a negative result and
  do not expand it further before publication-quality evidence exists.

## Phase 6: Outer-Loop World-Model Proof

Goal:

- close the loop between repairable world models and inner-loop adaptation.

Required changes:

- implement O0-O4 outer-loop tasks,
- enforce admission gates before RL exposure,
- version repaired worlds into the verified registry,
- run compile / repair / verify / admit inside train-loop boundaries

Primary files:

- curriculum pipeline
- verified registry tooling
- LLM compile / repair integration
- `src/pixie_solver/cli/main.py`

Exit criteria:

- newly admitted verified worlds are reloaded automatically into later cycles,
- repair success and regression survival are reported beside inner-loop metrics,
- held-out outer-loop family results are isolated cleanly.

## Phase 7: Paper-Grade Benchmark Runs

Goal:

- produce final evidence, not only infrastructure.

Required runs:

- all five internal baselines on all suites,
- adaptation-speed runs with abrupt world changes,
- held-out parameter runs,
- held-out family runs,
- strategy bad-advice robustness,
- uncertainty calibration and adaptive-search compute tradeoff,
- compile / repair outer-loop benchmark.

Artifacts:

- benchmark manifests,
- raw JSON reports,
- aggregate tables,
- adaptation plots,
- ablation tables,
- exact commands and seeds.

Exit criteria:

- claims can be reproduced from saved manifests and checkpoint paths alone.

## 7. Immediate Two-Sprint Plan

Sprint 1:

- finish Phase 0 benchmark metadata and CLI,
- implement blended value targets,
- add auxiliary targets,
- upgrade replay metadata,
- add bucket-level reporting to train-loop.

Sprint 2:

- implement world-aware replay buckets,
- reload verified world pools each cycle,
- add abrupt world-switch self-play blocks,
- build `v3` strategy token path,
- run first `v2` vs `v3` adaptation benchmark.

Do not expand `v4` during these two sprints unless `v3` already exists and has
clean benchmark results.

## 8. File Ownership By Workstream

Benchmark and metadata:

- `src/pixie_solver/eval/`
- `src/pixie_solver/training/dataset.py`
- `src/pixie_solver/cli/main.py`

Objectives and replay:

- `src/pixie_solver/training/train.py`
- `src/pixie_solver/training/selfplay.py`

Strategy path:

- `src/pixie_solver/strategy/`
- `src/pixie_solver/model/policy_value_v3.py`

Adaptive search and uncertainty:

- `src/pixie_solver/search/mcts.py`
- `src/pixie_solver/model/policy_value_v4.py`

Hypernetwork path:

- `src/pixie_solver/hypernet/`
- `src/pixie_solver/model/policy_value_v4.py`

Outer loop:

- `src/pixie_solver/curriculum/`
- `src/pixie_solver/llm/`
- verified registry tooling

## 9. Go / No-Go Gates

The project is on-track for a serious result if all of the following become
true:

- `v2` with stronger targets beats the current objective on recent-admission
  worlds,
- `v3` beats `v2` on adaptation under fixed simulations,
- uncertainty becomes calibrated enough to improve matched-budget strength,
- `v4` shows at least one measurable gain over `v3`,
- foundation strength does not collapse,
- held-out-family transfer is nontrivial,
- outer-loop repair succeeds without leaking broken worlds into RL.

If these do not happen, reduce scope and publish the strongest negative or
partial result instead of inflating the claim.

## 10. Bottom Line

The fastest path to a real SOTA claim is:

1. define the benchmark cleanly,
2. strengthen learning targets,
3. force adaptation in the curriculum,
4. prove `v3`,
5. only then try to prove `v4`,
6. close the outer-loop repair story,
7. run hard ablations and fixed-budget evaluations.

The benchmark and ablations are the product.
The architecture only matters if it wins there.
