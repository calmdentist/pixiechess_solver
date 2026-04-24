# Benchmark Manifests

Phase 0 uses JSON manifests to define fixed benchmark suites for model
evaluation.

For a serious run, start from
`data/benchmarks/phase0_suite_template.json` and replace the `frozen/*` paths
with the benchmark corpus you want to lock for the experiment.

An initial deterministic strategy-conditioned corpus is checked into
`data/benchmarks/frozen/phase0_serious_v0/manifest.json`.

## Manifest schema

```json
{
  "format_version": 1,
  "name": "phase0_suite",
  "description": "Optional free-form description.",
  "suites": [
    {
      "suite_id": "foundation",
      "description": "Optional suite description.",
      "games": "../runs/foundation_games.jsonl",
      "examples": "../runs/foundation_examples.jsonl",
      "filters": {
        "family_id": "foundation",
        "split": ["foundation", "train"]
      },
      "tags": ["foundation"]
    }
  ]
}
```

`games` and `examples` are both optional, but each suite must define at least
one of them.

- `games`: `SelfPlayGame` JSONL. Used for outcome summaries and can also supply
  examples when `examples` is omitted.
- `examples`: `SelfPlayExample` JSONL. Used for policy/value evaluation.
- `filters`: metadata filters applied to the loaded rows. Scalar filters match
  exact values; array filters match any listed value.
- `tags`: optional suite labels for downstream reporting.

## Intended suite taxonomy

Phase 0 should produce manifests that separate:

- `foundation`
- `known_mechanic`
- `recent_admission`
- `composition`
- `heldout_seen_family`
- `heldout_family`

All benchmark rows should already carry the canonical metadata fields stamped by
self-play and registry annotation:

- `world_model_digest`
- `family_id`
- `split`
- `novelty_tier`
- `admission_cycle`
- `strategy_digest`
- `search_budget`
- `model_architecture`

The checked-in `phase0_serious_v0` corpus currently uses two games per world and
deterministic world-specific strategy hypotheses so executor benchmarks actually
exercise strategy tokens instead of leaving them empty.
