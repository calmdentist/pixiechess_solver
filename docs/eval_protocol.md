# Evaluation Protocol

The evaluation stack is still intentionally minimal, but the intended ordering is fixed:

1. validate DSL programs and replay known piece examples,
2. run simulator invariants on all generated states,
3. measure search-only tactical performance,
4. compare model-guided search against the search-only baseline,
5. track piece-repair regressions with replay fixtures.

Current status:

- deterministic replay verification exists,
- search-only and model-guided search can now be compared on fixed states,
- the tactical fixture runner is still deferred.

Early work should favor deterministic fixtures and explicit traces over broad benchmark coverage.
