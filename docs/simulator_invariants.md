# Simulator Invariants

Even before the simulator exists, the core state contract already enforces these invariants:

- every active square is occupied by at most one piece instance,
- every piece instance references an existing piece class,
- piece instance ids match the keys in the state mapping,
- squares are normalized to lowercase algebraic notation,
- clocks are non-negative and fullmove count starts at 1,
- deterministic hashing is computed from canonical JSON serialization.

These invariants are the minimum contract that future move generation, hook resolution, and replay tooling must preserve.
