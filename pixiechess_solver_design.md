# pixiechess_solver_design.md

# PixieChess Solver Design

## 1. Goal

Design a **simple but powerful** PixieChess solver that can:

1. ingest special-piece descriptions,
2. convert them into executable piece programs in a DSL,
3. repair those programs over time when observed transitions disagree with predicted transitions,
4. and learn a strong move-prediction / value model conditioned on:
   - board state,
   - piece programs,
   - and piece instance state.

The guiding principle is:

> keep the learned system as simple as possible, and push rule ambiguity and semantic repair into a separate world-model loop.

The intended simplicity is closer in spirit to AlphaZero than to a heavily modular AlphaGo-style pipeline:
- no hand-coded strategic modules,
- no probabilistic runtime rules,
- no end-to-end learned rule inference inside the policy/value model,
- just an exact simulator, search, and a model that learns strategy under the current rule set.

---

## 2. Core Thesis

There are two different problems:

### A. Rule formalization / repair
What does each special piece actually do?

### B. Strategic play
Given the current rules, what is the best move?

These should be separated.

The solver should **not** try to jointly learn both with one monolithic end-to-end model.

Instead:

- use a frontier LLM via API to convert short natural-language piece descriptions into DSL programs,
- run the resulting simulator,
- detect mismatches between predicted and observed transitions,
- repair the DSL online using the LLM,
- and keep the move-selection model focused on strategy only.

This lets the learned model focus on things like:
- phasing pressure,
- exchange incentives,
- trigger-induced tactical threats,
- timer/counter-dependent planning,
- and rule-conditioned positional value.

---

## 3. High-Level Architecture

```text
Piece description
   ↓
LLM compiler → current DSL piece program
   ↓
Simulator
   ↓
Observed gameplay / official examples
   ↓
Mismatch detector
   ↓
LLM repair loop → updated DSL
   ↓
Verified simulator
   ↓
Self-play + search
   ↓
Policy/value training
   ↓
Inference: model + tree search
```

There are two loops:

### Loop 1: Rule loop
- English description → DSL
- simulator prediction
- mismatch detection
- LLM patch
- deploy updated DSL

### Loop 2: Strategy loop
- simulator defines legal actions and transitions
- search/self-play generates training data
- train policy/value network
- use model to guide search

The important constraint is:

> at runtime there is exactly one current executable DSL program per piece.

No multi-hypothesis DSL, no probabilistic rule execution, and no uncertainty branching inside the simulator for v1.

---

## 4. Why This Is the Right Simplification

A more ambitious design would try to:
- jointly encode DSL,
- infer rules from play inside the model,
- adapt latent concepts online while still uncertain about rules,
- and search over evolving rule beliefs.

That is interesting, but it makes the learning problem much less clean.

A simpler and stronger v1 is:

- **the simulator is the source of truth**
- **the LLM handles ambiguity outside the policy/value model**
- **the learned model only needs to play well under the current simulator**
- **the DSL is always crisp at any given time**

This is better for:
- sample efficiency,
- debugging,
- stable value learning,
- stable search targets,
- and forming reusable strategic abstractions.

---

## 5. Runtime / Simulator Design

## 5.1 Game state

The game state should consist of:

- standard chess board state
- active piece instances on board
- piece instance state
- event queue / hook resolution state
- global turn metadata

Suggested structure:

```python
GameState = {
    base_board_state,
    piece_instances,
    piece_instance_state,
    side_to_move,
    move_count,
    repetition_info,
    pending_events,
}
```

### 5.1.1 Base board state
This includes the usual AlphaZero-style information:
- piece occupancy
- color
- side to move
- castling rights if relevant
- repetition counters
- move count

### 5.1.2 Piece instances
Each active piece on the board has:
- piece class id
- color
- square
- instance variables

### 5.1.3 Piece instance state
This is critical for PixieChess.
Examples:
- cooldown timers
- counters
- charge levels
- flags
- once-per-turn activations
- memory fields for pieces that track a target or mode

---

## 6. DSL Design

Each special piece should be represented as a **program/class** in the DSL.

## 6.1 Piece program structure

A piece program should contain:

- `base_piece_type`
- `movement_modifiers`
- `capture_modifiers`
- `hooks`
- `conditions`
- `effects`
- `instance_state_schema`

Example conceptual shape:

```yaml
piece_id: war_automaton
base_piece_type: pawn

instance_state_schema: {}

movement_modifiers:
  - inherit_base

capture_modifiers:
  - inherit_base

hooks:
  - event: piece_captured
    condition:
      - forward_square_empty
    effect:
      - move_self_forward_one
```

## 6.2 Hooks

Hooks are how the simulator supports second-order updates after a move.

Examples:
- `on_move_commit`
- `on_piece_captured`
- `on_piece_died`
- `on_turn_start`
- `on_turn_end`
- `on_enter_square`
- `on_leave_square`

These hooks allow pieces like War Automaton to update the board after the initial move.

## 6.3 Effects

Effects should be simple primitives:
- move piece
- capture piece
- knock piece
- remove piece from board
- modify counter
- set flag
- emit event
- spawn effect zone
- change mode

## 6.4 Conditions

Conditions should also be primitive:
- square empty
- piece present
- edge of board
- piece color
- counter value
- cooldown zero
- event source/target type
- self on board

This is enough for most special pieces without making the language too large.

---

## 7. Rule Compilation and Repair

## 7.1 Initial compilation

When a new piece appears:
1. fetch 1-line piece description
2. call frontier LLM API
3. ask for DSL program
4. validate syntax and semantics
5. deploy as the current rule program

## 7.2 Why use an external frontier LLM

This is the right place to use a frontier LLM because:
- short piece descriptions are language-heavy and ambiguous,
- new pieces arrive online,
- rule repair is sparse and episodic,
- and this is not the main differentiating learned model.

So this module should be:
- external,
- online,
- repairable,
- and continuously updatable in production.

It does **not** need to be a model we train ourselves.

## 7.3 Mismatch detection

When gameplay is observed, compare:

- predicted next state from our simulator
- actual next state from PixieChess

If they differ:
- locate mismatch type
- identify implicated piece program
- generate structured diff
- send repair request to LLM

### 7.3.1 Example mismatch
Piece description: “Sumo rook knocks pieces back instead of capturing.”

Initial DSL hypothesis:
- piece gets pushed if square behind it is empty
- if edge reached, no effect

Observed transition:
- edge piece gets pushed off board

Repair:
- patch the DSL so “push beyond board” removes the target

## 7.4 Repair request format

The LLM should receive:

- piece description
- current DSL
- current board state
- move made
- predicted next state
- actual next state
- minimal diff summary
- known invariants

And return:
- patched DSL
- explanation of semantic change
- optional test cases

## 7.5 Verification after repair

Every patch must be validated by:
- DSL schema checks
- simulator checks
- replaying the mismatch example
- regression tests on earlier examples

## 7.6 Important design stance

Do **not** represent uncertainty inside the runtime DSL for v1.

At any point in time:
- there is one current executable piece program,
- the simulator is deterministic,
- the policy/value model trains against that exact environment.

This keeps training clean and sample-efficient.

---

## 8. Strategy Model: What Actually Needs to Be Learned

This is the most important part.

The learned model should answer:

> given the board state, piece programs, and piece instance states, what move is best and how good is the current position?

The model does **not** need to learn the rules procedurally from scratch.

The simulator already handles:
- legal move generation
- next-state transitions
- hook execution
- event cascades

The model’s job is to learn the **strategic consequences** of those rules.

Examples:
- “this rook phases through allies, so it threatens along its file even when screened”
- “capturing here activates the opponent’s war automaton”
- “this timer-based piece grows stronger if the position stays closed for two turns”
- “a knockback piece makes edge pressure more valuable than in orthodox chess”

This is the real target capability.

---

## 9. How This Differs From AlphaZero

AlphaZero assumes:

- fixed game rules
- fixed piece semantics
- fixed state representation
- fixed policy head over chess actions
- value function over a stationary game

PixieChess breaks several of these assumptions.

## 9.1 Fixed piece semantics no longer hold
The same board geometry can mean different things depending on active piece programs.

## 9.2 Some pieces have internal state
A piece may have:
- timers
- counters
- charge levels
- temporary modes

So position evaluation depends not just on board geometry but on piece-local runtime state.

## 9.3 Action semantics vary with rules
The move “Rook from a1 to a5” may:
- capture normally,
- push instead of capture,
- trigger a secondary event,
- or alter another piece’s state.

So a pure fixed policy tensor is a bad fit.

## 9.4 Position value is rule-conditioned
The value network must estimate:
- not “how winning is this board?”
- but “how winning is this board under these active piece programs and runtime states?”

---

## 10. Design Goal for the Learned Model

We want the model to stay **as simple as possible**, while still handling variable piece rules.

So the model should:

- operate on **board state**
- condition on **piece program embeddings**
- condition on **piece instance state**
- score **legal candidate moves**
- predict **position value**

That suggests a design simpler than a fully general symbolic-neural model, but more flexible than AlphaZero.

The model should learn features like:
- latent attack lines through allies for phasing pieces
- exchange incentives when death/capture triggers matter
- squares that are tactically unsafe only because of special mechanics
- delayed threats from timers and counters
- positional value shifts induced by nonstandard interaction rules

---

## 11. Proposed Policy/Value Model

Use a **transformer-based candidate-scoring policy/value network**.

It has three inputs:

### A. Piece program embeddings
Each active piece class gets embedded from its DSL program.

### B. Board / piece-instance tokens
Each occupied square is represented as a token containing:
- square
- color
- base piece type
- piece class id
- piece program embedding
- instance state embedding

### C. Legal move tokens
Instead of a fixed action head, enumerate legal moves from the simulator and score each move as a candidate.

This is the simplest clean generalization beyond AlphaZero.

---

## 12. Piece Program Encoder

We do not need a huge program interpreter inside the model.

A simple first design:

- normalize DSL into a compact canonical representation
- preserve field structure explicitly rather than collapsing everything into one flat string
- encode each major field separately, then pool/combine:
  - base piece type
  - movement modifiers
  - capture modifiers
  - hook types
  - condition/effect templates
  - instance state schema
- produce one embedding per piece class with a small transformer or MLP-based combiner

This is intentionally more structured than a raw text encoder, but much simpler than a full AST or graph encoder.

### 12.1 Why this is enough initially
The simulator already handles exact rule execution.
The model only needs a compact semantic embedding to condition strategy.

So the piece program encoder does **not** need to reconstruct exact execution internally.
It only needs to help the policy/value net understand:
- mobility profile
- tactical volatility
- delayed threats
- capture semantics
- event-triggered potential
- dependence on counters/timers

---

## 13. Board-State Encoder

Represent the board as tokens.

Each token corresponds to:
- one occupied square
- optionally special global tokens

### 13.1 Per-piece token features
Each piece token contains:
- square embedding
- color embedding
- base piece type embedding
- piece program embedding
- instance state embedding

### 13.2 Global tokens
Add a small number of global tokens:
- side to move
- castling / repetition / move count if needed
- total material summary
- active piece-class set summary

This lets the transformer reason over interactions between:
- board geometry
- piece semantics
- piece-local state

This is where the model should form abstractions like:
- “screened but not safe”
- “capture activates latent threat”
- “quiet move preserves future trigger”
- “edge pressure is stronger because of knockback mechanics”

---

## 14. Instance State Encoder

Some pieces require additional runtime information.

Encode instance state as:
- counters
- timers
- boolean flags
- mode embeddings

This can be:
- a small MLP over numeric state
- concatenated with piece token
- projected into model dimension

This is critical because in PixieChess the piece’s future power may depend on internal state.

---

## 15. Policy Head: Candidate Move Scoring

This is the biggest change from AlphaZero.

Instead of predicting over a fixed move tensor, do:

1. simulator enumerates legal moves
2. construct one token per legal move
3. score all legal move tokens
4. softmax over candidate scores

### 15.1 Move token features
Each move token should include:
- source square
- destination square
- moving piece token reference / embedding
- captured target type if any
- whether the move triggers hooks
- whether it changes piece instance state
- a shallow one-step consequence summary from simulator

### 15.2 One-step consequence summary
For each legal move, run a cheap 1-ply simulator update and extract:
- material delta
- king safety delta
- trigger count
- event-chain length
- affected squares summary
- whether a piece is removed/pushed/spawned
- instance-state deltas

This gives the policy head a strong signal without requiring the network to internally simulate everything from scratch.

This is a very high-leverage simplification.

### 15.3 Why candidate scoring is the right simplification
It avoids the complexity of designing a universal action tensor across arbitrary rule semantics.

The simulator already knows the legal move set, so let the model score candidates rather than invent them.

---

## 16. Value Head

The value head predicts:
- probability of win / draw / loss
- or scalar expected outcome

Input:
- pooled board representation
- global summary token
- active piece-program summary
- optional volatility features

The value head should be conditioned on:
- board state
- piece programs
- piece instance states

That is the clean minimal generalization of AlphaZero’s value head.

---

## 17. Model Forward Pass

Conceptually:

```text
DSL piece programs
   → piece program encoder
   → piece class embeddings

Current board state + piece instance states
   → piece tokens using piece class embeddings
   → board transformer
   → contextual board representation

Legal moves from simulator
   → move token builder
   → candidate move scorer conditioned on board representation

Outputs:
- policy over legal moves
- value of current position
```

This is simple, modular, and strong.

---

## 18. Search

Use **PUCT/MCTS** first.

The game is still an exact tree because:
- legal moves are enumerable,
- transitions are exact,
- terminal conditions are defined.

So there is no reason to abandon tree search in v1.

## 18.1 Why keep MCTS
It gives:
- tactical grounding
- robustness against model mistakes
- search targets for training
- AlphaZero-like simplicity

## 18.2 Modifications relative to AlphaZero
The main changes are not in the tree algorithm itself.
They are in:
- simulator semantics
- candidate-scoring policy head
- rule-conditioned inputs
- optional trigger-aware expansion heuristics

### 18.2.1 Trigger-aware search extensions
Some moves are unusually important because they cause:
- event cascades
- forced knockback sequences
- stateful power changes

So later we may add:
- deeper search on high-volatility lines
- quiescence-like treatment of trigger-rich branches

But v1 should keep the tree logic simple.

---

## 19. Training Data

Training examples come from self-play with search.

Each example should contain:
- board state
- active piece programs
- piece instance states
- legal move list
- search visit distribution over legal moves
- final game outcome

This is exactly the AlphaZero recipe, but with richer input structure and candidate-based actions.

---

## 20. Losses

Keep the loss simple.

## 20.1 Policy loss
Cross-entropy between:
- predicted policy over legal candidates
- target visit distribution from MCTS

```text
L_policy = CE(pi_search, pi_model)
```

## 20.2 Value loss
MSE or cross-entropy depending on value target choice.

If using scalar outcome in `[-1, 0, 1]`:
```text
L_value = MSE(v_pred, z)
```

If using win/draw/loss logits:
```text
L_value = CE(outcome_logits, outcome_target)
```

## 20.3 Optional auxiliary losses
Only add if useful.

Good simple auxiliaries:
- next-turn trigger count prediction
- material delta after best move
- event-chain length class
- king danger estimate

These can help the model learn tactical semantics faster.

But v1 can start without them.

## 20.4 Total loss
Keep the basic AlphaZero structure:

```text
L_total = L_policy + λ_value * L_value + λ_reg * L_reg
```

Where:
- `L_reg` is weight decay or standard regularization
- `λ_value` can start at 1.0

---

## 21. Training Process

## 21.1 Phase 1: Build simulator + rule loop
- implement DSL
- implement LLM compiler
- implement mismatch detector
- implement repair loop
- validate on 5–10 special pieces

## 21.2 Phase 2: Search-only baseline
- run MCTS using exact simulator
- no learned policy/value yet or very weak initialization
- verify tactics and rule interactions

## 21.3 Phase 3: Generate self-play data
- sample piece loadouts
- run self-play with search
- store visit distributions and outcomes

## 21.4 Phase 4: Train policy/value model
- train on self-play data
- evaluate on held-out positions and tactical suite
- plug into search as prior/value

## 21.5 Phase 5: Iterative self-play improvement
- use updated model to guide search
- generate stronger self-play data
- retrain

## 21.6 Phase 6: Production rule updates
When new pieces appear or rule mismatches are found:
- patch DSL via LLM repair loop
- update simulator
- continue self-play and adaptation

This keeps the rule system online and the strategy model grounded.

---

## 22. Why This Design Is Powerful

This design is powerful because it combines:

- exact rule execution
- online rule repair
- simple AlphaZero-style self-play
- flexible conditioning on variable piece semantics
- candidate-move scoring instead of brittle fixed action tensors

It avoids:
- trying to learn the rules implicitly from scratch
- probabilistic runtime semantics
- huge end-to-end multimodal complexity
- hand-coded piece-specific strategy modules

---

## 23. Why This Design Is Still Simple

It is simple because:

- the rule compiler is externalized to an API LLM
- the simulator is exact and modular
- the policy/value model is one main network
- the runtime DSL is always crisp
- the loss is just policy + value
- search remains basically AlphaZero-style
- the main innovation is in the input representation and candidate scoring

That is the right kind of simplicity.

---

## 24. Minimal v1 Model Specification

### Inputs
- piece program embeddings for active piece classes
- board tokens for all occupied squares
- piece instance state features
- legal move candidate tokens

### Core
- transformer over board tokens + global tokens
- cross-attention or concatenation from move tokens to board context

### Outputs
- score for each legal move
- scalar or categorical value

### Search
- PUCT/MCTS over exact simulator

### Training
- self-play
- policy imitation from visit counts
- value prediction from final outcomes

This is enough for a serious first system.

---

## 25. Key Open Questions

### 25.1 How rich should the piece program encoder be?
Start with a structured normalized DSL encoder.
Only move to AST/graph encoders if experiments show that rule generalization is bottlenecked by the encoder.

### 25.2 Should move scoring use shallow simulator summaries?
Likely yes.
This seems like a high-leverage simplification.

### 25.3 Should we add auxiliary tactical losses?
Only if training is data-inefficient.

### 25.4 Should MCTS be replaced?
Not initially.
Keep it.

---

## 26. Recommended Final Design

If forced to choose the cleanest v1:

### Rule side
- LLM API compiles/repairs piece programs in DSL
- simulator executes exact rules
- mismatch triggers an LLM patch to the DSL

### Model side
- rule-conditioned transformer
- structured normalized DSL encoder for piece programs
- board tokens + piece instance states
- candidate-move scoring head
- value head

### Search side
- AlphaZero-style PUCT/MCTS
- optional trigger-aware tweaks later

### Training side
- self-play
- policy/value loss
- iterative improvement

This is the simplest architecture that still captures what makes PixieChess different.

---

## 27. Final Summary

The central idea is:

> Keep rule understanding outside the learned model as much as possible, and let the learned model specialize in strategic evaluation under a repaired, executable world model.

That leads to a very clean system:

- **LLM for rule formalization and repair**
- **DSL simulator for exact next-state generation**
- **simple but expressive rule-conditioned policy/value network**
- **MCTS for tactical grounding**

The biggest departure from AlphaZero is not the search algorithm.
It is the move from:
- fixed piece semantics,
- fixed board interpretation,
- fixed move head

to:
- piece-program-conditioned representations,
- piece-instance-state-aware evaluation,
- and candidate-move scoring over legal moves.

And the most important thing the model should learn is not procedural rule execution itself, but strategic features like:
- “a phasing rook threatens through allies,”
- “captures may be bad because they activate reactive pieces,”
- “nonstandard movement changes what counts as a safe square,”
- and “piece-local timers/counters reshape positional value.”

That is the right balance of simplicity and power.
