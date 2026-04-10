# DSL Reference

This file is the short operator reference for the minimal DSL defined in [DSL.md](/Users/calmdentist/conductor/workspaces/pixiechess_solver/marseille/DSL.md).

## Top-Level Shape

```yaml
piece_id: ...
name: ...
base_piece_type: ...
instance_state_schema: []
movement_modifiers: []
capture_modifiers: []
hooks: []
metadata: {}
```

## Base Piece Types

- `pawn`
- `knight`
- `bishop`
- `rook`
- `queen`
- `king`

## State Field Types

- `bool`
- `int`
- `float`
- `str`

## Movement Modifier Ops

- `inherit_base`
- `phase_through_allies`
- `extend_range`
- `limit_range`

## Capture Modifier Ops

- `inherit_base`
- `replace_capture_with_push`

## Hook Events

- `move_committed`
- `piece_captured`
- `turn_start`
- `turn_end`

## Condition Ops

- `self_on_board`
- `square_empty`
- `square_occupied`
- `state_field_eq`
- `state_field_gte`
- `state_field_lte`
- `piece_color_is`

## Effect Ops

- `move_piece`
- `capture_piece`
- `set_state`
- `increment_state`
- `emit_event`

## Piece Refs

- `self`
- `event_source`
- `event_target`

## Square Refs

Absolute:

```yaml
absolute: e4
```

Relative:

```yaml
relative_to: self
offset: [0, 1]
```

`offset` uses mover-relative coordinates:

- first element: horizontal delta
- second element: forward/back delta
