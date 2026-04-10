from __future__ import annotations

from pixie_solver.core.piece import PieceClass


class DSLFeatureEncoder:
    def encode_piece_class(self, piece_class: PieceClass) -> list[float]:
        raise NotImplementedError("The DSL encoder lands in milestone M5.")
