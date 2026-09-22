"""A small, immutable-by-convention wrapper around python-shogi."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shogi


FEATURE_PLANES = 29  # 14 board planes, 14 hand-count planes, one side-to-move plane.


@dataclass
class GameState:
    board: shogi.Board

    @classmethod
    def initial(cls) -> "GameState":
        return cls(shogi.Board())

    @classmethod
    def from_sfen(cls, sfen: str) -> "GameState":
        return cls(shogi.Board(sfen))

    def copy(self) -> "GameState":
        return GameState(self.board.copy())

    def legal_moves(self) -> list[shogi.Move]:
        return list(self.board.legal_moves)

    def push(self, move: shogi.Move) -> None:
        if move not in self.board.legal_moves:
            raise ValueError(f"move is not legal: {move.usi()}")
        self.board.push(move)

    def is_game_over(self) -> bool:
        return self.board.is_game_over()

    def result(self) -> str:
        if self.board.is_checkmate():
            # The side to move has been checkmated.
            return "1-0" if self.board.turn == shogi.WHITE else "0-1"
        if self.board.is_game_over():
            return "1/2-1/2"
        return "*"

    def features(self) -> np.ndarray:
        features = np.zeros((FEATURE_PLANES, 9, 9), dtype=np.float32)
        for square in range(81):
            piece = self.board.piece_at(square)
            if piece is None:
                continue
            rank, file_ = divmod(square, 9)
            color_offset = 0 if piece.color == shogi.BLACK else 7
            features[color_offset + piece.piece_type - 1, rank, file_] = 1.0

        for color, offset in ((shogi.BLACK, 14), (shogi.WHITE, 21)):
            for piece_type in range(1, 8):
                count = self.board.pieces_in_hand[color].get(piece_type, 0)
                features[offset + piece_type - 1, :, :] = float(count) / 18.0
        features[28, :, :] = float(self.board.turn == shogi.BLACK)
        return features
