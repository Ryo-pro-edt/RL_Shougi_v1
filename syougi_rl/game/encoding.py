"""Fixed action vocabulary for python-shogi moves."""

from __future__ import annotations

import shogi

BOARD_SQUARES = 81
NORMAL_ACTION_SIZE = BOARD_SQUARES * BOARD_SQUARES * 2
DROP_PIECES = (
    shogi.PAWN,
    shogi.LANCE,
    shogi.KNIGHT,
    shogi.SILVER,
    shogi.GOLD,
    shogi.BISHOP,
    shogi.ROOK,
)
DROP_ACTION_SIZE = len(DROP_PIECES) * BOARD_SQUARES
ACTION_SIZE = NORMAL_ACTION_SIZE + DROP_ACTION_SIZE


def encode_move(move: shogi.Move) -> int:
    """Encode every USI move (normal, promotion, or drop) as a stable integer."""
    if move.drop_piece_type is not None:
        try:
            piece_index = DROP_PIECES.index(move.drop_piece_type)
        except ValueError as exc:
            raise ValueError(f"unsupported drop piece: {move.drop_piece_type}") from exc
        return NORMAL_ACTION_SIZE + piece_index * BOARD_SQUARES + move.to_square
    if move.from_square is None or not 0 <= move.from_square < BOARD_SQUARES:
        raise ValueError("normal moves need a board source square")
    promotion = 1 if move.promotion else 0
    return (move.from_square * BOARD_SQUARES + move.to_square) * 2 + promotion


def decode_move(action_id: int, board: shogi.Board | None = None) -> shogi.Move:
    """Decode an action ID. If a board is provided, legality is checked by callers."""
    if not 0 <= int(action_id) < ACTION_SIZE:
        raise ValueError(f"action ID out of range: {action_id}")
    action_id = int(action_id)
    if action_id >= NORMAL_ACTION_SIZE:
        drop_id = action_id - NORMAL_ACTION_SIZE
        piece_index, to_square = divmod(drop_id, BOARD_SQUARES)
        return shogi.Move(None, to_square, drop_piece_type=DROP_PIECES[piece_index])
    move_id, promotion = divmod(action_id, 2)
    from_square, to_square = divmod(move_id, BOARD_SQUARES)
    return shogi.Move(from_square, to_square, promotion=bool(promotion))
