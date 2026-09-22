import numpy as np
import shogi
import pytest

from syougi_rl.game.encoding import ACTION_SIZE, decode_move, encode_move
from syougi_rl.game.state import GameState


def test_initial_position_has_thirty_legal_moves_and_features():
    state = GameState.initial()

    assert len(state.legal_moves()) == 30
    features = state.features()
    assert isinstance(features, np.ndarray)
    assert features.shape == (43, 9, 9)
    assert features.dtype == np.float32


def test_promotion_move_round_trips_through_action_id():
    board = shogi.Board("4k4/9/9/9/9/9/P8/1P7/4K4 b - 1")
    move = shogi.Move.from_usi("9g9f")
    # Put a promotable pawn in the promotion zone and select its promoting move.
    board = shogi.Board("4k4/9/9/P8/9/9/9/9/4K4 b - 1")
    move = shogi.Move.from_usi("9d9c+")
    action = encode_move(move)
    decoded = decode_move(action, board)

    assert decoded == move


def test_drop_move_round_trips_through_action_id():
    board = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b P 1")
    move = shogi.Move.from_usi("P*5e")

    action = encode_move(move)
    decoded = decode_move(action, board)

    assert action >= ACTION_SIZE - 7 * 81
    assert decoded == move


def test_game_state_rejects_illegal_action_after_decode():
    state = GameState.initial()
    move = shogi.Move.from_usi("9a9b")

    assert move not in state.legal_moves()
    with pytest.raises(ValueError, match="legal"):
        decode_move(encode_move(move), state.board)
    with pytest.raises(ValueError, match="legal"):
        state.push(move)


def test_copy_is_independent_and_piece_planes_do_not_overlap_hands():
    state = GameState.initial()
    copied = state.copy()
    copied.push(shogi.Move.from_usi("9g9f"))

    assert len(state.legal_moves()) == 30
    assert copied.board.sfen() != state.board.sfen()
    # Black/white kings have dedicated board planes, not hand planes.
    assert state.features()[7].sum() == 1.0
    assert state.features()[21].sum() == 1.0


def test_promoted_piece_has_its_own_board_plane():
    state = GameState.from_sfen("4k4/9/9/9/9/9/9/2+B6/4K4 w - 1")

    features = state.features()

    assert features[12].sum() == 1.0  # black promoted bishop, piece type 13


def test_copy_preserves_fourfold_repetition_history():
    state = GameState.initial()
    for usi in ["7g7f", "3c3d", "7f7g", "3d3c"] * 3:
        state.board.push_usi(usi)

    copied = state.copy()

    assert state.board.is_fourfold_repetition()
    assert copied.board.is_fourfold_repetition()
