import numpy as np
import shogi

from syougi_rl.game.encoding import ACTION_SIZE, decode_move, encode_move
from syougi_rl.game.state import GameState


def test_initial_position_has_thirty_legal_moves_and_features():
    state = GameState.initial()

    assert len(state.legal_moves()) == 30
    features = state.features()
    assert isinstance(features, np.ndarray)
    assert features.shape == (29, 9, 9)
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
    assert decode_move(encode_move(move), state.board) == move
    try:
        state.push(move)
    except ValueError as exc:
        assert "legal" in str(exc).lower()
    else:
        raise AssertionError("illegal move was accepted")
