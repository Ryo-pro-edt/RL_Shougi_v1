from syougi_rl.gui.board import BoardController
from syougi_rl.game.state import GameState
from syougi_rl.engine.inference import list_checkpoints
from syougi_rl.gui.worker import InferenceWorker
import syougi_rl.gui.main as gui_main
from syougi_rl.gui.board import piece_label, piece_polygon, piece_text_color
from collections import Counter


def test_selecting_initial_pawn_returns_legal_marker_moves():
    controller = BoardController(GameState.initial())

    moves = controller.select(54)  # 9g

    assert moves
    assert all(move.from_square == 54 for move in moves)
    assert {move.to_square for move in moves} == {45}


def test_selecting_a_drop_returns_legal_drop_markers():
    state = GameState.from_sfen("4k4/9/9/9/9/9/9/9/4K4 b P 1")
    controller = BoardController(state)

    moves = controller.select_drop(1)  # pawn

    assert moves
    assert all(move.drop_piece_type == 1 for move in moves)


def test_cancel_selection_clears_markers():
    controller = BoardController(GameState.initial())
    controller.select(54)

    controller.clear_selection()

    assert controller.selected_square is None
    assert controller.highlighted_moves == []


def test_checkpoint_listing_handles_missing_directory(tmp_path):
    assert list_checkpoints(tmp_path / "missing") == []


def test_inference_worker_copies_state_at_construction():
    worker = InferenceWorker(None, GameState.initial(), simulations=1)

    assert worker.state.board.sfen() == GameState.initial().board.sfen()


def test_qt_platform_falls_back_to_offscreen_when_xcb_dependencies_are_missing(monkeypatch):
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setattr(gui_main, "_xcb_dependencies_available", lambda: False)
    monkeypatch.setenv("DISPLAY", ":9")

    selected = gui_main.configure_qt_platform()

    assert selected == "offscreen"


def test_piece_labels_are_japanese_including_promotions():
    assert piece_label(1) == "歩"
    assert piece_label(8) == "玉"
    assert piece_label(9) == "と"
    assert piece_label(13) == "馬"
    assert piece_text_color(1) == "#21170e"
    assert piece_text_color(9) == "#c62828"


def test_piece_polygon_points_up_for_self_and_down_for_opponent():
    own = piece_polygon(100, inverted=False)
    opponent = piece_polygon(100, inverted=True)

    assert min(point.y() for point in own) < max(point.y() for point in own)
    assert own[0].y() < own[2].y()
    assert opponent[0].y() > opponent[2].y()


def test_game_result_label_is_from_human_perspective():
    assert gui_main.game_result_label("1-0", 0) == "勝利"
    assert gui_main.game_result_label("1-0", 1) == "敗北"
    assert gui_main.game_result_label("0-1", 1) == "勝利"
    assert gui_main.game_result_label("1/2-1/2", 0) == "引き分け"


def test_check_status_is_added_when_side_to_move_is_in_check():
    state = GameState.from_sfen("4k4/9/9/9/9/9/9/4r4/4K4 b - 1")

    assert state.board.is_check()
    assert gui_main.status_with_check("あなたの手番", state) == "あなたの手番【王手】"


def test_hand_display_formats_opponent_captured_pieces_in_japanese():
    assert gui_main.format_hand(Counter({1: 2, 6: 1})) == "歩×2 角×1"
    assert gui_main.format_hand(Counter()) == "なし"
