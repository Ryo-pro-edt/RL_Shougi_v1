from syougi_rl.gui.board import BoardController
from syougi_rl.game.state import GameState
from syougi_rl.engine.inference import list_checkpoints
from syougi_rl.gui.worker import InferenceWorker
import syougi_rl.gui.main as gui_main
from syougi_rl.gui.board import piece_label


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
