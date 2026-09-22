"""PySide6 human-vs-checkpoint application."""

from __future__ import annotations

import ctypes.util
import os
from pathlib import Path

import shogi
from PySide6.QtCore import QThread, QTimer, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from syougi_rl.engine.inference import CheckpointEngine, list_checkpoints
from syougi_rl.gui.board import BoardController, ShogiBoardWidget, piece_label
from syougi_rl.gui.worker import InferenceWorker


def _xcb_dependencies_available() -> bool:
    """Return whether the Linux X11 libraries needed by Qt's xcb plugin exist."""
    required = ("xkbcommon-x11", "xcb-cursor", "xcb-icccm", "xcb-keysyms")
    return all(ctypes.util.find_library(name) for name in required)


def configure_qt_platform() -> str | None:
    """Select a safe Qt platform before QApplication is constructed.

    Windows uses the native platform. On Linux, an explicitly selected platform
    is respected; otherwise a missing X11 display/plugin dependency falls back
    to offscreen so CI/headless launches do not abort with a core dump.
    """
    if os.environ.get("QT_QPA_PLATFORM") or os.name == "nt":
        return os.environ.get("QT_QPA_PLATFORM")
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if not has_display or (os.environ.get("DISPLAY") and not _xcb_dependencies_available()):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        print("Qt display dependencies are unavailable; using QT_QPA_PLATFORM=offscreen", flush=True)
        return "offscreen"
    return None


def game_result_label(result: str, human_color: int) -> str:
    """Convert a python-shogi result into a human-facing outcome."""
    if result == "1/2-1/2":
        return "引き分け"
    if result not in {"1-0", "0-1"}:
        return "対局中"
    human_won = (result == "1-0") == (human_color == shogi.BLACK)
    return "勝利" if human_won else "敗北"


def status_with_check(text: str, state) -> str:
    """Append a visible 王手 marker when the side to move is checked."""
    if state.board.is_check() and "王手" not in text:
        return f"{text}【王手】"
    return text


def format_hand(hand) -> str:
    """Format a captured-piece Counter using Japanese piece labels."""
    parts = [f"{piece_label(piece_type)}×{hand.get(piece_type, 0)}" for piece_type in range(1, 8) if hand.get(piece_type, 0)]
    return " ".join(parts) if parts else "なし"


class GameWindow(QMainWindow):
    def __init__(self, checkpoint_dir: str | Path = "checkpoints") -> None:
        super().__init__()
        self.setWindowTitle("SyougiRL - Checkpoint対戦")
        self.controller = BoardController()
        self.engine: CheckpointEngine | None = None
        self.human_color = shogi.BLACK
        self.thread: QThread | None = None
        self.worker: InferenceWorker | None = None
        self._game_generation = 0
        self.ai_error = False
        self.game_over_announced = False
        self.thinking_phase = 0
        self.thinking_base = "AIが考えています"
        self.thinking_timer = QTimer(self)
        self.thinking_timer.setInterval(350)
        self.thinking_timer.timeout.connect(self._animate_thinking)

        self.checkpoint_box = QComboBox()
        self.checkpoint_box.addItems([str(path) for path in list_checkpoints(checkpoint_dir)])
        browse = QPushButton("Checkpointを選択")
        browse.clicked.connect(self.browse_checkpoint)
        self.side_box = QComboBox()
        self.side_box.addItem("先手（▲）", shogi.BLACK)
        self.side_box.addItem("後手（△）", shogi.WHITE)
        self.simulations = QSpinBox()
        self.simulations.setRange(1, 10_000)
        self.simulations.setValue(16)
        start = QPushButton("対局開始")
        start.clicked.connect(self.start_game)
        form = QFormLayout()
        form.addRow("Checkpoint", self.checkpoint_box)
        form.addRow("探索回数", self.simulations)
        form.addRow("手番", self.side_box)
        form.addRow(browse, start)

        self.hand_layout = QHBoxLayout()
        self.hand_buttons: dict[int, QPushButton] = {}
        for piece_type, label in ((1, "歩"), (2, "香"), (3, "桂"), (4, "銀"), (5, "金"), (6, "角"), (7, "飛")):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, p=piece_type: self.on_drop_piece(p))
            self.hand_buttons[piece_type] = button
            self.hand_layout.addWidget(button)
        self.opponent_hand_label = QLabel("なし")
        form.addRow("相手の持ち駒", self.opponent_hand_label)
        form.addRow("持ち駒", self.hand_layout)

        self.status = QLabel("Checkpointを選択して対局開始を押してください")
        self.board = ShogiBoardWidget(self.controller)
        self.board.square_clicked.connect(self.on_square_clicked)
        self.refresh_hand_buttons()
        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self.status)
        layout.addWidget(self.board, 1)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Checkpointを選択", "checkpoints", "PyTorch (*.pt)")
        if path:
            if self.checkpoint_box.findText(path) < 0:
                self.checkpoint_box.addItem(path)
            self.checkpoint_box.setCurrentText(path)

    def start_game(self) -> None:
        if self.thread is not None:
            QMessageBox.information(self, "AI探索中", "現在の対局が終わってから再開してください")
            return
        path = self.checkpoint_box.currentText()
        if not path:
            QMessageBox.warning(self, "Checkpoint未選択", "対局に使うCheckpointを選択してください")
            return
        try:
            self.engine = CheckpointEngine.from_checkpoint(path, device="auto")
        except Exception as exc:
            QMessageBox.critical(self, "Checkpointエラー", str(exc))
            return
        self.controller = BoardController()
        self.board.controller = self.controller
        self.human_color = self.side_box.currentData()
        self._game_generation += 1
        self.ai_error = False
        self.game_over_announced = False
        self._set_status(f"対局開始（推論デバイス: {self.engine.device.type}）")
        self.refresh_hand_buttons()
        self.board.update()
        if self.human_color == shogi.WHITE:
            self.request_ai_move()

    def on_square_clicked(self, square: int) -> None:
        if self.engine is None or self.ai_error or self.controller.state.is_game_over() or self.controller.state.board.turn != self.human_color:
            return
        if self.controller.highlighted_moves:
            candidates = self.controller.moves_for_destination(square)
            if candidates:
                move = candidates[0]
                if len(candidates) > 1:
                    answer = QMessageBox.question(self, "成り", "成りますか？", QMessageBox.Yes | QMessageBox.No)
                    move = next((item for item in candidates if item.promotion == (answer == QMessageBox.Yes)), move)
                self.controller.play_move(move)
                self.board.update()
                self.refresh_hand_buttons()
                self._after_human_move()
                return
        self.controller.select(square)
        self.board.update()

    def on_drop_piece(self, piece_type: int) -> None:
        if self.engine is None or self.ai_error or self.controller.state.is_game_over() or self.controller.state.board.turn != self.human_color:
            return
        self.controller.select_drop(piece_type)
        self.board.update()

    def refresh_hand_buttons(self) -> None:
        hand = self.controller.state.board.pieces_in_hand[self.human_color]
        opponent_color = shogi.WHITE if self.human_color == shogi.BLACK else shogi.BLACK
        self.opponent_hand_label.setText(format_hand(self.controller.state.board.pieces_in_hand[opponent_color]))
        enabled = self.engine is not None and not self.ai_error and self.controller.state.board.turn == self.human_color
        for piece_type, button in self.hand_buttons.items():
            label = button.text().split("×", 1)[0]
            count = hand.get(piece_type, 0)
            button.setText(f"{label}×{count}")
            button.setEnabled(enabled and count > 0)

    def _after_human_move(self) -> None:
        if self.controller.state.is_game_over():
            self._finish_game()
            return
        self.request_ai_move()

    def request_ai_move(self) -> None:
        if self.engine is None or self.thread is not None:
            return
        self._start_thinking()
        self.thread = QThread(self)
        generation = self._game_generation
        self.worker = InferenceWorker(self.engine, self.controller.state, self.simulations.value(), generation)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.move_ready.connect(self.on_ai_move)
        self.worker.error.connect(self.on_ai_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.on_ai_finished)
        self.thread.start()

    @Slot(object, int)
    def on_ai_move(self, move, generation: int) -> None:
        if generation != self._game_generation:
            return
        try:
            self.controller.state.push(move)
            self.board.update()
            self.refresh_hand_buttons()
            self._stop_thinking()
            if self.controller.state.is_game_over():
                self._finish_game()
            else:
                self._set_status("あなたの手番です")
        except ValueError as exc:
            self.on_ai_error(str(exc))

    @Slot(str)
    def on_ai_error(self, message: str) -> None:
        self._stop_thinking()
        self.ai_error = True
        self._set_status(f"AIエラー: {message}")
        QMessageBox.critical(self, "AIエラー", message)

    def on_ai_finished(self) -> None:
        self._stop_thinking()
        self.thread = None
        self.worker = None

    def _set_status(self, text: str) -> None:
        self.status.setText(status_with_check(text, self.controller.state))

    def _start_thinking(self) -> None:
        self.thinking_phase = 0
        self.thinking_timer.start()
        self._animate_thinking()

    def _animate_thinking(self) -> None:
        dots = "." * self.thinking_phase
        self._set_status(f"{self.thinking_base}{dots}")
        self.thinking_phase = (self.thinking_phase + 1) % 4

    def _stop_thinking(self) -> None:
        self.thinking_timer.stop()

    def _finish_game(self) -> None:
        self._stop_thinking()
        result = self.controller.state.result()
        outcome = game_result_label(result, self.human_color)
        self._set_status(f"終局: {outcome}（{result}）")
        if not self.game_over_announced:
            self.game_over_announced = True
            QMessageBox.information(self, outcome, f"対局結果：{outcome}\n{result}")

    def closeEvent(self, event) -> None:
        self._stop_thinking()
        if self.thread is not None:
            self.thread.quit()
            if not self.thread.wait(10_000):
                event.ignore()
                self.status.setText("AI探索終了待ちです。もう一度閉じてください")
                return
            self.thread = None
            self.worker = None
        event.accept()


def run_gui() -> int:
    from PySide6.QtWidgets import QApplication

    configure_qt_platform()
    app = QApplication.instance() or QApplication([])
    window = GameWindow()
    window.resize(700, 780)
    window.show()
    return app.exec()
