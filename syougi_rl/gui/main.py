"""PySide6 human-vs-checkpoint application."""

from __future__ import annotations

from pathlib import Path

import shogi
from PySide6.QtCore import QThread
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
from syougi_rl.gui.board import BoardController, ShogiBoardWidget
from syougi_rl.gui.worker import InferenceWorker


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
        self.status.setText(f"対局開始（推論デバイス: {self.engine.device.type}）")
        self.refresh_hand_buttons()
        self.board.update()
        if self.human_color == shogi.WHITE:
            self.request_ai_move()

    def on_square_clicked(self, square: int) -> None:
        if self.engine is None or self.ai_error or self.controller.state.board.turn != self.human_color:
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
        if self.engine is None or self.ai_error or self.controller.state.board.turn != self.human_color:
            return
        self.controller.select_drop(piece_type)
        self.board.update()

    def refresh_hand_buttons(self) -> None:
        hand = self.controller.state.board.pieces_in_hand[self.human_color]
        enabled = self.engine is not None and not self.ai_error and self.controller.state.board.turn == self.human_color
        for piece_type, button in self.hand_buttons.items():
            label = button.text().split("×", 1)[0]
            count = hand.get(piece_type, 0)
            button.setText(f"{label}×{count}")
            button.setEnabled(enabled and count > 0)

    def _after_human_move(self) -> None:
        if self.controller.state.is_game_over():
            self.status.setText(f"終局: {self.controller.state.result()}")
            return
        self.request_ai_move()

    def request_ai_move(self) -> None:
        if self.engine is None or self.thread is not None:
            return
        self.status.setText("AIが考えています…")
        self.thread = QThread(self)
        self.worker = InferenceWorker(self.engine, self.controller.state, self.simulations.value())
        generation = self._game_generation
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.move_ready.connect(lambda move, gen=generation: self.on_ai_move(move, gen))
        self.worker.error.connect(self.on_ai_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.on_ai_finished)
        self.thread.start()

    def on_ai_move(self, move, generation: int | None = None) -> None:
        if generation is not None and generation != self._game_generation:
            return
        try:
            self.controller.state.push(move)
            self.board.update()
            self.refresh_hand_buttons()
            self.status.setText("あなたの手番です")
            if self.controller.state.is_game_over():
                self.status.setText(f"終局: {self.controller.state.result()}")
        except ValueError as exc:
            self.on_ai_error(str(exc))

    def on_ai_error(self, message: str) -> None:
        self.ai_error = True
        self.status.setText(f"AIエラー: {message}")
        QMessageBox.critical(self, "AIエラー", message)

    def on_ai_finished(self) -> None:
        self.thread = None
        self.worker = None

    def closeEvent(self, event) -> None:
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(10_000)
            self.thread = None
            self.worker = None
        event.accept()


def run_gui() -> int:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    window = GameWindow()
    window.resize(700, 780)
    window.show()
    return app.exec()
