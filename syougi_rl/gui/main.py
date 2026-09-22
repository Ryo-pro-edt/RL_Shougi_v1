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

        self.status = QLabel("Checkpointを選択して対局開始を押してください")
        self.board = ShogiBoardWidget(self.controller)
        self.board.square_clicked.connect(self.on_square_clicked)
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
        self.status.setText(f"対局開始（推論デバイス: {self.engine.device.type}）")
        self.board.update()
        if self.human_color == shogi.WHITE:
            self.request_ai_move()

    def on_square_clicked(self, square: int) -> None:
        if self.engine is None or self.controller.state.board.turn != self.human_color:
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
                self._after_human_move()
                return
        self.controller.select(square)
        self.board.update()

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
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.move_ready.connect(self.on_ai_move)
        self.worker.error.connect(self.on_ai_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.on_ai_finished)
        self.thread.start()

    def on_ai_move(self, move) -> None:
        try:
            self.controller.state.push(move)
            self.board.update()
            self.status.setText("あなたの手番です")
            if self.controller.state.is_game_over():
                self.status.setText(f"終局: {self.controller.state.result()}")
        except ValueError as exc:
            self.on_ai_error(str(exc))

    def on_ai_error(self, message: str) -> None:
        self.status.setText(f"AIエラー: {message}")
        QMessageBox.critical(self, "AIエラー", message)

    def on_ai_finished(self) -> None:
        self.thread = None
        self.worker = None


def run_gui() -> int:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    window = GameWindow()
    window.resize(700, 780)
    window.show()
    return app.exec()
