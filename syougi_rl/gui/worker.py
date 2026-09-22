from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from syougi_rl.engine.inference import CheckpointEngine
from syougi_rl.game.state import GameState


class InferenceWorker(QObject):
    move_ready = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, engine: CheckpointEngine, state: GameState, simulations: int) -> None:
        super().__init__()
        self.engine = engine
        self.state = state.copy()
        self.simulations = int(simulations)

    @Slot()
    def run(self) -> None:
        try:
            self.move_ready.emit(self.engine.choose_move(self.state, self.simulations))
        except Exception as exc:  # forward worker errors to the GUI thread
            self.error.emit(str(exc))
        finally:
            self.finished.emit()
