"""Testable board-selection logic and a lightweight Qt board widget."""

from __future__ import annotations

from typing import Iterable

import shogi
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QFontDatabase, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from syougi_rl.game.state import GameState


PIECE_LABELS = {
    1: "歩",
    2: "香",
    3: "桂",
    4: "銀",
    5: "金",
    6: "角",
    7: "飛",
    8: "玉",
    9: "と",
    10: "杏",
    11: "圭",
    12: "全",
    13: "馬",
    14: "龍",
}


def piece_label(piece_type: int) -> str:
    """Return the standard Japanese glyph for a python-shogi piece type."""
    return PIECE_LABELS.get(int(piece_type), "?")


def japanese_font(size: int) -> QFont:
    """Prefer installed Japanese fonts while keeping Windows/Linux fallbacks."""
    available = set(QFontDatabase.families())
    for family in ("Yu Gothic UI", "Yu Gothic", "Meiryo", "Noto Sans CJK JP", "Noto Serif CJK JP", "MS Gothic"):
        if family in available:
            return QFont(family, size)
    return QFont("sans-serif", size)


def piece_polygon(cell: float, inverted: bool = False) -> QPolygonF:
    """Create a wooden shogi-piece pentagon centered at the origin."""
    half = cell * 0.42
    points = [
        QPointF(0.0, -cell * 0.45),
        QPointF(half, -cell * 0.16),
        QPointF(cell * 0.30, cell * 0.45),
        QPointF(-cell * 0.30, cell * 0.45),
        QPointF(-half, -cell * 0.16),
    ]
    if inverted:
        points = [QPointF(point.x(), -point.y()) for point in points]
    return QPolygonF(points)


class BoardController:
    def __init__(self, state: GameState | None = None) -> None:
        self.state = state or GameState.initial()
        self.selected_square: int | None = None
        self.selected_drop: int | None = None
        self.highlighted_moves: list[shogi.Move] = []

    def clear_selection(self) -> None:
        self.selected_square = None
        self.selected_drop = None
        self.highlighted_moves = []

    def select(self, square: int) -> list[shogi.Move]:
        moves = [move for move in self.state.legal_moves() if move.from_square == square]
        self.selected_square = square if moves else None
        self.selected_drop = None
        self.highlighted_moves = moves
        return moves

    def select_drop(self, piece_type: int) -> list[shogi.Move]:
        moves = [move for move in self.state.legal_moves() if move.drop_piece_type == piece_type]
        self.selected_square = None
        self.selected_drop = piece_type if moves else None
        self.highlighted_moves = moves
        return moves

    def moves_for_destination(self, square: int) -> list[shogi.Move]:
        return [move for move in self.highlighted_moves if move.to_square == square]

    def play_move(self, move: shogi.Move) -> None:
        if move not in self.highlighted_moves:
            raise ValueError("move is not one of the selected legal moves")
        self.state.push(move)
        self.clear_selection()


class ShogiBoardWidget(QWidget):
    square_clicked = Signal(int)

    def __init__(self, controller: BoardController, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setMinimumSize(540, 540)
        self.setMouseTracking(True)

    def _geometry(self) -> tuple[float, float, float]:
        size = min(self.width(), self.height())
        margin = 8.0
        return margin, margin, (size - margin * 2) / 9.0

    def square_at(self, position) -> int | None:
        x0, y0, cell = self._geometry()
        col = int((position.x() - x0) // cell)
        row = int((position.y() - y0) // cell)
        if 0 <= row < 9 and 0 <= col < 9:
            return row * 9 + col
        return None

    def mousePressEvent(self, event) -> None:
        square = self.square_at(event.position())
        if square is not None:
            self.square_clicked.emit(square)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        x0, y0, cell = self._geometry()
        painter.fillRect(self.rect(), QColor("#f2d28b"))
        painter.setPen(QPen(QColor("#5c3b1e"), 2))
        for i in range(10):
            painter.drawLine(QPointF(x0 + i * cell, y0), QPointF(x0 + i * cell, y0 + 9 * cell))
            painter.drawLine(QPointF(x0, y0 + i * cell), QPointF(x0 + 9 * cell, y0 + i * cell))

        marker_squares = {move.to_square for move in self.controller.highlighted_moves}
        for square in marker_squares:
            row, col = divmod(square, 9)
            painter.setBrush(QBrush(QColor(60, 150, 80, 120)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(x0 + (col + 0.5) * cell, y0 + (row + 0.5) * cell), cell * 0.13, cell * 0.13)

        if self.controller.selected_square is not None:
            row, col = divmod(self.controller.selected_square, 9)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor("#d12b2b"), 3))
            painter.drawRect(x0 + col * cell + 2, y0 + row * cell + 2, cell - 4, cell - 4)

        painter.setFont(japanese_font(max(12, int(cell * 0.55))))
        for square in range(81):
            piece = self.controller.state.board.piece_at(square)
            if piece is None:
                continue
            row, col = divmod(square, 9)
            center_x = x0 + (col + 0.5) * cell
            center_y = y0 + (row + 0.5) * cell
            painter.save()
            painter.translate(center_x, center_y)
            if piece.color == shogi.WHITE:
                painter.rotate(180)
            painter.setBrush(QBrush(QColor("#d7a45e")))
            painter.setPen(QPen(QColor("#6e431f"), max(1, int(cell * 0.025))))
            painter.drawPolygon(piece_polygon(cell, inverted=False))
            painter.setPen(QPen(QColor("#21170e")))
            painter.setFont(japanese_font(max(12, int(cell * 0.48))))
            painter.drawText(-cell / 2, -cell / 2, cell, cell, Qt.AlignCenter, piece_label(piece.piece_type))
            painter.restore()
