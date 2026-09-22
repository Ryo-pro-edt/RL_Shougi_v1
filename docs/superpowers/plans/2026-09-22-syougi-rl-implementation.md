# SyougiRL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Windows-ready PyTorch/python-shogi self-play learner and a PySide6 checkpoint-selecting GUI engine with legal-move markers.

**Architecture:** Keep game rules and action encoding independent from the neural model. Training calls a shared self-play/MCTS API from both `train.py` and `notebooks/train.ipynb`; the GUI loads the same model through an inference engine and runs AI search off the Qt event loop.

**Tech Stack:** Python 3.11+, PyTorch, python-shogi, PyYAML, PySide6, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-22-syougi-rl-design.md`

## Global Constraints

- Windows is the target desktop platform.
- `device: auto` selects CUDA first and CPU otherwise for training and inference.
- Checkpoints are `checkpoints/epoch_000001.pt`-style files and are CPU-loadable.
- GUI is for play only; learning is through `train.py` or `notebooks/train.ipynb`.
- python-shogi is authoritative for legal moves, promotions, drops, and game termination.
- GUI AI work must not block the Qt event loop.
- Every production feature has a test written and observed failing before implementation.

## Review Focus

- Invalid or incompatible Checkpoint: GUI reports the cause and does not start a game; test in Task 4.
- CPU-only machine loading a CUDA-created file: `map_location` succeeds; test in Task 3.
- Promotion and drops: legal destinations and action IDs round-trip; tests in Task 1.
- No CUDA despite `auto`: CPU is selected without import-time failure; test in Task 3.
- Worker exception during AI move: GUI returns to a consistent enabled state; test in Task 6.

### Task 1: Game representation and action encoding

**Files:**
- Create: `syougi_rl/game/state.py`
- Create: `syougi_rl/game/encoding.py`
- Create: `syougi_rl/game/__init__.py`
- Create: `tests/test_game.py`

**Interfaces:** `GameState.initial()`, `.legal_moves()`, `.push(move)`, `.is_game_over()`, `.result()`, `.features()`, `encode_move(move) -> int`, `decode_move(action_id, board) -> shogi.Move`.

- [ ] Write failing tests for initial legal moves, promotion, drops, feature shape, and action round-trip.
- [ ] Run `pytest tests/test_game.py -q` and observe missing-module failures.
- [ ] Implement a thin python-shogi wrapper and a deterministic fixed action vocabulary that includes normal moves, promotion variants, and drops; reject non-legal decoded moves.
- [ ] Run the focused test and then `pytest -q`; record output.
- [ ] Commit `feat: add shogi state and action encoding`.

### Task 2: Model, device selection, and Checkpoint I/O

**Files:**
- Create: `syougi_rl/model/network.py`
- Create: `syougi_rl/model/checkpoint.py`
- Create: `syougi_rl/model/device.py`
- Create: `syougi_rl/model/__init__.py`
- Create: `tests/test_model.py`

**Interfaces:** `select_device("auto"|"cuda"|"cpu") -> torch.device`, `PolicyValueNet(action_size)`, `save_checkpoint(path, model, optimizer, epoch, config)`, `load_checkpoint(path, model, optimizer=None, device="auto") -> dict`.

- [ ] Write failing tests for output shapes, auto CPU fallback, epoch-named metadata, and CPU loading of a saved checkpoint.
- [ ] Run `pytest tests/test_model.py -q` and observe missing-module failures.
- [ ] Implement a small residual CNN, safe device selection, and versioned `torch.save` payload loaded with `map_location`.
- [ ] Run focused and full tests; commit `feat: add policy value model and checkpoints`.

### Task 3: MCTS, replay, and short self-play training

**Files:**
- Create: `syougi_rl/training/mcts.py`
- Create: `syougi_rl/training/replay.py`
- Create: `syougi_rl/training/loop.py`
- Create: `syougi_rl/training/__init__.py`
- Create: `config/default.yaml`
- Create: `tests/test_training.py`

**Interfaces:** `MCTS(model, simulations, device).select_move(state)`, `ReplayBuffer.add(...)`, `train(config_path, overrides=None) -> list[pathlib.Path]`.

- [ ] Write failing tests for legal MCTS output, replay sampling, and a tiny run creating `epoch_000001.pt`.
- [ ] Run `pytest tests/test_training.py -q` and observe missing-module failures.
- [ ] Implement masked policy inference, bounded playouts, self-play examples, cross-entropy plus value loss, configurable update counts, and periodic epoch checkpoints.
- [ ] Run focused and full tests; commit `feat: add configurable self play training`.

### Task 4: CLI and Notebook entry points

**Files:**
- Create: `train.py`
- Create: `notebooks/train.ipynb`
- Create: `tests/test_entrypoints.py`

**Interfaces:** `python train.py --config config/default.yaml --device auto`; notebook imports `syougi_rl.training.train` and exposes the same config dictionary.

- [ ] Write failing tests for CLI argument parsing and config validation.
- [ ] Run the focused test and observe missing entrypoint failures.
- [ ] Implement CLI overrides for device, self-play games, simulations, updates, and checkpoint directory; create a minimal notebook with executable cells.
- [ ] Run parser tests and a tiny CLI training invocation; commit `feat: add CLI and notebook training entrypoints`.

### Task 5: Inference engine and checkpoint discovery

**Files:**
- Create: `syougi_rl/engine/inference.py`
- Create: `syougi_rl/engine/__init__.py`
- Create: `tests/test_engine.py`

**Interfaces:** `CheckpointEngine.from_checkpoint(path, device="auto")`, `.choose_move(state, simulations=16)`, `list_checkpoints(directory) -> list[pathlib.Path]`.

- [ ] Write failing tests for checkpoint listing, legal move selection, and invalid checkpoint errors.
- [ ] Run focused tests and observe missing-module failures.
- [ ] Implement load/compatibility validation and a no-UI inference API using the same MCTS; propagate descriptive exceptions.
- [ ] Run focused and full tests; commit `feat: add checkpoint inference engine`.

### Task 6: PySide6 play GUI with legal markers

**Files:**
- Create: `syougi_rl/gui/main.py`
- Create: `syougi_rl/gui/board.py`
- Create: `syougi_rl/gui/worker.py`
- Create: `syougi_rl/gui/__init__.py`
- Create: `play_gui.py`
- Create: `tests/test_gui_logic.py`

**Interfaces:** `BoardController.select(square) -> list[shogi.Move]`, `.select_drop(piece) -> list[shogi.Move]`, `GameWindow`, `InferenceWorker` Qt signals `move_ready`, `error`, `finished`.

- [ ] Write failing non-visual tests for marker destinations, promotion choice data, checkpoint selection model, and worker error state.
- [ ] Run focused tests and observe missing-module failures (skip tests when PySide6 is unavailable only via an explicit import guard).
- [ ] Implement a QGraphics/QWidget board, coordinate mapping, side panel, checkpoint file dialog, promotion dialog, translucent legal markers, and worker-thread AI calls.
- [ ] Run tests and a Qt offscreen smoke test; commit `feat: add checkpoint play GUI`.

### Task 7: README, packaging, and final verification

**Files:**
- Create: `README.md`
- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `.gitignore`
- Modify: `docs/superpowers/specs/2026-09-22-syougi-rl-design.md` only if implementation decisions require clarification.

- [ ] Write failing documentation smoke checks for referenced files and commands.
- [ ] Run the checks and observe missing-document failures.
- [ ] Document Windows venv setup, CUDA PyTorch selection, CLI/Notebook training, checkpoint naming, GUI launch, settings, architecture, tests, and limitations; add package metadata and ignores.
- [ ] Run `pytest -q`, `python -m compileall syougi_rl train.py play_gui.py`, and tiny CPU training; commit `docs: document setup and architecture`.

## Plan self-review

All spec sections map to tasks: game/legality (1), model/checkpoints/device (2), training/config (3), CLI/Notebook (4), inference (5), GUI/markers/threading (6), README/Windows setup/testing (7). No TODO/TBD placeholders remain. Interfaces use `GameState`, `PolicyValueNet`, `CheckpointEngine`, and training entry points consistently. Review-focus failure modes have explicit tests in Tasks 1, 2, 5, and 6.
