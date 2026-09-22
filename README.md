# SyougiRL

Windows向けの小規模な将棋強化学習サンプルです。PyTorchで方策・価値ネットワークを自己対戦学習し、生成したCheckpointをPySide6のGUIで選んで人間と対局できます。GUIは学習を実行せず、対戦と合法手の表示に集中します。

## できること

- `python-shogi` にルール、合法手、成り、持ち駒、終局判定を委譲
- AlphaZero型の自己対戦、合法手マスク、MCTS、方策/価値更新
- `device: auto` でRTX 4070などCUDA GPUを優先し、利用できなければCPUへフォールバック
- `epoch_000001.pt` 形式のCPU互換Checkpoint
- GUIで任意のCheckpointを選択し、駒・持ち駒の合法な行き先を半透明マーカー表示
- 先手/後手、探索回数、自己対戦局数や更新回数を設定ファイル/CLIで変更

## アーキテクチャ

```text
train.py / notebooks/train.ipynb
          |
  training.loop --- MCTS --- game.state/encoding --- python-shogi
          |
  model.network -> checkpoints/epoch_XXXXXX.pt
          |
  play_gui.py -> engine.inference -> MCTS/model
          |
  PySide6 GUI (BoardController + InferenceWorker)
```

`syougi_rl.game` は29チャネルの盤面特徴量と、通常手・成り・駒打ちを含む13,689個の固定行動IDを提供します。`syougi_rl.model` は方策・価値Residual CNNとCheckpoint I/O、`syougi_rl.training` は自己対戦と更新、`syougi_rl.engine` はGUI非依存の推論、`syougi_rl.gui` は盤面表示とワーカースレッドを担当します。

## Windowsセットアップ

PowerShellでPython 3.11以上の仮想環境を作成します。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

RTX 4070を学習に使う場合は、先に [PyTorch公式のインストール選択ページ](https://pytorch.org/get-started/locally/) でWindows / Pip / Python / 使用可能なCUDA版を選び、そのコマンドを実行してください。CUDA版を固定したコマンドは環境差で動かなくなるため、このプロジェクトでは固定していません。その後、残りの依存関係を入れます。

```powershell
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`False CPU` と表示されても、学習と対戦はCPUで継続できます。

## 学習

短時間の動作確認は既定設定で実行できます。

```powershell
python train.py --config config/default.yaml --device auto
```

生成先は `checkpoints/epoch_000001.pt` です。学習量をCLIで上書きできます。

```powershell
python train.py --config config/default.yaml --device cuda `
  --epochs 20 --self-play-games 50 --mcts-simulations 64 `
  --updates-per-epoch 100 --batch-size 64
```

設定ファイルには `epochs`, `self_play_games`, `max_moves`, `mcts_simulations`, `updates_per_epoch`, `batch_size`, `learning_rate`, `checkpoint_every`, `checkpoint_dir` を指定できます。CUDAが使用不能な環境で `--device cuda` を指定した場合も、警告後にCPUへ切り替わります。

Jupyterからも同じAPIを実行できます。

```powershell
jupyter notebook notebooks/train.ipynb
```

Notebookは独自実装を持たず、`syougi_rl.training.train` を呼び出します。

## GUI対戦

```powershell
python play_gui.py
```

起動画面で `checkpoints/epoch_000001.pt` などのCheckpointを選び、先手/後手と探索回数を指定して「対局開始」を押します。推論も `auto` が既定で、CUDA利用可能ならGPU、不可ならCPUです。自分の駒をクリックすると合法な移動先がドットで表示されます。持ち駒ボタンをクリックすると合法な打ち先が表示されます。成りが選べる手では確認ダイアログを表示します。

## テスト

```powershell
python -m pytest -q
```

GUIを表示できないCIやリモート環境ではQtをオフスクリーンにします。

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m pytest -q
```

## 既知の制約

これは短時間の動作確認を目的とした小型実装です。既定の自己対戦量では強い棋力を保証しません。長時間学習では`mcts_simulations`、自己対戦局数、更新回数、ネットワーク規模を増やしてください。GUIの棋譜保存や持ち駒専用パネルは、コア学習・対戦機能と分離して今後拡張できます。
