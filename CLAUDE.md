# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 絶対厳守ルール（違反禁止）

### 学習実行は必ずGCPで行うこと
- **ローカルMacでの学習実行は禁止**（テストモード以外）
- Macはスリープでプロセスが死ぬ。過去に2日間の学習時間を無駄にした
- ローカルで許可されるのは `--test-mode --steps 10000` のみ
- 学習を開始する前に必ず下記「GCP Training」セクションの手順に従うこと
- **インスタンス名**: `ofc-training-v2`（`ofc-training` ではない）
- **Python環境**: `venv_linux`（`.venv` はMac用、Linux非互換）

### 既存の手順を必ず確認してから実行
- このファイルの該当セクションを読んでから作業を開始すること
- 過去の成功パターンを踏襲し、独自の方法を勝手に試さないこと

## Session Checklist

Before starting work, check:
1. **`NEXT_ACTIONS.md`** - Pending tasks and next steps
2. **`docs/learning/04_current_status.md`** - Current training status

## Project Overview

Open-Face Chinese Poker (Pineapple) 3-Max AI using Deep Reinforcement Learning.
- 54-card deck (52 standard + 2 Jokers)
- 3-player simultaneous play
- Fantasy Land with Ultimate Rules (14-17 cards based on top hand)

## Build Commands

```bash
# Build C++ extension (required after C++ changes)
python setup.py build_ext --inplace

# Verify build
python -c "import ofc_engine as ofc; print('Engine loaded')"

# C++ unit tests
make test

# Python tests
python tests/test_joker.py
python tests/test_fl_solver.py
```

## Training Commands

**ローカルMacでの学習は禁止。必ずGCPで実行すること。**

```bash
# ローカルはテストのみ許可
NUM_ENVS=2 python src/python/train_phase10_fl_stay.py --test-mode --steps 10000
```

### GCP Training（必ずこの手順に従う）

```bash
# 1. インスタンス起動
gcloud compute instances start ofc-training-v2 --zone=asia-northeast1-b

# 2. ファイル転送
gcloud compute scp --recurse --zone=asia-northeast1-b \
  src models setup.py \
  ofc-training-v2:~/ofc-training/

# 3. C++エンジンリビルド（venv_linux を使う）
gcloud compute ssh ofc-training-v2 --zone=asia-northeast1-b --command="\
  cd ~/ofc-training && source venv_linux/bin/activate && \
  pip install -e . --force-reinstall --no-deps"

# 4. 学習開始
gcloud compute ssh ofc-training-v2 --zone=asia-northeast1-b --command="\
  cd ~/ofc-training && source venv_linux/bin/activate && \
  NUM_ENVS=4 nohup python3 src/python/train_phase10_fl_stay.py \
  --steps 200000000 > training.log 2>&1 &"

# 5. 動作確認（必須！省略するな）
sleep 30
gcloud compute ssh ofc-training-v2 --zone=asia-northeast1-b --command="\
  tail -30 ~/ofc-training/training.log"

# 6. 進捗確認
gcloud compute ssh ofc-training-v2 --zone=asia-northeast1-b --command="\
  grep 'Step' ~/ofc-training/training.log | tail -5"

# 7. モデルダウンロード
gcloud compute scp --recurse --zone=asia-northeast1-b \
  ofc-training-v2:~/ofc-training/models/phase10/ models/phase10/

# 8. インスタンス停止（学習完了後、必ず実行）
gcloud compute instances stop ofc-training-v2 --zone=asia-northeast1-b
```

## Architecture

### Layer 1: C++ Game Engine (`src/cpp/`)

Header-only implementation with key files:
- `game.hpp` - Game state machine, player management, FL handling
- `board.hpp` - Top(3)/Middle(5)/Bottom(5) slot management
- `evaluator.hpp` - Hand ranking with Joker support
- `solver.hpp` - Fantasy Land optimal placement solver
- `pybind/bindings.cpp` - Python bindings via pybind11

Card representation uses 64-bit bitboards for O(1) hand evaluation.

### Layer 2: Python Environment (`src/python/`)

- `ofc_3max_env.py` - PettingZoo AECEnv for 3-player multi-agent training
- `ofc_env.py` - Gymnasium single-player environment

Key features:
- Action masking via MaskablePPO (filters invalid placements)
- Continuous games with button rotation
- FL state inheritance between games

### Layer 3: Training Scripts (`src/python/train_*.py`)

Current: `train_phase85_selfplay.py` - Self-play with Ultimate Rules FL
- Uses Stable-Baselines3 MaskablePPO
- 4 parallel environments (SubprocVecEnv)
- Auto-checkpointing every 100k steps
- Discord notifications via webhook

## Key Concepts

### Ultimate Rules (Fantasy Land)

FL card distribution varies by top hand strength:
| Top Hand | Cards | Difficulty |
|----------|-------|------------|
| QQ | 14 | Standard |
| KK | 15 | Easier |
| AA | 16 | Easy |
| Trips | 17 | Easiest |

### Game Phases

```
PHASE_INITIAL_DEAL → PHASE_TURN (×4) → PHASE_SHOWDOWN
(5 cards)            (3 cards each)    (scoring)
```

### Observation Space (881 features)

Board state for all players, current hand, opponent visible cards, FL status indicators.

## Environment Variables

```bash
NUM_ENVS=4                    # Parallel training environments
CLOUD_PROVIDER=gcs            # or s3
GCS_BUCKET=bucket-name
DISCORD_WEBHOOK_URL=...       # Notifications every 100k steps
```

## Server Notes

- Use `venv_linux` on GCP (not `.venv` which is Mac)
- Checkpoint auto-cleanup keeps latest 2 + every 1M milestone
- Training auto-resumes from latest checkpoint

## Current Performance (Phase 9 FL Mastery - Final)

| Metric | Phase 9 (250M) | Phase 9 (150M) | Target |
|--------|------------------:|---------------------:|-------:|
| Foul Rate | 16.8% | 17.6% | < 20% ✅ |
| Mean Score | +12.66 | +12.58 | > +10 ✅ |
| FL Entry Rate | 22.8% | 21.2% | > 15% ✅ |
| FL Stay Rate | 8.0% | 8.2% | > 5% ✅ |
| Win Rate | 75.8% | 75.4% | > 70% ✅ |

**Latest Model**: `models/phase9/p9_fl_mastery_250000000.zip`
**ONNX Model**: `models/onnx/ofc_ai.onnx` (2.47MB, for Node.js inference)

### Greedy FL Solver

Training uses `greedy_fl_solver.py` (Monte Carlo sampling) instead of C++ brute-force
solver for FL placement. The brute-force solver in `solver.hpp` has O(4^n) complexity
with no pruning, causing FPS=2 during training. The greedy solver achieves 95% quality
at 14,000x speed for 17-card hands.

- Training: `fl_solver_mode='greedy'` (fast, approximate)
- Evaluation: Use default brute-force (accurate)

### GCP Training (Phase 8.5b)

```bash
# Instance: ofc-training-v2 (e2-standard-8, asia-northeast1-b)
# Training: 100M → 150M steps

# Check progress
gcloud compute ssh ofc-training-v2 --zone=asia-northeast1-b \
  --command="grep 'Step' ~/ofc-training/training.log | tail -5"

# Download model after completion
gcloud compute scp ofc-training-v2:~/ofc-training/models/p85_full_fl_*.zip \
  models/phase85b/ --zone=asia-northeast1-b

# Run evaluation
python scripts/evaluate_phase85b.py --compare

# Stop instance (save costs)
gcloud compute instances stop ofc-training-v2 --zone=asia-northeast1-b
```

## Mix Poker App Integration

学習済みAIを `/Users/naoai/Desktop/mix-poker-app` のCPUプレイヤーとして統合する手順。

### アーキテクチャ

```
mix-poker-app (Node.js)  ←→  OFC AI Server (Python/FastAPI)
     │                              │
     └── Socket.IO ─────────────────┘
           or HTTP POST
```

mix-poker-appのOFCGameEngine.tsがBotアクションを必要とする時、Python AIサーバーに問い合わせ。

### ファイル構成

```
OFC NN/
├── models/phase9/p9_fl_mastery_150000000.zip  # 学習済みモデル
└── src/python/
    ├── ai_server.py           # FastAPI推論サーバー（作成する）
    ├── ofc_3max_env.py        # 環境（観測生成に利用）
    └── ofc_engine.cpython-*.so # C++エンジン
```

### モデル配置

```bash
# 最新モデルをコピー
cp models/phase9/p9_fl_mastery_150000000.zip models/production/ofc_ai_latest.zip
```

### AIサーバー起動

```bash
cd /Users/naoai/試作品一覧/OFC\ NN
source .venv/bin/activate
python src/python/ai_server.py --model models/production/ofc_ai_latest.zip --port 8765
```

### mix-poker-app側の呼び出し

`server/OFCBot.ts` を修正し、AIサーバーへHTTPリクエスト:

```typescript
// AIサーバーへのリクエスト例
const response = await fetch('http://localhost:8765/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    phase: 'initial',  // 'initial' | 'pineapple' | 'fantasyland'
    cards: ['As', 'Kh', 'Qd', 'Jc', '9s'],
    board: { top: [], middle: [], bottom: [] },
    opponentBoards: [...]
  })
});
const { placements, discard } = await response.json();
```

### カード表記の変換

| mix-poker-app | OFC NN (C++) |
|---------------|--------------|
| `A♠` / `As` | Card index 0-51 |
| `K♥` / `Kh` | `ofc.Card(ofc.HEART, ofc.KING)` |

変換関数を `ai_server.py` に実装。

### 注意事項

- OFC NNは54枚デッキ（Joker 2枚）、mix-poker-appは52枚デッキ
- mix-poker-appは2人対戦、OFC NNは3人対戦で学習
- AI推論は10-50ms程度、タイムアウト設定に注意
