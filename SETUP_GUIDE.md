# OFC Pineapple AI V2 — 環境構築ガイド

別PCへのプロジェクト移行用手順書。

---

## 前提条件

- **OS**: macOS (Apple Silicon / Intel) or Linux (Ubuntu 22.04+)
- **Python**: 3.9+ (3.10 推奨)
- **C++ コンパイラ**: clang++ (macOS) or g++ (Linux), C++17 対応
- **Git**: 2.30+
- **ディスク容量**: 約 2GB (プロジェクト + venv + モデル)

---

## Step 1: リポジトリのクローン

```bash
git clone <your-repo-url> "OFC NN"
cd "OFC NN"
```

もしくは既存のプロジェクトフォルダをコピー:
```bash
cp -r "/path/to/OFC NN" "/new/path/OFC NN"
cd "/new/path/OFC NN"
```

---

## Step 2: Python 仮想環境の作成

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## Step 3: 依存パッケージのインストール

### 方法A: ロックファイルから（推奨 — 完全再現）
```bash
pip install -r requirements-lock.txt
```

### 方法B: 最小依存から（バージョン違いの可能性あり）
```bash
pip install -r requirements.txt
```

### PyTorch について
`requirements-lock.txt` は CPU 版 PyTorch を含んでいます。
GPU (CUDA) を使う場合は先に PyTorch を個別インストール:
```bash
# CUDA 12.x の場合
pip install torch --index-url https://download.pytorch.org/whl/cu121
# その後、残りのパッケージ
pip install -r requirements.txt
```

---

## Step 4: C++ ゲームエンジンのビルド

```bash
python setup.py build_ext --inplace
```

成功確認:
```bash
python -c "import ofc_engine as ofc; print('OK:', ofc.HandRank.ROYAL_FLUSH)"
```

### ビルドに失敗する場合

**macOS (Xcode Command Line Tools が必要)**:
```bash
xcode-select --install
```

**Linux (build-essential が必要)**:
```bash
sudo apt-get install build-essential python3-dev
```

---

## Step 5: 動作確認

### テスト実行
```bash
# C++ エンジン網羅テスト (124テスト)
python tests/test_evaluator_comprehensive.py

# Joker テスト
python tests/test_joker.py
```

### 学習のテスト実行 (10K ステップ、数十秒で完了)
```bash
python v2/train_v2.py --test-mode --reward-config C
```

### ルールベースエージェントのテスト
```bash
cd "OFC NN"
PYTHONPATH="." python v2/rule_based_agent.py
```

---

## Step 6: 学習の開始

### Config C (確定済み報酬設定) でゼロからスタート
```bash
# Discord Webhook 設定
export DISCORD_WEBHOOK_URL='https://discord.com/api/webhooks/1461388510140301315/Ofc9ok3IDLgRhFR7Oa2iA4-CHHL9uFvFbIPT9G9tbA5X5r-wG-XMbnkg7ubocLJJQ5Pf'

# NUM_ENVS はCPUコア数に合わせて調整
NUM_ENVS=4 python v2/train_v2.py --reward-config C --steps 5000000
```

### Step 2 (Self-Play 拡張比較) の場合

Config D (ベースライン):
```bash
NUM_ENVS=4 python v2/train_v2.py --reward-config C --steps 10000000 \
  --run-name D --pool-size 5 --latest-prob 0.8 --rule-based-prob 0.0
```

Config E (拡張版):
```bash
NUM_ENVS=4 python v2/train_v2.py --reward-config C --steps 10000000 \
  --run-name E --pool-size 15 --latest-prob 0.6 --rule-based-prob 0.1
```

### チェックポイントからの再開
チェックポイントが `models/v2_configC/` にあれば自動検出して再開します。
別の run から再開する場合:
```bash
python v2/train_v2.py --reward-config C --steps 10000000 \
  --run-name D --resume-from models/v2_configC/v2_c_5000000.zip
```

---

## ディレクトリ構成

```
OFC NN/
├── CLAUDE.md              # プロジェクト運営ルール
├── SETUP_GUIDE.md         # このファイル
├── setup.py               # C++ ビルド設定
├── requirements.txt       # 最小依存
├── requirements-lock.txt  # 完全再現用ロック
├── src/
│   ├── cpp/               # C++ ゲームエンジン
│   │   ├── evaluator.hpp  # ハンド評価 (ACE=14 修正済み)
│   │   ├── board.hpp      # Top/Mid/Bot 管理
│   │   ├── game.hpp       # ゲーム進行・FL処理
│   │   ├── solver.hpp     # FL ブルートフォースソルバー
│   │   └── pybind/        # Python バインディング
│   └── python/
│       ├── ofc_3max_env.py    # PettingZoo 3人環境
│       ├── ofc_env.py         # Gymnasium 単体環境
│       ├── greedy_fl_solver.py # FL 近似ソルバー
│       └── notifier.py        # Discord 通知
├── v2/
│   ├── train_v2.py        # V2 学習スクリプト
│   ├── rule_based_agent.py # ルールベースエージェント
│   └── evaluate_benchmark.py # ベンチマーク評価
├── tests/
│   ├── test_evaluator_comprehensive.py  # 124テスト
│   └── test_joker.py      # Joker テスト
├── models/                # 学習済みモデル
├── docs/                  # 実験記録・テンプレート
└── archive/v1/            # V1 全記録
```

---

## 現在の進捗状況 (2026-02-27)

### 完了済み
- Phase 0: C++ エンジン整備、報酬設計、環境構築
- Phase 1 Step 1: 報酬アブレーション (A/B/C) → **Config C 採用確定**
  - Config C 5M 結果: Foul 33.2%, Score +5.86, FL追求差 +2.0% (OK)
- C++ エンジン網羅テスト: 124/124 パス

### 次のタスク
- Phase 1 Step 2: Self-Play プール拡張比較 (D vs E)
  - D: pool=5, latest=0.8, rule_based=0% (ベースライン)
  - E: pool=15, latest=0.6, rule_based=10% (拡張版)
  - 各 5M → 10M ステップ（Config C の 5M から再開）
  - ローカル学習に移行: Config C をゼロから再学習 → 5M 到達後に D/E 分岐

### 重要な注意
- **V1 モデル (models/phase9, phase10_gcp, onnx) は ACE=0 バグあり** — 使用不可
- GCPのチェックポイント (Config C 5M) は削除済み — ローカルで再学習が必要

---

## トラブルシューティング

### `import ofc_engine` で ModuleNotFoundError
```bash
python setup.py build_ext --inplace
```

### `from ofc_3max_env import ...` で ModuleNotFoundError
プロジェクトルートから実行するか、PYTHONPATH を設定:
```bash
PYTHONPATH=".:src/python" python your_script.py
```

### macOS でサーマルスロットリング
`NUM_ENVS` を下げる (CPU コア数 - 2 程度):
```bash
NUM_ENVS=2 python v2/train_v2.py ...
```

### Discord 通知が届かない
環境変数を確認:
```bash
echo $DISCORD_WEBHOOK_URL
# 空なら設定
export DISCORD_WEBHOOK_URL='https://discord.com/api/webhooks/1461388510140301315/Ofc9ok3IDLgRhFR7Oa2iA4-CHHL9uFvFbIPT9G9tbA5X5r-wG-XMbnkg7ubocLJJQ5Pf'
```
