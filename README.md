# OFC Pineapple AI

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C++-17-00599C?logo=cplusplus&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C?logo=pytorch&logoColor=white)
![SB3](https://img.shields.io/badge/Stable--Baselines3-MaskablePPO-blue)
![Tests](https://img.shields.io/badge/Tests-124%2F124%20passed-brightgreen)
![License](https://img.shields.io/badge/License-Private-red)

Open-Face Chinese Poker (Pineapple) 3-Max の強化学習AIプロジェクト。

## 概要

- 3人同時対戦の OFC Pineapple を MaskablePPO で学習
- C++ ゲームエンジン + pybind11 による高速シミュレーション
- 54枚デッキ（Joker 2枚）、Fantasy Land (Ultimate Rules) 対応

## 現在の状況 — V2 再学習中

V1 で ACE=0 バグが発覚し、250M+ ステップの学習結果が無効に。
V2 としてゼロから再設計・再学習を進行中。

### V2 完了済み
- **Phase 0**: C++ エンジン整備、報酬設計 3案 (A/B/C)
- **Phase 1 Step 1**: 報酬アブレーション → **Config C (条件付きシェーピング) 採用確定**
  - 5M 結果: Foul 33.2%, Score +5.86, FL追求差 +2.0%
- **C++ エンジン網羅テスト**: 124/124 パス（ACE, 全役比較, Joker, FL条件, スコア計算, フォール判定）

### V2 次のタスク
- **Phase 1 Step 2**: Self-Play プール拡張比較 (D: baseline vs E: extended)
- **Phase 2**: 行動空間分離 + 30M 学習
- **Phase 3**: Embedding + 250M 本学習

## 技術スタック

| レイヤー | 技術 |
|---------|------|
| ゲームエンジン | C++17 (Header-only) + pybind11 |
| 強化学習 | MaskablePPO (sb3-contrib) |
| 環境 | PettingZoo AECEnv (3人マルチエージェント) |
| 並列化 | SubprocVecEnv (spawn) |
| 通知 | Discord Webhooks |
| デプロイ | ONNX Runtime (Node.js) |

## クイックスタート

```bash
# 1. 仮想環境
python3 -m venv .venv && source .venv/bin/activate

# 2. 依存パッケージ（完全再現）
pip install -r requirements-lock.txt

# 3. C++ エンジンビルド
python setup.py build_ext --inplace
python -c "import ofc_engine as ofc; print('OK')"

# 4. テスト（124テスト全パス確認）
python tests/test_evaluator_comprehensive.py

# 5. 学習テスト（数十秒で完了）
python v2/train_v2.py --test-mode --reward-config C

# 6. 本学習（Config C）
DISCORD_WEBHOOK_URL='...' NUM_ENVS=4 python v2/train_v2.py --reward-config C --steps 5000000
```

詳細は [SETUP_GUIDE.md](SETUP_GUIDE.md) を参照。

## プロジェクト構成

```
OFC NN/
├── src/
│   ├── cpp/                        # C++ ゲームエンジン
│   │   ├── evaluator.hpp           #   ハンド評価 (ACE=14 修正済み)
│   │   ├── board.hpp               #   Top(3)/Mid(5)/Bot(5) 管理
│   │   ├── game.hpp                #   ゲーム進行・FL処理
│   │   ├── solver.hpp              #   FL ブルートフォースソルバー
│   │   └── pybind/bindings.cpp     #   Python バインディング
│   └── python/
│       ├── ofc_3max_env.py         #   PettingZoo 3人環境
│       ├── greedy_fl_solver.py     #   FL 近似ソルバー
│       └── notifier.py             #   Discord 通知
├── v2/
│   ├── train_v2.py                 # V2 学習スクリプト
│   ├── rule_based_agent.py         # ルールベースエージェント (Safe/Aggressive)
│   └── evaluate_benchmark.py       # ベンチマーク評価
├── tests/
│   ├── test_evaluator_comprehensive.py  # C++ エンジン 124テスト
│   └── test_joker.py               # Joker テスト
├── docs/
│   ├── templates/                  # レポートテンプレート
│   ├── experiments/                # 実験記録
│   └── learning/                   # 技術学習ノート
├── archive/v1/                     # V1 全記録 (git除外)
├── setup.py                        # C++ ビルド設定
├── requirements.txt                # 最小依存
├── requirements-lock.txt           # 完全再現用ロック
├── CLAUDE.md                       # プロジェクト運営ルール
├── SETUP_GUIDE.md                  # 環境構築ガイド
└── RESUME_PROMPT.md                # Claude Code 移行用プロンプト
```

## V2 学習コマンド

```bash
# Config C（確定済み報酬設定）でゼロから
NUM_ENVS=4 python v2/train_v2.py --reward-config C --steps 5000000

# Step 2: D（ベースライン）
python v2/train_v2.py --reward-config C --steps 10000000 \
  --run-name D --pool-size 5 --latest-prob 0.8 --rule-based-prob 0.0

# Step 2: E（拡張版 — Self-Play プール拡大 + ルールベース混合）
python v2/train_v2.py --reward-config C --steps 10000000 \
  --run-name E --pool-size 15 --latest-prob 0.6 --rule-based-prob 0.1

# チェックポイントからの再開（自動検出）
python v2/train_v2.py --reward-config C --steps 10000000
```

## V1 最終結果（参考値、ACE バグあり）

| 指標 | Phase 9 (250M) |
|------|:-:|
| Foul Rate | 16.8% |
| FL Entry | 22.8% |
| Win Rate | 75.8% |

V1 の全記録は `archive/v1/PROJECT_HISTORY.md` に保存。

## ライセンス

Private - All Rights Reserved
