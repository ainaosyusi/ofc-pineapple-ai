# OFC Pineapple AI

Open-Face Chinese Poker (Pineapple) の深層強化学習AIプロジェクト。

## 概要

3人対戦（3-Max）のOFC Pineappleをプレイする強化学習AIを開発しています。
C++ゲームエンジン + MaskablePPO (Stable-Baselines3) による高速学習を実現。

## 現在の状況

### Phase 9 完了・Phase 10 学習中 (2026-02-07)

**Phase 9 (FL Mastery)** が250Mステップで完了し、全目標を達成しました。
現在は **Phase 10 (FL Stay向上)** を学習中です。

#### Phase 9 最終結果

| 指標 | Phase 9 (250M) | 目標 | 評価 |
|:---|---:|:---|:---|
| Foul Rate | **16.8%** | <20% | ✅ プロレベル |
| Mean Score | **+12.66** | >+10 | ✅ 達成 |
| FL Entry Rate | **22.8%** | >15% | ✅ 達成 |
| FL Stay Rate | **8.0%** | >5% | ✅ 達成 |
| Win Rate | **75.8%** | >70% | ✅ 達成 |

#### Phase 10 進行中

- **目標**: FL Stay Rate 8% → 15%+
- **手法**: 修正greedy_fl_solverでのFine-tuning
- **現在**: 〜400k / 50M ステップ

## 主な特徴

- **高性能C++エンジン**: pybind11によるPythonバインディング、学習時 900-1000 FPS
- **MaskablePPO**: 有効アクションのみを選択するアクション・マスキング
- **3人対戦 (3-Max)**: 複数プレイヤーの戦略的インタラクション
- **ジョーカー対応**: 54カードデッキ（ジョーカー2枚含む）
- **Fantasy Land**: Ultimate Rules対応（QQ=14枚, KK=15枚, AA=16枚, Trips=17枚）
- **Self-Play**: 同一モデルとの対戦による継続的強化
- **ONNX Export**: Node.js統合用のONNXモデル出力対応

## 技術スタック

- **ゲームエンジン**: C++ (Header-only) + pybind11
- **強化学習**: Stable-Baselines3, sb3-contrib (MaskablePPO)
- **環境**: PettingZoo AECEnv (マルチエージェント)
- **並列化**: SubprocVecEnv (spawn method)
- **クラウド**: GCP GCE (e2-standard-8)
- **通知**: Discord Webhooks
- **デプロイ**: ONNX Runtime (Node.js)

## プロジェクト構造

```
OFC NN/
├── src/
│   ├── cpp/                    # C++ゲームエンジン (Header-only)
│   │   ├── game.hpp            # ゲームロジック、FL処理
│   │   ├── board.hpp           # ボード管理 (bitboard)
│   │   ├── evaluator.hpp       # 役判定 (Joker対応)
│   │   ├── solver.hpp          # FL最適配置ソルバー
│   │   └── pybind/bindings.cpp # Pythonバインディング
│   └── python/                 # 学習・評価スクリプト
│       ├── ofc_3max_env.py     # 3人対戦PettingZoo環境
│       ├── greedy_fl_solver.py # 高速FL配置ソルバー
│       ├── train_phase10_fl_stay.py  # Phase 10学習
│       └── notifier.py         # Discord通知
├── docs/
│   ├── learning/               # 学習ドキュメント
│   ├── research/               # 研究レポート
│   ├── reports/                # フェーズ別レポート
│   └── blog/                   # 開発ブログ
├── models/                     # 学習済みモデル
│   ├── phase9/                 # Phase 9 チェックポイント
│   ├── phase10/                # Phase 10 チェックポイント
│   └── onnx/                   # ONNX変換済みモデル
├── scripts/                    # 評価・テストスクリプト
└── gcp/                        # GCPセットアップ
```

## クイックスタート

### 1. エンジンのビルド

```bash
python setup.py build_ext --inplace
python -c "import ofc_engine as ofc; print('Engine loaded')"
```

### 2. AIのテスト実行

```bash
# AIのプレイを表示
python src/python/visual_demo.py --games 1

# 100ゲームの統計
python src/python/visual_demo.py --stats 100
```

### 3. ローカル学習

```bash
# Phase 10 テスト (少量ステップ)
NUM_ENVS=2 python src/python/train_phase10_fl_stay.py --steps 10000
```

### 4. GCPデプロイ

```bash
# インスタンス起動
gcloud compute instances start ofc-training --zone=asia-northeast1-b

# コード転送
gcloud compute scp --recurse --zone=asia-northeast1-b \
    src models setup.py ofc-training:~/ofc-training/

# 学習開始
gcloud compute ssh ofc-training --zone=asia-northeast1-b \
    --command="cd ~/ofc-training && NUM_ENVS=4 nohup python3 \
    src/python/train_phase10_fl_stay.py > training.log 2>&1 &"
```

## 学習フェーズ履歴

| Phase | 説明 | Steps | Foul Rate | FL Entry | Win Rate |
|:---|:---|---:|---:|---:|---:|
| Phase 1-4 | 基礎学習 | 20M | 25% | - | - |
| Phase 5 | 3-Max導入 | 30M | 38% | - | - |
| Phase 7 | 並列学習 | 20M | 26% | 1% | 65% |
| Phase 8 | Self-Play | 100M | 22% | 8% | 69% |
| Phase 8.5 | FL導入 | 100M | 22% | 8% | 69% |
| Phase 8.5b | Solver修正 | 150M | 18% | 21% | 75% |
| **Phase 9** | **FL Mastery** | **250M** | **16.8%** | **22.8%** | **75.8%** |
| Phase 10 | FL Stay向上 | 進行中 | - | - | - |

## 評価基準

### 人間レベル比較

| レベル | Foul Rate | Royalty | FL Entry |
|:---|---:|---:|---:|
| 初心者 | 40-50% | 1-2 | 0-5% |
| 中級者 | 25-35% | 3-5 | 5-10% |
| 上級者 | 15-25% | 5-8 | 10-20% |
| プロ | 10-20% | 7-12 | 15-30% |

**現在のAI (Phase 9)**: プロレベル (Foul 16.8%, FL Entry 22.8%)

## 保存済みモデル

| ファイル | Steps | 用途 |
|:---|---:|:---|
| `models/phase9/p9_fl_mastery_250000000.zip` | 250M | 最新推奨モデル |
| `models/onnx/ofc_ai.onnx` | - | Node.js統合用 |

## ドキュメント

- [現在の開発状況](docs/learning/04_current_status.md)
- [Phase 9 レポート](docs/reports/phase9_fl_mastery_report.md)
- [評価基準](docs/research/evaluation_metrics.md)
- [次のアクション](NEXT_ACTIONS.md)

## ライセンス

Private - All Rights Reserved

---
