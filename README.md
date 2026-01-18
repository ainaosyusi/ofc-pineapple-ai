# OFC Pineapple AI

Open-Face Chinese Poker (Pineapple) の深層強化学習AIプロジェクト。

## 概要

3人対戦（3-Max）のOFC Pineappleをプレイする強化学習AIを開発しています。
C++ゲームエンジン + MaskablePPO (Stable-Baselines3) による高速学習を実現。

## 現在の状況

### マルチバリアント並行学習中 (2026-01-18)

3つの異なるAIバリアントをGCPで並行学習しています：

| インスタンス | バリアント | 特徴 |
|:---|:---|:---|
| ofc-training | **Phase 8 Self-Play** | 過去モデルとの対戦学習 |
| ofc-aggressive | **Aggressive** | FL重視、高リスク高リターン |
| ofc-teacher | **Teacher Learning** | ルールベース教師の模倣 |

### Phase 7 完了実績

| 指標 | 値 | 評価 |
|:---|---:|:---|
| 総学習ステップ | 20,000,000 | - |
| ファウル率 | **25.8%** | 中級〜上級者レベル |
| Mean Royalty | **7.56** | 上級者レベル |
| FL Entry Rate | 1.1% | 改善余地あり |
| 勝率 (vs Random) | 65-68% | 安定した優位性 |

## 主な特徴

- **高性能C++エンジン**: pybind11によるPythonバインディング、学習時 4,500-12,000 FPS
- **MaskablePPO**: 有効アクションのみを選択するアクション・マスキング
- **3人対戦 (3-Max)**: 複数プレイヤーの戦略的インタラクション
- **ジョーカー対応**: 54カードデッキ（ジョーカー2枚含む）
- **Self-Play**: 過去モデルプールとの対戦による継続的強化
- **マルチバリアント学習**: 異なる報酬設計で多様な戦略を探索

## 技術スタック

- **ゲームエンジン**: C++ + pybind11
- **強化学習**: Stable-Baselines3, sb3-contrib (MaskablePPO)
- **環境**: PettingZoo AECEnv (マルチエージェント)
- **並列化**: SubprocVecEnv
- **クラウド**: GCP GCE (n2-standard-4)
- **通知**: Discord Webhooks

## プロジェクト構造

```
OFC NN/
├── src/
│   ├── cpp/                    # C++ゲームエンジン
│   │   ├── game.hpp            # ゲームロジック
│   │   ├── board.hpp           # ボード管理
│   │   ├── evaluator.hpp       # 役判定
│   │   └── pybind/             # Pythonバインディング
│   └── python/                 # 学習スクリプト
│       ├── ofc_3max_env.py     # 3人対戦環境
│       ├── train_phase8_selfplay.py    # Self-Play学習
│       ├── train_variant_aggressive.py # Aggressive学習
│       ├── train_variant_teacher.py    # Teacher学習
│       ├── visual_demo.py      # 視覚デモ
│       └── disk_cleanup.py     # ディスク管理
├── docs/
│   ├── learning/               # 学習ドキュメント
│   └── research/               # 研究レポート
├── models/                     # 学習済みモデル
└── gcp/                        # GCPセットアップ
```

## クイックスタート

### 1. エンジンのビルド

```bash
python setup.py build_ext --inplace
```

### 2. 視覚デモの実行

```bash
# AIのプレイを表示
python src/python/visual_demo.py --games 1

# 100ゲームの統計
python src/python/visual_demo.py --stats 100
```

### 3. ローカル学習

```bash
# Phase 8 Self-Play (テストモード)
NUM_ENVS=2 python src/python/train_phase8_selfplay.py --test-mode --steps 10000
```

### 4. GCPデプロイ

```bash
# GCPインスタンス作成
gcloud compute instances create ofc-training \
    --zone=asia-northeast1-b \
    --machine-type=n2-standard-4 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud

# 学習開始
ssh naoai@INSTANCE_IP "cd ~/ofc-training && source venv_linux/bin/activate && \
    NUM_ENVS=4 nohup python3 src/python/train_phase8_selfplay.py > training.log 2>&1 &"
```

## 学習フェーズ履歴

| Phase | 説明 | ファウル率 | Royalty |
|:---|:---|---:|---:|
| Phase 1 | ファウル回避基礎 | 37.8% | 0.34 |
| Phase 2 | 役作り基礎 | 32.0% | 0.26 |
| Phase 3 | 2人Self-Play | 58-63% | 0.5-0.8 |
| Phase 4 | ジョーカー対応 | **25.1%** | 0.85 |
| Phase 5 | 3人Self-Play | 38.5% | 0.78 |
| Phase 7 | 並列学習 (GCP) | **25.8%** | **7.56** |
| Phase 8 | Multi-Variant | 進行中 | - |

## 評価基準

### 人間レベル比較

| レベル | ファウル率 | Royalty |
|:---|---:|---:|
| 初心者 | 40-50% | 1-2 |
| 中級者 | 25-35% | 3-5 |
| 上級者 | 15-25% | 5-8 |
| プロ | 10-20% | 7-12 |

**現在のAI**: 中級〜上級者レベル

## 監視コマンド

```bash
# ログ確認
ssh naoai@35.243.93.32 "tail -50 ~/ofc-training/training.log"

# プロセス確認
ssh naoai@35.243.93.32 "ps aux | grep python | grep -v grep"

# ディスク使用量
ssh naoai@35.243.93.32 "df -h /home"
```

## コスト見積もり

| 構成 | コスト/時間 | 20Mステップ |
|:---|---:|---:|
| n2-standard-4 × 1 | $0.19 | ~$2 |
| n2-standard-4 × 3 | $0.57 | ~$5 |

## ドキュメント

- [評価基準](docs/research/evaluation_metrics.md)
- [ロードマップ](docs/research/roadmap_and_expectations.md)
- [学習結果サマリー](docs/research/training_results_summary.md)
- [現在の開発状況](docs/learning/04_current_status.md)

## ライセンス

Private - All Rights Reserved

---
