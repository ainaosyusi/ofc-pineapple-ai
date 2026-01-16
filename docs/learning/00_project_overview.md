# OFC Pineapple AI 開発 - プロジェクト概要

## 📅 最終更新: 2026-01-16

### プロジェクト進捗サマリー

| フェーズ | 項目 | 状態 | ファウル率 | ロイヤリティ |
|:---|:---|:---:|:---:|:---:|
| Phase 1 | C++コアエンジン + pybind11 | ✅ | - | - |
| Phase 1 | ファウル回避学習 | ✅ | 37.8% | 0.34 |
| Phase 2 | Gymnasium環境 + PPO学習 | ✅ | 32.0% | 0.26 |
| Phase 3 | Self-Play学習 (2人対戦) | ✅ | 58-63% | 0.00 |
| Phase 3 Enhanced | 確率考慮型報酬 | ✅ | 58.0% | 0.00 |
| **Phase 4** | **ジョーカー対応 (54カード)** | ✅ | **25.1%** | **0.85** |
| **Phase 5** | **3人対戦 (3-Max)** | 🚀 学習中 | (TBD) | (TBD) |

---

## 🏆 ベストモデル: Phase 4 Joker

| 項目 | 値 |
|:---|:---:|
| モデルファイル | `ofc_phase4_joker_..._10500000_steps.zip` |
| ファウル率 | **25.10%** |
| 平均ロイヤリティ | **0.85** |
| FL突入率 | 1.10% |

---

## 📊 技術スタック

| 言語/ツール | 用途 |
|:---|:---|
| C++17 | ゲームエンジン（高速処理） |
| pybind11 | Python-C++バインディング |
| Python 3.9+ | 学習環境・スクリプト |
| Stable-Baselines3 | PPO学習アルゴリズム |
| sb3-contrib | MaskablePPO（アクションマスキング）|
| PettingZoo | マルチエージェント環境 |
| AWS EC2 | 長期学習実行環境 |
| Discord Webhooks | 学習進捗通知 |

---

## 📁 主要ファイル構成

```
OFC NN/
├── src/
│   ├── cpp/                    # C++ゲームエンジン
│   │   ├── card.hpp            # カード表現（54枚対応）
│   │   ├── deck.hpp            # デッキ管理
│   │   ├── board.hpp           # ボード管理
│   │   ├── evaluator.hpp       # 役判定（ジョーカー対応）
│   │   ├── game.hpp            # ゲームエンジン（3人対応）
│   │   └── pybind/bindings.cpp # Python連携
│   │
│   └── python/                 # Python学習環境
│       ├── ofc_env.py          # シングルエージェント環境
│       ├── ofc_phase1_env.py   # Phase 1 ファウル回避環境
│       ├── multi_ofc_env.py    # 2人対戦環境
│       ├── ofc_3max_env.py     # 3人対戦環境 (Phase 5)
│       ├── train_phase*.py     # 各フェーズの学習スクリプト
│       └── evaluate_model.py   # モデル評価
│
├── models/                     # 学習済みモデル
│   ├── phase1/
│   ├── phase2/
│   ├── phase3/
│   ├── phase4/                 # ジョーカー対応
│   └── phase5/                 # 3-Max (学習中)
│
└── docs/
    ├── learning/               # 開発学習ノート
    └── research/               # 研究レポート
```

---

## 🔜 次のステップ

1. **Phase 5学習完了** - 2000万ステップ（推定14.7時間）
2. **モデル評価** - 3人対戦での勝率・ファウル率計測
3. **Phase 6検討** - MCTS統合、より高度な戦略

---

## 📝 関連ドキュメント

- [01_bitboard_basics.md](./01_bitboard_basics.md) - Bitboard技術解説
- [02_docker_basics.md](./02_docker_basics.md) - Docker基礎
- [03_aws_ec2_setup.md](./03_aws_ec2_setup.md) - AWS EC2セットアップ
- [04_current_status.md](./04_current_status.md) - 現在の開発状況
- [研究レポート](../research/ofc_ai_training_report.md) - 詳細なトレーニング分析
