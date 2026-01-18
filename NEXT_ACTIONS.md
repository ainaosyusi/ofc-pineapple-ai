# OFC AI - 次のアクション (必読)

**最終更新: 2026-01-18**

---

## ⚠️ 重要: Phase 8.5 Full FL学習準備完了

Phase 8のHybrid Agent統合テストでEndgame Solverに問題を発見。
方針転換し、**Phase 8.5: Full FL学習**を開始する準備が整った。

---

## 🚀 Phase 8.5: Full FL Training + Ultimate Rules

### 概要

実践形式の学習を行い、**Ultimate Rules**（FL枚数変動）を導入。

| 機能 | Phase 8 | Phase 8.5 |
|:---|:---:|:---:|
| FL Turns処理 | ✅ | ✅ |
| FL状態引き継ぎ | ❌ | ✅ |
| ボタンローテーション | ❌ | ✅ |
| **Ultimate Rules** | ❌ | ✅ |
| 実践形式 | 部分的 | 完全 |

### Ultimate Rules: FL配布枚数

FL突入時のTopハンドによって配布枚数が変化：

| Top Hand | FL Cards | 継続難易度 |
|:---|---:|:---|
| QQ | 14枚 | 標準 |
| KK | 15枚 | やや容易 |
| AA | 16枚 | 容易 |
| Trips (222+) | 17枚 | 最も容易 |

強いハンドでFL突入するほど、枚数が多く継続しやすい。

### 期待される効果

1. **FL Entry戦略の深化**: 強いTopを狙うリスク/リワードの学習
2. **FL継続チェーンの学習**: 17枚 → 連続FL継続の可能性
3. **Mean Scoreの大幅向上**: FL効果で+15以上のスコアが頻発

### 学習スクリプト

```bash
# ローカルテスト
cd /Users/naoai/試作品一覧/OFC\ NN
python3 src/python/train_phase85_full_fl.py --test-mode

# GCP本番学習
ssh naoai@INSTANCE_IP
cd ~/ofc-training && source venv_linux/bin/activate
NUM_ENVS=4 nohup python3 src/python/train_phase85_full_fl.py > training.log 2>&1 &
```

### 監視指標

| 指標 | Phase 8 Best | 目標 |
|:---|---:|---:|
| Foul Rate | 20.8% | < 20% |
| FL Entry Rate | 3.2% | > 5% |
| FL Stay Rate | N/A | > 30% |
| High Score (≥15) Rate | N/A | > 5% |
| Mean Score | +7.87 | > +10 |

---

## 📋 完了したアクション

### ✅ 1. 最良チェックポイントの保存

**状態**: 完了

```
models/phase8/
├── p8_selfplay_5000000.zip  (推奨ベースモデル)
├── p8_selfplay_9800000.zip  (最新)
└── aggressive_1000000.zip   (FL重視版)
```

### ✅ 2. GCPインスタンス停止

**状態**: 完了

### ✅ 3. Hybrid Agent統合テスト

**状態**: 完了（問題発見）

### ✅ 4. Ultimate Rules実装

**状態**: 完了

FL配布枚数の変動を実装：

- `src/cpp/game.hpp`: `start_with_fl_cards()` メソッド追加
- `src/cpp/pybind/bindings.cpp`: Python API公開
- `src/python/ofc_3max_env.py`: FL枚数計算＆引き継ぎ

テスト結果:
```
QQ on top: 14 cards ✅
KK on top: 15 cards ✅
AA on top: 16 cards ✅
Trips on top: 17 cards ✅
```

### ✅ 5. Phase 8.5環境修正

**状態**: 完了

- ボタンローテーション: 自動実行
- FLプレイヤー除外: ターン順から除外
- FL状態引き継ぎ: `continuous_games=True`で有効

---

## 🔬 統合テスト結果

| 設定 | ファウル率 | 評価 |
|:---|---:|:---|
| **Pure NN** (ベースライン) | 22-24% | Phase 8学習結果と一致 |
| **Hybrid (NN-only)** | 23.5% | Pure NNと同等 |
| **Hybrid (保守的設定)** | 25.0% | +3% ファウル |
| **Hybrid (デフォルト)** | 43.0% | Endgame Solverが悪影響 |

### 発見した問題

1. **Endgame Solverのバグ** (修正済み)
   - アクションエンコーディングの不整合
   - モード判定ロジックの不一致

2. **Endgame Solver自体の品質問題** (未解決)
   - 修正後もファウル率を悪化させる
   - 全探索ロジックまたは評価関数に問題あり

### 現在の推奨設定

**Pure NN**または**HybridConfig（デフォルト設定）**を推奨:

```python
# デフォルトでEndgameとMCTSは無効化済み
hybrid_config = HybridConfig()  # そのまま使用可能

# または明示的に設定
hybrid_config = HybridConfig(
    endgame_max_remaining=0,      # Endgame無効化
    critical_fl_threshold=1.0,    # MCTS無効化
    critical_royalty_threshold=100,
    critical_foul_risk=1.0,
)
```

**注意**: `HybridConfig`のデフォルト値は2026-01-18に更新され、
Endgame SolverとMCTSはデフォルトで無効化されています。

---

## 📊 Phase 8 最終結果

| バリアント | 総ステップ | ファウル率 (Best) | Mean Score | FL Entry |
|:---|---:|---:|---:|---:|
| **Self-Play** | 9.8M | **20.8%** | +7.87 | 3.2% |
| **Aggressive** | 6.8M | 21.4% | +7.81 | 3.2% |
| **Teacher** | 6.8M | 24.8% | +7.58 | 2.0% |

### Phase 7 → Phase 8 改善

| 指標 | Phase 7 | Phase 8 (Best) | 変化 |
|:---|---:|---:|:---|
| ファウル率 | 25.8% | **20.8%** | **-5.0%** |
| Mean Score | +7.56 | +7.87 | +0.31 |
| FL Entry | 1.1% | 3.2% | +2.1% |

---

## 🎯 Hybrid Agent構築計画

### アーキテクチャ

```
Round 1 (初手5枚):   RL Policy Network     → 大局観・定石
Round 2-3 (ドロー):  RL + 軽量MCTS         → リスクバランス
Round 4 (ドロー):    MCTS (探索増加)       → 確定的リスク回避
Round 5 (最終):      完全解析ソルバー       → ミスゼロの確定
```

### 実装済みコンポーネント

| コンポーネント | ファイル | 状態 |
|:---|:---|:---|
| HybridInferenceAgent | `src/python/hybrid_agent.py` | ✅ |
| EndgameSolver | `src/python/endgame_solver.py` | ✅ |
| MCTSFLAgent | `src/python/mcts_agent.py` | ✅ |
| FantasySolver (C++) | `src/cpp/solver.hpp` | ✅ |
| FL確率計算 (C++) | `ofc_engine` | ✅ |

### 期待される改善（要再評価）

| コンポーネント | 当初期待 | 実測 | 状態 |
|:---|:---|:---|:---|
| 最良RLモデル | 20.8% | 22-24% | ✅ 動作確認 |
| + Round 5 Solver | -5% | **+20%** | ❌ 改善必要 |
| + MCTS補正 | -2% | 効果不明 | ⚠️ 要検証 |

---

## 📋 次のアクション

### 1. Endgame Solver改善 [優先度: 高]

**状態**: 試行済み - 改善効果なし

#### 試行した改善

1. **アクションエンコーディング修正** ✅
   - `endgame_solver.py:352-368` - 正しいエンコーディングに修正
   - 修正後は全アクションが有効になった

2. **評価関数の改善** ✅
   - `endgame_solver.py:213-326` - 包括的な評価関数に書き換え
   - ファウル検出、行順序チェック、FL考慮、ロイヤリティボーナス等を追加

3. **NN検証モード** ✅
   - `hybrid_agent.py:337-414` - NNを検証し問題がある場合のみ上書き
   - 結果: それでもPure NNより6.5%悪化

#### 結論

現在の評価関数では、NNが数百万ゲームから学習した「暗黙の知識」を再現できない。
Solverが「良い」と判断する選択が、実際には次のターン以降で問題を起こす。

#### 今後の対応案

1. **NNの価値関数を利用**: Solverの評価にNNのvalue outputを組み込む
2. **Endgame専用NN訓練**: 終盤局面に特化したNNを別途訓練
3. **より保守的な介入条件**: NNが明らかにファウルを選ぶ場合のみ介入

### 2. MCTS単体評価 [優先度: 中]

**状態**: 未着手

検証内容:
- MCTSのみを使用した場合の効果測定
- シミュレーション数の最適化
- FL確率計算の精度検証

### 3. 対人テスト [優先度: 低]

**状態**: 待機中

現在のPure NNモデルで基本的な対人テストは可能

---

## 🗓️ 今後のロードマップ

### Phase 8.5: Full FL Training (現在)

1. ✅ 環境修正（ボタンローテーション、FL除外）
2. ✅ Phase 8.5学習スクリプト作成
3. ⬜ GCPデプロイ＆学習開始
4. ⬜ 学習結果分析

### Phase 9: Hybrid Agent改善（Phase 8.5後）

1. NNの価値関数をSolverに組み込む
2. Endgame専用NNの訓練
3. MCTS効果の再検証

### Phase 10: 最終評価

1. 対人テスト
2. UI/デモ作成
3. ドキュメント整備

---

## 📝 学んだこと（Phase 8）

1. **純粋RLの限界**
   - Explained Variance 0.15-0.22（低い）
   - ドロー運の分散が大きく、NNだけでは期待値予測が困難

2. **報酬シェーピングの限界**
   - FL突入率は改善できるが
   - ファウル率の根本的改善には寄与しない

3. **次のステップ**
   - 「学習」から「推論の質」へ軸足を移す
   - Solver/MCTSで計算力を補完

---

## 📁 関連ドキュメント

| ファイル | 説明 |
|:---|:---|
| `docs/research/phase8_training_analysis.md` | Phase 8詳細分析レポート |
| `docs/learning/07_phase8_multivariant.md` | Phase 8学習ドキュメント |
| `docs/learning/04_current_status.md` | 現在の開発状況 |

---

*このファイルは `CLAUDE.md` で参照が強制されています。*
