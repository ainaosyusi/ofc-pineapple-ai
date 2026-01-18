# OFC Pineapple AI - 現在の開発状況

## 📅 最終更新: 2026-01-18 (Phase 8完了)

> **⚠️ 重要**: 次のフェーズは Hybrid Agent (RL + Solver) の構築

---

## 🚀 現在のステータス: Phase 8完了 → Hybrid Agent構築へ移行

Phase 8のMulti-Variant学習が**学習プラトー（停滞）**に到達。
純粋な強化学習の限界を確認し、**Hybrid Agent**への移行を決定。

### Phase 8 最終結果

| バリアント | 総ステップ | ファウル率 (Best) | Mean Score | FL Entry |
|:---|---:|---:|---:|---:|
| **Self-Play** | 9.8M | **20.8%** | +7.87 | 3.2% |
| **Aggressive** | 6.8M | 21.4% | +7.81 | 3.2% |
| **Teacher** | 6.8M | 24.8% | +7.58 | 2.0% |

### 停滞の理由

- **Explained Variance**: 0.15-0.22（低い）
- OFCのドロー運による分散が大きく、NNだけでは期待値予測が困難
- → **推論時の探索（Solver/MCTS）**で補完が必要

---

## 📊 Phase 7 → Phase 8 比較

| 指標 | Phase 7 | Phase 8 (Best) | 変化 |
|:---|---:|---:|:---|
| **ファウル率** | 25.8% | **20.8%** | -5.0% ⬆️ |
| **Mean Score** | +7.56 | +7.87 | +0.31 ⬆️ |
| **FL Entry** | 1.1% | 3.2% | +2.1% ⬆️ |

---

## 🎯 次のフェーズ: Hybrid Agent

### アーキテクチャ

```
Round 1 (初手5枚):   RL Policy Network     → 大局観・定石
Round 2-3 (ドロー):  RL + 軽量MCTS         → リスクバランス
Round 4 (ドロー):    MCTS (探索増加)       → 確定的リスク回避
Round 5 (最終):      完全解析ソルバー       → ミスゼロの確定
```

### 期待される改善

| コンポーネント | 効果 | 累積ファウル率 |
|:---|:---|---:|
| 最良RLモデル | ベースライン | 20.8% |
| + Round 5 Solver | -5% | ~15-16% |
| + MCTS補正 | -2% | ~13-14% |
| **目標** | | **< 15%** |

### 実装済みコンポーネント

| コンポーネント | ファイル | 状態 |
|:---|:---|:---|
| HybridInferenceAgent | `src/python/hybrid_agent.py` | ✅ 実装済 |
| EndgameSolver | `src/python/endgame_solver.py` | ✅ 実装済 |
| MCTSFLAgent | `src/python/mcts_agent.py` | ✅ 実装済 |
| FantasySolver (C++) | `src/cpp/solver.hpp` | ✅ 実装済 |
| FL確率計算 (C++) | `ofc_engine` | ✅ リビルド済 |

---

## 📊 フェーズ別進捗

```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 7 ──► Phase 8 ──► Hybrid
 ファウル     役作り      Self-Play    ジョーカー    3人対戦     並列学習     Multi-      RL+Solver
回避学習    基礎       (2人)       対応       Self-Play   完了       Variant     構築中
37.8%      32.0%      58-63%     25.1%      38.5%      25.8%      20.8% 🏆    目標<15%
```

### 達成したマイルストーン

| フェーズ | 目標 | 結果 | 評価 |
|:---|:---|:---|:---:|
| Phase 1 | ファウル回避の基礎習得 | 37.8% | ✅ |
| Phase 2 | 役作りと報酬最大化 | 32.0%, Royalty 0.26 | ✅ |
| Phase 3 | 対戦型戦略の習得 | 58-63%（攻撃的） | ⚠️ |
| Phase 4 | ジョーカー活用 | 25.1%, Royalty 0.85 | ✅ |
| Phase 5 | 3人戦の戦略習得 | 38.5%, 1150万ステップ | ✅ |
| Phase 7 | 並列学習 (GCP) | 25.8%, Royalty 7.56 | ✅ |
| **Phase 8** | Multi-Variant学習 | **20.8%**, Score +7.87 | 🏆 |
| Hybrid | RL + Solver | 目標: <15% | 🔜 |

---

## 🔧 次のステップ

1. [x] Phase 8 Multi-Variant学習（3インスタンス並行）
2. [x] 学習プラトー到達の確認
3. [x] HybridInferenceAgent実装
4. [x] C++エンジン リビルド（MCTS関数有効化）
5. [ ] **最良チェックポイントの保存・選定**
6. [ ] **Round 5 Solverの統合テスト**
7. [ ] **Hybrid Agent vs Pure RL 対戦評価**
8. [ ] 対人テスト
9. [ ] UI/デモ作成

---

## 📁 保存されたモデル

### Phase 8 推奨モデル

| バリアント | チェックポイント | ファウル率 | 用途 |
|:---|:---|---:|:---|
| **Self-Play** | `p8_selfplay_5000000.zip` | ~21% | Hybrid Agent基盤（推奨） |
| **Aggressive** | `aggressive_1000000.zip` | ~24% | FL重視版 |

### GCPインスタンス

| インスタンス | IP | 状態 |
|:---|:---|:---|
| ofc-training | 35.243.93.32 | 学習完了（停止推奨） |
| ofc-aggressive | 34.146.34.141 | 学習完了（停止推奨） |
| ofc-teacher | 35.200.57.236 | 学習完了（停止推奨） |

---

## 📞 コマンドリファレンス

### Hybrid Agent テスト
```bash
cd "/Users/naoai/試作品一覧/OFC NN"
source .venv/bin/activate
python src/python/hybrid_agent.py --test
```

### GCPインスタンス停止
```bash
gcloud compute instances stop ofc-training --zone=asia-northeast1-b
gcloud compute instances stop ofc-aggressive --zone=asia-northeast1-b
gcloud compute instances stop ofc-teacher --zone=asia-northeast1-b
```

### チェックポイント取得
```bash
scp naoai@35.243.93.32:~/ofc-training/models/p8_selfplay_5000000.zip ./models/
```

---

## 📁 関連ドキュメント

| ファイル | 説明 |
|:---|:---|
| `docs/research/phase8_training_analysis.md` | Phase 8詳細分析レポート |
| `docs/learning/07_phase8_multivariant.md` | Phase 8学習ドキュメント |
| `NEXT_ACTIONS.md` | 次のアクション一覧 |
| `CLAUDE.md` | プロジェクトガイドライン |

---

*Last updated: 2026-01-18*
