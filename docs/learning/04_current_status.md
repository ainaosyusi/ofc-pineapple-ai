# OFC Pineapple AI - 現在の開発状況

## 最終更新: 2026-02-07

> **現在の状態**: Phase 9 完了 ✅ / Phase 10 学習中 (FL Stay向上)

---

## 最新結果

### Phase 9: FL Mastery (250M) - 完了 ✅

Phase 9では報酬シェーピングを改善し、FL Entry/Stayに追加報酬を付与。
250Mステップで全目標を達成しました。

| 指標 | Phase 8.5b (150M) | Phase 9 (250M) | 目標 | 達成 |
|:-----|---:|---:|---:|:---:|
| Foul Rate | 22.7% | **16.8%** | < 20% | ✅ |
| Mean Score | +9.04 | **+12.66** | > +10 | ✅ |
| FL Entry Rate | 7.9% | **22.8%** | > 15% | ✅ |
| FL Stay Rate | 3.0% | **8.0%** | > 5% | ✅ |
| Win Rate | 71.2% | **75.8%** | > 70% | ✅ |

**成功要因**:
- FL Entry報酬: 30ポイント
- FL Stay報酬: 60ポイント
- 継続的Self-Play学習

### Phase 10: FL Stay向上 (進行中)

Phase 9ではFL Stay Rate 8%を達成したが、目標は15%+。
greedy_fl_solverのFL Stay探索を修正し、Fine-tuning中。

| 指標 | Phase 9 (250M) | Phase 10 現在 | 目標 |
|:-----|---:|---:|---:|
| FL Entry Rate | 22.8% | ~18% | 27%+ |
| FL Stay Rate | 8.0% | ~7% | 15%+ |

**修正内容**:
- `greedy_fl_solver.py`: FL Stay探索を完全書き換え
  - Trips on Top 探索を追加
  - Quads on Bottom 探索を優先
- Discord通知を100kステップごとに送信

---

## フェーズ別進捗

```
Phase 1 → Phase 2 → ... → Phase 8.5b → Phase 9 → Phase 10
 ファウル   役作り        FL継続      FL Mastery   FL Stay
回避学習   基礎          完了        完了 ✅      進行中
37.8%     32.0%        22.7%       16.8%       -
```

### 達成マイルストーン

| フェーズ | 目標 | 結果 | 評価 |
|:--------|:-----|:-----|:----:|
| Phase 1 | ファウル回避の基礎習得 | 37.8% | ✅ |
| Phase 2 | 役作りと報酬最大化 | 32.0% | ✅ |
| Phase 3 | 対戦型戦略の習得 | 58-63% | ⚠️ |
| Phase 4 | ジョーカー活用 | 25.1% | ✅ |
| Phase 5 | 3人戦の戦略習得 | 38.5% | ✅ |
| Phase 7 | 並列学習 (GCP) | 25.8% | ✅ |
| Phase 8 | Multi-Variant学習 | 20.8% | ✅ |
| Phase 8.5 | Ultimate Rules FL | 22.0%, FL 8.2% | ✅ |
| Phase 8.5b | FL継続学習 | 22.7%, FL Stay 3.0% | ✅ |
| ExIt実験 | MCTS Expert Iteration | Foul 53.3% (失敗) | ❌ |
| **Phase 9** | **FL Mastery** | **16.8%, FL Entry 22.8%** | **✅** |
| Phase 10 | FL Stay向上 | 進行中 | ⏳ |

---

## 保存されたモデル

### 推奨モデル

| モデル | ステップ | 説明 |
|:-------|:--------:|:-----|
| `models/phase9/p9_fl_mastery_250000000.zip` | 250M | **最新推奨** |
| `models/onnx/ofc_ai.onnx` | - | Node.js統合用 |

### Phase 10 (学習中)

| モデル | ステップ | 説明 |
|:-------|:--------:|:-----|
| `models/phase10/p10_fl_stay_100000.zip` | 100k | 最初のチェックポイント |

### 過去モデル

| モデル | フェーズ | 特徴 |
|:-------|:---------|:-----|
| `models/phase9/p9_fl_mastery_150000000.zip` | Phase 9 | 中間チェックポイント |
| `models/phase85b/` | Phase 8.5b | Solver修正版 |
| `models/exit/` | ExIt実験 | 失敗、使用非推奨 |

---

## 次のステップ

1. [x] Phase 8.5 100Mステップ学習完了
2. [x] FL Stay Rate 0% バグ修正 (`continuous_games=True`)
3. [x] FantasySolver ボトルネック修正 (Greedy FL Solver)
4. [x] Phase 8.5b GCP学習完了 (150M steps)
5. [x] Expert Iteration実験 (結果: 失敗)
6. [x] **Phase 9 FL Mastery完了 (250M steps)**
7. [x] **ONNX変換・mix-poker-app統合**
8. [ ] **Phase 10 FL Stay向上 (進行中)**
9. [ ] Phase 10完了後: 対人テスト・UI改善
10. [ ] Phase 11: Hold'em AI (将来)

---

## デプロイ状況

### mix-poker-app 統合

OFC AI v1.1.0が `mix-poker-app` に統合済み:
- ONNX形式でNode.jsから直接推論
- 推論速度: 10-50ms/手
- ファイル: `mix-poker-app/server/models/ofc_ai.onnx`

### GCPインスタンス

| インスタンス | 状態 | 用途 |
|:-------------|:-----|:-----|
| ofc-training | 停止推奨 | Phase 10はローカルで実行中 |

---

## 技術的知見

### PPO学習の飽和

Phase 8.5bの実験から、PPO Self-Playは110M以降で飽和することが判明。
これを突破したのがPhase 9の報酬シェーピング改善:

```python
# Phase 9 報酬設計
fl_entry_bonus = 30.0  # FL Entry達成時
fl_stay_bonus = 60.0   # FL Stay達成時
```

### FL Stay条件

FL Stayには以下のいずれかが必要:
1. **Trips on Top** (任意ランク) - Bot >= Mid >= Tripsが必要で非常に困難
2. **Quads on Bottom** - 実用的なパス、Joker活用が鍵

### Greedy FL Solver

ブルートフォースソルバー (O(4^n)) は17枚手でFPS=2まで低下。
Monte Carloサンプリングによるgreedy_fl_solverで95%品質、14,000倍高速化。

---

## 関連ドキュメント

| ファイル | 説明 |
|:---------|:-----|
| [phase9_fl_mastery_report.md](../reports/phase9_fl_mastery_report.md) | Phase 9 最終レポート |
| [phase85b_solver_fix_report.md](../reports/phase85b_solver_fix_report.md) | Phase 8.5b Solver修正 |
| [expert_iteration_experiment.md](../reports/expert_iteration_experiment.md) | ExIt実験レポート (失敗分析) |
| [CLAUDE.md](../../CLAUDE.md) | プロジェクトガイドライン |
| [NEXT_ACTIONS.md](../../NEXT_ACTIONS.md) | 次のアクション |

---

*Last updated: 2026-02-07*
