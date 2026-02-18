# OFC Pineapple AI - 現在の開発状況

## 最終更新: 2026-02-19

> **現在の状態**: Phase 10 完了 ✅ (FL Stay向上、150M steps)

---

## 最新結果

### Phase 10: FL Stay Training (150M) - 完了 ✅

greedy_fl_solver v3を導入し、Flush/Straight検出を追加。
Phase 9から全指標が大幅に改善。

| 指標 | Phase 9 (250M) | Phase 10 (150M) | 改善 |
|:-----|:--------------:|:---------------:|:----:|
| Foul Rate | 27.2% | **18.0%** | **-9.2%** |
| Mean Score | +17.92 | **+36.08** | **+18.16** |
| Mean Royalty | +3.00 | **+6.10** | **+3.10** |
| FL Entry Rate | 17.8% | **33.6%** | **+15.8%** |
| FL Stay Rate | 5.0% | **15.8%** | **+10.8%** |
| FL Stay/Entry | 28.1% | **47.0%** | **+18.9%** |
| Win Rate | 53.0% | **55.2%** | **+2.2%** |

**成功要因**:
- greedy_fl_solver v3: Flush/Straight検出、高ロイヤリティ組み合わせ探索
- FL Entry報酬: 50ポイント、FL Stay報酬: 100ポイント
- GCP e2-standard-8 で150M steps学習

### Phase 10 学習推移

| Metric | P10 (50M) | P10 (100M) | P10 (150M) |
|:-------|:---------:|:----------:|:----------:|
| Foul Rate | 19.2% | 18.2% | **18.0%** |
| Mean Score | +38.43 | +31.72 | **+36.08** |
| FL Entry | 34.8% | 31.8% | **33.6%** |
| FL Stay | 16.8% | 12.2% | **15.8%** |

50M stepsで大部分が収束。100Mで一時的に低下するが、150Mで安定。

---

## フェーズ別進捗

```
Phase 1 → Phase 2 → ... → Phase 8.5b → Phase 9 → Phase 10
 ファウル   役作り        FL継続      FL Mastery   FL Stay
回避学習   基礎          完了        完了 ✅      完了 ✅
37.8%     32.0%        22.7%       16.8%       18.0%
```

### 達成マイルストーン

| フェーズ | 目標 | 結果 | 評価 |
|:--------|:-----|:-----|:----:|
| Phase 1 | ファウル回避の基礎習得 | 37.8% | ✅ |
| Phase 2 | 役作りと報酬最大化 | 32.0% | ✅ |
| Phase 3 | 対戦型戦略の習得 | 58-63% | ✅ |
| Phase 4 | ジョーカー活用 | 25.1% | ✅ |
| Phase 5 | 3人戦の戦略習得 | 38.5% | ✅ |
| Phase 7 | 並列学習 (GCP) | 25.8% | ✅ |
| Phase 8 | Multi-Variant学習 | 20.8% | ✅ |
| Phase 8.5 | Ultimate Rules FL | 22.0%, FL 8.2% | ✅ |
| Phase 8.5b | FL継続学習 | 22.7%, FL Stay 3.0% | ✅ |
| ExIt実験 | MCTS Expert Iteration | Foul 53.3% (失敗) | ❌ |
| **Phase 9** | **FL Mastery** | **16.8%, FL Entry 22.8%** | **✅** |
| **Phase 10** | **FL Stay向上** | **18.0%, FL Stay 15.8%** | **✅** |

---

## 保存されたモデル

### 推奨モデル

| モデル | ステップ | 説明 |
|:-------|:--------:|:-----|
| `models/phase10_gcp/p10_fl_stay_150000000.zip` | 150M | **最新推奨** |
| `models/onnx/ofc_ai.onnx` | - | Node.js統合用 (Phase 9ベース) |

### Phase 10 チェックポイント

| モデル | ステップ | 説明 |
|:-------|:--------:|:-----|
| `models/phase10_gcp/p10_fl_stay_150000000.zip` | 150M | 最終モデル |
| `models/phase10_gcp/p10_fl_stay_100000000.zip` | 100M | 中間 |
| `models/phase10_gcp/p10_fl_stay_50000000.zip` | 50M | 初期収束 |

### 過去モデル

| モデル | フェーズ | 特徴 |
|:-------|:---------|:-----|
| `models/phase9/p9_fl_mastery_250000000.zip` | Phase 9 | FL Mastery |
| `models/phase85b/` | Phase 8.5b | Solver修正版 |

---

## 次のステップ

1. [x] Phase 8.5 100Mステップ学習完了
2. [x] FL Stay Rate 0% バグ修正 (`continuous_games=True`)
3. [x] FantasySolver ボトルネック修正 (Greedy FL Solver)
4. [x] Phase 8.5b GCP学習完了 (150M steps)
5. [x] Expert Iteration実験 (結果: 失敗)
6. [x] **Phase 9 FL Mastery完了 (250M steps)**
7. [x] **ONNX変換・mix-poker-app統合**
8. [x] **Phase 10 FL Stay向上完了 (150M steps)**
9. [ ] Phase 10 ONNX変換・mix-poker-app更新
10. [ ] 対人テスト・UI改善
11. [ ] Phase 11: Hold'em AI (将来)

---

## デプロイ状況

### mix-poker-app 統合

OFC AI v1.1.0が `mix-poker-app` に統合済み:
- ONNX形式でNode.jsから直接推論
- 推論速度: 10-50ms/手
- ファイル: `mix-poker-app/server/models/ofc_ai.onnx`
- **TODO**: Phase 10モデルでONNX更新

### GCPインスタンス

**全リソース削除済み** (2026-02-19)
- ofc-training-v2: 削除
- aigis-trading-bot: 削除
- ディスク・ファイアウォール: 全削除

---

## 技術的知見

### greedy_fl_solver v3

v3ではFlush/Straight検出を追加し、FL Stay率を大幅に改善:

| 枚数 | ソルバー単体FL Stay |
|:---:|:---:|
| 14枚 | 29% |
| 15枚 | 32% |
| 16枚 | 50% |
| 17枚 | 66% |

### FL Stay条件

FL Stayには以下のいずれかが必要:
1. **Trips on Top** (任意ランク) - Mid >= Trips, Bot >= Mid が必須
2. **Quads on Bottom** - Joker + Trips が実用的なパス

---

## 関連ドキュメント

| ファイル | 説明 |
|:---------|:-----|
| [phase10_fl_stay_report.md](../reports/phase10_fl_stay_report.md) | Phase 10 最終レポート |
| [phase9_fl_mastery_report.md](../reports/phase9_fl_mastery_report.md) | Phase 9 レポート |
| [CLAUDE.md](../../CLAUDE.md) | プロジェクトガイドライン |
| [NEXT_ACTIONS.md](../../NEXT_ACTIONS.md) | 次のアクション |

---

*Last updated: 2026-02-19*
