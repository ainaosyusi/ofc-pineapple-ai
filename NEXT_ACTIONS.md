# OFC AI - 次のアクション

**最終更新: 2026-02-19**

---

## 現在の状態: Phase 10 完了 ✅

- **Phase 9 (250M steps)** 完了 ✅
- **Phase 10 (150M steps)** 完了 ✅ - FL Stay向上
- GCPリソース全削除済み（料金ゼロ）

---

## Phase 10: FL Stay Training - 完了 ✅

### 評価結果 (500 games, deterministic)

| 指標 | Phase 9 (250M) | Phase 10 (150M) | 改善 |
|:-----|:--------------:|:---------------:|:----:|
| Foul Rate | 27.2% | **18.0%** | -9.2% |
| Mean Score | +17.92 | **+36.08** | +18.16 |
| Mean Royalty | +3.00 | **+6.10** | +3.10 |
| FL Entry Rate | 17.8% | **33.6%** | +15.8% |
| FL Stay Rate | 5.0% | **15.8%** | +10.8% |
| FL Stay/Entry | 28.1% | **47.0%** | +18.9% |
| Win Rate | 53.0% | **55.2%** | +2.2% |

### 学習ログ最終値 (150M steps)

| 指標 | 値 |
|:-----|:---:|
| Foul Rate | 21.0% |
| FL Entry | 32.5% |
| FL Stay | 18.1% |
| Mean Reward | +17.66 |
| FPS | ~300 |

### 主要改善: greedy_fl_solver v3

| 枚数 | FL Stay | 平均Royalty |
|:---:|:---:|:---:|
| 14枚 | 29% | 8.7 |
| 15枚 | 32% | 9.7 |
| 16枚 | 50% | 13.1 |
| 17枚 | 66% | 17.5 |

---

## 次のタスク

### 1. Phase 10 ONNX変換・mix-poker-app更新

```bash
# ONNX変換
python scripts/export_onnx.py --model models/phase10_gcp/p10_fl_stay_150000000.zip --output models/onnx/ofc_ai_v2.onnx

# mix-poker-app にコピー
cp models/onnx/ofc_ai_v2.onnx /Users/naoai/Desktop/mix-poker-app/server/models/ofc_ai.onnx
```

### 2. 対人テスト

Phase 10モデルをmix-poker-appでテストプレイ

### 3. Phase 11 (将来)

Hold'em AI

---

## 完了した実験

### Phase 10: FL Stay Training ✅

- greedy_fl_solver v3 導入
- 150M steps on GCP
- FL Stay 5.0% → 15.8%

### Phase 9: FL Mastery (250M) ✅

| 指標 | 250M | 150M | 100M |
|:---|---:|---:|---:|
| Foul Rate | 16.8% | 17.6% | 22.0% |
| FL Entry | 22.8% | 21.2% | 8.2% |
| FL Stay | 8.0% | 8.2% | 0.0% |

### Phase 8.5b: Self-Play (150M) ✅
### ExIt v1/v2: ❌ 失敗・放棄

---

## 保存済みモデル

| ファイル | ステップ | 説明 |
|:---|---:|:---|
| `models/phase10_gcp/p10_fl_stay_150000000.zip` | 150M | Phase 10 最終 ⭐ |
| `models/phase10_gcp/p10_fl_stay_100000000.zip` | 100M | Phase 10 中間 |
| `models/phase10_gcp/p10_fl_stay_50000000.zip` | 50M | Phase 10 初期 |
| `models/phase9/p9_fl_mastery_250000000.zip` | 250M | Phase 9 |
| `models/onnx/ofc_ai.onnx` | - | ONNX (Phase 9ベース) |

---

## 教訓

1. **PPO self-playは110M以降飽和** - 報酬シェーピングで突破可能
2. **ExItは高品質エキスパートが必須** - 100-800 MCTS simsでは不十分
3. **FL報酬シェーピングは有効** - FL Entry 8%→34%, FL Stay 5%→16%
4. **ONNX変換でNode.js統合可能** - Python不要、推論10-50ms
5. **GCPインスタンスは学習完了後すぐ停止** - コスト注意
6. **FL Stay は数学的制約が厳しい** - Trips on Top には Mid も Trips+ が必須
7. **FL Solver は Flush/Straight 優先探索が重要** - ランダム探索では見逃す
8. **チェックポイントは数値ソート必須** - アルファベット順バグ注意
9. **学習は必ずGCPで実行** - ローカルMacはスリープで死ぬ

---

*このファイルは `CLAUDE.md` で参照が強制されています。*
