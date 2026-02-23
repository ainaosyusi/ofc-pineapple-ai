# Phase 9 FL Mastery 学習レポート (250M Steps) - 完了

**期間**: 2026-01-29 〜 2026-02-07
**総ステップ数**: 250,000,000
**学習時間**: 約100時間
**ステータス**: **完了** ✅

---

## Executive Summary

Phase 9はOFC Pineapple AIプロジェクトのマイルストーンです。
FL特化報酬設計とネットワーク拡張により、**すべての目標を達成**しました。

### 最終結果 (250M)

| 指標 | Phase 9 (250M) | Phase 8.5 (100M) | 目標 | 達成 |
|:---|---:|---:|:---|:---:|
| **Foul Rate** | **16.8%** | 22.0% | <20% | ✅ |
| **Mean Score** | **+12.66** | +8.43 | >+10 | ✅ |
| **FL Entry Rate** | **22.8%** | 8.2% | >15% | ✅ |
| **FL Stay Rate** | **8.0%** | 0.0% | >5% | ✅ |
| **Win Rate** | **75.8%** | 68.8% | >70% | ✅ |

---

## 学習進捗

| ステップ | Foul Rate | Mean Score | FL Entry | FL Stay | Win Rate |
|---:|---:|---:|---:|---:|---:|
| 50M | 19.6% | +12.32 | 17.8% | 7.6% | 68.5% |
| 100M | 18.0% | +12.52 | 22.2% | 9.0% | 75.1% |
| 150M | 17.6% | +12.58 | 21.2% | 8.2% | 75.4% |
| **250M** | **16.8%** | **+12.66** | **22.8%** | **8.0%** | **75.8%** |

### 学習曲線の特徴

1. **Foul Rate**: 50M→250Mで19.6%→16.8%と継続的に改善
2. **FL Entry**: 100M付近で22%に到達し、以降安定
3. **Win Rate**: 100M以降で75%超を維持

---

## Phase 9 の主要な変更点

### 1. ネットワーク拡張
- Phase 8.5: `[64, 64]` MLP
- Phase 9: `[512, 256, 128]` MLP (881次元入力に対応)

### 2. FL特化報酬設計

**中間報酬 (`_apply_intermediate_reward`)**
| ステージ | 条件 | 報酬 (Phase 8.5 → Phase 9) |
|:---|:---|:---|
| Stage 1 | Q/K/A on Top | 3.0 → **5.0** |
| Stage 2 | QQ/KK/AA ペア | 5.0 → **10.0** |
| Stage 3 | Trips on Top | 3.0 → **8.0** |

- ファウルペナルティ: FL追求中は 0.5 → **0.2** に軽減
- Bottom強化ボーナス: Straight+ で +2.0、Full House+ で +3.0

**最終報酬 (`_calculate_final_rewards`)**
| イベント | 報酬 (Phase 8.5 → Phase 9) |
|:---|:---|
| FL Entry | 25.0 → **30.0** |
| FL Stay | 20.0 → **60.0** |
| AA+ Top | - → **+10.0** |
| Trips Top | - → **+10.0 追加** |

### 3. ハイパーパラメータ調整
| パラメータ | Phase 8.5 | Phase 9 |
|:---|---:|---:|
| gamma | 0.995 | **0.998** |
| n_steps | 4,096 | **8,192** |
| batch_size | 512 | **1,024** |
| n_epochs | - | **5** |
| ent_coef | 0.01 | **0.03** |

### 4. バグ修正
- `HandRank` 比較演算子を pybind11 に追加 (`py::arithmetic()`)
- Bottom強化ボーナスでTypeErrorになる問題を解決

---

## 学習パラメータ詳細 (最終値)

- **entropy_loss**: -0.77 (健全、崩壊なし)
- **explained_variance**: 0.49〜0.54 (価値関数が適切に学習)
- **approx_kl**: 0.006〜0.007 (安定した更新)
- **clip_fraction**: 0.045〜0.048 (適切なクリッピング)
- **FPS**: 400〜900 (平均約600)

---

## モデルファイル

| ファイル | ステップ | 用途 |
|:---|---:|:---|
| `models/phase9/p9_fl_mastery_250000000.zip` | 250M | **最終・推奨** |
| `models/phase9/p9_fl_mastery_150000000.zip` | 150M | 中間チェックポイント |
| `models/onnx/ofc_ai.onnx` | - | Node.js統合用 |

---

## デプロイ

### ONNX変換

```bash
python scripts/export_onnx.py \
    --model models/phase9/p9_fl_mastery_250000000.zip \
    --output models/onnx/ofc_ai.onnx
```

### mix-poker-app統合

OFC AI v1.1.0として `mix-poker-app` に統合済み:
- ファイル: `mix-poker-app/server/models/ofc_ai.onnx`
- 推論速度: 10-50ms/手
- Python依存なしでNode.jsから直接推論

---

## 人間レベル比較

| レベル | Foul Rate | FL Entry | 現在のAI |
|:---|---:|---:|:---|
| 初心者 | 40-50% | 0-5% | - |
| 中級者 | 25-35% | 5-10% | - |
| 上級者 | 15-25% | 10-20% | - |
| **プロ** | 10-20% | 15-30% | **←16.8%, 22.8%** |

**Phase 9 AIはプロレベルに到達**

---

## 考察

### 成功要因

1. **FL特化報酬**: FL Entry/Stay達成時の大きなボーナスがモデルの行動を誘導
2. **ネットワーク拡張**: 881次元の観測空間に対応する表現力
3. **継続学習**: 250Mステップの長期学習で戦略が洗練
4. **Self-Play**: 同一モデル対戦による継続的な改善

### 残された課題

1. **FL Stay Rate 8%**: 目標の5%は達成したが、理論上15-20%まで改善可能
2. **FL Stay条件の困難さ**:
   - Trips on Top: Bot >= Mid >= Tripsが必要で非常に困難
   - Quads on Bottom: 実用的だがJoker依存が高い

### Phase 10への移行

Phase 10ではFL Stay Rateを8%→15%に向上させるため:
1. `greedy_fl_solver.py`のFL Stay探索を改修
2. Trips on Top/Quads on Bottomの探索を優先
3. 50Mステップのfine-tuning

---

## 結論

Phase 9は250Mステップの学習を完了し、**すべての目標を達成**しました:

- **Foul Rate 16.8%** (目標<20%) ✅
- **Mean Score +12.66** (目標>+10) ✅
- **FL Entry Rate 22.8%** (目標>15%) ✅
- **FL Stay Rate 8.0%** (目標>5%) ✅
- **Win Rate 75.8%** (目標>70%) ✅

これはOFC Pineapple AIプロジェクトの重要なマイルストーンであり、
プロレベルのAIを実現しました。

---

*レポート作成: 2026-02-07*
*最終モデル: p9_fl_mastery_250000000.zip*
