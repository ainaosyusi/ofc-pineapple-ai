# Phase 8.5 Full FL 学習レポート (100M Steps)

**期間**: 2026-01-19 〜 2026-01-24
**総ステップ数**: 100,000,004
**総ゲーム数**: 10,240,000

---

## 最終結果

| 指標 | 値 | Phase 8 Best | 変化 |
|:---|---:|---:|:---|
| **Foul Rate** | 22.0% | 20.8% | +1.2% |
| **Mean Score** | +8.43 | +7.87 | +0.56 |
| **Mean Royalty** | 1.40 | - | - |
| **FL Entry Rate** | 8.2% | 3.2% | +5.0% |
| **FL Stay Rate** | 0.0% | N/A | **バグ** |
| **High Score (≥15) Rate** | 16.6% | - | - |
| **Win Rate** | 68.8% | - | - |

---

## 発見されたバグ

### FL Stay Rate 0% の原因

`train_phase85_full_fl.py` の152-156行目で `continuous_games=False` が設定されていた：

```python
# Phase 8.5: 一時的にcontinuous_games=False（パフォーマンス調査用）
self.env = OFC3MaxEnv(
    enable_fl_turns=True,
    continuous_games=False  # ← これが原因
)
```

**影響**:
- FL状態が次のゲームに引き継がれない
- `in_fantasy_land=True` でゲーム開始することがない
- `stayed_fl` が常に False
- FL継続戦略の学習が全く行われていない

### ボタン分布の異常

```
Button Dist: {0: 10240000, 1: 0, 2: 0}
```

全ゲームでボタン位置が0のみ。これも `continuous_games=False` が原因で、ボタンローテーションが機能していない。

---

## 学習曲線の傾向

### Foul Rate
- 初期: 25%前後
- 100M時点: 22-27%で推移
- 最良: 20.8%（局所的）

### Mean Score
- 安定して +7 〜 +8.5 の範囲
- FL Entry効果でスコア向上の兆候あり

### FL Entry Rate
- 4% 〜 8% の範囲で推移
- Phase 8 (3.2%) より明らかに向上

---

## 次のアクション

### 1. バグ修正
`continuous_games=True` に変更し、以下を有効化：
- FL状態の引き継ぎ
- ボタンローテーション
- FL継続戦略の学習

### 2. 新しい学習 Phase 8.5b
現在の100Mチェックポイントから継続し、修正版で追加学習：
- 目標: さらに50M〜100Mステップ
- 期待: FL Stay Rate > 30%

### 3. 監視指標
| 指標 | 目標 |
|:---|---:|
| Foul Rate | < 20% |
| FL Entry Rate | > 8% |
| FL Stay Rate | > 30% |
| Mean Score | > +10 |

---

## インフラ情報

### 使用したGCPインスタンス
- `ofc-training` (e2-standard-8) - 削除済み
- `ofc-fl-specialist` (e2-standard-8) - 削除済み

### 保存されたモデル
- `models/p85_full_fl_100000004.zip` (最終チェックポイント)

### 学習ログ
- `plots/phase85/raw_metrics.txt`
- `plots/phase85/training_curves.png`

---

*Report generated: 2026-01-26*
