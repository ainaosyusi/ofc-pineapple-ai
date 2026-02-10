# Phase 8.5 Full FL Training Report - Final

**Date**: 2026-01-23
**Status**: **Completed** (100,000,004 steps)

> **Note (2026-02-07)**: このレポートはPhase 8.5の記録です。
> 後続のPhase 8.5b (150M) およびPhase 9 (250M) で大幅に性能が向上しています。
> 最新の結果は [Phase 9 レポート](../reports/phase9_fl_mastery_report.md) を参照してください。

---

## Executive Summary

Phase 8.5は**Ultimate Rules Fantasy Land**を完全サポートした100Mステップの大規模学習を完了しました。
これはOFC Pineapple AIプロジェクト史上最大の学習規模であり、約1,024万ゲームの自己対戦を通じてモデルを訓練しました。

### Key Achievements

| Metric | Phase 8 (Best) | Phase 8.5 (Final) | Change |
|:-------|:--------------:|:-----------------:|:------:|
| Total Steps | 9.8M | **100.0M** | +10x |
| Foul Rate | **20.8%** | 22.0% | +1.2% |
| Mean Score | +7.87 | **+8.43** | +0.56 |
| FL Entry Rate | 3.2% | **8.2%** | +5.0% |
| Mean Royalty | 0.87 | **1.40** | +0.53 |
| Win Rate | 65.3% | **68.8%** | +3.5% |

---

## Training Configuration

| Parameter | Value |
|:----------|:------|
| Environment | OFC Pineapple 3-Max (Ultimate Rules) |
| Algorithm | MaskablePPO |
| Parallel Envs | 6 (SubprocVecEnv) |
| n_steps | 2048 |
| batch_size | 256 |
| learning_rate | 1e-4 (decay from 3e-4) |
| Total Steps | 100,000,004 |
| Training Time | ~22.5 hours |
| Average FPS | 631 |

### Ultimate Rules FL Configuration

| Top Hand | Cards Dealt | FL Continuation Requirement |
|:---------|:-----------:|:----------------------------|
| QQ | 14 | QQ+ on top row |
| KK | 15 | QQ+ on top row |
| AA | 16 | QQ+ on top row |
| Trips | 17 | QQ+ on top row |

---

## Final Performance Metrics

### Primary Metrics

```
Step 100,000,004 - Phase 8.5 Full FL
══════════════════════════════════════
Games Played:     10,240,000
Foul Rate:        22.0%
Mean Score:       +8.43
Mean Royalty:     1.40
FL Entry Rate:    8.2%
FL Stay Rate:     0.0% (solver disabled)
High Score Rate:  16.6%
Win Rate:         68.8%
FPS:              631-1233
══════════════════════════════════════
```

### PPO Training Metrics (Final Iteration)

| Metric | Value |
|:-------|:------|
| approx_kl | 0.014 |
| clip_fraction | 0.073 |
| entropy_loss | -0.183 |
| explained_variance | 0.247 |
| policy_gradient_loss | -0.018 |
| value_loss | 71.2 |
| n_updates | 70,910 |

---

## Training Progress Analysis

### Foul Rate Trend

- 学習初期（48-50M）: 24-25%で変動
- 中盤（50-70M）: 22-25%で安定
- 終盤（70-100M）: 21-23%で収束
- **最終値**: 22.0%

Phase 8の20.8%より若干高いが、FL重視の戦略による副作用。

### FL Entry Rate Trend

- Phase 8: 3.2%
- Phase 8.5初期: 4-5%
- Phase 8.5中盤: 5-7%
- **Phase 8.5最終: 8.2%**

**+156%の改善** - FL突入を積極的に狙う戦略を学習。

### Mean Score Trend

- Phase 8: +7.87
- Phase 8.5最終: **+8.43**

**+0.56ポイント改善** - より高いロイヤリティを獲得する手作りを習得。

---

## Technical Achievements

### 1. Continuous Games with FL State Preservation

```python
continuous_games=True  # FL状態がゲーム間で引き継がれる
```

これにより、FL継続の学習が可能になった。

### 2. Dead Agent Handling Fix

```python
def step(self, action):
    # 状態の整合性チェック
    if self.agent_selection not in self.terminations:
        self._reinitialize_state()
```

PettingZoo環境での終了エージェント問題を解決。

### 3. Safe Opponent Loop

```python
def _play_until_my_turn_or_end(self):
    max_iterations = 1000
    iterations = 0
    while iterations < max_iterations:
        if all(self.env.terminations.values()):
            break
        # ...
```

無限ループ防止機構を実装。

---

## Comparison with Previous Phases

| Phase | Steps | Foul Rate | Mean Score | FL Entry | Focus |
|:------|------:|:---------:|:----------:|:--------:|:------|
| Phase 7 | 20M | 25.8% | +7.56 | 1.1% | Parallel Training |
| Phase 8 | 9.8M | **20.8%** | +7.87 | 3.2% | Multi-Variant |
| **Phase 8.5** | 100M | 22.0% | **+8.43** | **8.2%** | Ultimate Rules FL |

### Key Observations

1. **Foul Rate vs FL Entry Trade-off**
   - FL重視戦略はファウル率を若干上昇させる
   - ただしスコア期待値は大幅に向上

2. **Win Rate Improvement**
   - Self-Play Pool相手に68.8%勝率
   - 過去モデルとの対戦でも強さを示す

3. **Royalty Points**
   - 1.40ロイヤリティ/ゲームは過去最高
   - 役作りの質が向上

---

## Model Artifacts

### Saved Models

| Checkpoint | Steps | Size | Location |
|:-----------|------:|-----:|:---------|
| `p85_full_fl_100000004.zip` | 100M | 1.69MB | `models/` |
| `p85_full_fl_99800004.zip` | 99.8M | 1.69MB | GCP |

### Training Logs

- TensorBoard: `gcp_backup/phase85_full_fl/`
- Raw Logs: `plots/phase85/raw_metrics.txt`

### Visualization

- Training Curves: `plots/phase85/training_curves.png`
- Phase Comparison: `plots/phase85/phase_comparison.png`

---

## Known Limitations

### 1. FL Stay Rate = 0%

Fantasy Land継続（FL中にQQ+を作る）は学習できていない。

**原因**: `enable_fl_turns=False`でソルバーを無効化したため、FL中の手札配置は学習対象外。

**対策**: FL Specialist学習（Phase 9）で専用モデルを訓練中。

### 2. Explained Variance

0.247と低め（理想は0.8以上）。

**原因**: OFCの高い分散（運要素）により、状態価値の予測が困難。

**示唆**: 純粋なRLの限界。Hybrid Agent（RL + Solver）が必要。

---

## Next Steps (実施済み)

1. **Phase 8.5b (150M)** - ✅ 完了
   - Greedy FL Solver導入でFPS問題解決
   - FL Stay Rate 3.0%達成

2. **Phase 9 FL Mastery (250M)** - ✅ 完了
   - FL Entry 22.8%、FL Stay 8.0%達成
   - Foul Rate 16.8%（プロレベル）
   - 報酬シェーピング改善が成功

3. **Phase 10 FL Stay向上** - 進行中
   - FL Stay Rate 15%+を目標
   - greedy_fl_solver改修中

---

## Conclusion

Phase 8.5は100Mステップの大規模学習を完了し、以下を達成しました：

- **FL Entry Rate 8.2%** - Phase 8比+156%
- **Mean Score +8.43** - 過去最高
- **Win Rate 68.8%** - 安定した強さ

純粋なRLとしては優れた結果ですが、FL継続やファウル完全回避にはSolver統合が必要です。
次フェーズでHybrid Agentを完成させ、実用レベルのAIを目指します。

---

*Report generated: 2026-01-24*
*Model: p85_full_fl_100000004.zip*
