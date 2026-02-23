# Expert Iteration (ExIt) 実験レポート

**日付**: 2026-01-27
**状態**: 完了 (失敗 - Catastrophic Forgetting)

---

## 概要

Phase 8.5bのPPOモデル (100M steps) をベースに、MCTS Expert Iterationによる
ポリシー改善を試みた。結果、Foul Rateが25.9%→53.3%に悪化し、FL能力も喪失。
PPO self-playの方が優れた学習パラダイムであることが確認された。

---

## 実験設定

| パラメータ | 値 |
|:---|:---|
| ベースモデル | `models/backup/p85_full_fl_100000004_backup.zip` (100M steps) |
| イテレーション数 | 5 |
| ゲーム数/イテレーション | 500 |
| MCTSシミュレーション数 | 100 |
| 制限時間 | 1000ms |
| 学習エポック | 10 |
| 評価ゲーム数 | 200 |
| 出力先 | `models/exit/` |

### 実行コマンド

```bash
python src/python/expert_iteration.py \
  --model models/backup/p85_full_fl_100000004_backup \
  --iterations 5 --games-per-iter 500 --simulations 100 \
  --time-limit 1000 --train-epochs 10 --eval-games 200 \
  --output models/exit --save-data --verbose
```

---

## 結果

### イテレーション別メトリクス

| Iter | Foul Rate | Mean Score | Win Rate | FL Entry | Top1-Acc | Policy Loss |
|:---:|---:|---:|---:|---:|---:|---:|
| Init | 25.9% | -1.11 | 35.2% | 6.7% | - | - |
| 1 | 41.1% | +0.20 | 41.7% | 1.2% | 0.509 | 1.887 |
| 2 | 50.9% | -0.29 | 36.4% | 0.6% | 0.505 | 1.883 |
| 3 | 55.7% | +0.03 | 33.9% | 0.9% | 0.508 | 1.859 |
| 4 | 44.1% | +2.38 | 50.0% | 0.0% | 0.509 | 1.863 |
| 5 | 53.3% | +0.37 | 40.0% | 0.0% | 0.504 | 1.858 |

### ベースラインとの比較

| 指標 | Init (ExIt) | Final (ExIt) | GCP 8.5b (150M) | 変化 (ExIt) |
|:---|---:|---:|---:|:---|
| Foul Rate | 25.9% | 53.3% | 22.7% | **+27.4% (悪化)** |
| Mean Score | -1.11 | +0.37 | +9.04 | +1.48 (微改善) |
| Win Rate | 35.2% | 40.0% | 71.2% | +4.8% (微改善) |
| FL Entry | 6.7% | 0.0% | 7.9% | **-6.7% (喪失)** |

### 学習ロス推移

| Iter | Train P-Loss | Val P-Loss | Train V-Loss | Val V-Loss |
|:---:|---:|---:|---:|---:|
| 1 | 1.695 | 1.858 | 6.729 | 10.523 |
| 2 | 1.654 | 1.852 | 6.419 | 10.284 |
| 3 | 1.633 | 1.831 | 6.222 | 10.139 |
| 4 | 1.649 | 1.835 | 6.306 | 10.301 |
| 5 | 1.695 | 1.858 | 6.729 | 10.523 |

### グラフ

`plots/expert_iteration_results.png` に4指標の比較グラフを保存済み。

---

## 失敗原因の分析

### 1. MCTSエキスパートの質が不十分

100シミュレーションでは3人OFCの正確な手を生成できない。
- OFCの行動空間は各ターン3択×5カード配置 = 複雑な組合せ
- 100 simsでは浅い探索しかできず、「エキスパート」手が実際にはランダムに近い
- 結果: 高フォールド率のデータで学習してしまう

### 2. Catastrophic Forgetting

教師あり学習がPPOで獲得した保守的プレイスタイルを上書き:
- PPOは100Mステップかけて「ファウルを避ける」方策を学習
- ExItの教師データは品質が低く、ファウル率43.5%のゲームデータ
- 10エポックの教師あり学習でPPO方策が破壊される

### 3. FL配置パターンの消失

MCTSはFL特有の配置パターン（Top QQ+を維持する配置）を探索しない:
- MCTSのrolloutが均一ランダムで、FL条件を考慮しない
- FL EntryRate: 6.7% → 0.0% に完全消失

---

## 教訓と今後の方針

### ExItを改善するなら

1. **MCTS品質向上**: 1000+ simulations、FL-awareなrollout policy
2. **部分的学習**: 全パラメータを更新せず、最終層のみfine-tune
3. **KL-divergenceペナルティ**: ベースモデルからの乖離を制限
4. **データフィルタリング**: ファウルしたゲームのデータを除外

### 代替アプローチ

PPO self-playが飽和した現状（Foul 22.7%, Score +9.04）を突破するには:
1. **報酬シェーピングの改善**: ファウルペナルティの増加
2. **Curriculum Learning**: 段階的に対戦相手を強化
3. **Population-Based Training**: 複数エージェントの並列進化
4. **Network Architecture**: より大きなモデル or Attention層の追加

---

## 保存されたアーティファクト

| ファイル | 説明 |
|:---|:---|
| `models/exit/exit_final.zip` | 最終モデル (Iter 5、品質低) |
| `models/exit/exit_iter_001.zip` ~ `005.zip` | 各イテレーションのモデル |
| `models/exit/data_iter_001.npz` ~ `005.npz` | 各イテレーションの学習データ |
| `plots/expert_iteration_results.png` | 結果グラフ |

---

*Report generated: 2026-01-27*
