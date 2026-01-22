# Phase 8.5: Full FL (Ultimate Rules) Training

**作成日**: 2026-01-19  
**ステータス**: 途中終了（48.6M / 50M steps）

---

## 概要

Phase 8.5では、**Ultimate RulesのFantasy Land学習**を導入し、継続ゲームとポジションローテーションを前提にした強化学習を実施した。

**主な変更点**
- **Continuous Games**: FL状態をゲーム間で維持
- **Button Rotation**: 毎ゲームでポジションが回る
- **Ultimate Rules**: QQ=14 / KK=15 / AA=16 / Trips=17 枚

---

## 学習設定

| 項目 | 値 |
|:---|:---|
| 環境 | OFC Pineapple 3-Max |
| アルゴリズム | MaskablePPO |
| 並列環境数 | 6 |
| n_steps | 2048 |
| batch_size | 256 |
| learning_rate | 3e-4 |
| 目標ステップ | 50,000,000 |

---

## 学習結果（中断時点）

| 指標 | 値 |
|:---|:---|
| **進捗** | 97.0%（48,600,972 / 50,000,000） |
| **ゲーム数** | 約9,700,000 |
| **ファウル率** | 21.8% |
| **Mean Score** | +8.44 |
| **Mean Royalty** | 1.34 |
| **Win Rate** | 65.3% |
| **FPS** | 約753 |

---

## 保存モデル

| チェックポイント | ステップ | サイズ |
|:---|---:|---:|
| `p85_full_fl_48400968.zip` | 48.4M | 1.69 MB |
| `p85_full_fl_48600972.zip` | 48.6M | 1.69 MB |

---

## 主要な問題と対処

### 1. Dead Agent Error

```
ValueError: when an agent is dead, the only valid action is None
```

**原因**: 終了したエージェントに対して `step(action)` が送られていた。  
**対処**: 終了済みの場合は `step(None)` を送る。

### 2. _play_opponents の無限ループ

**原因**: 特定状態で終了判定が成立せずループが継続。  
**対処**: `max_iterations` ガードを追加。

### 3. 修正後の進捗停止

**状況**: ワーカーは動作するが、メインプロセスがバッチ収集待ちで停止。  
**メモ**: SubprocVecEnv内の終端処理に起因する可能性が高い。

---

## 生成物

**モデル**
- `gcp_backup/p85_full_fl_48400968.zip`
- `gcp_backup/p85_full_fl_48600972.zip`

**ログ**
- `gcp_backup/phase85_full_fl/`（TensorBoard）
- `gcp_backup/training*.log`

**グラフ**
- `plots/phase85/training_progress.png`
- `plots/phase85/performance.png`
- `plots/phase85/learning.png`

---

## 次のアクション

1. Dead agent処理の恒久修正と回帰テスト
2. DummyVecEnvでの再現テスト追加
3. 48.6Mチェックポイントから学習再開
4. Phase 8とPhase 8.5の比較評価

---

*Document created: 2026-01-19*
