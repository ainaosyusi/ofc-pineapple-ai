# Phase 8.5b FantasySolver 性能修正レポート

**日付**: 2026-01-26 → 2026-01-27 (完了)
**状態**: 完了 (150M Steps)

---

## 概要

Phase 8.5b の GCP 学習開始時、FantasySolver の指数的探索がボトルネックとなり
学習速度が FPS=2 まで低下していた (150Mステップに約2.4年)。

Monte Carlo sampling ベースの Greedy FL Solver を実装し、FPS を 410 に改善。
100M ステップのチェックポイントから学習を再開した。

---

## 問題の詳細

### 症状

- GCPインスタンスでの学習開始後、46分間ログ出力なし
- プロセスは CPU 100% で動作中
- 実際には FPS=2 で動作 (Stable-Baselines3 の verbose=1 出力が途中から出力)
- 最初の callback レポート (100kステップ) に到達するまで約14時間必要

### 根本原因

`src/cpp/solver.hpp` の FantasySolver が O(4^n) の全探索を実行:

```
14カード: C(14,5) * C(9,5) * C(4,3) ≈ 100万通り → 0.67秒/回
15カード: ≈ 700万通り → 4.8秒/回
16カード: ≈ 3400万通り → 推定34秒/回
17カード: ≈ 1.7億通り → 推定240秒/回
```

`solver.hpp:116` にスコア枝刈りの TODO コメントがあるが**未実装**:
```cpp
// TODO: もし現在のスコア + 最大ロイヤリティ < best_score なら return
```

`continuous_games=True` + `enable_fl_turns=True` の環境では、
FL突入のたびにこのソルバーが呼ばれ、学習が実質停止状態だった。

---

## 解決策: Greedy FL Solver

### 設計

`src/python/greedy_fl_solver.py` を新規作成。3フェーズのアルゴリズム:

1. **Phase 1: FL-Stay Targeted Search**
   - QQ+/KK+/AA+ のペアを Top に配置
   - 残りカードを B/M に分配して validity チェック
   - FL Stay に最適化した構造的サンプリング

2. **Phase 2: Royalty-aware Structured Search**
   - ペア・セットを同じ列にグルーピング
   - ランクでソートした構造的配置

3. **Phase 3: Random Sampling**
   - ランダムシャッフルで広範囲探索
   - 500サンプルでカバレッジ確保

### 実装の変更点

| ファイル | 変更内容 |
|:---|:---|
| `src/python/greedy_fl_solver.py` | 新規 - Greedy FL Solver |
| `src/python/ofc_3max_env.py` | `fl_solver_mode` パラメータ追加 |
| `src/python/train_phase85_full_fl.py` | `fl_solver_mode='greedy'` 指定 |

### 発見したバグ

`greedy_fl_solver.py` v2 初版で `_State` オブジェクトの参照が分離しており、
Phase 1 (FL-Stay search) の結果が Phase 2/3 に引き継がれていなかった。

修正: 全フェーズで同一の `_State` オブジェクトを共有するように変更。

---

## ベンチマーク結果

### Solver 速度比較

| カード数 | Brute-force | Greedy | 高速化 |
|:---:|---:|---:|---:|
| 14 | 672ms | 15ms | **44x** |
| 15 | 4,808ms | 15ms | **320x** |
| 16 | ~34,000ms | 15ms | **~2,300x** |
| 17 | ~240,000ms | 17ms | **~14,000x** |

### FL Stay 品質

14カードハンド50回テスト:
- Brute-force: 38% stay-per-entry
- Greedy v2: **36%** stay-per-entry (95%品質)

### 学習 FPS

| 状態 | FPS | 150Mステップ所要時間 |
|:---|---:|---:|
| Brute-force Solver | 2 | ~2.4年 |
| Greedy Solver | **410** | **~34時間** |

---

## Phase 8.5b 初期メトリクス (100.1M時点)

| 指標 | Phase 8.5 (100M) | Phase 8.5b (100.1M) | 変化 |
|:---|---:|---:|:---|
| Foul Rate | 22.0% | 21.8% | -0.2% |
| Mean Score | +8.43 | +9.51 | +1.08 |
| Mean Royalty | 1.40 | 1.76 | +0.36 |
| FL Entry Rate | 8.2% | 5.4% | -2.8% |
| FL Stay-per-Entry | 0% | **29.6%** | **+29.6%** |
| Win Rate | 68.8% | 71.5% | +2.7% |

FL Stay-per-Entry が 0% → 29.6% に改善。`continuous_games=True` + Greedy Solver により
FL継続学習が機能している。

---

## 最終結果 (150M Steps, 2026-01-27)

**学習完了**: 150,000,000 ステップ / FPS ~621 / 約22時間

| 指標 | 初期(100.1M) | 最終(直近5M平均) | 目標 | 達成 |
|:---|---:|---:|---:|:---:|
| Foul Rate | 21.8% | 22.7% | < 20% | ❌ |
| Mean Score | +9.51 | +9.04 | > +10 | ❌ |
| Mean Royalty | 1.76 | 1.76 | > 2.0 | ❌ |
| FL Entry Rate | 5.4% | 7.9% | > 8% | ❌ (ほぼ達成) |
| FL Stay Rate | 1.6% | 3.0% | > 2% | ✅ |
| FL Stay-per-Entry | 29.6% | ~38% | > 30% | ✅ |
| Win Rate | 71.5% | 71.2% | > 70% | ✅ |
| 総ステップ | 100.1M | **150M** | 150M | ✅ |

### トレンド分析

110M以降、全指標がプラトーに達しており追加学習の効果は飽和。
PPO self-play単体ではFoul Rate 20%以下の壁を突破できない。

### グラフ

`plots/gcp_phase85b_training.png` に6指標の学習曲線を保存済み。

---

## インフラ情報

### GCPインスタンス

- **名前**: ofc-training-v2
- **タイプ**: e2-standard-8 (8 vCPU, 32 GB RAM)
- **ゾーン**: asia-northeast1-b
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.10 + venv_linux

### チェックポイント管理

- ベースモデル: `models/p85_full_fl_100000000.zip` (100万の倍数にリネーム済み)
- クリーンアップ: 最新2つ + 100万の倍数を保持
- バックアップ: ローカル `models/backup/p85_full_fl_100000004_backup.zip`

---

*Report generated: 2026-01-26*
