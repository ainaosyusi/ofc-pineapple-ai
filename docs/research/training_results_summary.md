# OFC Pineapple AI - 学習結果サマリー

## 概要

Open-Face Chinese Poker (Pineapple) AIの深層強化学習による開発結果をまとめる。

---

## 最新モデル: Phase 7 (2026-01-18完了)

### 性能指標

| 指標 | 値 | 評価 |
|:---|---:|:---|
| 総学習ステップ | 20,000,000 | - |
| ファウル率 | **25.8%** | 中級〜上級者レベル |
| Mean Royalty | **7.56** | 上級者レベル |
| FL Entry Rate | 1.1% | 改善余地あり |
| 勝率 (vs Random) | 65-68% | 安定した優位性 |
| 平均スコア | +6.5 | - |

### モデルファイル

```
models/p7_parallel_20000000.zip
```

### 学習環境

- **アルゴリズム**: MaskablePPO (sb3-contrib)
- **並列環境**: SubprocVecEnv × 4
- **インスタンス**: GCP n2-standard-4
- **学習速度**: 4,500-12,000 FPS
- **学習時間**: 約7-8時間

---

## プレイ例 (視覚的デモ)

### 典型的な勝利パターン

```
  [AI] Score: +11 | Royalty: 2
  Top:    4♠ 2♥ 7♦
  Middle: 7♠ 5♥ J♥ J♣ jk      (Trips with Joker)
  Bottom: J♠ T♥ A♥ A♣ JK      (Trips with Joker)
  Hands: High / Trips / Trips

  [Random 1] Score: +3 | Royalty: 0
  [Random 2] Score: -14 | FOUL!
```

**観察ポイント**:
- ジョーカーを効果的に使用してTrips（3カード）を構築
- ファウルを回避しながら確実にポイントを稼ぐ
- ランダム相手のファウルを誘発（相対的優位性）

### AIの戦略的特徴

1. **ジョーカー活用**: BottomやMiddleでTrips/フルハウスを完成させる
2. **安全策優先**: 無理なFantasyland狙いより確実なファウル回避
3. **役の強さ順序**: Top < Middle < Bottom を厳守

---

## 学習フェーズ履歴

### Phase 1: ファウル回避基礎 (完了)

| 項目 | 値 |
|:---|---:|
| ステップ | 1,000,000 |
| ファウル率 | 37.8% |
| Mean Royalty | 0.34 |

**学習内容**: 役の強さ順序（Top < Middle < Bottom）の基本理解

### Phase 2: 役作り基礎 (完了)

| 項目 | 値 |
|:---|---:|
| ステップ | 2,000,000 |
| ファウル率 | 32.0% |
| Mean Royalty | 0.26 |

**学習内容**: ペア、ストレート、フラッシュ等の基本役構築

### Phase 3: 2人Self-Play (完了)

| 項目 | 値 |
|:---|---:|
| ステップ | 5,000,000 |
| ファウル率 | 58-63% |
| Mean Royalty | 0.5-0.8 |

**学習内容**: 対戦型戦略（攻撃的傾向あり）

**問題点**: ファウル率が上昇（攻撃的すぎる戦略）

### Phase 4: ジョーカー対応 (完了)

| 項目 | 値 |
|:---|---:|
| ステップ | 10,500,000 |
| ファウル率 | **25.1%** |
| Mean Royalty | 0.85 |

**学習内容**: 54カードデッキ（ジョーカー2枚）への対応

**成果**: ファウル率大幅改善、ジョーカーを「保険」として活用

### Phase 5: 3人Self-Play (完了)

| 項目 | 値 |
|:---|---:|
| ステップ | 11,500,000 |
| ファウル率 | 38.5% |
| Mean Royalty | 0.78 |

**学習内容**: 3人対戦（3-Max）環境への適応

### Phase 7: 並列学習 (完了)

| 項目 | 値 |
|:---|---:|
| ステップ | **20,000,000** |
| ファウル率 | **25.8%** |
| Mean Royalty | **7.56** |
| FL Entry Rate | 1.1% |

**学習内容**: 大規模並列学習による収束

**成果**: 過去最高性能を達成

---

## Phase 8: Self-Play (進行中)

### 目的

ランダム相手から過去モデルとの対戦へ移行し、より高度な戦略を獲得。

### 技術的アプローチ

```python
class SelfPlayOpponentManager:
    def __init__(self, pool_size=5, latest_prob=0.8):
        self.model_pool = []  # 過去モデル保持
        self.latest_prob = 0.8  # 最新モデル使用確率

    def select_opponent(self):
        if random.random() < self.latest_prob:
            return self.current_weights, "latest"
        return random.choice(self.model_pool), "past"
```

### 期待される改善

| 指標 | Phase 7 | Phase 8予測 |
|:---|---:|---:|
| ファウル率 | 25.8% | 20-25% |
| Mean Royalty | 7.56 | 8-10 |
| FL Rate | 1.1% | 3-5% |
| 勝率 | 65-68% | 70-80% |

### 進捗監視

```bash
# Discord通知項目
- Foul Rate: ファウル率
- Mean Score: 平均スコア（進捗の主要指標）
- Mean Royalty: 平均ロイヤリティ
- Win Rate: 勝率
```

---

## 評価基準

### 人間レベル比較

| レベル | ファウル率 | Mean Royalty |
|:---|---:|---:|
| 初心者 | 40-50% | 1-2 |
| 中級者 | 25-35% | 3-5 |
| 上級者 | 15-25% | 5-8 |
| プロ | 10-20% | 7-12 |

**Phase 7モデル評価**: 中級〜上級者レベル

### 達成基準表

| レベル | ファウル率 | Royalty | FL Rate | 該当Phase |
|:---|---:|---:|---:|:---|
| Bronze | < 40% | > 0.5 | > 0% | Phase 1-3 |
| Silver | < 30% | > 1.0 | > 1% | Phase 4-5 |
| **Gold** | < 25% | > 2.0 | > 3% | **Phase 7 現在** |
| Platinum | < 20% | > 3.0 | > 5% | Phase 8 目標 |
| Diamond | < 15% | > 5.0 | > 10% | 最終目標 |

---

## 視覚的強さ確認方法

### デモスクリプト実行

```bash
cd "/Users/naoai/試作品一覧/OFC NN"

# 1ゲームの視覚的表示
python src/python/visual_demo.py --games 1

# 複数ゲームの統計
python src/python/visual_demo.py --stats 100

# 特定モデルを指定
python src/python/visual_demo.py --model models/p7_parallel_20000000.zip --stats 100
```

### 出力例

```
==================================================
  STATISTICS SUMMARY
==================================================
  Total Games:    100
  Foul Rate:      25.0%
  Mean Royalty:   1.01
  FL Entry Rate:  2.0%
  Win Rate:       65.0% (vs Random)
  Avg Score:      +6.54 +/- 10.2
==================================================

  [Strength Assessment]
  Foul Control: Good (Advanced)
  Hand Building: Basic
```

---

## 技術スタック

| 項目 | 技術 |
|:---|:---|
| ゲームエンジン | C++ + pybind11 |
| RL フレームワーク | Stable-Baselines3, sb3-contrib |
| アルゴリズム | MaskablePPO |
| 環境 | PettingZoo AECEnv |
| 並列化 | SubprocVecEnv |
| クラウド | GCP GCE (n2-standard-4) |

---

## 関連ファイル

| ファイル | 説明 |
|:---|:---|
| `src/python/visual_demo.py` | 視覚的デモスクリプト |
| `src/python/train_phase8_selfplay.py` | Phase 8学習スクリプト |
| `docs/research/evaluation_metrics.md` | 評価基準詳細 |
| `docs/research/roadmap_and_expectations.md` | ロードマップ |
| `models/p7_parallel_20000000.zip` | Phase 7最終モデル |

---

## 更新履歴

| 日付 | 内容 |
|:---|:---|
| 2026-01-18 | Phase 7完了、Phase 8開始。学習結果サマリー作成。 |
