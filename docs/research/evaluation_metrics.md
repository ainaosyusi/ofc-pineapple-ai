# OFC Pineapple AI 評価基準ドキュメント

## 概要

本ドキュメントはOFC (Open-Face Chinese) Pineapple AIの性能評価に用いる基準と尺度を定義する。
研究目的での再現性と比較可能性を確保するため、評価指標の定義と解釈を明確化する。

---

## 1. ロイヤリティスコアリング表（JOPTルール）

### 1.1 役別ロイヤリティボーナス

| 役 | Bottom | Middle | Top |
|:---|---:|---:|---:|
| ペア(66-AA) | — | — | 1-9点 |
| トリップス(222-AAA) | — | — | 10-22点 |
| ストレート | 2点 | 4点 | — |
| フラッシュ | 4点 | 8点 | — |
| フルハウス | 6点 | 12点 | — |
| 4カード | 10点 | 20点 | — |
| ストレートフラッシュ | 15点 | 30点 | — |
| ロイヤルフラッシュ | 25点 | 50点 | — |

### 1.2 Topペア・トリップス詳細

| Top役 | ロイヤリティ |
|:---|---:|
| 66 | 1点 |
| 77 | 2点 |
| 88 | 3点 |
| 99 | 4点 |
| TT | 5点 |
| JJ | 6点 |
| QQ | 7点 |
| KK | 8点 |
| AA | 9点 |
| 222 | 10点 |
| 333 | 11点 |
| ... | ... |
| AAA | 22点 |

### 1.3 Fantasyland条件

- **進入条件**: TopでQQ以上を達成（ファウルなし）
- **ボーナス**: 次ゲームで14枚一括配置
- **再進入条件**:
  - Top: トリップス
  - Middle: フルハウス以上
  - Bottom: 4カード以上

---

## 2. Mean Royaltyの解釈尺度

### 2.1 評価レンジ

| 範囲 | 評価 | 説明 |
|---:|:---|:---|
| 0-1点 | 低 | 役が成立しない、またはハイカードのみ |
| 1-3点 | 基礎 | ペア・ストレート程度の基本役 |
| 3-6点 | 中級 | フラッシュ・フルハウス含む安定した役作り |
| 6-10点 | 上級 | 複数段で強い役、Top QQ以上達成 |
| 10点以上 | 超人 | 4カード・SF級の高役を頻繁に達成 |

### 2.2 理論的境界値

- **理論的最大値**: 97点
  - Top AAA: 22点
  - Middle RF: 50点
  - Bottom RF: 25点

- **実用的上限**: 15-20点（非常に好調なゲーム）

- **期待値目安**:
  - ランダムプレイ: 0-0.5点
  - 基本戦略: 2-4点
  - 最適戦略: 5-8点

---

## 3. Phase別実績データ

### 3.1 学習フェーズ推移

| Phase | Steps | Foul Rate | Mean Royalty | 備考 |
|:---|---:|---:|---:|:---|
| Phase 1 | 1M | 37.8% | 0.34 | ファウル回避基礎 |
| Phase 2 | 2M | 32.0% | 0.26 | 役作り基礎 |
| Phase 3 | 5M | 58-63% | 0.5-0.8 | 2P Self-Play（攻撃的） |
| Phase 4 | 10.5M | 25.1% | 0.85 | Joker導入 |
| Phase 5 | 11.5M | 38.5% | 0.78 | 3人Self-Play |
| **Phase 7** | **20M** | **25.8%** | **7.56** | 並列学習完了 |

### 3.2 観測項目

- **Foul Rate**: ファウル（役の強さ逆転）発生率
- **Mean Royalty**: 1ゲームあたりの平均ロイヤリティ獲得
- **FL Entry Rate**: Fantasyland進入率
- **Win Rate**: 対戦勝率（Self-Play時）

---

## 4. 参考: 人間プレイヤーの推定値

### 4.1 スキルレベル別目安

| レベル | ファウル率 | Mean Royalty | 特徴 |
|:---|---:|---:|:---|
| 初心者 | 40-50% | 1-2点 | 役の強さ順序を理解していない |
| 中級者 | 25-35% | 3-5点 | 基本戦略を習得、確率計算が甘い |
| 上級者 | 15-25% | 5-8点 | 確率計算、ブロッキング戦略習得 |
| プロ | 10-20% | 7-12点 | GTO寄りのプレイ、FL狙い精度高 |

### 4.2 Phase 7モデルの位置づけ

Phase 7完了時点（20Mステップ）:
- **Foul Rate 25.8%**: 中級者〜上級者レベル
- **Mean Royalty 7.56**: 上級者レベル

→ 全体として**上級者相当**の性能を達成

---

## 5. 評価プロトコル

### 5.1 統計的有意性の確保

- 最低ゲーム数: 1,000ゲーム
- 推奨ゲーム数: 10,000ゲーム以上
- 報告項目: 平均値 ± 標準偏差

### 5.2 対戦相手の標準化

| 評価タイプ | 対戦相手 | 用途 |
|:---|:---|:---|
| 絶対評価 | ランダムエージェント | 基礎性能測定 |
| 相対評価 | 過去バージョン | 学習進捗確認 |
| 自己対戦 | 同一モデル | 戦略均衡確認 |
| 人間対戦 | 人間プレイヤー | 実用性評価 |

### 5.3 報告フォーマット

```
Model: [モデル名]
Steps: [学習ステップ数]
Games: [評価ゲーム数]
Opponent: [対戦相手タイプ]

Results:
- Foul Rate: XX.X% ± X.X%
- Mean Royalty: X.XX ± X.XX
- FL Entry Rate: X.X%
- Win Rate: XX.X% (Self-Play時のみ)
```

---

## 6. コード内での実装

### 6.1 統計収集クラス

```python
class GameStatistics:
    def __init__(self):
        self.fouls = deque(maxlen=1000)
        self.royalties = deque(maxlen=1000)
        self.fl_entries = deque(maxlen=1000)
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def record_game(self, fouled: bool, royalty: float, entered_fl: bool, score: float):
        self.fouls.append(1.0 if fouled else 0.0)
        self.royalties.append(royalty)
        self.fl_entries.append(1.0 if entered_fl else 0.0)

        if score > 0:
            self.wins += 1
        elif score < 0:
            self.losses += 1
        else:
            self.draws += 1

    @property
    def foul_rate(self) -> float:
        return np.mean(self.fouls) * 100 if self.fouls else 0.0

    @property
    def mean_royalty(self) -> float:
        return np.mean(self.royalties) if self.royalties else 0.0

    @property
    def fl_rate(self) -> float:
        return np.mean(self.fl_entries) * 100 if self.fl_entries else 0.0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses + self.draws
        return self.wins / total * 100 if total > 0 else 0.0
```

### 6.2 Discord通知フォーマット

```python
metrics = {
    'games': total_games,
    'foul_rate': f"{foul_rate:.1f}%",
    'mean_royalty': f"{mean_royalty:.2f}",
    'fl_rate': f"{fl_rate:.1f}%",
    'win_rate': f"{win_rate:.1f}%",  # Self-Play時
    'fps': fps
}
```

---

## 更新履歴

| 日付 | 内容 |
|:---|:---|
| 2026-01-18 | 初版作成（Phase 8開始に伴い評価基準を文書化） |
