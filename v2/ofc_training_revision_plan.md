# OFC AI 学習システム 修正計画書

**作成日**: 2026-02-26
**対象ドキュメント**: `09_training_system_deep_dive.md`
**入力**: スペシャリストA・B 2名のレビュー意見
**目的**: Claude Code への提出用 — 実装方針と優先順位の明確化

---

## 1. エキスパートレビュー総括

### 共通して指摘された重大課題（A・B両者が一致）

以下の4点は両スペシャリストが独立して指摘しており、**最優先で対処すべき構造的課題**である。

| # | 課題 | Aの指摘 | Bの指摘 | 深刻度 |
|---|------|---------|---------|--------|
| 1 | **報酬ハッキング（Reward Hacking）** | FL接近報酬の合計が大きすぎ、評価基盤が弱い | FL接近ボーナス合計+23 vs フォールペナルティ-6.2 → 差し引き+16.8で「わざとフォール」が黒字 | 🔴 Critical |
| 2 | **行動空間の意味ブレ** | Street1とStreet2-5で同じ243枠に異なる意味を押し込んでいる | Action 42がフェーズで意味が劇的に切り替わり収束速度が低下 | 🟠 High |
| 3 | **カード表現の限界** | one-hot中心は冗長。役形成に直結する集約特徴量を増やすべき | One-hotではA♠とK♠の距離がA♠と2♥と同じ。Embedding層が有効 | 🟠 High |
| 4 | **Self-Playプールの不足** | pool_size=5では3人戦メタの偏りが発生 | pool_size=5は小さすぎる。対数サンプリングで破滅的忘却を防ぐ | 🟡 Medium |
| 5 | **FLソルバー依存** | 最重要局面をAI自身が学ばない設計 | ソルバーの癖にAI戦略が引っ張られる | 🟡 Medium |

### A氏のみの追加指摘

| # | 課題 | 内容 | 深刻度 |
|---|------|------|--------|
| 6 | **player_0固定学習の非対称性** | 座順の癖を過学習する可能性。学習主体のローテーションが必要 | 🟡 Medium |
| 7 | **学習監視指標の不足** | 平均スコアだけでは「なぜ強く/弱くなったか」が不明 | 🟡 Medium |
| 8 | **評価基盤の先行整備** | 30M回す前に評価系を固めないと開発リスクになる | 🟠 High |

---

## 2. 修正項目の詳細と実装方針

### 2.1 🔴 [Critical] 報酬ハッキングの防止

**問題の本質**:
現在のFL接近ボーナスの合計は最大+23（Stage1: +5, Stage2: +10, Stage3: +8）。
一方フォールペナルティはC++スコアで-6、中間報酬の緩和ペナルティで-0.2。
AIは「Topに無理やりTripsを作って+23のボーナスをもらい、フォールして-6.2の罰則を受けても+16.8の黒字」という学習をする可能性が極めて高い。

**修正方針（2つのアプローチを段階実施）**:

#### アプローチA: ポテンシャルベース報酬（推奨・優先実装）

```python
# 修正前: 各ステップで即時付与
def _apply_intermediate_reward(self):
    stage = self._get_fl_approach_stage()
    if stage >= 1: reward += 5.0   # Stage1
    if stage >= 2: reward += 10.0  # Stage2
    if stage >= 3: reward += 8.0   # Stage3

# 修正後: ゲーム終了時にフォールしていない場合のみ付与
def _calculate_final_rewards(self):
    if not fouled:
        # FL接近報酬をゲーム終了時にまとめて付与
        stage = self._get_fl_approach_stage()
        reward += fl_approach_bonus[stage]
    else:
        # フォール時はFL接近報酬を一切付与しない
        pass
```

#### アプローチB: スケール大幅縮小（簡易実装・即座に適用可能）

```python
# FL接近報酬をフォールペナルティより圧倒的に小さくする
# 合計が最大でも +2.3 程度（フォール-6.0の1/3以下）
fl_stage1_reward = 0.5   # 旧: 5.0 → 1/10に縮小
fl_stage2_reward = 1.0   # 旧: 10.0 → 1/10に縮小
fl_stage3_reward = 0.8   # 旧: 8.0 → 1/10に縮小
```

**実装優先度**: まずアプローチBで即座にスケール修正 → アプローチAをPhase2で実装

---

### 2.2 🟠 [High] 行動空間の再設計

**問題の本質**:
243次元の出力層に、Street 1（5枚→3列振り分け）と Street 2-5（2枚配置+1枚捨て）の
意味が異なるアクションが混在。ニューラルネットの学習安定性を損なう。

**修正方針（3段階）**:

#### Step 1: 観測空間のcurrent_street改善（即座に実装可能）

```python
# 修正前: current_streetは単一スカラー値
game_state[0] = current_street  # 1〜5の数値

# 修正後: One-hotベクトル（5次元）に変更
# game_stateの[0]を5次元に拡張（観測空間は881→885次元）
current_street_onehot = np.zeros(5)
current_street_onehot[current_street - 1] = 1.0
```

#### Step 2: アクション空間の分離（中期実装）

```python
# 案1: 270次元方式（B氏提案）
# Street 1用: index 0-242 (243アクション)
# Street 2-5用: index 243-269 (27アクション)
# フェーズに応じてマスクで完全切り替え
action_space = Discrete(270)

# 案2: 階層型行動（A氏提案）
# 捨て札選択（3択）→ 配置選択（9択）の2段階
# MaskablePPOとの統合にやや工夫が必要
```

**推奨**: Step 1を即座に実装。Step 2は案1（270次元方式）をPhase2で実装。
270次元方式は既存のMaskablePPOパイプラインとの互換性が高い。

---

### 2.3 🟠 [High] 観測空間の改善

**問題の本質**:
881次元のone-hot中心構成は冗長。カード間の関係性（同ランク、同スート）を
AIが自力で膨大なステップ数をかけて学習する必要がある。

**修正方針（2段階）**:

#### Step 1: extended_fl_obsの標準ON化（即座に実装可能）

```python
# 現在はデフォルトFalseになっている以下を標準ONに変更
extended_fl_obs = True  # +6次元（live_queens, live_kings, live_aces, etc.）

# さらに追加すべき集約特徴量:
# - 各スートの未見枚数（4次元）
# - 各列の現在のハンドランク（3次元、正規化）
# - フラッシュアウツ数（各列・各スート、最大3次元）
# - ストレートアウツ数（各列、3次元）
```

#### Step 2: Embedding層の導入（中期実装）

```python
# カード表現を「ランク(13次元) × スート(4次元)」の組み合わせに変更
# ネットワーク入力直後にEmbedding層を挟む

# policy_kwargsにカスタムfeature extractorを指定
policy_kwargs = dict(
    features_extractor_class=OFCFeatureExtractor,
    features_extractor_kwargs=dict(
        card_embedding_dim=16,  # 各カードを16次元に埋め込み
    ),
    net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
)
```

**推奨**: Step 1を即座に実装。Step 2はPhase3以降（ネットワーク構造変更のため大改修）。

---

### 2.4 🟡 [Medium] Self-Play プールの拡張

**問題の本質**:
pool_size=5は3人戦OFCではメタの偏りが発生しやすい。
「自分の分身にだけ強いAI」になるリスク。

**修正方針**:

```python
# === Self-Play Opponent Pool 修正 ===

# 1. プールサイズの拡張
pool_size = 15  # 旧: 5 → 15に拡張

# 2. 対数的サンプリング（Historical Sampling）
# 線形保存ではなく、古いモデルを対数的に残す
# 例: 200K前, 400K前, 800K前, 1.6M前, 3.2M前...
def _should_keep_model(self, steps_ago):
    """対数的間隔でモデルを保持"""
    milestones = [200_000 * (2 ** i) for i in range(10)]
    return any(abs(steps_ago - m) < 100_000 for m in milestones)

# 3. 対戦相手の構成比率変更
latest_prob = 0.6      # 旧: 0.8 → 0.6（最新モデル）
historical_prob = 0.3  # 過去モデル（対数サンプリング）
rule_based_prob = 0.1  # 固定ルールベース相手（新規追加）

# 4. ルールベース相手の追加（A氏提案）
class RuleBasedOpponent:
    """固定的な戦略を持つ相手（学習偏り防止用）"""
    def predict(self, obs, masks):
        # 安全重視: フォール回避優先
        # FL非追求: 常に安全な配置
        pass
```

---

### 2.5 🟡 [Medium] FLソルバー依存の段階的解消

**修正方針（2段階計画）**:

```
Phase A（現行維持）:
  - FL配置はgreedy solverに委任
  - AIは「FLに入る前」の戦略のみ学習
  - 速度優先で問題なし

Phase B（将来計画）:
  - FL配置を別モデルで学習
  - greedy solverの出力を教師データとして使用
  - 段階的にモデル出力の比率を上げる

具体的なPhase B設計:
  1. greedy solverの配置ログを収集
  2. FL配置専用モデルを教師あり学習で事前訓練
  3. 本体AIのFL Entry判断は引き続きPPOが担当
  4. FL配置モデルをRL（PPO）でファインチューニング
```

**注意点（B氏）**: ソルバーが特定のFL構成を見落とす場合、AIもそのFLへの突入を
無意識に避けるようになる。ソルバーの品質評価テストも並行して実施すること。

---

### 2.6 🟡 [Medium] 学習主体のローテーション

**修正方針**:

```python
# 修正前: player_0のみ固定で学習
# → 座順の癖を過学習するリスク

# 修正後: エピソードごとに学習主体をローテーション
class ParallelOFCEnv:
    def reset(self):
        # 学習対象プレイヤーをランダムに選択
        self.learning_player = random.choice([0, 1, 2])
        # 観測・報酬をこのプレイヤー視点で返す
```

---

### 2.7 🟡 [Medium] 学習監視指標の拡充

**追加すべき指標**:

```python
# === 既存指標 ===
# Foul Rate, FL Entry, FL Stay, Mean Score, FPS

# === 追加指標（A氏提案） ===
additional_metrics = {
    # 対戦相手タイプ別勝率
    "win_rate_vs_latest":    deque(maxlen=200),
    "win_rate_vs_historical": deque(maxlen=200),
    "win_rate_vs_rule_based": deque(maxlen=200),

    # ポジション別平均スコア
    "score_as_btn":  deque(maxlen=200),
    "score_as_sb":   deque(maxlen=200),
    "score_as_bb":   deque(maxlen=200),

    # 列別ロイヤリティ期待値
    "royalty_top":    deque(maxlen=500),
    "royalty_mid":    deque(maxlen=500),
    "royalty_bot":    deque(maxlen=500),

    # FL戦略の詳細分析
    "ev_when_fl_pursuing":     deque(maxlen=200),
    "ev_when_not_fl_pursuing": deque(maxlen=200),
    "fl_attempt_foul_rate":    deque(maxlen=200),  # FL狙い時のフォール率
}
```

---

### 2.8 🟠 [High] 評価基盤の先行整備

**A氏の最重要提言**: 学習本体が強いぶん、評価の弱さがそのまま開発リスクになる。

```
評価基盤に必要なもの:

1. 固定ベンチマーク相手群
   - ルールベースAgent（安全重視型）
   - ルールベースAgent（攻撃型・FL追求型）
   - 過去の固定チェックポイント（1M, 5M, 10M時点など）
   - ランダムAgent（下限ベースライン）

2. 評価プロトコル
   - 各ベンチマーク相手と1000ゲーム以上の対局
   - 全ポジション（BTN/SB/BB）で均等に評価
   - 統計的有意差の検定（95%信頼区間）

3. 報酬アブレーション
   - FL接近ボーナスON/OFF
   - フォールペナルティ強度の変化
   - 最終報酬のみ vs 中間報酬あり
```

---

## 3. 実装ロードマップ（修正版）

### Phase 0: 基盤整備（学習開始前） ← **新設・最優先**

```
0-1. ルール・スコアの完全検証
     - C++エンジンの全スコア計算パスのユニットテスト
     - エッジケース（Joker、FL Stay条件等）の網羅テスト

0-2. 固定ベンチマーク相手の作成
     - RuleBasedAgent（安全型/攻撃型）の実装
     - 評価スクリプトの作成（1000ゲーム自動対局）

0-3. 報酬ハッキング修正（アプローチB: スケール縮小）
     - FL接近ボーナスを1/10に縮小
     - フォールペナルティとのバランス確認

0-4. 観測空間の即時改善
     - extended_fl_obs = True をデフォルト化
     - current_street の One-hot化（5次元）

0-5. 監視指標の拡充
     - 追加メトリクスの実装
     - Discord通知フォーマットの更新
```

### Phase 1: 小規模検証学習（1M〜5Mステップ）

```
1-1. 報酬アブレーション実験
     - 3種類の報酬設定で1Mステップずつ比較
     - 「わざとフォール」が発生しないことを確認

1-2. Self-Playプール拡張の検証
     - pool_size=15 + 対数サンプリング
     - ルールベース相手10%混合

1-3. ベンチマーク評価の実施
     - 各設定のAIをベンチマーク相手群と対局
     - 最良の設定を選定

1-4. 学習主体ローテーションの検証
     - player_0固定 vs ローテーション を比較
```

### Phase 2: 構造改善 + 中規模学習（5M〜30Mステップ）

```
2-1. 行動空間の分離実装
     - 270次元方式（Street1: 0-242, Street2-5: 243-269）
     - MaskablePPOマスクの対応修正

2-2. 報酬ハッキング修正（アプローチA: ポテンシャルベース）
     - FL接近報酬をゲーム終了時に条件付き付与
     - Phase 1の結果と比較

2-3. 中規模学習の実施
     - 最良設定で30Mステップ
     - 定期的にベンチマーク評価を実施
```

### Phase 3: 高度化 + 本学習（30M〜250Mステップ）

```
3-1. カード表現のEmbedding化
     - カスタムFeatureExtractorの実装
     - ランク(13次元) × スート(4次元) → Embedding(16次元)

3-2. FL配置モデルの学習（Phase B）
     - greedy solverの配置ログ収集
     - FL専用モデルの教師あり事前訓練

3-3. 本学習の実施
     - 全改善を統合した最終版で250Mステップ
     - 継続的ベンチマーク評価
```

---

## 4. パラメータ変更サマリー

| パラメータ | 旧値 | 新値 | 変更理由 | Phase |
|-----------|------|------|---------|-------|
| `fl_stage1_reward` | 5.0 | **0.5** | 報酬ハッキング防止 | 0 |
| `fl_stage2_reward` | 10.0 | **1.0** | 報酬ハッキング防止 | 0 |
| `fl_stage3_reward` | 8.0 | **0.8** | 報酬ハッキング防止 | 0 |
| `extended_fl_obs` | False | **True** | 集約特徴量の活用 | 0 |
| `current_street表現` | スカラー | **One-hot(5次元)** | フェーズ認識の強化 | 0 |
| `pool_size` | 5 | **15** | メタ偏り防止 | 1 |
| `latest_prob` | 0.8 | **0.6** | 多様な相手との対戦 | 1 |
| `rule_based_prob` | 0.0 | **0.1** | 偏り防止（新設） | 1 |
| `action_space` | Discrete(243) | **Discrete(270)** | 意味の固定化 | 2 |
| `learning_player` | 0固定 | **ローテーション** | 座順過学習防止 | 1 |

---

## 5. リスク評価と対策

| リスク | 影響度 | 対策 |
|--------|--------|------|
| 報酬スケール縮小でFL学習が遅くなる | 中 | FL Entry Rateを監視。0%が続けばent_coefを微調整 |
| 観測空間拡張で学習速度低下 | 低 | 追加次元は少数。影響は限定的 |
| Self-Playプール拡張でメモリ増加 | 低 | モデルweightのみ保存。15モデルでも数百MB |
| 270次元方式で既存コードとの互換性問題 | 中 | Phase 2まで延期。Phase 1は243維持 |
| ローテーション学習で収束速度低下 | 中 | Phase 1で1M検証してから採用判断 |

---

## 6. 成功判定基準（ベンチマーク）

### Phase 0 完了条件
- [ ] C++エンジンのユニットテスト全パス
- [ ] ベンチマーク対局スクリプトが動作
- [ ] 報酬スケール修正が適用済み

### Phase 1 完了条件
- [ ] 1Mステップで「わざとフォール」が発生しない（Foul Rate < 30%）
- [ ] FL Entry Rate > 5%（FL学習が進んでいる）
- [ ] ベンチマーク相手に対して安定した勝率

### Phase 2 完了条件
- [ ] 30Mステップで Foul Rate < 20%
- [ ] FL Entry Rate > 10%
- [ ] ランダム相手に対して勝率 > 80%
- [ ] ルールベース相手に対して勝率 > 60%

### Phase 3 完了条件
- [ ] 250Mステップで安定したプレイ品質
- [ ] FL Stay Rate > 5%
- [ ] 全ポジションで均等な勝率（±5%以内）
