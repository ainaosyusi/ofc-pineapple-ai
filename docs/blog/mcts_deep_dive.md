# AlphaZeroに学ぶ：MCTS（モンテカルロ木探索）でポーカーAIを強化する

## はじめに

AlphaGo、AlphaZero、MuZero...これらのAIの共通点は何でしょうか？

答えは**MCTS（Monte Carlo Tree Search）** です。

強化学習で「直感」を学んだニューラルネットワークに、「思考」を追加することで、超人的な性能を実現しています。

本記事では、OFC Pineapple AIにMCTSを導入した実験の詳細と、そこから得られた教訓を共有します。

---

## Policy Network だけでは足りない

### 現状の問題

強化学習で訓練したPolicy Networkは「直感型」のプレイヤーです。

```
観測 → Policy Network → アクション
```

これは高速ですが、以下の限界があります：

1. **先読みしない**：現在の状態だけで判断
2. **確率的**：最も良さそうな手を選ぶが、確証はない
3. **学習データ依存**：見たことない状況に弱い

人間のエキスパートは違います。「この手を打ったらどうなるか」を何手も先まで読みます。

### MCTSで「思考」を追加

```
観測 → Policy Network → 候補手（Top-K）
           ↓
       各候補手をシミュレーション
           ↓
       最も良い結果の手を選択
```

---

## MCTSの基本構造

### アルゴリズム概要

```
1. 選択（Selection）
   - 現在のノードから、UCB1スコアが最大の子ノードを選ぶ
   
2. 展開（Expansion）
   - 未探索のアクションでノードを展開
   
3. シミュレーション（Simulation）
   - ゲーム終了までランダム（または Policy）でプレイ
   
4. バックプロパゲーション（Backpropagation）
   - 結果を親ノードに伝播し、統計を更新
```

### OFCへの適用

```python
class MCTSAgent:
    def predict_with_search(self, obs, action_mask, simulations=100):
        # 1. Policyから候補手を取得
        candidates = self.get_policy_actions(obs, action_mask, k=3)
        
        # 2. 各候補手をシミュレーション
        scores = {}
        for action, policy_prob in candidates:
            sim_score = self.simulate_rollout(action)
            
            # Policyとシミュレーション結果を組み合わせ
            combined = (
                self.policy_weight * policy_prob +
                (1 - self.policy_weight) * sim_score
            )
            scores[action] = combined
        
        # 3. 最良手を選択
        return max(scores, key=scores.get)
```

---

## 実験1: ランダムロールアウト

### 仮説

「弱いモデル + MCTS > 強いモデル（直感のみ）」

これがAlphaZeroの成功の鍵でした。では、ポーカーでも成り立つでしょうか？

### 設定

```python
MCTSAgent(
    top_k=3,           # 候補手数
    simulations=30,    # シミュレーション回数
    policy_weight=0.3  # Policy vs Simulation の重み
)
```

ロールアウト：ランダムにゲーム終了までプレイ

### 結果

| 対戦 | MCTS | Policy-only |
|------|------|-------------|
| 勝率 | **25%** | **75%** |
| 平均スコア | -15.25 | +15.25 |

**惨敗。** なぜでしょうか？

---

## 分析：なぜランダムロールアウトは失敗したか

### 問題1: シミュレーションの質

ランダムにプレイすると、ほぼ確実にファウルします。

```
シミュレーション結果:
- ファウル率: ~90%
- 平均スコア: -25〜-30
```

どの手を打っても結果が似たようなものになり、**手の良し悪しを区別できない**。

### 問題2: ゲームの特性

チェスや囲碁と違い、OFCには以下の特徴があります：

1. **不確定要素が大きい**：次に来るカードがわからない
2. **ファウルの崖**：少しのミスで全てが台無し
3. **長期的影響**：序盤の決定が終盤に響く

ランダムプレイではこれらを考慮できません。

---

## 実験2: ヒューリスティックロールアウト

### 改良

ランダムの代わりに、簡単なルールで評価：

```python
def _estimate_policy_value(self, action):
    row = action % 3  # 0=TOP, 1=MID, 2=BOT
    
    # ヒューリスティック: Bottomは安全、Topは危険
    row_scores = {
        0: 0.0,   # TOP: リスク高
        1: 2.0,   # MID: 中程度
        2: 5.0    # BOT: 安全
    }
    
    return row_scores.get(row, 0) + noise
```

### 結果

| 対戦 | MCTS | Policy-only |
|------|------|-------------|
| 勝率 | **27%** | **73%** |

わずか2%の改善。まだPolicy-onlyに負けています。

### 教訓

**単純なヒューリスティックでは、学習済みPolicy Networkの価値を超えられない。**

---

## 解決策：C++エンジンでの高速シミュレーション

### 必要な機能

真のMCTSを実現するには、**ゲーム状態をコピーして先読み**する必要があります。

```cpp
// game.hpp に追加
class GameEngine {
public:
    // 状態のシリアライズ
    std::vector<uint8_t> serialize() const;
    bool deserialize(const std::vector<uint8_t>& data);
    
    // 状態のクローン（MCTS用）
    GameEngine clone() const;
    
    // 残りカード枚数（終盤判定用）
    int remaining_cards_in_board(int player) const;
};
```

### 実装のポイント

1. **バイナリシリアライズ**：約113バイトで全状態を保存
2. **高速クローン**：シリアライズ→デシリアライズで複製
3. **Python連携**：pybind11でシームレスに呼び出し

```python
# Pythonから使用
engine = ofc.GameEngine(2)
engine.start_new_game(42)

# 状態をコピー
cloned = engine.clone()

# 別の手を試す
cloned.apply_action(action)
score = cloned.evaluate()
```

---

## 終盤ソルバー：全探索による完全解析

### アイデア

残りカードが5枚以下なら、**全パターンを探索して最適解を見つけられる**。

```python
class EndgameSolver:
    def __init__(self, max_remaining=5):
        self.max_remaining = max_remaining
    
    def can_solve(self, engine, player=0):
        remaining = engine.remaining_cards_in_board(player)
        return remaining <= self.max_remaining
    
    def solve(self, engine, player=0):
        best_action, best_score = None, float('-inf')
        
        for action in all_possible_actions():
            cloned = engine.clone()
            cloned.apply(action)
            score = evaluate(cloned)
            
            if score > best_score:
                best_action, best_score = action, score
        
        return best_action
```

### MCTSとの統合

```python
def predict_with_search(self, obs, action_mask, engine=None):
    # 終盤なら完全解析
    if self.solver and engine:
        if self.solver.can_solve(engine):
            return self.solver.solve(engine)
    
    # 序中盤はMCTS
    return self._mcts_search(obs, action_mask)
```

---

## 今後の方向性

### 1. Policy-guided Rollout

ランダムではなく、**シミュレーション中もPolicy Networkを使用**。

```python
def simulate_with_policy(self, engine, n_sims=50):
    total_score = 0
    for _ in range(n_sims):
        cloned = engine.clone()
        while not cloned.is_game_over():
            obs = cloned.get_observation()
            action = self.model.predict(obs)
            cloned.apply(action)
        total_score += cloned.get_score()
    return total_score / n_sims
```

### 2. AlphaZero式学習

MCTSで生成したデータで再学習。

```
1. MCTSでゲームをプレイ
2. (状態, MCTS選択手, 結果) を記録
3. このデータでPolicy/Value Networkを再学習
4. 繰り返し
```

### 3. 確率特徴量の追加

C++でフラッシュ完成率などを計算し、観測空間に追加。

```python
observation = {
    'board': board_state,
    'hand': hand_cards,
    'flush_probability': engine.calculate_flush_prob(),  # NEW
    'straight_probability': engine.calculate_straight_prob(),  # NEW
}
```

---

## まとめ

| アプローチ | MCTS勝率 | 考察 |
|------------|----------|------|
| ランダムロールアウト | 25% | シミュレーションがノイジー |
| ヒューリスティック | 27% | 単純すぎる評価 |
| 終盤ソルバー | - | 残り5枚以下で完全解 |
| Policy-guided (予定) | ? | 期待大 |

MCTSの威力を発揮するには、**質の高いシミュレーション**が必要。

次のステップ：
1. C++エンジンでのPolicy-guided rollout実装
2. AlphaZero式のセルフプレイ学習
3. 確率計算のC++実装

---

## 参考文献

- Silver et al. (2017) "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
- Browne et al. (2012) "A Survey of Monte Carlo Tree Search Methods"
- Coulom (2006) "Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search"

---

*この記事はOFC Pineapple AI開発の途中経過です。最終結果は別記事で報告予定。*
