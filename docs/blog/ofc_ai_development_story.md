# ポーカーAIを強化学習で作ってみた：OFC Pineapple 開発記

## はじめに

「AIにポーカーを教えることはできるのか？」

この素朴な疑問から始まったプロジェクトが、気づけば深層強化学習、クラウドコンピューティング、そしてモンテカルロ木探索にまで発展していました。

本記事では、**OFC Pineapple（オープンフェイス・チャイニーズ・ポーカー・パイナップル）** という複雑なカードゲームを強化学習で攻略しようとした開発の軌跡を、技術的な詳細とともにお伝えします。

---

## OFC Pineapple とは？

OFC Pineappleは、52枚のデッキから配られたカードを3つの列に配置して役を作るポーカーゲームです。

```
┌───────────────────────────┐
│ Top    : [  ] [  ] [  ]   │ ← 3枚（弱い役）
│ Middle : [  ] [  ] [  ] [  ] [  ] │ ← 5枚（中程度）
│ Bottom : [  ] [  ] [  ] [  ] [  ] │ ← 5枚（強い役）
└───────────────────────────┘
```

**重要なルール：**
- Bottom ≥ Middle ≥ Top の強さ順序を守らないと **「ファウル」** で負け
- 一度置いたカードは動かせない
- 毎ターン3枚配られ、2枚配置・1枚捨て

つまり、**最初は弱い役しか見えない状態で、強い役が作れるスペースを確保しつつ、将来の可能性を残す**という長期的な戦略が必要です。

これがAIにとって難しい理由：
1. **不完全情報ゲーム**：相手の手札は見えない
2. **長期計画**：13ターンかけて完成させる
3. **確率計算**：残りデッキから何が来るかを予測

---

## Phase 1: ファウルと戦う

### 最初の挑戦

まず作ったのは、「ファウルしない」ことだけを目標にしたAIです。

```python
# 報酬設計 v1
reward = -30 if foul else royalty_points
```

結果は惨憺たるものでした。

| 指標 | 値 |
|------|-----|
| ファウル率 | **87.5%** |
| 勝率 | 0% |
| 学習時間 | 9分 |

AIはほぼ毎回ファウルしていました。なぜでしょうか？

### 問題の発見

理由は明快でした：**報酬がゲーム終了時にしか与えられない**。

13ターンもかけてゲームを進め、最後にやっと「ファウルだった」と教えられても、AIはどこで間違えたのかわかりません。これは「**報酬の希薄性問題（Sparse Reward Problem）**」と呼ばれます。

### 解決策：Curriculum Learning

そこで採用したのが**Curriculum Learning（段階的学習）**です。

```
Phase 1: ファウル回避に集中（報酬 -30 for foul）
    ↓
Phase 2: ロイヤリティ獲得を追加
    ↓
Phase 3: 対戦相手との駆け引き
```

一度に全てを学ばせるのではなく、まず「ファウルしない」という基本を徹底的に学ばせ、それができてから次のステップに進む。人間の学習と同じですね。

---

## Phase 2: AWSクラウドで本格学習

### なぜクラウドが必要だったのか

ローカルPCでは500,000ステップに9分かかりましたが、Phase 1の目標達成には**5,000,000ステップ（約90分）** が必要でした。さらにPhase 3では**10,000,000ステップ**を予定しています。

**EC2 m7i-flex.large インスタンス**を使うことで：
- 24時間連続稼働が可能
- GPUなしでも十分（PPOはCPU効率が良い）
- コスト：約$0.05/時間

### Docker化による再現性確保

```yaml
# docker-compose.yml
services:
  training:
    build: .
    container_name: ofc-training-phase1
    volumes:
      - ./models:/app/models
    command: ["python", "train_phase1.py", "--steps", "5000000"]
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

これにより：
- どのマシンでも同じ環境を再現
- モデルの自動保存
- リソース制限で安全に稼働

### Phase 1 結果

5,000,000ステップ後：

| 指標 | 開始時 | 終了時 |
|------|--------|--------|
| ファウル率 | 87.5% | **22.9%** |
| 平均ロイヤリティ | 0 | 0.07 |

約65%ポイントの改善！AIはファウルを避ける基本を身につけました。

---

## Phase 2: ロイヤリティを求めて

ファウル回避ができるようになったので、次は**ロイヤリティ（ボーナス点）**の獲得を目指します。

### 報酬の調整

```python
# Phase 2 報酬設計
reward = -30 if foul else (royalty_bonus + win_bonus)
```

OFCでは強い役を作るとボーナスがもらえます：
- Top: スリーカード → +10点以上
- Middle: ストレートフラッシュ → +30点
- Bottom: フォーカード → +10点

### Phase 2 結果

| 指標 | Phase 1終了時 | Phase 2終了時 |
|------|--------------|--------------|
| ファウル率 | 22.9% | **39.0%** |
| 平均ロイヤリティ | 0.07 | **0.14** |

ファウル率が上がったのは、**リスクを取ってロイヤリティを狙うようになった**から。これは想定通りの「J-curve効果」です。

---

## Phase 3: Self-Play - AIが自分自身と戦う

### なぜSelf-Playが必要か

ここまでのAIは「一人でカードを並べる」練習しかしていません。しかし実際のゲームでは、相手がいます。相手のカードを見て、自分の戦略を調整する必要があります。

**Self-Play**では、AIが自分自身のコピーと対戦することで、対戦環境での戦略を学習します。

### 「Latest vs Pool」戦略

単純に最新のモデルと対戦させるとどうなるか？

→ **直近の戦略にだけ強くなり、多様な戦略に弱くなる**

これを防ぐために「Latest vs Pool」戦略を採用：

```
対戦相手の選び方:
├─ 80%: 過去のモデルプールからランダム選択
└─ 20%: 最新モデル
```

こうすることで、過去の自分にも勝てる堅牢なAIが育ちます。

### Self-Play 環境の実装

```python
class SelfPlayEnvPhase3(gym.Env):
    def __init__(self, pool_ratio=0.8, win_bonus=10.0):
        self.opponent_pool = []  # 過去モデルのプール
        self.latest_opponent = None
        self.win_bonus = win_bonus
        self.pool_ratio = pool_ratio
    
    def reset(self):
        # 80%の確率でプールから、20%で最新モデル
        if random() < self.pool_ratio and self.opponent_pool:
            self.active_opponent = choice(self.opponent_pool)
        else:
            self.active_opponent = self.latest_opponent
```

---

## 高度な研究：推論時探索

### MCTSの導入

ここまでのAIは「直感型」です。観測を見て、即座に行動を決める。

しかし、人間のトッププレイヤーは違います。「この手を打ったらどうなるか」を何手も先まで読みます。

これを実現するのが**Monte Carlo Tree Search（MCTS）**です。

```python
def predict_with_search(obs, model, simulations=100):
    # 1. モデルから有力な候補手を取得
    candidates = model.get_top_k_actions(obs, k=3)
    
    # 2. 各候補手についてシミュレーション
    scores = {}
    for action in candidates:
        avg_score = simulate_rollout(action, n=simulations)
        scores[action] = avg_score
    
    # 3. 最良の手を選択
    return max(scores, key=scores.get)
```

### 実験結果

| 手法 | 勝率 vs Policy-only |
|------|---------------------|
| ランダムロールアウト | 25% |
| ヒューリスティックロールアウト | 27% |

**教訓：** 単純なロールアウトでは学習済みPolicyに勝てない。真のMCTS改善には、C++エンジンでの高速シミュレーションが必要。

---

## 終盤ソルバー：完全解析

### アイデア

残り5枚未満の局面では、全パターンを探索して**理論上の最適解**を導出できます。

```python
class EndgameSolver:
    def solve(self, engine, player=0):
        if engine.remaining_cards_in_board(player) > 5:
            return None  # 全探索不可
        
        # 全配置パターンを試す
        best_action, best_score = None, float('-inf')
        for action in all_possible_actions():
            cloned = engine.clone()
            cloned.apply(action)
            score = evaluate(cloned)
            if score > best_score:
                best_action, best_score = action, score
        
        return best_action
```

### C++エンジンの拡張

終盤ソルバーにはゲーム状態のコピーが必要です。そこでC++エンジンにシリアライズ機能を追加：

```cpp
// game.hpp
std::vector<uint8_t> GameEngine::serialize() const;
bool GameEngine::deserialize(const std::vector<uint8_t>& data);
GameEngine GameEngine::clone() const;
```

これにより、任意の局面をコピーして「if-then」シミュレーションが可能に。

---

## 技術スタック

| レイヤー | 技術 | 選定理由 |
|----------|------|----------|
| ゲームエンジン | C++ | 高速なビットボード演算 |
| Python連携 | pybind11 | シームレスなC++/Python統合 |
| 環境 | PettingZoo | マルチエージェント対応 |
| 学習 | MaskablePPO | 無効アクションのマスキング |
| インフラ | AWS EC2 + Docker | 再現性と長時間学習 |
| 通知 | Discord Webhook | 学習進捗のリアルタイム監視 |

---

## 学んだこと

### 1. 報酬設計が全て
強化学習の成否は報酬設計で決まる。「何を目標にするか」を明確に定義することが最重要。

### 2. Curriculum Learningの威力
複雑なタスクは分解して段階的に学ばせる。人間の学習と同じ原則がAIにも適用できる。

### 3. Self-Playの落とし穴
単純なSelf-Playは局所解に陥りやすい。モデルプールで多様性を確保。

### 4. 探索と学習の組み合わせ
直感（Policy）だけでは限界がある。思考（Search）を組み合わせることで超人レベルへ。

---

## 今後の展望

1. **AlphaZero的アプローチ**: MCTSで生成したデータで再学習
2. **確率特徴量の追加**: フラッシュ完成率などをC++で計算し入力に追加
3. **Jokerルール対応**: Phase 4として実装予定

---

## おわりに

「ポーカーAIを作る」という一見シンプルなプロジェクトが、深層強化学習の奥深さを体感する旅になりました。

GitHubで全コードを公開しています：
[OFC-NN リポジトリ](https://github.com/xxx/OFC-NN)

質問やフィードバックはお気軽に！

---

*この記事は開発の途中経過をまとめたものです。Phase 3の結果は後日追記予定。*
