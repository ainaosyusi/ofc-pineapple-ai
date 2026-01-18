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

## Phase 4: ジョーカー対応 - JOPTルール

### 54枚デッキへの拡張

JOPT（Japan Open Poker Tour）のルールに準拠するため、**ジョーカー2枚**を追加した54枚デッキに対応しました。

```cpp
// evaluator.hpp
// ジョーカーをワイルドカードとして評価
HandRank evaluateWithJokers(const Card cards[], int count, int jokerCount) {
    // ジョーカー0枚: 通常評価
    // ジョーカー1枚: 最も有利な役に昇格
    // ジョーカー2枚: さらに強い役へ
}
```

### Phase 4 結果（1050万ステップ）

| 指標 | 値 |
|------|-----|
| **ファウル率** | **25.1%** (全Phase最良) |
| **平均ロイヤリティ** | **0.85** |
| **FL突入率** | **1.1%** |

ジョーカーは「困った時の保険」として効果的に機能し、ファウル率を大幅に改善しました。

---

## Phase 5: 3人対戦 (3-Max)

### マルチエージェント環境

2人対戦から3人対戦（3-Max）へ拡張。Triangle Scoringによる複雑な利害関係を学習させます。

```python
# 観測空間の拡張（約720次元）
{
    'my_board': 162,
    'my_hand': 270,
    'next_opponent_board': 162,  # 下家
    'prev_opponent_board': 162,  # 上家
    'my_discards': 54,
    'unseen_probability': 54,
    'position_info': 3,
}
```

### 学習される能力

1. **アウツ・ブロック**: 相手が集めているスートを避ける
2. **ポジション戦略**: 後出し有利を活用
3. **リスク管理**: 2人相手のファウルは×2のダメージ

---

## Phase 7: 並列学習 - GCPへの移行

### AWSからGCPへ

AWS EC2の不安定性（SSH接続断、学習中断）を受け、GCP GCEへ移行しました。

### SubprocVecEnvによる並列化

単純にマシンスペックを上げてもFPSは大きく改善しませんでした。そこで**SubprocVecEnv**による並列環境を導入。

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

NUM_ENVS = 4
env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
model = MaskablePPO("MultiInputPolicy", env, batch_size=256, ...)
```

### 劇的なパフォーマンス改善

| 構成 | インスタンス | FPS | コスト/時間 |
|------|-------------|-----|------------|
| シングル | n2-standard-16 | 186 | $0.76 |
| **並列4環境** | **n2-standard-4** | **4,494-12,382** | **$0.19** |

**約30-90倍のFPS向上、コスト1/4に削減！**

### 学習進捗（2026/01/17）

```
Progress: 12.5% (2,500,000 / 20,000,000)
Foul Rate: 34.0% (改善中)
FPS: 4,494-12,382
Estimated Cost: ~$2 (約300円)
```

---

## Web UI & Discord Bot

### ブラウザで対戦

FastAPIベースのWebインターフェースを実装しました。

```bash
python src/python/web_ui.py
# http://localhost:8000 でアクセス
```

### Discord Bot

`/play` コマンドでAIと対戦できるDiscord Botも作成。

```
/play   - 新しいゲームを開始
/board  - 現在のボードを表示
/status - 学習状況を確認
```

---

## 技術スタック（最新版）

| レイヤー | 技術 | 選定理由 |
|----------|------|----------|
| ゲームエンジン | C++ (pybind11) | 高速なビットボード演算 |
| 環境 | PettingZoo AECEnv | マルチエージェント対応 |
| 学習 | MaskablePPO (sb3-contrib) | 無効アクションのマスキング |
| 並列化 | SubprocVecEnv | CPUコアの効率的活用 |
| インフラ | GCP GCE (n2-standard-4) | コスト効率とGCS連携 |
| 通知 | Discord Webhook | 100kステップごとに進捗通知 |

---

## フェーズ別成績まとめ

| Phase | 概要 | ファウル率 | 特記事項 |
|-------|------|-----------|---------|
| 1 | ファウル回避 | 37.8% | 基礎習得 |
| 2 | 役作り | 32.0% | 安定 |
| 3 | 2人Self-Play | 58-63% | 攻撃的 |
| **4** | **ジョーカー** | **25.1%** | **🏆 最良** |
| 5 | 3人Self-Play | 38.5% | 3-Max対応 |
| 7 | 並列学習 | 34.0%→ | 進行中 |

---

## 学んだこと（追記）

### 5. 並列化の効果
マシンスペックを上げるより、並列環境で効率的にCPUを使う方が効果的。特に環境シミュレーションがボトルネックになるゲームAIでは顕著。

### 6. クラウドの選択
AWS/GCP両対応のコードを書いておくと、障害時の移行がスムーズ。抽象化レイヤー（`cloud_storage.py`）を設けることで切り替えが容易に。

---

## 今後の展望

1. [ ] Phase 7完了（20Mステップ到達）
2. [ ] 最終モデルの対人テスト
3. [ ] Webデモの公開
4. [ ] 論文/技術記事の執筆

---

## おわりに

「ポーカーAIを作る」という一見シンプルなプロジェクトが、深層強化学習、分散コンピューティング、そしてシステム設計にまで発展しました。

現在も**GCP上で並列学習が進行中**です。Phase 4で達成した**ファウル率25.1%**を超えることを目指して学習を続けています。

GitHubで全コードを公開しています：
[OFC-NN リポジトリ](https://github.com/xxx/OFC-NN)

質問やフィードバックはお気軽に！

---

*最終更新: 2026-01-17 - Phase 7 並列学習を追記*
