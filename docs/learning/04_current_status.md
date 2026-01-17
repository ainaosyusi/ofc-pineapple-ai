# OFC Pineapple AI - 現在の開発状況

## 📅 最終更新: 2026-01-18 17:00

---

## 🚀 現在のステータス: マルチバリアント並行学習中

3つの異なるAIバリアントをGCP上で並行学習中。

### 稼働中インスタンス

| インスタンス | IP | バリアント | 状態 |
|:---|:---|:---|:---|
| ofc-training | 35.243.93.32 | Phase 8 Self-Play | ✅ 学習中 |
| ofc-aggressive | 34.146.34.141 | Aggressive | ✅ 学習中 |
| ofc-teacher | 35.200.57.236 | Teacher Learning | ✅ 学習中 |

### 各バリアントの特徴

| バリアント | 特徴 | 報酬設計 |
|:---|:---|:---|
| **Phase 8 Self-Play** | 過去モデルとの対戦 | 標準スコア + FL +15 |
| **Aggressive** | FL重視、高リスク高リターン | FL +25, Royalty ×1.5 |
| **Teacher** | ルールベース教師の模倣 | スコア + 教師一致ボーナス |

### コスト
- n2-standard-4 × 3 = **約$0.57/時間**
- 20Mステップ完了まで: **約$4-5/バリアント**

---

## 📊 Phase 7 完了実績

| 指標 | 値 |
|:---|---:|
| 総ステップ | 20,000,000 |
| ファウル率 | **25.8%** |
| Mean Royalty | **7.56** |
| FL Entry Rate | 1.1% |
| 勝率 (vs Random) | 65-68% |

### 並列学習の成果
| 比較項目 | 以前 (シングル) | 現在 (並列4環境) |
|:---|:---|:---|
| インスタンス | n2-standard-16 | n2-standard-4 |
| FPS | 129-186 | **4,494-12,382** |
| コスト/時間 | $0.76 | **$0.19** |
| 改善率 | - | **約30-90倍** |

---

## 📊 フェーズ別進捗

```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 7 ──► Phase 8
 ファウル     役作り      Self-Play    ジョーカー    3人対戦     並列学習     Self-Play
回避学習    基礎       (2人)       対応       Self-Play   完了        (3人)
37.8%      32.0%      58-63%     25.1%      38.5%      25.8% 🏆   準備中
```

### 達成したマイルストーン

| フェーズ | 目標 | 結果 | 評価 |
|:---|:---|:---|:---:|
| Phase 1 | ファウル回避の基礎習得 | 37.8% | ✅ |
| Phase 2 | 役作りと報酬最大化 | 32.0%, Royalty 0.26 | ✅ |
| Phase 3 | 対戦型戦略の習得 | 58-63%（攻撃的） | ⚠️ |
| Phase 4 | ジョーカー活用 | 25.1%, Royalty 0.85 | ✅ |
| Phase 5 | 3人戦の戦略習得 | 38.5%, 1150万ステップ | ✅ |
| **Phase 7** | 並列学習 (GCP) | **25.8%**, Royalty **7.56**, 20Mステップ完了 | 🏆 |
| Phase 8 | Self-Play (3人) | 準備完了、学習開始待ち | 🔜 |

---

## 🎯 Phase 8 の特徴 (Self-Play)

### 技術的アプローチ
- **Self-Play**: 過去モデルとの対戦学習
- **対戦相手選択**: 最新80% / 過去モデル20%の確率選択
- **モデルプール**: 5世代を保持、200kステップごとに更新
- **拡張統計**: 勝率、対戦相手タイプ別勝率の追跡

### Self-Play実装
```python
class SelfPlayOpponentManager:
    def __init__(self, pool_size=5, latest_prob=0.8):
        self.model_pool = []
        self.pool_size = pool_size
        self.latest_prob = latest_prob

    def select_opponent(self):
        if not self.model_pool or random.random() < self.latest_prob:
            return self.current_weights, "latest"
        return random.choice(self.model_pool), "past"
```

### Phase 8 ハイパーパラメータ
| 項目 | Phase 7 | Phase 8 | 変更理由 |
|:---|---:|---:|:---|
| total_steps | 20M | 20M | 継続 |
| learning_rate | 1e-4 | 1e-4 | 安定 |
| opponent_update_freq | — | 200K | Self-Play導入 |
| pool_size | — | 5 | 多様性確保 |
| latest_prob | — | 0.8 | 最新優先 |

### 新規追加統計
- `wins`, `losses`, `draws`: 勝敗記録
- `vs_latest_winrate`: 最新モデル対戦時の勝率
- `vs_past_winrate`: 過去モデル対戦時の勝率
- `opponent_pool_size`: プールサイズの推移

---

## 🎯 Phase 7 の特徴 (並列学習) - 完了

### 技術的アプローチ
- **並列環境**: SubprocVecEnv (4並列)
- **アルゴリズム**: MaskablePPO
- **対戦相手**: ランダムプレイ（軽量化のため）
- **過去モデルプール**: 5世代を保持、ランダム選択

### 最終結果 (20Mステップ完了)
| 指標 | 値 |
|:---|:---|
| ファウル率 | **25.8%** |
| Mean Royalty | **7.56** |
| FL Entry Rate | 1.1% |
| 学習速度 | 4,494-12,382 FPS |

### 自動化機能
- 自動レジューム: 中断後も最新チェックポイントから再開
- 自動クリーンアップ: 最新2世代と100万ステップごとの節目を保持
- Discord通知: 100kステップごとに進捗レポート
- チェックポイント保存: 200kステップごと

---

## 🌐 クラウド移行対応 (2026-01-17)

### AWS/GCP両対応
| 項目 | AWS | GCP |
|:---|:---|:---|
| ストレージ | S3 | GCS |
| コンピュート | EC2 | GCE |
| セットアップ | `aws/setup_ec2.sh` | `gcp/setup_gce.sh` |
| 学習スクリプト | `train_aws_phase7.py` | `train_gcp_phase7.py` |

### 環境変数設定
```bash
# GCP
export CLOUD_PROVIDER=gcs
export GCS_BUCKET=your-bucket-name
export GCP_PROJECT=your-project-id

# AWS
export CLOUD_PROVIDER=s3
export S3_BUCKET=your-bucket-name
export AWS_REGION=ap-northeast-1
```

---

## 🎯 Phase 5 の特徴

### 観測空間（7チャンネル、約720次元）

```python
{
    'my_board': 162,           # 自分のボード
    'my_hand': 270,            # 手札
    'next_opponent_board': 162, # 下家のボード
    'prev_opponent_board': 162, # 上家のボード
    'my_discards': 54,         # 自分の捨て札
    'unseen_probability': 54,  # 確率マップ
    'position_info': 3,        # ポジション（BTN/SB/BB）
}
```

### AIが学習する能力

1. **アウツ・ブロック戦略**
   - 「上家がハートを集めているから、自分はフラッシュに行かない」
   - 「枯れているスートを避ける」

2. **ポジション戦略**
   - 「最後の手番だから、相手の配置を見てから安全策を取る」
   - 後出し有利を活かした判断

3. **リスク管理**
   - 「2人相手にファウルすると×2のダメージ → より慎重に」

---

## 📁 関連ファイル

| ファイル | 説明 |
|:---|:---|
| `src/python/ofc_3max_env.py` | 3人対戦環境（PettingZoo互換） |
| `src/python/train_phase5_3max.py` | Phase 5 学習スクリプト |
| `src/python/ofc_phase1_env.py` | Phase 1/4 環境 |
| `src/cpp/game.hpp` | C++ゲームエンジン（3人対応） |

---

## 📈 学習曲線の傾向

### Phase 4 (ジョーカー対応) - ベストモデル

| ステップ | ファウル率 | Mean Royalty |
|:---|:---:|:---:|
| 0 | 69% | 0.1 |
| 5M | 35% | 0.7 |
| 10.5M | **25.1%** | **0.85** |

### 分析
- ジョーカーは「役完成の保険」として効果的に機能
- 攻撃的なPhase 3よりも安定したプレイスタイルを獲得
- Fantasyland突入率(1.1%)は改善の余地あり

---

## 🔧 次のステップ

1. [x] Phase 5 学習完了（1150万ステップ到達）
2. [x] GCP移行対応（スクリプト・ドキュメント整備）
3. [x] GCPでの並列学習開始（4並列、FPS大幅向上）
4. [x] Phase 7 学習完了（20Mステップ、ファウル率25.8%、Royalty 7.56）
5. [x] 評価基準ドキュメント整備 (`docs/research/evaluation_metrics.md`)
6. [x] Phase 8 Self-Play実装（`train_phase8_selfplay.py`）
7. [ ] Phase 8 学習開始・完了 - **次のタスク**
8. [ ] 最終モデルの評価・対人テスト
9. [ ] UI/デモ作成

---

## 📞 学習監視コマンド

### ローカルテスト（Phase 8）
```bash
cd "/Users/naoai/試作品一覧/OFC NN"
NUM_ENVS=2 python src/python/train_phase8_selfplay.py --test-mode --steps 10000
```

### GCP (GCE)
```bash
# ログ確認
ssh -i ~/.ssh/google_compute_engine naoai@35.243.93.32 "tail -50 ~/ofc-training/training.log"

# プロセス確認
ssh -i ~/.ssh/google_compute_engine naoai@35.243.93.32 "ps aux | grep python"

# Phase 8 学習開始
cd ~/ofc-training && source venv_linux/bin/activate
NUM_ENVS=4 nohup python3 src/python/train_phase8_selfplay.py > training.log 2>&1 &
```

### GCPへのデプロイ
```bash
# スクリプトをGCPに転送
scp "/Users/naoai/試作品一覧/OFC NN/src/python/train_phase8_selfplay.py" naoai@35.243.93.32:~/ofc-training/src/python/
```

---

## 📁 関連ファイル

### Phase 8 関連（新規）
| ファイル | 説明 |
|:---|:---|
| `src/python/train_phase8_selfplay.py` | **Phase 8 Self-Play学習スクリプト** |
| `docs/research/evaluation_metrics.md` | 評価基準ドキュメント |

### インフラ・共通
| ファイル | 説明 |
|:---|:---|
| `gcp/setup_gce.sh` | GCE環境セットアップスクリプト |
| `src/python/gcs_utils.py` | GCS連携モジュール |
| `src/python/cloud_storage.py` | AWS/GCP抽象化層 |
| `src/python/train_gcp_phase7_parallel.py` | Phase 7並列学習スクリプト（完了） |
| `CLAUDE.md` | プロジェクトガイドライン |
