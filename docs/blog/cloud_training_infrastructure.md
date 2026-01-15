# 強化学習をAWSでスケールさせる：OFC AI クラウド学習インフラ構築記

## はじめに

強化学習の実験には膨大な計算リソースが必要です。ローカルPCでは限界があり、クラウドへのスケールが必須になります。

本記事では、ポーカーAI「OFC Pineapple AI」のAWS EC2での学習環境構築について、具体的な設定とトラブルシューティングを解説します。

---

## なぜクラウドが必要か

### 学習時間の現実

| ステップ数 | ローカル (M1 Mac) | EC2 (m7i-flex.large) |
|-----------|------------------|----------------------|
| 500K | 9分 | 7分 |
| 5M | 90分 | 70分 |
| 10M | 3時間 | 2.3時間 |

一見差は小さいですが、**24時間連続稼働**と**複数実験の並列実行**がクラウドの真価です。

---

## システム構成

```
┌─────────────────────────────────────┐
│           AWS Infrastructure         │
├─────────────────────────────────────┤
│                                      │
│  EC2 (m7i-flex.large)               │
│  ├── Docker Container               │
│  │   ├── Python 3.9                 │
│  │   ├── PyTorch                    │
│  │   ├── Stable-Baselines3          │
│  │   └── OFC Engine (C++)           │
│  │                                  │
│  └── Volumes                        │
│      ├── /models (永続化)           │
│      └── /logs                      │
│                                      │
├─────────────────────────────────────┤
│  通知                               │
│  └── Discord Webhook                │
└─────────────────────────────────────┘
```

---

## Docker化のポイント

### Dockerfile（マルチステージビルド）

```dockerfile
# ビルドステージ
FROM python:3.9-slim AS builder

RUN apt-get update && apt-get install -y \
    build-essential cmake g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY setup.py .
COPY src/cpp/ src/cpp/
RUN python setup.py build_ext --inplace

# 実行ステージ
FROM python:3.9-slim

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /app/*.so /app/
COPY --from=builder /app/build /app/build
COPY src/ src/
```

**ポイント：**
- マルチステージビルドで最終イメージを小さく
- build-essential等はビルド時のみ必要
- `.so` ファイル（C++コンパイル済み）をコピー

### docker-compose.yml

```yaml
version: '3.8'

services:
  phase3:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ofc-training-phase3
    
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    
    environment:
      - TOTAL_TIMESTEPS=10000000
      - DISCORD_WEBHOOK_URL=${DISCORD_WEBHOOK_URL}
    
    command: [
      "python", "src/python/train_aws_phase3.py",
      "--steps", "10000000",
      "--batch-size", "128"
    ]
    
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 8G
    
    restart: unless-stopped
```

**ポイント：**
- `volumes`: モデルの永続化
- `environment`: 環境変数で設定注入
- `deploy.resources.limits`: リソース制限
- `restart: unless-stopped`: 自動再起動

---

## EC2セットアップ手順

### 1. インスタンス起動

```bash
# SSH接続
ssh -i ofc-training-key.pem ubuntu@<EC2_IP>

# Docker Compose インストール
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo usermod -aG docker ubuntu
```

### 2. コード転送

```bash
# ローカルから rsync
rsync -avz -e "ssh -i ofc-training-key.pem" \
  --exclude 'models/*.zip' \
  --exclude '__pycache__' \
  ./ ubuntu@<EC2_IP>:/home/ubuntu/OFC-NN/
```

### 3. 学習開始

```bash
cd /home/ubuntu/OFC-NN
sudo docker-compose up -d phase3
```

### 4. ログ監視

```bash
# リアルタイムログ
sudo docker logs -f ofc-training-phase3

# 定期確認（Foul Rate）
sudo docker logs ofc-training-phase3 | grep "Foul Rate" | tail -n 5
```

---

## Discord通知の設定

### Webhook URL取得

1. Discordサーバーでチャンネル設定を開く
2. 「連携サービス」→「ウェブフック」
3. 新しいウェブフックを作成しURLをコピー

### Python実装

```python
class TrainingNotifier:
    def __init__(self, discord_webhook=None):
        self.discord_webhook = discord_webhook or os.getenv("DISCORD_WEBHOOK_URL")
    
    def send_progress(self, step, total_steps, metrics):
        message = f"📊 Progress: {step/total_steps*100:.1f}%\n"
        message += f"Foul Rate: {metrics['foul_rate']:.1f}%\n"
        message += f"Win Rate: {metrics['win_rate']:.1f}%"
        
        payload = {
            "embeds": [{
                "description": message,
                "color": 0x0099ff
            }]
        }
        requests.post(self.discord_webhook, json=payload)
```

### 通知タイミング

| イベント | タイミング |
|----------|-----------|
| 🚀 学習開始 | 起動時 |
| 📊 進捗報告 | 10万ステップ毎 |
| 💾 チェックポイント | 20万ステップ毎 |
| ✅ 学習完了 | 終了時 |
| ❌ エラー | 例外発生時 |

---

## トラブルシューティング

### 問題1: ディスク容量不足

```bash
# 症状
Error: no space left on device

# 解決
docker system prune -a  # 未使用イメージ削除
df -h  # 容量確認
```

### 問題2: CPU制限エラー

```bash
# 症状
range of CPUs is from 0.01 to 2.00, as there are only 2 CPUs available

# 解決：docker-compose.yml で制限を調整
deploy:
  resources:
    limits:
      cpus: '2'  # インスタンスのCPU数以下に
```

### 問題3: Pythonバージョン不一致

```bash
# 症状
ModuleNotFoundError: No module named 'xxx'

# 解決
# Dockerfile でバージョンを明示
FROM python:3.9-slim  # ローカルと合わせる
```

---

## コスト最適化

### インスタンス選定

| インスタンス | vCPU | メモリ | 時間単価 | 用途 |
|-------------|------|--------|---------|------|
| t3.micro | 2 | 1GB | $0.01 | テスト |
| m7i-flex.large | 2 | 8GB | $0.05 | Phase 1-2 |
| c6a.xlarge | 4 | 8GB | $0.08 | Phase 3 |

**ポイント：**
- GPUは不要（PPOはCPU効率が良い）
- メモリは8GB推奨（大きなバッチサイズ用）
- Spotインスタンスで50%コスト削減可能

### 自動停止設定

```bash
# 学習終了後に自動停止
aws ec2 stop-instances --instance-ids i-xxxx

# または Dockerコンテナ終了時にシャットダウン
# train_script.py の最後に:
import subprocess
subprocess.run(["sudo", "shutdown", "-h", "now"])
```

---

## まとめ

AWS + Dockerで強化学習をスケールさせるポイント：

1. **Docker化で再現性確保**：どの環境でも同じ結果
2. **Volume永続化**：モデルを失わない
3. **Webhook通知**：離れていても進捗把握
4. **リソース制限**：コスト管理

次回は、学習したモデルの評価方法について解説します。

---

*技術的な質問はコメント欄またはGitHub Issuesまで！*
