# GCP クイックスタートガイド

OFC Pineapple AI を GCP で実行するための手順書です。

## 1. 前提条件

- Google Cloud アカウント
- gcloud CLI インストール済み
- 適切な権限を持つサービスアカウント

## 2. GCE インスタンス作成

```bash
# プロジェクト設定
export PROJECT_ID=your-project-id
export ZONE=asia-northeast1-b
export INSTANCE_NAME=ofc-training

gcloud config set project $PROJECT_ID

# インスタンス作成 (CPU版)
gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=n2-standard-4 \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-ssd \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud

# GPU版の場合 (オプション)
gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=n1-standard-4 \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-ssd \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --maintenance-policy=TERMINATE
```

## 3. セットアップ

```bash
# インスタンスにSSH接続
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE

# セットアップスクリプトを実行
# (ローカルからアップロード後)
bash setup_gce.sh
```

## 4. プロジェクトファイルの転送

```bash
# ローカルから
gcloud compute scp --recurse "./OFC NN" $INSTANCE_NAME:/app/ofc-training --zone=$ZONE

# または rsync
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- "mkdir -p /app/ofc-training"
rsync -avz -e "gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --" \
    "./OFC NN/" ":/app/ofc-training/"
```

## 5. Python環境セットアップ

```bash
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE

cd /app/ofc-training
source .venv/bin/activate

# 依存関係インストール
pip install -r requirements.txt

# C++拡張のビルド
python setup.py build_ext --inplace
```

## 6. GCS バケット設定

```bash
# バケット作成
gsutil mb -l asia-northeast1 gs://ofc-training-bucket

# 環境変数設定
cat >> .env << 'EOF'
GCS_BUCKET=ofc-training-bucket
GCP_PROJECT=your-project-id
DISCORD_WEBHOOK_URL=your-webhook-url
EOF
```

## 7. 学習開始

```bash
# 方法1: ヘルパースクリプト
ofc-start

# 方法2: 直接実行
source .venv/bin/activate
nohup python src/python/train_gcp_phase7.py > training.log 2>&1 &
```

## 8. 監視コマンド

```bash
# ステータス確認
ofc-status

# ログ追跡
ofc-logs

# 停止
ofc-stop
```

## 9. インスタンス管理

```bash
# 停止 (課金停止)
gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE

# 再開
gcloud compute instances start $INSTANCE_NAME --zone=$ZONE

# 削除
gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE
```

## 10. コスト最適化

### プリエンプティブル VM (最大80%安価)
```bash
gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=n2-standard-4 \
    --boot-disk-size=50GB \
    --preemptible \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud
```

**注意**: プリエンプティブルVMは24時間で停止、またはリソース不足時に中断されます。
自動レジューム機能により、再起動後に学習を継続できます。

## トラブルシューティング

### GCS認証エラー
```bash
gcloud auth application-default login
```

### ディスク容量不足
```bash
# 古いチェックポイントを確認
ls -la models/p7_mcts_*.zip

# 手動クリーンアップ
python scripts/cleanup_checkpoints.py models/ --keep 5
```

### プロセスが起動しない
```bash
# ログ確認
cat training.log

# Python環境確認
which python
python --version
```
