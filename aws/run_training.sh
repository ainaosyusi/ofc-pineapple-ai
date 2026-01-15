#!/bin/bash
# ===========================================
# OFC Pineapple AI - Training Runner
# ===========================================
# EC2上で学習を実行するスクリプト
# 
# 機能:
#   - 環境変数の読み込み
#   - S3からの既存チェックポイントのダウンロード（再開用）
#   - 学習の実行
#   - 完了/エラー時の通知
#   - オプション: 完了後のインスタンス停止

set -e

APP_DIR="${APP_DIR:-/app/ofc-training}"
cd $APP_DIR

# -------------------------------------------
# Load Environment Variables
# -------------------------------------------
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "[Config] Loaded .env"
fi

# -------------------------------------------
# Configuration
# -------------------------------------------
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
OPPONENT_UPDATE_FREQ="${OPPONENT_UPDATE_FREQ:-50000}"
NOTIFICATION_INTERVAL="${NOTIFICATION_INTERVAL:-50000}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-false}"
UPLOAD_TO_S3="${UPLOAD_TO_S3:-false}"

echo "=============================================="
echo "OFC Pineapple AI - Training"
echo "=============================================="
echo "Timesteps: $TOTAL_TIMESTEPS"
echo "Opponent Update: $OPPONENT_UPDATE_FREQ"
echo "Notification Interval: $NOTIFICATION_INTERVAL"
echo "Auto Shutdown: $AUTO_SHUTDOWN"
echo "Upload to S3: $UPLOAD_TO_S3"
echo ""

# -------------------------------------------
# Download Latest Checkpoint (if resuming)
# -------------------------------------------
if [ "$UPLOAD_TO_S3" = "true" ] && [ -n "$S3_BUCKET" ]; then
    echo "[S3] Checking for existing checkpoints..."
    
    # 最新のチェックポイントをダウンロード
    LATEST=$(aws s3 ls s3://$S3_BUCKET/ofc-training/checkpoints/ 2>/dev/null | sort | tail -1 | awk '{print $4}')
    
    if [ -n "$LATEST" ]; then
        echo "[S3] Found checkpoint: $LATEST"
        aws s3 cp s3://$S3_BUCKET/ofc-training/checkpoints/$LATEST ./models/
        export RESUME_FROM="./models/$LATEST"
        echo "[S3] Downloaded to: $RESUME_FROM"
    else
        echo "[S3] No existing checkpoints found. Starting fresh."
    fi
fi

# -------------------------------------------
# Start Training
# -------------------------------------------
echo ""
echo "[Training] Starting..."

# Python環境のPATH設定
export PYTHONPATH=$APP_DIR:$PYTHONPATH

# 学習実行
python src/python/train_phase1.py \
    --timesteps $TOTAL_TIMESTEPS \
    --lr 0.0003

TRAINING_EXIT_CODE=$?

# -------------------------------------------
# Post-Training Actions
# -------------------------------------------
echo ""
echo "[Training] Completed with exit code: $TRAINING_EXIT_CODE"

# Upload final checkpoint to S3
if [ "$UPLOAD_TO_S3" = "true" ] && [ -n "$S3_BUCKET" ]; then
    echo "[S3] Uploading final checkpoints..."
    aws s3 sync ./models/ s3://$S3_BUCKET/ofc-training/checkpoints/ --exclude "*" --include "*.zip"
    aws s3 sync ./logs/ s3://$S3_BUCKET/ofc-training/logs/ --exclude "*" --include "*.log"
fi

# Auto-shutdown
if [ "$AUTO_SHUTDOWN" = "true" ]; then
    echo "[System] Auto-shutdown enabled. Shutting down in 60 seconds..."
    echo "Cancel with: sudo shutdown -c"
    sudo shutdown -h +1
fi

exit $TRAINING_EXIT_CODE
