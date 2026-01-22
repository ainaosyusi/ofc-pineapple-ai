#!/bin/bash
# OFC AI Phase 8.5 - Training Monitor
# Usage: ./check_training.sh [lines]

INSTANCE_NAME="ofc-training"
ZONE="asia-northeast1-b"
REMOTE_DIR="~/ofc-training"
LINES="${1:-50}"

echo "============================================"
echo "OFC AI Phase 8.5 - Training Status"
echo "============================================"

# Check instance status
STATUS=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format="value(status)" 2>/dev/null || echo "NOT_FOUND")
echo "Instance: $INSTANCE_NAME ($STATUS)"
echo ""

if [ "$STATUS" != "RUNNING" ]; then
    echo "Instance is not running. Start with: ./start_training.sh"
    exit 0
fi

# Check if training process is running
echo "Process status:"
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    ps aux | grep -E 'train_phase85|python3.*train' | grep -v grep || echo 'No training process found'
"

echo ""
echo "Latest log ($LINES lines):"
echo "--------------------------------------------"
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    tail -$LINES $REMOTE_DIR/training.log 2>/dev/null || echo 'No log file found'
"

echo ""
echo "--------------------------------------------"
echo "Checkpoints:"
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    ls -lh $REMOTE_DIR/models/p85_*.zip 2>/dev/null | tail -5 || echo 'No Phase 8.5 checkpoints yet'
"
