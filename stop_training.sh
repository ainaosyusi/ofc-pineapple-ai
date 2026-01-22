#!/bin/bash
# OFC AI Phase 8.5 - Stop Training and Instance
# Usage: ./stop_training.sh [--keep-running]

INSTANCE_NAME="ofc-training"
ZONE="asia-northeast1-b"
KEEP_RUNNING="${1:-}"

echo "============================================"
echo "OFC AI Phase 8.5 - Stop Training"
echo "============================================"

# Check instance status
STATUS=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

if [ "$STATUS" != "RUNNING" ]; then
    echo "Instance is not running."
    exit 0
fi

# Kill training process
echo "Stopping training process..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    pkill -f 'train_phase85' || echo 'No training process to kill'
    sleep 2
    ps aux | grep -E 'train_phase85|python3.*train' | grep -v grep || echo 'Training stopped'
"

# Stop instance unless --keep-running
if [ "$KEEP_RUNNING" != "--keep-running" ]; then
    echo ""
    echo "Stopping instance..."
    gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE
    echo "Instance stopped. Billing paused."
else
    echo ""
    echo "Instance kept running (--keep-running flag)"
fi

echo ""
echo "Done. To download models: ./sync_models.sh"
