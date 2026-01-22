#!/bin/bash
# OFC AI Phase 8.5 - Sync Models from GCP
# Usage: ./sync_models.sh [--all]

INSTANCE_NAME="ofc-training"
ZONE="asia-northeast1-b"
REMOTE_DIR="~/ofc-training"
LOCAL_DIR="models"
SYNC_ALL="${1:-}"

echo "============================================"
echo "OFC AI Phase 8.5 - Model Sync"
echo "============================================"

# Check instance status
STATUS=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

if [ "$STATUS" != "RUNNING" ]; then
    echo "Starting instance for sync..."
    gcloud compute instances start $INSTANCE_NAME --zone=$ZONE
    sleep 30
fi

mkdir -p $LOCAL_DIR

echo "Remote checkpoints:"
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    ls -lh $REMOTE_DIR/models/p85_*.zip 2>/dev/null || echo 'No Phase 8.5 checkpoints'
"

echo ""
if [ "$SYNC_ALL" = "--all" ]; then
    echo "Downloading all Phase 8.5 checkpoints..."
    gcloud compute scp --recurse --zone=$ZONE \
        "$INSTANCE_NAME:$REMOTE_DIR/models/p85_*.zip" \
        $LOCAL_DIR/ 2>/dev/null || echo "No files to download"
else
    echo "Downloading latest checkpoint..."
    LATEST=$(gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
        ls -t $REMOTE_DIR/models/p85_*.zip 2>/dev/null | head -1
    ")
    if [ -n "$LATEST" ]; then
        gcloud compute scp --zone=$ZONE \
            "$INSTANCE_NAME:$LATEST" \
            $LOCAL_DIR/
        echo "Downloaded: $(basename $LATEST)"
    else
        echo "No checkpoints to download"
    fi
fi

echo ""
echo "Local models:"
ls -lh $LOCAL_DIR/p85_*.zip 2>/dev/null || echo "No Phase 8.5 models locally"

echo ""
echo "Done."
