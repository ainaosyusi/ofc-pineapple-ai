#!/bin/bash
# OFC AI Phase 8.5 - GCP Training Launcher
# Usage: ./start_training.sh [steps] [envs]
#
# Examples:
#   ./start_training.sh              # Default: 50M steps, 4 envs
#   ./start_training.sh 20000000     # 20M steps
#   ./start_training.sh 50000000 8   # 50M steps, 8 envs

set -e

# Configuration
INSTANCE_NAME="ofc-training"
ZONE="asia-northeast1-b"
REMOTE_DIR="~/ofc-training"
STEPS="${1:-50000000}"
ENVS="${2:-4}"

echo "============================================"
echo "OFC AI Phase 8.5 - GCP Training Launcher"
echo "============================================"
echo "Instance: $INSTANCE_NAME"
echo "Zone: $ZONE"
echo "Steps: $STEPS"
echo "Envs: $ENVS"
echo ""

# 1. Start instance if not running
echo "[1/5] Checking instance status..."
STATUS=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

if [ "$STATUS" = "NOT_FOUND" ]; then
    echo "ERROR: Instance $INSTANCE_NAME not found in $ZONE"
    exit 1
elif [ "$STATUS" != "RUNNING" ]; then
    echo "Starting instance..."
    gcloud compute instances start $INSTANCE_NAME --zone=$ZONE
    echo "Waiting for instance to boot..."
    sleep 30
else
    echo "Instance already running"
fi

# 2. Sync source code
echo ""
echo "[2/5] Syncing source code..."
gcloud compute scp --recurse --zone=$ZONE \
    src setup.py CLAUDE.md NEXT_ACTIONS.md \
    $INSTANCE_NAME:$REMOTE_DIR/

# 3. Sync models (if any local Phase 8.5 checkpoints exist)
echo ""
echo "[3/5] Syncing models..."
if ls models/p85_*.zip 1>/dev/null 2>&1; then
    gcloud compute scp --recurse --zone=$ZONE \
        models/p85_*.zip \
        $INSTANCE_NAME:$REMOTE_DIR/models/
    echo "Phase 8.5 checkpoints synced"
elif ls models/p8_*.zip 1>/dev/null 2>&1; then
    gcloud compute scp --recurse --zone=$ZONE \
        models/p8_*.zip \
        $INSTANCE_NAME:$REMOTE_DIR/models/
    echo "Phase 8 checkpoints synced (as base)"
else
    echo "No local checkpoints to sync"
fi

# 4. Rebuild C++ extension
echo ""
echo "[4/5] Rebuilding C++ extension on remote..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    cd $REMOTE_DIR && \
    source venv_linux/bin/activate && \
    pip install -e . --force-reinstall --no-deps
"

# 5. Start training
echo ""
echo "[5/5] Starting training..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    cd $REMOTE_DIR && \
    source venv_linux/bin/activate && \
    NUM_ENVS=$ENVS nohup python3 src/python/train_phase85_full_fl.py --steps $STEPS > training.log 2>&1 &
    echo 'Training started in background'
    sleep 2
    tail -5 training.log
"

echo ""
echo "============================================"
echo "Training started successfully!"
echo ""
echo "Useful commands:"
echo "  ./check_training.sh     # Check progress"
echo "  ./stop_training.sh      # Stop instance"
echo "  ./sync_models.sh        # Download models"
echo "============================================"
