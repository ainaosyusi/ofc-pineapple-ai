#!/bin/bash
# Remote training launcher script

cd ~/ofc-training
source venv_linux/bin/activate

# Kill existing training
pkill -f train_phase85 2>/dev/null || true
sleep 2

# Start training with DummyVecEnv (4 environments for stability)
export PYTHONUNBUFFERED=1
export NUM_ENVS=4

echo "Starting Phase 8.5 training (4 envs, 50M steps)..."
nohup python3 -u src/python/train_phase85_full_fl.py --steps 50000000 > training.log 2>&1 &

echo "Training started with PID: $!"
