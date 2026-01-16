#!/bin/bash
# Phase 4 Joker Training - Auto Continue Script
# 現在の学習完了後、追加で2500万ステップの学習を自動開始

set -e

cd /home/ubuntu/OFC-NN
source ./.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/OFC-NN:/home/ubuntu/OFC-NN/src/python

echo "======================================================"
echo "🃏 Phase 4 Joker Training - Extended Run (25M+ steps)"
echo "======================================================"
echo "Start time: $(date)"
echo ""

# 追加学習: 2500万ステップ
STEPS=25000000

echo "Starting extended training: $STEPS steps"
echo ""

./.venv/bin/python3 src/python/train_aws_phase4_joker.py \
    --steps $STEPS \
    --save-freq 500000 \
    --notify-freq 500000

echo ""
echo "======================================================"
echo "✅ Extended training complete!"
echo "End time: $(date)"
echo "======================================================"
