# OFC Pineapple AI Project - Claude Guidelines

## Project Overview

Open-Face Chinese Poker (Pineapple) AI using Deep Reinforcement Learning.
54-card deck (including 2 Jokers), 3-player (3-Max) self-play training.

## Tech Stack

- **Game Engine**: C++ with pybind11 bindings
- **RL Framework**: Stable-Baselines3, sb3-contrib (MaskablePPO)
- **Environment**: PettingZoo AECEnv (multi-agent)
- **Cloud**: AWS EC2 / GCP GCE (dual support)

## Directory Structure

```
OFC NN/
├── src/
│   ├── cpp/                    # C++ game engine
│   │   ├── game.hpp            # Main game logic
│   │   ├── evaluator.hpp       # Hand evaluation (Joker support)
│   │   └── pybind/             # Python bindings
│   └── python/                 # Python training code
│       ├── ofc_3max_env.py     # 3-player environment
│       ├── train_gcp_phase7_parallel.py  # Parallel training (current)
│       ├── cloud_storage.py    # AWS/GCP abstraction
│       └── notifier.py         # Discord notifications
├── docs/
│   ├── learning/               # Learning documentation
│   └── research/               # Research reports
├── models/                     # Saved model checkpoints
├── gcp/                        # GCP setup scripts
└── aws/                        # AWS setup scripts
```

## Current Training Status

### Phase 7: Parallel Training (SubprocVecEnv)
- **Instance**: GCP n2-standard-4 (4 vCPU, 16GB)
- **Parallel Envs**: 4 (SubprocVecEnv)
- **FPS**: ~4,500-12,000
- **Goal**: 20,000,000 steps

### Key Metrics
- Foul Rate: Target < 25%
- Mean Royalty: Target > 1.0
- Fantasyland Entry: Target > 5%

## Commands

### Check Training Status (GCP)
```bash
# SSH to instance
gcloud compute ssh ofc-training --zone=asia-northeast1-b

# Check log
ssh -i ~/.ssh/google_compute_engine naoai@INSTANCE_IP "tail -50 ~/ofc-training/training.log"

# Check processes
ssh -i ~/.ssh/google_compute_engine naoai@INSTANCE_IP "ps aux | grep python"
```

### Start/Stop Training
```bash
# Start parallel training
cd ~/ofc-training && source venv_linux/bin/activate
NUM_ENVS=4 nohup python3 src/python/train_gcp_phase7_parallel.py > training.log 2>&1 &

# Stop training
pkill -f train_gcp
```

### Cloud Instance Management
```bash
# GCP
gcloud compute instances start ofc-training --zone=asia-northeast1-b
gcloud compute instances stop ofc-training --zone=asia-northeast1-b

# AWS
python3 -c "import boto3; boto3.client('ec2', region_name='ap-northeast-1').stop_instances(InstanceIds=['i-xxx'])"
```

## Training Phases

| Phase | Description | Status | Foul Rate |
|-------|-------------|--------|-----------|
| 1 | Foul avoidance | Done | 37.8% |
| 2 | Hand building | Done | 32.0% |
| 3 | 2P Self-Play | Done | 58-63% |
| 4 | Joker support | Done | **25.1%** |
| 5 | 3P Self-Play | Done | 38.5% |
| 7 | MCTS Distill + Parallel | **Running** | ~34% |

## Important Notes

1. **Venv on Server**: Use `venv_linux` (not `.venv` which is Mac's)
2. **Checkpoint Cleanup**: Auto-cleanup keeps latest 2 + every 1M milestone
3. **Discord Notifications**: Every 100k steps via webhook
4. **Auto-Resume**: Training auto-resumes from latest checkpoint

## Environment Variables

```bash
# Cloud provider selection
CLOUD_PROVIDER=gcs   # or s3
GCS_BUCKET=bucket-name
S3_BUCKET=bucket-name

# Training config
NUM_ENVS=4           # Parallel environments
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

## Cost Estimation (GCP)

| Instance | Price/hour | FPS | Cost for 20M steps |
|----------|-----------|-----|-------------------|
| n2-standard-4 | ~$0.19 | ~5,000 | ~$2 |
| n2-standard-16 | ~$0.76 | ~6,500 | ~$6 |

Parallel training on n2-standard-4 is most cost-effective.
