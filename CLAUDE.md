# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session Checklist

Before starting work, check:
1. **`NEXT_ACTIONS.md`** - Pending tasks and next steps
2. **`docs/learning/04_current_status.md`** - Current training status

## Project Overview

Open-Face Chinese Poker (Pineapple) 3-Max AI using Deep Reinforcement Learning.
- 54-card deck (52 standard + 2 Jokers)
- 3-player simultaneous play
- Fantasy Land with Ultimate Rules (14-17 cards based on top hand)

## Build Commands

```bash
# Build C++ extension (required after C++ changes)
python setup.py build_ext --inplace

# Verify build
python -c "import ofc_engine as ofc; print('Engine loaded')"

# C++ unit tests
make test

# Python tests
python tests/test_joker.py
python tests/test_fl_solver.py
```

## Training Commands

```bash
# Local test (short run)
NUM_ENVS=2 python src/python/train_phase85_selfplay.py --test-mode --steps 10000

# GCP training
gcloud compute instances start ofc-training --zone=asia-northeast1-b
gcloud compute scp --recurse --zone=asia-northeast1-b src models setup.py ofc-training:~/ofc-training/
gcloud compute ssh ofc-training --zone=asia-northeast1-b --command="cd ~/ofc-training && pip install -e . --force-reinstall --no-deps"
gcloud compute ssh ofc-training --zone=asia-northeast1-b --command="cd ~/ofc-training && NUM_ENVS=4 nohup python3 src/python/train_phase85_selfplay.py --steps 50000000 > training.log 2>&1 &"

# Check training log
gcloud compute ssh ofc-training --zone=asia-northeast1-b --command="tail -50 ~/ofc-training/training.log"

# Stop instance
gcloud compute instances stop ofc-training --zone=asia-northeast1-b
```

## Architecture

### Layer 1: C++ Game Engine (`src/cpp/`)

Header-only implementation with key files:
- `game.hpp` - Game state machine, player management, FL handling
- `board.hpp` - Top(3)/Middle(5)/Bottom(5) slot management
- `evaluator.hpp` - Hand ranking with Joker support
- `solver.hpp` - Fantasy Land optimal placement solver
- `pybind/bindings.cpp` - Python bindings via pybind11

Card representation uses 64-bit bitboards for O(1) hand evaluation.

### Layer 2: Python Environment (`src/python/`)

- `ofc_3max_env.py` - PettingZoo AECEnv for 3-player multi-agent training
- `ofc_env.py` - Gymnasium single-player environment

Key features:
- Action masking via MaskablePPO (filters invalid placements)
- Continuous games with button rotation
- FL state inheritance between games

### Layer 3: Training Scripts (`src/python/train_*.py`)

Current: `train_phase85_selfplay.py` - Self-play with Ultimate Rules FL
- Uses Stable-Baselines3 MaskablePPO
- 4 parallel environments (SubprocVecEnv)
- Auto-checkpointing every 100k steps
- Discord notifications via webhook

## Key Concepts

### Ultimate Rules (Fantasy Land)

FL card distribution varies by top hand strength:
| Top Hand | Cards | Difficulty |
|----------|-------|------------|
| QQ | 14 | Standard |
| KK | 15 | Easier |
| AA | 16 | Easy |
| Trips | 17 | Easiest |

### Game Phases

```
PHASE_INITIAL_DEAL → PHASE_TURN (×4) → PHASE_SHOWDOWN
(5 cards)            (3 cards each)    (scoring)
```

### Observation Space (881 features)

Board state for all players, current hand, opponent visible cards, FL status indicators.

## Environment Variables

```bash
NUM_ENVS=4                    # Parallel training environments
CLOUD_PROVIDER=gcs            # or s3
GCS_BUCKET=bucket-name
DISCORD_WEBHOOK_URL=...       # Notifications every 100k steps
```

## Server Notes

- Use `venv_linux` on GCP (not `.venv` which is Mac)
- Checkpoint auto-cleanup keeps latest 2 + every 1M milestone
- Training auto-resumes from latest checkpoint

## Current Performance (Phase 8)

| Metric | Value |
|--------|-------|
| Foul Rate | 20.8% |
| FL Entry | 3.2% |
| FPS | 4,500-12,000 |
