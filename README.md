# OFC Pineapple AI (Enhanced)

Advanced Reinforcement Learning for Open-Face Chinese Poker (Pineapple variant) using C++ Engine and MaskablePPO.

## Overview

This project aims to build a world-class OFC Pineapple AI capable of strategic decision-making and optimal royalty extraction. It leverages a high-performance C++ game engine for simulations and Stable-Baselines3 (MaskablePPO) for deep reinforcement learning.

## Key Features

- **High-Performance C++ Engine**: Core game logic and hand evaluation implemented in C++ for maximum speed (~1,600 FPS during training).
- **MaskablePPO**: Utilizes Action Masking to eliminate illegal moves, significantly accelerating the learning process.
- **Probability Features**: Observation space includes real-time calculations of Flush and Straight completion rates.
- **MCTS Integration**: Monte Carlo Tree Search with Policy-guided rollouts for forward-looking strategic planning.
- **Endgame Solver**: Exhaustive search for the final streets (<= 5 cards) ensuring perfect play in the late game.
- **Self-Play Architecture**: Training through "Latest vs Pool" competitive self-play.
- **Auto-Curriculum**: Automated feedback system that monitors training progress and provides strategic insights via Discord/Slack.

## Tech Stack

- **Core**: C++, Python (pybind11)
- **RL Framework**: Stable-Baselines3 (sb3-contrib)
- **Infrastructure**: AWS EC2 (m7i-flex.large), Docker, Docker Compose
- **Monitoring**: Discord/Slack Webhooks, Tensorboard

## Project Structure

- `src/cpp/`: High-performance game engine and evaluation logic.
- `src/python/`: Training scripts, MCTS implementation, and Gym environments.
- `docs/research/`: Deep dives into the RL strategies and mathematical models.
- `docs/blog/`: Development journey and technical summaries.

## How to Run

1. **Build Engine**:
   ```bash
   python setup.py build_ext --inplace
   ```
2. **Train Locally**:
   ```bash
   python src/python/train_enhanced_phase3.py --steps 1000000
   ```
3. **Deploy to AWS**:
   ```bash
   python src/python/deploy_enhanced.py
   ```

---
*Created by Advanced Agentic Coding Team*
