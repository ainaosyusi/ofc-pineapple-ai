# Phase 8.5 Full FL Training Report

**Date**: 2026-01-19
**Status**: Interrupted (48.6M / 50M steps)

---

## Overview

Phase 8.5 introduced **Ultimate Rules** Fantasy Land training with the following features:
- **Continuous Games**: FL status preserved across games
- **Button Rotation**: Position changes each game
- **Ultimate Rules FL Cards**: QQ=14, KK=15, AA=16, Trips=17 cards

---

## Training Configuration

| Parameter | Value |
|:----------|:------|
| Environment | OFC Pineapple 3-Max |
| Algorithm | MaskablePPO |
| Parallel Envs | 6 |
| n_steps | 2048 |
| batch_size | 256 |
| learning_rate | 3e-4 |
| Total Target | 50,000,000 steps |

---

## Training Progress

### Final Status (Before Interruption)

| Metric | Value |
|:-------|:------|
| **Progress** | 97.0% (48,600,972 / 50,000,000) |
| **Games Played** | ~9,700,000 |
| **Foul Rate** | 21.8% |
| **Mean Score** | +8.44 |
| **Mean Royalty** | 1.34 |
| **Win Rate** | 65.3% |
| **FPS** | ~753 |

### Checkpoints Saved

| Checkpoint | Steps | File Size |
|:-----------|:------|:----------|
| p85_full_fl_48400968.zip | 48.4M | 1.69 MB |
| p85_full_fl_48600972.zip | 48.6M | 1.69 MB |

---

## Comparison with Previous Phases

| Metric | Phase 8 (Self-Play) | Phase 8.5 (Full FL) | Change |
|:-------|:-------------------:|:-------------------:|:------:|
| Foul Rate | 20.8% | 21.8% | +1.0% |
| Mean Score | +7.87 | +8.44 | +0.57 |
| FL Entry | 3.2% | N/A | - |

---

## Training Issues Encountered

### 1. Dead Agent Error

When resuming from checkpoints, the environment occasionally raised:
```
ValueError: when an agent is dead, the only valid action is None
```

**Root Cause**: The `ParallelOFCEnv.step()` method was sending actions to terminated agents.

**Fix Applied**:
```python
def step(self, action):
    if self.env.terminations.get(self.learning_agent, False):
        self.env.step(None)
    else:
        self.env.step(action)
    self._play_opponents()
```

### 2. Infinite Loop in _play_opponents

The `_play_opponents()` method could enter an infinite loop when all agents were in specific states.

**Fix Applied**: Added max_iterations guard:
```python
def _play_opponents(self):
    max_iterations = 50
    for _ in range(max_iterations):
        if all(self.env.terminations.values()):
            break
        # ... rest of logic
```

### 3. Training Stall After Bug Fixes

After applying bug fixes, training would start but progress updates stopped appearing. Investigation showed workers were running but the main process was waiting for batch collection.

---

## TensorBoard Metrics Analysis

Based on 3,965 data points extracted from MaskablePPO_10:

### Available Metrics
- `time/fps`: Training speed
- `train/approx_kl`: KL divergence
- `train/clip_fraction`: PPO clip fraction
- `train/entropy_loss`: Policy entropy
- `train/explained_variance`: Value function accuracy
- `train/loss`: Combined loss
- `train/policy_gradient_loss`: Policy loss
- `train/value_loss`: Value function loss

### Observations

1. **Explained Variance**: Remained low (0.15-0.22), indicating difficulty in value prediction due to high game variance
2. **Entropy**: Gradually decreased, showing policy convergence
3. **FPS**: Averaged ~750-800 during stable training

---

## Files Preserved

### Models
- `gcp_backup/p85_full_fl_48400968.zip`
- `gcp_backup/p85_full_fl_48600972.zip`

### Logs
- `gcp_backup/phase85_full_fl/` (TensorBoard logs)
- `gcp_backup/training*.log`

### Graphs
- `plots/phase85/training_progress.png`
- `plots/phase85/performance.png`
- `plots/phase85/learning.png`

---

## Lessons Learned

1. **Checkpoint Resume Issues**: PettingZoo environments require careful handling of terminated agents when resuming from checkpoints
2. **SubprocVecEnv Complexity**: Multiple worker processes can mask issues that only appear at specific game states
3. **Testing Before Production**: Changes should be tested locally before deployment to GCP

---

## Next Steps (Recommended)

1. Fix the dead agent handling properly
2. Add comprehensive unit tests for edge cases
3. Implement DummyVecEnv fallback for debugging
4. Resume training from 48.6M checkpoint once issues are resolved

---

*Report generated: 2026-01-19*
