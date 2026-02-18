#!/usr/bin/env python3
"""
Phase 10 学習曲線プロット
training_phase10_gcp_full.log から抽出
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_log(logfile):
    """ログからメトリクスを抽出"""
    steps, foul_rates, fl_entries, fl_stays, rewards = [], [], [], [], []

    current_step = None
    with open(logfile) as f:
        for line in f:
            m = re.search(r'\[Step ([\d,]+)\]', line)
            if m:
                current_step = int(m.group(1).replace(',', ''))
                continue

            if current_step is None:
                continue

            m = re.search(r'Foul Rate: ([\d.]+)%', line)
            if m:
                steps.append(current_step)
                foul_rates.append(float(m.group(1)))

            m = re.search(r'FL Entry Rate: ([\d.]+)%', line)
            if m:
                fl_entries.append(float(m.group(1)))

            m = re.search(r'FL Stay Rate: ([\d.]+)%', line)
            if m:
                fl_stays.append(float(m.group(1)))

            m = re.search(r'Mean Reward: ([+-]?[\d.]+)', line)
            if m:
                rewards.append(float(m.group(1)))
                current_step = None  # reset for next block

    # Align lengths
    min_len = min(len(steps), len(foul_rates), len(fl_entries), len(fl_stays), len(rewards))
    return {
        'steps': np.array(steps[:min_len]),
        'foul_rate': np.array(foul_rates[:min_len]),
        'fl_entry': np.array(fl_entries[:min_len]),
        'fl_stay': np.array(fl_stays[:min_len]),
        'reward': np.array(rewards[:min_len]),
    }


def smooth(data, window=50):
    """移動平均"""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def plot_training_curves(data, output_path):
    """4パネルの学習曲線を描画"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 10: FL Stay Training (150M steps, greedy_fl_solver v3)', fontsize=14, fontweight='bold')

    steps_m = data['steps'] / 1e6  # million steps
    w = 50

    # 1. Foul Rate
    ax = axes[0][0]
    ax.plot(steps_m, data['foul_rate'], alpha=0.15, color='red')
    s = smooth(data['foul_rate'], w)
    ax.plot(steps_m[:len(s)], s, color='red', linewidth=2)
    ax.axhline(y=20, color='gray', linestyle='--', alpha=0.5, label='Target: <20%')
    ax.set_title('Foul Rate')
    ax.set_ylabel('%')
    ax.set_xlabel('Steps (M)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. FL Entry Rate
    ax = axes[0][1]
    ax.plot(steps_m, data['fl_entry'], alpha=0.15, color='blue')
    s = smooth(data['fl_entry'], w)
    ax.plot(steps_m[:len(s)], s, color='blue', linewidth=2)
    ax.axhline(y=27, color='gray', linestyle='--', alpha=0.5, label='Target: >27%')
    ax.set_title('FL Entry Rate')
    ax.set_ylabel('%')
    ax.set_xlabel('Steps (M)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. FL Stay Rate
    ax = axes[1][0]
    ax.plot(steps_m, data['fl_stay'], alpha=0.15, color='green')
    s = smooth(data['fl_stay'], w)
    ax.plot(steps_m[:len(s)], s, color='green', linewidth=2)
    ax.axhline(y=15, color='gray', linestyle='--', alpha=0.5, label='Target: >15%')
    ax.axhline(y=8, color='orange', linestyle=':', alpha=0.5, label='Phase 9 baseline: 8%')
    ax.set_title('FL Stay Rate')
    ax.set_ylabel('%')
    ax.set_xlabel('Steps (M)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Mean Reward
    ax = axes[1][1]
    ax.plot(steps_m, data['reward'], alpha=0.15, color='purple')
    s = smooth(data['reward'], w)
    ax.plot(steps_m[:len(s)], s, color='purple', linewidth=2)
    ax.axhline(y=12.66, color='orange', linestyle=':', alpha=0.5, label='Phase 9 baseline: +12.66')
    ax.set_title('Mean Reward')
    ax.set_ylabel('Score')
    ax.set_xlabel('Steps (M)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    logfile = 'training_phase10_gcp_full.log'
    if not os.path.exists(logfile):
        print(f"Error: {logfile} not found")
        return

    data = parse_log(logfile)
    print(f"Parsed {len(data['steps'])} data points")
    print(f"Step range: {data['steps'][0]:,} - {data['steps'][-1]:,}")

    # 最終メトリクス（最後50ポイントの平均）
    n = min(50, len(data['steps']))
    print(f"\nFinal metrics (last {n} points avg):")
    print(f"  Foul Rate:    {np.mean(data['foul_rate'][-n:]):.1f}%")
    print(f"  FL Entry:     {np.mean(data['fl_entry'][-n:]):.1f}%")
    print(f"  FL Stay:      {np.mean(data['fl_stay'][-n:]):.1f}%")
    print(f"  Mean Reward:  {np.mean(data['reward'][-n:]):+.2f}")

    plot_training_curves(data, 'plots/phase10/phase10_training_curves.png')


if __name__ == "__main__":
    main()
