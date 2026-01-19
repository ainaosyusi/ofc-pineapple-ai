#!/usr/bin/env python3
"""
Phase 8.5 TensorBoard Data Extraction and Graph Generation
"""

import os
import sys
from pathlib import Path

# Add tensorboard to path
try:
    from tensorboard.backend.event_processing import event_accumulator
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    print("Installing required packages...")
    os.system("pip install tensorboard pandas matplotlib -q")
    from tensorboard.backend.event_processing import event_accumulator
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')


def extract_scalars(log_dir):
    """Extract scalar data from TensorBoard log directory"""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    scalars = {}
    for tag in ea.Tags().get('scalars', []):
        events = ea.Scalars(tag)
        scalars[tag] = [(e.step, e.value) for e in events]

    return scalars


def create_graphs(data, output_dir):
    """Create graphs from extracted data"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Define metrics to plot
    metrics_groups = {
        'training_progress': ['rollout/ep_len_mean', 'rollout/ep_rew_mean'],
        'performance': ['train/entropy_loss', 'train/policy_gradient_loss', 'train/value_loss'],
        'learning': ['train/explained_variance', 'train/approx_kl', 'train/clip_fraction'],
    }

    for group_name, metrics in metrics_groups.items():
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            if metric in data:
                steps, values = zip(*data[metric])
                ax.plot(steps, values, 'b-', alpha=0.7)
                ax.set_xlabel('Steps')
                ax.set_ylabel(metric.split('/')[-1])
                ax.set_title(metric)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{group_name}.png', dpi=150)
        plt.close()
        print(f"Saved {group_name}.png")


def main():
    # Find the largest log file
    log_dirs = [
        "gcp_backup/phase85_full_fl/MaskablePPO_10",
        "logs/phase85_full_fl",
    ]

    all_data = {}

    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            print(f"Processing {log_dir}...")
            try:
                data = extract_scalars(log_dir)
                all_data.update(data)
                print(f"  Found {len(data)} metrics")
            except Exception as e:
                print(f"  Error: {e}")

    if all_data:
        print("\nAvailable metrics:")
        for key in sorted(all_data.keys()):
            print(f"  - {key}: {len(all_data[key])} points")

        create_graphs(all_data, "plots/phase85")
        print("\nGraphs saved to plots/phase85/")
    else:
        print("No data found!")


if __name__ == "__main__":
    main()
