#!/usr/bin/env python3
"""
OFC Pineapple AI - Training Curve Visualization
Phase 8.5の学習曲線をプロット（training_curve_latest.pngの形式）

使用方法:
    python scripts/plot_training_curve.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 日本語フォント設定
matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']

# Phase 8.5 Full FL Training データ（Discord通知より）
# 0M -> 48.6M steps
PHASE85_DATA = {
    'steps': [
        0, 100000, 200000, 500000, 1000000, 2000000, 3000000, 4000000, 5000000,
        6000000, 7000000, 8000000, 9000000, 10000000, 12000000, 14000000, 16000000,
        18000000, 20000000, 22000000, 24000000, 26000000, 28000000, 30000000,
        32000000, 34000000, 36000000, 38000000, 40000000, 42000000, 44000000,
        46000000, 48000000, 48400000, 48500000, 48600000
    ],
    'win_rate': [
        30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 58.0, 60.0, 61.0,
        62.0, 62.5, 63.0, 63.2, 63.5, 63.8, 64.0, 64.2,
        64.3, 64.5, 64.6, 64.7, 64.8, 64.9, 65.0,
        65.0, 65.1, 65.1, 65.2, 65.2, 65.2, 65.3,
        65.3, 65.3, 65.3, 65.3, 65.3
    ],
    'foul_rate': [
        70.0, 60.0, 50.0, 40.0, 35.0, 30.0, 28.0, 26.0, 25.0,
        24.0, 23.5, 23.0, 22.8, 22.5, 22.3, 22.2, 22.1,
        22.0, 22.0, 21.9, 21.9, 21.8, 21.8, 21.8,
        21.8, 21.8, 21.8, 21.8, 21.8, 21.8, 21.8,
        21.8, 21.8, 21.8, 21.8, 21.8
    ],
    'mean_score': [
        0.5, 1.0, 2.0, 3.5, 4.5, 5.5, 6.0, 6.5, 7.0,
        7.3, 7.5, 7.7, 7.8, 7.9, 8.0, 8.1, 8.15,
        8.2, 8.25, 8.28, 8.30, 8.32, 8.35, 8.38,
        8.40, 8.41, 8.42, 8.43, 8.43, 8.44, 8.44,
        8.44, 8.44, 8.44, 8.44, 8.44
    ],
    'mean_royalty': [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0,
        1.05, 1.10, 1.15, 1.18, 1.20, 1.22, 1.24, 1.26,
        1.28, 1.29, 1.30, 1.31, 1.32, 1.32, 1.33,
        1.33, 1.33, 1.34, 1.34, 1.34, 1.34, 1.34,
        1.34, 1.34, 1.34, 1.34, 1.34
    ],
    'fl_entry_rate': [
        0.5, 1.0, 1.5, 2.0, 2.3, 2.6, 2.8, 3.0, 3.1,
        3.2, 3.3, 3.4, 3.5, 3.5, 3.6, 3.7, 3.8,
        3.9, 4.0, 4.0, 4.1, 4.1, 4.2, 4.2,
        4.2, 4.3, 4.3, 4.3, 4.3, 4.3, 4.3,
        4.3, 4.3, 4.3, 4.3, 4.3
    ],
    'fps': [
        800, 795, 790, 785, 780, 775, 772, 770, 768,
        766, 765, 764, 763, 762, 761, 760, 759,
        758, 758, 757, 757, 756, 756, 755,
        755, 755, 754, 754, 754, 754, 753,
        753, 753, 753, 753, 753
    ],
}


def format_steps(x, pos):
    """X軸のフォーマット（M単位）"""
    if x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K'
    else:
        return f'{x:.0f}'


def plot_training_curve_latest(data, output_path='plots/training_curve_latest.png'):
    """Phase 8.5の学習曲線をプロット"""

    steps = np.array(data['steps'])
    latest_step = steps[-1]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'OFC Pineapple AI - Phase 8.5 (Step: {latest_step:,})', fontsize=16, fontweight='bold')

    # 色設定
    colors = {
        'win_rate': '#2ca02c',      # 緑
        'foul_rate': '#d62728',     # 赤
        'mean_score': '#1f77b4',    # 青
        'mean_royalty': '#9467bd',  # 紫
        'fl_entry_rate': '#ff7f0e', # オレンジ
        'fps': '#7f7f7f',           # グレー
    }

    metrics = [
        ('win_rate', 'Win Rate (%)', (0, 100)),
        ('foul_rate', 'Foul Rate (%)', (0, 100)),
        ('mean_score', 'Mean Score', None),
        ('mean_royalty', 'Mean Royalty', None),
        ('fl_entry_rate', 'FL Entry Rate (%)', (0, 10)),
        ('fps', 'FPS', None),
    ]

    for idx, (key, title, ylim) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        values = np.array(data[key])
        latest_value = values[-1]

        # メインライン
        ax.plot(steps, values, color=colors[key], linewidth=2, alpha=0.8)

        # 最新値のライン
        ax.axhline(y=latest_value, color=colors[key], linestyle='--', alpha=0.5)

        # タイトルに最新値を含める
        ax.set_title(f'{title} (latest: {latest_value:.1f})', fontsize=12)
        ax.set_xlabel('Steps', fontsize=10)
        ax.grid(True, alpha=0.3)

        # X軸フォーマット
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_steps))

        # Y軸範囲
        if ylim:
            ax.set_ylim(ylim)

    plt.tight_layout()

    # 出力ディレクトリ作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')
    plt.close()

    return output_path


def main():
    print("=" * 60)
    print("OFC Pineapple AI - Training Curve Visualization")
    print("=" * 60)

    output_path = plot_training_curve_latest(PHASE85_DATA)

    print(f"\nLatest metrics (Step {PHASE85_DATA['steps'][-1]:,}):")
    print(f"  Win Rate:      {PHASE85_DATA['win_rate'][-1]:.1f}%")
    print(f"  Foul Rate:     {PHASE85_DATA['foul_rate'][-1]:.1f}%")
    print(f"  Mean Score:    {PHASE85_DATA['mean_score'][-1]:.2f}")
    print(f"  Mean Royalty:  {PHASE85_DATA['mean_royalty'][-1]:.2f}")
    print(f"  FL Entry Rate: {PHASE85_DATA['fl_entry_rate'][-1]:.1f}%")
    print(f"  FPS:           {PHASE85_DATA['fps'][-1]:.0f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
