"""
OFC Pineapple AI - Learning Curve Visualization
TensorBoardログまたはDiscord通知から学習曲線を生成

使用方法:
    python src/python/plot_learning_curves.py --log logs/phase7_parallel/
    python src/python/plot_learning_curves.py --csv training_metrics.csv
    python src/python/plot_learning_curves.py --manual  # 手動データ入力
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 日本語フォント設定
matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio', 'sans-serif']

# フェーズごとの学習データ（手動記録）
PHASE_DATA = {
    'Phase 1': {
        'description': 'ファウル回避学習',
        'steps': [0, 50000, 100000],
        'foul_rate': [90.0, 50.0, 37.8],
        'royalty': [0.0, 0.1, 0.34],
        'color': '#1f77b4',
    },
    'Phase 2': {
        'description': '役作り基礎',
        'steps': [0, 100000, 200000],
        'foul_rate': [40.0, 35.0, 32.0],
        'royalty': [0.1, 0.2, 0.26],
        'color': '#ff7f0e',
    },
    'Phase 3': {
        'description': 'Self-Play (2人)',
        'steps': [0, 500000, 1000000],
        'foul_rate': [35.0, 55.0, 58.0],
        'royalty': [0.3, 0.1, 0.0],
        'color': '#2ca02c',
    },
    'Phase 4 (Joker)': {
        'description': 'ジョーカー対応',
        'steps': [0, 5000000, 10500000],
        'foul_rate': [69.0, 35.0, 25.1],
        'royalty': [0.1, 0.7, 0.85],
        'fl_rate': [0.0, 0.5, 1.1],
        'color': '#d62728',
    },
    'Phase 5 (3-Max)': {
        'description': '3人対戦',
        'steps': [0, 5000000, 11500000],
        'foul_rate': [60.0, 45.0, 38.5],
        'royalty': [0.2, 0.5, 0.78],
        'color': '#9467bd',
    },
    'Phase 7 (Parallel)': {
        'description': '並列学習',
        'steps': [2300000, 2400000, 2500000, 2600000, 2700000, 2800000, 2900000,
                  3000000, 3100000, 3200000, 3300000, 3400000, 3500000],
        'foul_rate': [37.6, 32.8, 34.0, 39.2, 34.6, 33.4, 35.6,
                     32.2, 35.4, 39.6, 32.8, 33.6, 37.8],
        'royalty': [4.39, 5.29, 5.17, 3.73, 4.21, 4.94, 4.66,
                   4.49, 4.18, 4.17, 4.86, 4.95, 4.00],
        'fps': [12382, 6468, 4494, 3513, 2908, 2520, 2240,
               2028, 1865, 1731, 1624, 1536, 1460],
        'color': '#8c564b',
    },
}


def plot_foul_rate_comparison():
    """フェーズ別ファウル率比較"""
    fig, ax = plt.subplots(figsize=(12, 6))

    for phase, data in PHASE_DATA.items():
        ax.plot(data['steps'], data['foul_rate'],
                marker='o', label=f"{phase}: {data['description']}",
                color=data['color'], linewidth=2, markersize=8)

    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel('Foul Rate (%)', fontsize=12)
    ax.set_title('OFC Pineapple AI - Foul Rate by Phase', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    # Phase 4の達成ラインを追加
    ax.axhline(y=25.1, color='red', linestyle='--', alpha=0.5, label='Phase 4 Best (25.1%)')

    plt.tight_layout()
    return fig


def plot_royalty_comparison():
    """フェーズ別ロイヤリティ比較"""
    fig, ax = plt.subplots(figsize=(12, 6))

    for phase, data in PHASE_DATA.items():
        if 'royalty' in data:
            ax.plot(data['steps'], data['royalty'],
                    marker='s', label=f"{phase}",
                    color=data['color'], linewidth=2, markersize=8)

    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel('Average Royalty', fontsize=12)
    ax.set_title('OFC Pineapple AI - Royalty by Phase', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_phase7_detail():
    """Phase 7 並列学習の詳細"""
    data = PHASE_DATA['Phase 7 (Parallel)']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Foul Rate
    ax1 = axes[0, 0]
    ax1.plot(data['steps'], data['foul_rate'], 'b-o', linewidth=2, markersize=10)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Foul Rate (%)')
    ax1.set_title('Foul Rate Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 70)

    # Royalty
    ax2 = axes[0, 1]
    ax2.plot(data['steps'], data['royalty'], 'g-s', linewidth=2, markersize=10)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Mean Royalty')
    ax2.set_title('Mean Royalty Over Time')
    ax2.grid(True, alpha=0.3)

    # FPS
    ax3 = axes[1, 0]
    ax3.bar(range(len(data['fps'])), data['fps'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax3.set_xticks(range(len(data['steps'])))
    ax3.set_xticklabels([f"{s/1e6:.1f}M" for s in data['steps']])
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('FPS')
    ax3.set_title('Training Speed (FPS)')
    ax3.grid(True, alpha=0.3, axis='y')

    # Summary Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    table_data = [
        ['Metric', 'Start', 'Current', 'Change'],
        ['Foul Rate', f"{data['foul_rate'][0]:.1f}%", f"{data['foul_rate'][-1]:.1f}%",
         f"{data['foul_rate'][-1] - data['foul_rate'][0]:+.1f}%"],
        ['Royalty', f"{data['royalty'][0]:.2f}", f"{data['royalty'][-1]:.2f}",
         f"{data['royalty'][-1] - data['royalty'][0]:+.2f}"],
        ['FPS', f"{data['fps'][0]:,}", f"{data['fps'][-1]:,}",
         f"{data['fps'][-1]/data['fps'][0]:.1f}x"],
    ]
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                       colWidths=[0.25, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    # ヘッダー行のスタイル
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax4.set_title('Phase 7 Summary', fontsize=14, fontweight='bold', pad=20)

    plt.suptitle('Phase 7: Parallel Training Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_all_phases_summary():
    """全フェーズの最終成績サマリー"""
    phases = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4\n(Joker)', 'Phase 5\n(3-Max)', 'Phase 7\n(Parallel)']
    foul_rates = [37.8, 32.0, 58.0, 25.1, 38.5, 34.0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(phases, foul_rates, color=colors, edgecolor='black', linewidth=1.5)

    # 最小値をハイライト
    min_idx = foul_rates.index(min(foul_rates))
    bars[min_idx].set_edgecolor('gold')
    bars[min_idx].set_linewidth(4)

    # 値をバーの上に表示
    for i, (bar, rate) in enumerate(zip(bars, foul_rates)):
        height = bar.get_height()
        label = f'{rate}%'
        if i == min_idx:
            label += ' ★'
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Final Foul Rate (%)', fontsize=12)
    ax.set_title('OFC Pineapple AI - Final Foul Rate by Phase', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 70)
    ax.grid(True, alpha=0.3, axis='y')

    # ターゲットラインを追加
    ax.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Target (20%)')
    ax.legend()

    plt.tight_layout()
    return fig


def load_tensorboard_logs(log_dir: str) -> dict:
    """TensorBoardログからデータを読み込み"""
    try:
        from tensorboard.backend.event_processing import event_accumulator

        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()

        data = {}
        for tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            data[tag] = {
                'steps': [e.step for e in events],
                'values': [e.value for e in events],
            }
        return data
    except Exception as e:
        print(f"[Warning] Failed to load TensorBoard logs: {e}")
        return {}


def save_figures(output_dir: str = "plots"):
    """全ての図を保存"""
    os.makedirs(output_dir, exist_ok=True)

    figures = [
        ("foul_rate_comparison", plot_foul_rate_comparison),
        ("royalty_comparison", plot_royalty_comparison),
        ("phase7_detail", plot_phase7_detail),
        ("all_phases_summary", plot_all_phases_summary),
    ]

    for name, plot_func in figures:
        fig = plot_func()
        path = os.path.join(output_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Learning Curve Visualization")
    parser.add_argument("--log", type=str, help="TensorBoard log directory")
    parser.add_argument("--output", type=str, default="plots", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    print("=" * 60)
    print("OFC Pineapple AI - Learning Curve Visualization")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # TensorBoardログがあれば読み込み
    if args.log and os.path.exists(args.log):
        print(f"\nLoading TensorBoard logs from: {args.log}")
        tb_data = load_tensorboard_logs(args.log)
        if tb_data:
            print(f"  Found {len(tb_data)} metrics")

    # 図を保存
    print(f"\nGenerating plots to: {args.output}/")
    save_figures(args.output)

    if args.show:
        plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
