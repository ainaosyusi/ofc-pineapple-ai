#!/usr/bin/env python3
"""
Phase 10 FL Stay 学習結果評価
Phase 9 (250M) vs Phase 10 (50M / 100M / 150M) 比較
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src/python'))

import numpy as np
import torch
torch.distributions.Distribution.set_default_validate_args(False)
from sb3_contrib import MaskablePPO
import ofc_engine as ofc
from ofc_3max_env import OFC3MaxEnv


def evaluate_model(model_path, num_games=500, label=""):
    """モデルを評価"""
    print(f"Loading: {model_path}")
    model = MaskablePPO.load(model_path)
    print(f"  Evaluating {num_games} games...")

    env = OFC3MaxEnv(continuous_games=True, fl_solver_mode='greedy')

    stats = {
        'games': 0, 'fouls': 0, 'fl_entries': 0, 'fl_stays': 0,
        'total_score': 0, 'total_royalty': 0, 'wins': 0,
    }

    for game_num in range(num_games):
        env.reset()
        done = False

        while not done:
            current_agent = env.agent_selection
            if env.terminations[current_agent] or env.truncations[current_agent]:
                env.step(None)
                done = all(env.terminations.values()) or all(env.truncations.values())
                continue

            obs = env.observe(current_agent)
            mask = env.action_masks(current_agent)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            env.step(int(action))
            done = all(env.terminations.values()) or all(env.truncations.values())

        player = env.engine.player(0)
        board = player.board
        stats['games'] += 1

        if board.is_foul():
            stats['fouls'] += 1
        else:
            if board.qualifies_for_fl():
                stats['fl_entries'] += 1
                if board.can_stay_fl():
                    stats['fl_stays'] += 1
            stats['total_royalty'] += board.calculate_royalties()

        reward = env._cumulative_rewards.get('player_0', 0)
        stats['total_score'] += reward
        if reward > 0:
            stats['wins'] += 1

        if (game_num + 1) % 100 == 0:
            print(f"    {game_num + 1}/{num_games}")

    return stats


def print_comparison(results):
    """比較表を表示"""
    print("\n" + "=" * 80)
    print("Phase 10 FL Stay Training - Evaluation Results")
    print("=" * 80)

    header = f"{'Metric':<20}"
    for name in results:
        header += f" {name:>14}"
    print(header)
    print("-" * 80)

    def row(label, key, pct=True, lower_better=False):
        line = f"{label:<20}"
        for name, stats in results.items():
            g = stats['games']
            v = stats[key] / g * (100 if pct else 1)
            if pct:
                line += f" {v:>13.1f}%"
            else:
                line += f" {v:>+14.2f}"
        print(line)

    row("Foul Rate", "fouls")
    row("Mean Score", "total_score", pct=False)
    row("Mean Royalty", "total_royalty", pct=False)
    row("FL Entry Rate", "fl_entries")
    row("FL Stay Rate", "fl_stays")
    row("Win Rate", "wins")

    # FL Stay of entries
    line = f"{'FL Stay/Entry':<20}"
    for name, stats in results.items():
        if stats['fl_entries'] > 0:
            v = stats['fl_stays'] / stats['fl_entries'] * 100
            line += f" {v:>13.1f}%"
        else:
            line += f" {'N/A':>14}"
    print(line)


def main():
    num_games = 500

    models = {
        'P9 (250M)': 'models/phase9/p9_fl_mastery_250000000.zip',
        'P10 (50M)': 'models/phase10_gcp/p10_fl_stay_50000000.zip',
        'P10 (100M)': 'models/phase10_gcp/p10_fl_stay_100000000.zip',
        'P10 (150M)': 'models/phase10_gcp/p10_fl_stay_150000000.zip',
    }

    results = {}
    for name, path in models.items():
        if os.path.exists(path):
            stats = evaluate_model(path, num_games, name)
            results[name] = stats
        else:
            print(f"  SKIP (not found): {path}")

    print_comparison(results)


if __name__ == "__main__":
    main()
