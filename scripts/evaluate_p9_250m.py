#!/usr/bin/env python3
"""
Phase 9 250M モデル評価スクリプト
150M モデルとの比較
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src/python'))

import numpy as np
from sb3_contrib import MaskablePPO
import ofc_engine as ofc
from ofc_3max_env import OFC3MaxEnv

def evaluate_model(model_path, num_games=500):
    """モデルを評価"""
    print(f"Loading: {model_path}")
    model = MaskablePPO.load(model_path)
    print("Model loaded.\n")

    env = OFC3MaxEnv(continuous_games=True, fl_solver_mode='greedy')

    stats = {
        'games': 0,
        'fouls': 0,
        'fl_entries': 0,
        'fl_stays': 0,
        'total_score': 0,
        'total_royalty': 0,
        'wins': 0,
        'high_scores': 0,  # >=15
    }

    print(f"Running {num_games} games...")
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

        # player_0の結果を集計
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

            royalty = board.calculate_royalties()
            stats['total_royalty'] += royalty

        # スコア計算（簡易版）
        reward = env._cumulative_rewards.get('player_0', 0)
        stats['total_score'] += reward
        if reward >= 15:
            stats['high_scores'] += 1
        if reward > 0:
            stats['wins'] += 1

        if (game_num + 1) % 100 == 0:
            print(f"  {game_num + 1}/{num_games}")

    return stats

def print_results(name, stats):
    """結果を表示"""
    games = stats['games']
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"Games: {games}")
    print(f"Foul Rate: {100*stats['fouls']/games:.1f}%")
    print(f"Mean Score: {stats['total_score']/games:+.2f}")
    print(f"Mean Royalty: {stats['total_royalty']/games:.2f}")
    print(f"FL Entry Rate: {100*stats['fl_entries']/games:.1f}%")
    if stats['fl_entries'] > 0:
        print(f"FL Stay Rate: {100*stats['fl_stays']/stats['fl_entries']:.1f}% (of entries)")
    print(f"High Score (≥15) Rate: {100*stats['high_scores']/games:.1f}%")
    print(f"Win Rate: {100*stats['wins']/games:.1f}%")

def main():
    num_games = 500

    # 250M モデル評価
    stats_250m = evaluate_model("models/phase9/p9_fl_mastery_250000000.zip", num_games)
    print_results("Phase 9 (250M steps)", stats_250m)

    # 150M モデル評価（比較用）
    stats_150m = evaluate_model("models/phase9/p9_fl_mastery_150000000.zip", num_games)
    print_results("Phase 9 (150M steps)", stats_150m)

    # 比較表
    print("\n" + "="*60)
    print("COMPARISON: 250M vs 150M")
    print("="*60)
    print(f"{'Metric':<25} {'250M':>12} {'150M':>12} {'Change':>12}")
    print("-"*60)

    def compare(name, key, is_lower_better=False):
        v250 = stats_250m[key] / stats_250m['games'] * (100 if 'rate' in name.lower() or 'foul' in name.lower() else 1)
        v150 = stats_150m[key] / stats_150m['games'] * (100 if 'rate' in name.lower() or 'foul' in name.lower() else 1)
        diff = v250 - v150
        sign = '-' if (diff > 0 and is_lower_better) or (diff < 0 and not is_lower_better) else '+'
        if 'rate' in name.lower() or 'foul' in name.lower():
            print(f"{name:<25} {v250:>11.1f}% {v150:>11.1f}% {sign}{abs(diff):>10.1f}%")
        else:
            print(f"{name:<25} {v250:>+12.2f} {v150:>+12.2f} {sign}{abs(diff):>11.2f}")

    compare("Foul Rate", "fouls", is_lower_better=True)
    compare("FL Entry Rate", "fl_entries")
    compare("High Score Rate", "high_scores")
    compare("Win Rate", "wins")

    # メモ
    print("\n" + "="*60)
    print("Training Log Final Stats (250M)")
    print("="*60)
    print("Foul Rate: 16.8%")
    print("Mean Score: +12.66")
    print("Mean Royalty: 2.88")
    print("FL Entry Rate: 22.8%")
    print("FL Stay Rate: 8.0%")
    print("Win Rate: 75.8%")

if __name__ == "__main__":
    main()
