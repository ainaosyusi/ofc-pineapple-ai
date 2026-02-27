#!/usr/bin/env python3
"""
OFC Pineapple AI - V2 Benchmark Evaluation Script
Phase 0-5: 固定ベンチマーク相手との対局評価

使い方:
    # ルールベース相手の評価（モデルなしでも動作確認可能）
    python v2/evaluate_benchmark.py --model models/v2_configA/v2_a_1000000.zip --games 1000

    # 複数モデル比較
    python v2/evaluate_benchmark.py \
        --model models/v2_configA/v2_a_1000000.zip \
        --model models/v2_configB/v2_b_1000000.zip \
        --games 1000

    # ルールベース同士のベースライン計測
    python v2/evaluate_benchmark.py --baseline-only --games 500
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# プロジェクトルートをパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src', 'python'))

import ofc_engine as ofc
from ofc_3max_env import OFC3MaxEnv
from v2.rule_based_agent import SafeAgent, AggressiveAgent, RandomAgent, BaseRuleAgent


class ModelAgent:
    """学習済みモデルを使うエージェント"""

    def __init__(self, model_path: str):
        from sb3_contrib import MaskablePPO
        self.model = MaskablePPO.load(model_path)
        self.name = os.path.basename(model_path).replace('.zip', '')

    def select_action(self, env, agent_name: str) -> int:
        obs = env.observe(agent_name)
        mask = env.action_masks(agent_name)
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)


class BenchmarkEvaluator:
    """ベンチマーク評価の実行と集計"""

    def __init__(self, num_games: int = 1000, fl_solver_mode: str = 'greedy'):
        self.num_games = num_games
        self.fl_solver_mode = fl_solver_mode

    def evaluate_vs_opponent(
        self,
        model_agent,
        opponent_agent,
        opponent_name: str,
    ) -> Dict:
        """モデル vs 固定相手を評価。3人対戦で model=P0, opponent=P1,P2"""
        env = OFC3MaxEnv(
            enable_fl_turns=True,
            continuous_games=False,
            fl_solver_mode=self.fl_solver_mode,
            extended_fl_obs=True,
        )

        stats = {
            'games': 0,
            'scores': [],
            'fouls': 0,
            'fl_entries': 0,
            'fl_stays': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'royalties': [],
            'position_scores': {0: [], 1: [], 2: []},  # BTN, SB, BB
            'position_fouls': {0: 0, 1: 0, 2: 0},
            'position_games': {0: 0, 1: 0, 2: 0},
            'opp_fouls': 0,
            'opp_games': 0,
        }

        errors = 0
        start_time = time.time()

        # 各ポジションで均等にプレイ（3ポジション × N/3ゲーム）
        games_per_position = self.num_games // 3
        remaining = self.num_games - games_per_position * 3

        for pos in range(3):
            n_games = games_per_position + (1 if pos < remaining else 0)
            for _ in range(n_games):
                try:
                    result = self._play_one_game(
                        env, model_agent, opponent_agent, model_position=pos
                    )
                    if result is None:
                        errors += 1
                        continue

                    score, fouled, entered_fl, stayed_fl, royalty, opp_fouled = result

                    stats['games'] += 1
                    stats['scores'].append(score)
                    stats['royalties'].append(royalty)
                    stats['position_scores'][pos].append(score)
                    stats['position_games'][pos] += 1

                    if fouled:
                        stats['fouls'] += 1
                        stats['position_fouls'][pos] += 1
                    if entered_fl:
                        stats['fl_entries'] += 1
                    if stayed_fl:
                        stats['fl_stays'] += 1

                    if score > 0:
                        stats['wins'] += 1
                    elif score < 0:
                        stats['losses'] += 1
                    else:
                        stats['draws'] += 1

                    if opp_fouled:
                        stats['opp_fouls'] += 1
                    stats['opp_games'] += 1

                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"  [Error] Game error: {e}")

        elapsed = time.time() - start_time
        stats['elapsed_sec'] = elapsed
        stats['errors'] = errors
        stats['opponent'] = opponent_name

        return self._compute_summary(stats)

    def _play_one_game(
        self,
        env,
        model_agent,
        opponent_agent,
        model_position: int,
    ) -> Optional[Tuple]:
        """1ゲーム実行。model_position でモデルの座席を指定"""
        env.reset()

        # エージェントの配置（model_position の席にモデル、他は opponent）
        agents = {}
        for i, name in enumerate(env.possible_agents):
            if i == model_position:
                agents[name] = model_agent
            else:
                agents[name] = opponent_agent

        max_steps = 500  # 無限ループ防止
        step = 0

        while not all(env.terminations.values()):
            step += 1
            if step > max_steps:
                return None

            agent_name = env.agent_selection
            if env.terminations.get(agent_name, False):
                env.step(None)
                continue

            pidx = env.agent_name_mapping[agent_name]
            ps = env.engine.player(pidx)
            if ps.board.total_placed() >= 13 or ps.in_fantasy_land:
                env.step(None)
                continue

            agent = agents[agent_name]
            action = agent.select_action(env, agent_name)
            env.step(action)

        # 結果集計
        result = env.engine.result()
        model_name = env.possible_agents[model_position]
        model_idx = model_position

        score = result.get_score(model_idx)
        fouled = result.is_fouled(model_idx)
        royalty = result.get_royalty(model_idx)

        # FL entry/stay
        entered_fl = False
        stayed_fl = False
        ps = env.engine.player(model_idx)
        board = ps.board
        if not fouled and board.qualifies_for_fl():
            entered_fl = True
        if hasattr(result, 'stayed_fl'):
            stayed_fl = result.stayed_fl(model_idx)

        # 相手のフォール（2人分）
        opp_fouled = False
        for i in range(3):
            if i != model_idx and result.is_fouled(i):
                opp_fouled = True

        return score, fouled, entered_fl, stayed_fl, royalty, opp_fouled

    def _compute_summary(self, stats: Dict) -> Dict:
        """統計サマリーを計算"""
        n = stats['games']
        if n == 0:
            return stats

        scores = np.array(stats['scores'])
        royalties = np.array(stats['royalties'])

        summary = {
            'opponent': stats['opponent'],
            'games': n,
            'errors': stats['errors'],
            'elapsed_sec': stats['elapsed_sec'],
            # Core metrics
            'foul_rate': stats['fouls'] / n * 100,
            'mean_score': float(np.mean(scores)),
            'score_std': float(np.std(scores)),
            'mean_royalty': float(np.mean(royalties)),
            'fl_entry_rate': stats['fl_entries'] / n * 100,
            'fl_stay_rate': stats['fl_stays'] / n * 100,
            'fl_stay_per_entry': (
                stats['fl_stays'] / stats['fl_entries'] * 100
                if stats['fl_entries'] > 0 else 0.0
            ),
            'win_rate': stats['wins'] / n * 100,
            'wins': stats['wins'],
            'losses': stats['losses'],
            'draws': stats['draws'],
            # Opp foul rate
            'opp_foul_rate': (
                stats['opp_fouls'] / stats['opp_games'] * 100
                if stats['opp_games'] > 0 else 0.0
            ),
            # Position breakdown
            'position_scores': {},
            'position_foul_rates': {},
        }

        pos_names = {0: 'BTN', 1: 'SB', 2: 'BB'}
        for pos in range(3):
            pn = pos_names[pos]
            pg = stats['position_games'][pos]
            if pg > 0:
                summary['position_scores'][pn] = float(np.mean(stats['position_scores'][pos]))
                summary['position_foul_rates'][pn] = stats['position_fouls'][pos] / pg * 100
            else:
                summary['position_scores'][pn] = 0.0
                summary['position_foul_rates'][pn] = 0.0

        # 95% 信頼区間 (score)
        if n > 1:
            se = stats['scores']
            summary['score_ci95'] = 1.96 * float(np.std(se)) / np.sqrt(n)
        else:
            summary['score_ci95'] = 0.0

        return summary


def print_results(model_name: str, results: List[Dict]):
    """結果を表形式で表示"""
    print(f"\n{'='*80}")
    print(f"  Benchmark Results: {model_name}")
    print(f"{'='*80}")

    # ヘッダー
    header = (
        f"{'Opponent':<15} {'Games':>6} {'Foul%':>7} {'Score':>8} "
        f"{'±CI95':>7} {'WinR%':>7} {'FL Ent%':>8} {'FL Stay%':>9}"
    )
    print(f"\n{header}")
    print('-' * 80)

    for r in results:
        line = (
            f"{r['opponent']:<15} {r['games']:>6} {r['foul_rate']:>6.1f}% "
            f"{r['mean_score']:>+7.1f} {r['score_ci95']:>6.1f} "
            f"{r['win_rate']:>6.1f}% {r['fl_entry_rate']:>7.1f}% "
            f"{r['fl_stay_rate']:>8.1f}%"
        )
        print(line)

    print()

    # ポジション別詳細
    print("  Position Breakdown:")
    print(f"  {'Opponent':<15} {'BTN Score':>10} {'SB Score':>10} {'BB Score':>10} "
          f"{'BTN Foul%':>10} {'SB Foul%':>10} {'BB Foul%':>10}")
    print('  ' + '-' * 75)

    for r in results:
        ps = r.get('position_scores', {})
        pf = r.get('position_foul_rates', {})
        line = (
            f"  {r['opponent']:<15} "
            f"{ps.get('BTN', 0):>+9.1f} {ps.get('SB', 0):>+9.1f} {ps.get('BB', 0):>+9.1f} "
            f"{pf.get('BTN', 0):>9.1f}% {pf.get('SB', 0):>9.1f}% {pf.get('BB', 0):>9.1f}%"
        )
        print(line)

    print()

    # 追加情報
    for r in results:
        fps = r['games'] / r['elapsed_sec'] if r['elapsed_sec'] > 0 else 0
        print(
            f"  vs {r['opponent']}: "
            f"W/L/D={r['wins']}/{r['losses']}/{r['draws']}  "
            f"Royalty={r['mean_royalty']:.1f}  "
            f"OppFoul={r['opp_foul_rate']:.1f}%  "
            f"Errors={r['errors']}  "
            f"{fps:.0f} games/sec"
        )

    print()


def print_comparison(all_results: Dict[str, List[Dict]]):
    """複数モデルの比較テーブル"""
    if len(all_results) < 2:
        return

    model_names = list(all_results.keys())
    opponents = [r['opponent'] for r in all_results[model_names[0]]]

    print(f"\n{'='*80}")
    print(f"  Model Comparison")
    print(f"{'='*80}")

    for opp in opponents:
        print(f"\n  vs {opp}:")
        print(f"  {'Model':<30} {'Foul%':>7} {'Score':>8} {'WinR%':>7} {'FL Ent%':>8}")
        print('  ' + '-' * 65)

        for mname in model_names:
            r = next((x for x in all_results[mname] if x['opponent'] == opp), None)
            if r:
                print(
                    f"  {mname:<30} {r['foul_rate']:>6.1f}% "
                    f"{r['mean_score']:>+7.1f} {r['win_rate']:>6.1f}% "
                    f"{r['fl_entry_rate']:>7.1f}%"
                )

    print()


def run_baseline_evaluation(evaluator: BenchmarkEvaluator):
    """ルールベース同士のベースライン計測"""
    agents = {
        'Safe': SafeAgent(),
        'Aggressive': AggressiveAgent(),
        'Random': RandomAgent(),
    }

    print("\nBaseline Evaluation: Rule-Based Agents vs Each Other")
    print("=" * 60)

    for name, agent in agents.items():
        results = []
        for opp_name, opp_agent in agents.items():
            if opp_name == name:
                continue
            print(f"  Evaluating {name} vs {opp_name}...", end='', flush=True)
            r = evaluator.evaluate_vs_opponent(agent, opp_agent, opp_name)
            results.append(r)
            print(f" Score={r['mean_score']:+.1f} Foul={r['foul_rate']:.1f}%")

        print_results(f"Rule-Based: {name}", results)


def main():
    parser = argparse.ArgumentParser(description='OFC V2 Benchmark Evaluation')
    parser.add_argument(
        '--model', action='append', default=[],
        help='Path to model .zip file (can specify multiple times)'
    )
    parser.add_argument(
        '--games', type=int, default=1000,
        help='Number of games per opponent (default: 1000)'
    )
    parser.add_argument(
        '--baseline-only', action='store_true',
        help='Only run rule-based baseline evaluation'
    )
    parser.add_argument(
        '--fl-solver', default='greedy',
        choices=['greedy', 'default'],
        help='FL solver mode (default: greedy)'
    )
    parser.add_argument(
        '--opponents', nargs='+',
        default=['random', 'safe', 'aggressive'],
        choices=['random', 'safe', 'aggressive'],
        help='Opponent types to evaluate against'
    )
    args = parser.parse_args()

    evaluator = BenchmarkEvaluator(
        num_games=args.games,
        fl_solver_mode=args.fl_solver,
    )

    if args.baseline_only:
        run_baseline_evaluation(evaluator)
        return

    if not args.model:
        print("Error: --model required (or use --baseline-only)")
        parser.print_help()
        sys.exit(1)

    opponent_map = {
        'random': ('Random', RandomAgent()),
        'safe': ('Safe', SafeAgent()),
        'aggressive': ('Aggressive', AggressiveAgent()),
    }

    all_results = {}

    for model_path in args.model:
        if not os.path.exists(model_path):
            print(f"Warning: Model not found: {model_path}")
            continue

        print(f"\nLoading model: {model_path}")
        model_agent = ModelAgent(model_path)
        model_name = model_agent.name

        results = []
        for opp_key in args.opponents:
            opp_name, opp_agent = opponent_map[opp_key]
            print(f"  Evaluating vs {opp_name} ({args.games} games)...", end='', flush=True)
            r = evaluator.evaluate_vs_opponent(model_agent, opp_agent, opp_name)
            results.append(r)
            print(
                f" Score={r['mean_score']:+.1f} Foul={r['foul_rate']:.1f}% "
                f"Win={r['win_rate']:.1f}%"
            )

        print_results(model_name, results)
        all_results[model_name] = results

    if len(all_results) > 1:
        print_comparison(all_results)


if __name__ == '__main__':
    main()
