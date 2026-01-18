"""
OFC Pineapple AI - Phase Comparison Evaluation
Phase 4 (Joker, 2P) vs Phase 7 (Parallel, 3P) の対戦評価

使用方法:
    python src/python/evaluate_phase_comparison.py --games 1000
    python src/python/evaluate_phase_comparison.py --mode head2head --games 500
"""

import os
import sys
import argparse
import random
import numpy as np
from collections import defaultdict
from datetime import datetime

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from sb3_contrib import MaskablePPO
from ofc_3max_env import OFC3MaxEnv
from ofc_phase1_env import OFCPhase1Env


class ModelWrapper:
    """モデルのラッパー（ロード失敗時のフォールバック対応）"""
    def __init__(self, model_path: str, name: str = None):
        self.path = model_path
        self.name = name or os.path.basename(model_path)
        self.model = None
        self.is_random = False

    def load(self, env):
        """モデルをロード"""
        if not os.path.exists(self.path):
            print(f"[Warning] Model not found: {self.path}, using random policy")
            self.is_random = True
            return False
        try:
            self.model = MaskablePPO.load(self.path, env=env)
            return True
        except Exception as e:
            print(f"[Warning] Failed to load {self.path}: {e}, using random policy")
            self.is_random = True
            return False

    def predict(self, obs, action_masks):
        """行動を予測"""
        if self.is_random or self.model is None:
            valid_actions = np.where(action_masks)[0]
            return random.choice(valid_actions) if len(valid_actions) > 0 else 0
        action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
        return action


class Phase7Evaluator:
    """Phase 7 (3-Max) 環境での評価"""

    def __init__(self):
        self.env = OFC3MaxEnv()

    def evaluate_single_model(self, model: ModelWrapper, num_games: int = 1000) -> dict:
        """単一モデルをランダム相手に評価"""
        print(f"\n[Evaluating] {model.name} vs Random (3-Max)")

        # モデルロード用の一時環境
        temp_env = self._create_single_agent_env()
        model.load(temp_env)

        stats = defaultdict(list)

        for game_idx in range(num_games):
            result = self._play_game_vs_random(model)
            for k, v in result.items():
                stats[k].append(v)

            if (game_idx + 1) % 100 == 0:
                foul_rate = sum(stats['fouled']) / len(stats['fouled']) * 100
                print(f"  Game {game_idx + 1}/{num_games}: Foul Rate = {foul_rate:.1f}%")

        return {
            'model': model.name,
            'games': num_games,
            'foul_rate': np.mean(stats['fouled']) * 100,
            'avg_score': np.mean(stats['score']),
            'avg_royalty': np.mean(stats['royalty']),
            'fl_rate': np.mean(stats['entered_fl']) * 100,
            'win_rate': np.mean(stats['won']) * 100,
        }

    def evaluate_head_to_head(self, model1: ModelWrapper, model2: ModelWrapper,
                               num_games: int = 500) -> dict:
        """2モデルの直接対戦評価（3人目はランダム）"""
        print(f"\n[Head-to-Head] {model1.name} vs {model2.name}")

        temp_env = self._create_single_agent_env()
        model1.load(temp_env)
        model2.load(temp_env)

        stats = {
            model1.name: defaultdict(list),
            model2.name: defaultdict(list),
        }

        for game_idx in range(num_games):
            # 交互にポジションを入れ替え
            if game_idx % 2 == 0:
                players = [model1, model2, None]  # None = Random
            else:
                players = [model2, model1, None]

            result = self._play_game_multi(players)

            for i, player in enumerate(players):
                if player is not None:
                    for k, v in result[i].items():
                        stats[player.name][k].append(v)

            if (game_idx + 1) % 100 == 0:
                print(f"  Game {game_idx + 1}/{num_games}")

        return {
            model1.name: {
                'games': num_games,
                'foul_rate': np.mean(stats[model1.name]['fouled']) * 100,
                'avg_score': np.mean(stats[model1.name]['score']),
                'win_rate': np.mean(stats[model1.name]['won']) * 100,
            },
            model2.name: {
                'games': num_games,
                'foul_rate': np.mean(stats[model2.name]['fouled']) * 100,
                'avg_score': np.mean(stats[model2.name]['score']),
                'win_rate': np.mean(stats[model2.name]['won']) * 100,
            }
        }

    def _create_single_agent_env(self):
        """シングルエージェント評価用の環境"""
        from gymnasium import spaces
        import gymnasium as gym

        class SingleAgentWrapper(gym.Env):
            def __init__(self):
                self.env = OFC3MaxEnv()
                self.learning_agent = "player_0"
                self.observation_space = self.env.observation_space(self.learning_agent)
                self.action_space = self.env.action_space(self.learning_agent)

            def reset(self, seed=None, options=None):
                self.env.reset(seed=seed)
                return self.env.observe(self.learning_agent), {}

            def step(self, action):
                self.env.step(action)
                obs = self.env.observe(self.learning_agent)
                done = self.env.terminations.get(self.learning_agent, False)
                return obs, 0, done, False, {}

            def action_masks(self):
                return self.env.action_masks(self.learning_agent)

        return SingleAgentWrapper()

    def _play_game_vs_random(self, model: ModelWrapper) -> dict:
        """モデル vs ランダム2人の1ゲーム"""
        self.env.reset()
        learning_agent = "player_0"

        while not all(self.env.terminations.values()):
            agent = self.env.agent_selection

            if self.env.terminations.get(agent, False):
                self.env.step(None)
                continue

            if agent == learning_agent:
                obs = self.env.observe(agent)
                mask = self.env.action_masks(agent)
                action = model.predict(obs, mask)
            else:
                valid = self.env.get_valid_actions(agent)
                action = random.choice(valid) if valid else 0

            self.env.step(action)

        result = self.env.engine.result()
        player_idx = 0  # player_0

        scores = [result.get_score(i) for i in range(3)]
        won = scores[player_idx] == max(scores) and scores.count(scores[player_idx]) == 1

        return {
            'fouled': result.is_fouled(player_idx),
            'score': result.get_score(player_idx),
            'royalty': result.get_royalty(player_idx) if hasattr(result, 'get_royalty') else 0,
            'entered_fl': result.entered_fl(player_idx),
            'won': won,
        }

    def _play_game_multi(self, players: list) -> list:
        """複数モデルでの1ゲーム（Noneはランダム）"""
        self.env.reset()
        agent_to_player = {
            "player_0": players[0],
            "player_1": players[1],
            "player_2": players[2],
        }

        while not all(self.env.terminations.values()):
            agent = self.env.agent_selection

            if self.env.terminations.get(agent, False):
                self.env.step(None)
                continue

            player = agent_to_player[agent]

            if player is not None:
                obs = self.env.observe(agent)
                mask = self.env.action_masks(agent)
                action = player.predict(obs, mask)
            else:
                valid = self.env.get_valid_actions(agent)
                action = random.choice(valid) if valid else 0

            self.env.step(action)

        result = self.env.engine.result()
        scores = [result.get_score(i) for i in range(3)]
        max_score = max(scores)

        results = []
        for i in range(3):
            won = scores[i] == max_score and scores.count(scores[i]) == 1
            results.append({
                'fouled': result.is_fouled(i),
                'score': result.get_score(i),
                'won': won,
            })

        return results


def find_models():
    """利用可能なモデルを検索"""
    models = {}

    # Phase 4 (Joker)
    phase4_path = "models/phase4/ofc_phase4_joker_20260115_190744_10500000_steps.zip"
    if os.path.exists(phase4_path):
        models['phase4'] = phase4_path

    # Phase 7 (Parallel) - ローカル
    import glob
    p7_files = glob.glob("models/p7_parallel_*.zip") + glob.glob("models/p7_mcts_*.zip")
    if p7_files:
        latest = max(p7_files, key=lambda f: int(f.split('_')[-1].replace('.zip', '')))
        models['phase7'] = latest

    return models


def print_results(results: dict):
    """結果を表形式で表示"""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    print(f"\n{'Model':<50} | {'Foul%':<8} | {'Score':<8} | {'Win%':<8}")
    print("-" * 80)

    for name, data in results.items():
        if isinstance(data, dict) and 'foul_rate' in data:
            print(f"{name:<50} | {data['foul_rate']:>6.1f}% | {data.get('avg_score', 0):>7.1f} | {data.get('win_rate', 0):>6.1f}%")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Phase Comparison Evaluation")
    parser.add_argument("--mode", choices=["single", "head2head", "all"], default="all",
                        help="Evaluation mode")
    parser.add_argument("--games", type=int, default=500, help="Number of games")
    parser.add_argument("--phase4", type=str, help="Path to Phase 4 model")
    parser.add_argument("--phase7", type=str, help="Path to Phase 7 model")
    args = parser.parse_args()

    print("=" * 60)
    print("OFC Pineapple AI - Phase Comparison Evaluation")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # モデル検索
    models = find_models()

    if args.phase4:
        models['phase4'] = args.phase4
    if args.phase7:
        models['phase7'] = args.phase7

    print(f"\nFound models:")
    for name, path in models.items():
        print(f"  - {name}: {path}")

    if not models:
        print("[Error] No models found!")
        return

    evaluator = Phase7Evaluator()
    results = {}

    # 単一評価
    if args.mode in ["single", "all"]:
        for name, path in models.items():
            model = ModelWrapper(path, name)
            result = evaluator.evaluate_single_model(model, args.games)
            results[name] = result

    # 直接対戦
    if args.mode in ["head2head", "all"] and len(models) >= 2:
        model_list = list(models.items())
        for i, (name1, path1) in enumerate(model_list):
            for name2, path2 in model_list[i+1:]:
                m1 = ModelWrapper(path1, name1)
                m2 = ModelWrapper(path2, name2)
                h2h_result = evaluator.evaluate_head_to_head(m1, m2, args.games // 2)
                results[f"{name1}_vs_{name2}"] = h2h_result

    print_results(results)

    # 結果をファイルに保存
    output_path = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(output_path, 'w') as f:
        f.write(f"OFC Pineapple AI - Evaluation Results\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Games: {args.games}\n\n")
        for name, data in results.items():
            f.write(f"{name}:\n")
            if isinstance(data, dict):
                for k, v in data.items():
                    f.write(f"  {k}: {v}\n")
            f.write("\n")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
