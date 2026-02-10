"""
OFC Pineapple AI - Phase 8.5b 評価スクリプト

Phase 8.5b (150Mステップ) モデルの評価。
オプションで100Mベースラインとの比較も実行。

使用法:
    cd "/Users/naoai/試作品一覧/OFC NN"
    python scripts/evaluate_phase85b.py
    python scripts/evaluate_phase85b.py --games 2000
    python scripts/evaluate_phase85b.py --compare  # 100Mモデルとの比較
"""

import os
import sys
import argparse
import random
import numpy as np
from collections import defaultdict
from datetime import datetime

# パス設定
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'python'))

from sb3_contrib import MaskablePPO
from ofc_3max_env import OFC3MaxEnv


class EvalWrapper:
    """評価用の環境ラッパー（ParallelOFCEnvの簡易版）"""

    def __init__(self, enable_fl_turns=True, continuous_games=True):
        self.env = OFC3MaxEnv(
            enable_fl_turns=enable_fl_turns,
            continuous_games=continuous_games
        )
        self.learning_agent = "player_0"
        self.observation_space = self.env.observation_space(self.learning_agent)
        self.action_space = self.env.action_space(self.learning_agent)
        self.current_reward = 0

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        self.current_reward = 0
        self._play_opponents()
        return self.env.observe(self.learning_agent), {}

    def step(self, action):
        if self.env.terminations.get(self.learning_agent, False):
            self.env.step(None)
        else:
            self.env.step(action)
        self._play_opponents()

        obs = self.env.observe(self.learning_agent)
        reward = self.env._cumulative_rewards.get(self.learning_agent, 0) - self.current_reward
        self.current_reward += reward

        terminated = self.env.terminations.get(self.learning_agent, False)
        truncated = self.env.truncations.get(self.learning_agent, False)

        info = {}
        if terminated or truncated:
            res = self.env.engine.result()
            info = {
                'score': res.get_score(0),
                'royalty': res.get_royalty(0),
                'fouled': res.is_fouled(0),
                'entered_fl': res.entered_fl(0),
                'stayed_fl': res.stayed_fl(0),
            }
        return obs, reward, terminated, truncated, info

    def _play_opponents(self):
        while (not all(self.env.terminations.values())
               and self.env.agent_selection != self.learning_agent):
            agent = self.env.agent_selection
            if self.env.terminations.get(agent, False):
                self.env.step(None)
                continue
            valid_actions = self.env.get_valid_actions(agent)
            if not valid_actions:
                self.env.step(0)
                continue
            self.env.step(random.choice(valid_actions))

    def action_masks(self):
        if self.env.terminations.get(self.learning_agent, False):
            mask = np.zeros(243, dtype=np.int8)
            mask[0] = 1
            return mask
        return self.env.action_masks(self.learning_agent)


def evaluate_model(model_path, num_games=1000, label=None):
    """モデルを評価してメトリクスを返す"""
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return None

    name = label or os.path.basename(model_path)
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"Path: {model_path}")
    print(f"Games: {num_games}")
    print(f"{'='*60}")

    env = EvalWrapper(enable_fl_turns=True, continuous_games=True)
    model = MaskablePPO.load(model_path, env=env)

    # メトリクス収集
    scores = []
    royalties = []
    fouls = 0
    fl_entries = 0
    fl_stays = 0
    wins = 0
    losses = 0
    draws = 0
    high_score_games = 0  # score >= 15
    errors = 0

    for i in range(num_games):
        try:
            obs, _ = env.reset()
            done = False
            while not done:
                mask = env.action_masks()
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            if 'score' in info:
                score = info['score']
                scores.append(score)
                royalties.append(info.get('royalty', 0))
                if info.get('fouled', False):
                    fouls += 1
                if info.get('entered_fl', False):
                    fl_entries += 1
                if info.get('stayed_fl', False):
                    fl_stays += 1
                if score > 0:
                    wins += 1
                elif score < 0:
                    losses += 1
                else:
                    draws += 1
                if score >= 15:
                    high_score_games += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [Warning] Game {i} error: {e}")

        # 進捗表示
        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{num_games} games completed")

    total = len(scores)
    if total == 0:
        print(f"[ERROR] No games completed successfully")
        return None

    # 結果計算
    foul_rate = fouls / total * 100
    mean_score = np.mean(scores)
    mean_royalty = np.mean(royalties)
    fl_entry_rate = fl_entries / total * 100
    fl_stay_rate = fl_stays / total * 100
    fl_stay_per_entry = (fl_stays / fl_entries * 100) if fl_entries > 0 else 0
    win_rate = wins / total * 100
    high_score_rate = high_score_games / total * 100

    results = {
        'name': name,
        'total_games': total,
        'foul_rate': foul_rate,
        'mean_score': mean_score,
        'mean_royalty': mean_royalty,
        'fl_entry_rate': fl_entry_rate,
        'fl_stay_rate': fl_stay_rate,
        'fl_stay_per_entry': fl_stay_per_entry,
        'win_rate': win_rate,
        'high_score_rate': high_score_rate,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'fl_entries': fl_entries,
        'fl_stays': fl_stays,
        'errors': errors,
    }

    # 結果表示
    print(f"\n--- Results: {name} ---")
    print(f"  Games:           {total} (errors: {errors})")
    print(f"  Foul Rate:       {foul_rate:.1f}%")
    print(f"  Mean Score:      {mean_score:+.2f}")
    print(f"  Mean Royalty:    {mean_royalty:.2f}")
    print(f"  FL Entry Rate:   {fl_entry_rate:.1f}% ({fl_entries} entries)")
    print(f"  FL Stay Rate:    {fl_stay_rate:.1f}% (per-game)")
    print(f"  FL Stay/Entry:   {fl_stay_per_entry:.1f}% ({fl_stays}/{fl_entries})")
    print(f"  Win Rate:        {win_rate:.1f}% (W:{wins} L:{losses} D:{draws})")
    print(f"  High Score Rate: {high_score_rate:.1f}% (score >= 15)")

    return results


def print_comparison(results_list):
    """複数モデルの比較表を出力"""
    if len(results_list) < 2:
        return

    print(f"\n{'='*80}")
    print("Comparison Summary")
    print(f"{'='*80}")

    headers = ['Metric', *[r['name'] for r in results_list], 'Change']
    metrics = [
        ('Foul Rate', 'foul_rate', '{:.1f}%', 'lower'),
        ('Mean Score', 'mean_score', '{:+.2f}', 'higher'),
        ('Mean Royalty', 'mean_royalty', '{:.2f}', 'higher'),
        ('FL Entry Rate', 'fl_entry_rate', '{:.1f}%', 'higher'),
        ('FL Stay/Entry', 'fl_stay_per_entry', '{:.1f}%', 'higher'),
        ('Win Rate', 'win_rate', '{:.1f}%', 'higher'),
        ('High Score Rate', 'high_score_rate', '{:.1f}%', 'higher'),
    ]

    # ヘッダー
    col_w = 20
    header_line = f"{'Metric':<{col_w}}"
    for r in results_list:
        header_line += f" | {r['name']:>{col_w}}"
    header_line += f" | {'Change':>{col_w}}"
    print(header_line)
    print("-" * len(header_line))

    # 各メトリクス
    base = results_list[0]
    latest = results_list[-1]
    for label, key, fmt, direction in metrics:
        line = f"{label:<{col_w}}"
        for r in results_list:
            line += f" | {fmt.format(r[key]):>{col_w}}"
        # 変化量
        diff = latest[key] - base[key]
        sign = '+' if diff >= 0 else ''
        improved = (diff < 0 and direction == 'lower') or (diff > 0 and direction == 'higher')
        marker = ' **' if improved else ''
        line += f" | {sign}{fmt.format(diff):>{col_w-1}}{marker}"
        print(line)

    print(f"{'='*80}")


def save_results(results_list, output_dir):
    """結果をファイルに保存"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'eval_phase85b_{timestamp}.txt')

    with open(filepath, 'w') as f:
        f.write(f"Phase 8.5b Evaluation Results\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")

        for r in results_list:
            f.write(f"Model: {r['name']}\n")
            f.write(f"  Games:           {r['total_games']}\n")
            f.write(f"  Foul Rate:       {r['foul_rate']:.1f}%\n")
            f.write(f"  Mean Score:      {r['mean_score']:+.2f}\n")
            f.write(f"  Mean Royalty:    {r['mean_royalty']:.2f}\n")
            f.write(f"  FL Entry Rate:   {r['fl_entry_rate']:.1f}%\n")
            f.write(f"  FL Stay Rate:    {r['fl_stay_rate']:.1f}%\n")
            f.write(f"  FL Stay/Entry:   {r['fl_stay_per_entry']:.1f}%\n")
            f.write(f"  Win Rate:        {r['win_rate']:.1f}%\n")
            f.write(f"  High Score Rate: {r['high_score_rate']:.1f}%\n")
            f.write(f"\n")

    print(f"\nResults saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Phase 8.5b Model Evaluation")
    parser.add_argument("--games", type=int, default=1000, help="Number of games per model (default: 1000)")
    parser.add_argument("--compare", action="store_true", help="Compare with 100M baseline model")
    parser.add_argument("--model", type=str, default=None, help="Path to specific model to evaluate")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    # モデルパスの候補
    phase85b_candidates = [
        "models/phase85b/",  # GCPからダウンロード先
        "models/",
    ]

    results = []

    # --- 100Mベースライン評価（比較モード時）---
    if args.compare:
        baseline_paths = [
            "models/backup/p85_full_fl_100000004_backup.zip",
            "models/p85_full_fl_100000000.zip",
        ]
        for path in baseline_paths:
            if os.path.exists(path):
                res = evaluate_model(path, num_games=args.games, label="Phase 8.5 (100M)")
                if res:
                    results.append(res)
                break
        else:
            print("[Warning] 100M baseline model not found. Skipping comparison.")

    # --- Phase 8.5b モデル評価 ---
    if args.model:
        model_path = args.model
    else:
        # 自動検出: phase85bディレクトリまたはmodelsディレクトリから最新のp85_full_flを探す
        import glob
        model_path = None
        for candidate_dir in phase85b_candidates:
            if not os.path.isdir(candidate_dir):
                continue
            checkpoints = glob.glob(os.path.join(candidate_dir, "p85_full_fl_*.zip"))
            if checkpoints:
                # ステップ数が最大のものを選択
                def get_step(f):
                    try:
                        return int(os.path.basename(f).split('_')[-1].replace('.zip', ''))
                    except ValueError:
                        return 0
                latest = max(checkpoints, key=get_step)
                step = get_step(latest)
                # 100Mベースラインとは別のモデルを選択（110M以上）
                if step > 110_000_000:
                    model_path = latest
                    break

        if model_path is None:
            print("[ERROR] Phase 8.5b model not found.")
            print("Download from GCP first:")
            print('  gcloud compute scp ofc-training-v2:~/ofc-training/models/p85_full_fl_*.zip \\')
            print('    "models/phase85b/" --zone=asia-northeast1-b')
            if not results:
                return
        else:
            step_str = os.path.basename(model_path).split('_')[-1].replace('.zip', '')
            step_m = int(step_str) / 1_000_000
            res = evaluate_model(model_path, num_games=args.games, label=f"Phase 8.5b ({step_m:.0f}M)")
            if res:
                results.append(res)

    # --- 比較表 ---
    if len(results) >= 2:
        print_comparison(results)

    # --- 目標達成チェック ---
    if results:
        latest = results[-1]
        print(f"\n{'='*60}")
        print("Target Achievement Check")
        print(f"{'='*60}")
        targets = [
            ('Foul Rate', latest['foul_rate'], '< 20%', latest['foul_rate'] < 20),
            ('Mean Score', latest['mean_score'], '> +10', latest['mean_score'] > 10),
            ('Mean Royalty', latest['mean_royalty'], '> 2.0', latest['mean_royalty'] > 2.0),
            ('FL Entry Rate', latest['fl_entry_rate'], '> 8%', latest['fl_entry_rate'] > 8),
            ('FL Stay/Entry', latest['fl_stay_per_entry'], '> 30%', latest['fl_stay_per_entry'] > 30),
            ('Win Rate', latest['win_rate'], '> 70%', latest['win_rate'] > 70),
        ]
        achieved = 0
        for name, value, target, ok in targets:
            status = 'PASS' if ok else 'FAIL'
            print(f"  [{status}] {name}: {value:.1f} (target: {target})")
            if ok:
                achieved += 1
        print(f"\n  Result: {achieved}/{len(targets)} targets achieved")

    # --- 結果保存 ---
    if results:
        save_results(results, os.path.join(PROJECT_ROOT, "plots", "phase85b"))

    print("\nDone.")


if __name__ == "__main__":
    main()
