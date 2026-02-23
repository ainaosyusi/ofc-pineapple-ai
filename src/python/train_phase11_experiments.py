#!/usr/bin/env python3
"""
Phase 11: FL Entry Rate改善 — 4実験比較

各実験を個別に実行して、どの変更がFL Entry Rateに最も効果的か検証する。

実験一覧:
  1. FL Entry報酬増額 (+50 → +150)
  2. 観測空間拡張 (FLアウツ情報追加, 881→887次元)
  3. オープニングFL追求報酬強化 (Stage1: +5→+15, Stage2: +10→+25)
  4. ファウルペナルティ軽減 (FL追求中: 0.2→0.1, 通常: 0.5→0.3)

Usage:
    # 実験1を実行
    python src/python/train_phase11_experiments.py --experiment 1 --steps 50000000

    # ローカルテスト (全実験、10kステップ)
    NUM_ENVS=2 python src/python/train_phase11_experiments.py --experiment 1 --test-mode --steps 10000

    # ベースライン (Phase 10と同じ設定で再学習)
    python src/python/train_phase11_experiments.py --experiment 0 --steps 50000000

環境変数:
    NUM_ENVS: 並列環境数 (デフォルト: 4)
"""

import os
import sys
import time
import argparse
import glob
import re
import numpy as np
from datetime import datetime
from collections import deque
import torch
torch.distributions.Distribution.set_default_validate_args(False)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import gymnasium as gym

from ofc_3max_env import OFC3MaxEnv
from notifier import TrainingNotifier

NUM_ENVS = int(os.getenv("NUM_ENVS", "4"))
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")


# ========================================
# 実験設定
# ========================================

EXPERIMENTS = {
    0: {
        'name': 'baseline',
        'description': 'Phase 10と同じ設定（ベースライン）',
        'reward_config': {},  # 全てデフォルト
        'extended_fl_obs': False,
        'wrapper_fl_entry_bonus': 50.0,
        'wrapper_fl_stay_bonus': 100.0,
        'can_finetune': True,
    },
    1: {
        'name': 'fl_reward_boost',
        'description': 'FL Entry報酬を大幅増額 (+50→+150)',
        'reward_config': {
            'fl_entry_bonus': 150.0,   # 50 → 150
            'fl_aa_bonus': 20.0,       # 10 → 20
            'fl_trips_bonus': 20.0,    # 10 → 20
        },
        'extended_fl_obs': False,
        'wrapper_fl_entry_bonus': 150.0,  # wrapper側も増額
        'wrapper_fl_stay_bonus': 100.0,
        'can_finetune': True,
    },
    2: {
        'name': 'extended_obs',
        'description': '観測空間にFLアウツ情報を追加 (881→887次元)',
        'reward_config': {},
        'extended_fl_obs': True,
        'wrapper_fl_entry_bonus': 50.0,
        'wrapper_fl_stay_bonus': 100.0,
        'can_finetune': False,  # 観測空間が変わるため既存モデル不可
    },
    3: {
        'name': 'opening_reward',
        'description': 'オープニングでのFL追求報酬を強化',
        'reward_config': {
            'fl_stage1_reward': 15.0,  # 5 → 15 (Q/K/AをTopに置く)
            'fl_stage2_reward': 25.0,  # 10 → 25 (QQ/KK/AA成立)
            'fl_stage3_reward': 12.0,  # 8 → 12 (Trips成立)
        },
        'extended_fl_obs': False,
        'wrapper_fl_entry_bonus': 50.0,
        'wrapper_fl_stay_bonus': 100.0,
        'can_finetune': True,
    },
    4: {
        'name': 'reduced_foul',
        'description': 'ファウルペナルティを軽減（リスクテイク促進）',
        'reward_config': {
            'foul_penalty_fl': 0.1,      # 0.2 → 0.1 (FL追求中)
            'foul_penalty_normal': 0.3,  # 0.5 → 0.3 (通常時)
        },
        'extended_fl_obs': False,
        'wrapper_fl_entry_bonus': 50.0,
        'wrapper_fl_stay_bonus': 100.0,
        'can_finetune': True,
    },
}


# ========================================
# Self-Play環境
# ========================================

class Phase11SelfPlayEnv(gym.Env):
    """Phase 11 Self-Play環境: 実験パラメータ対応"""

    def __init__(self, env_id: int = 0, experiment_id: int = 0):
        super().__init__()

        exp = EXPERIMENTS[experiment_id]
        self.env = OFC3MaxEnv(
            enable_fl_turns=True,
            continuous_games=True,
            fl_solver_mode='greedy',
            reward_config=exp['reward_config'],
            extended_fl_obs=exp['extended_fl_obs'],
        )
        self.env_id = env_id
        self.experiment_id = experiment_id
        self.learning_agent = "player_0"

        self.observation_space = self.env.observation_space(self.learning_agent)
        self.action_space = self.env.action_space(self.learning_agent)
        self.current_reward = 0

        # Wrapper側のFL報酬（env側とは別に上乗せ）
        self.fl_entry_bonus = exp['wrapper_fl_entry_bonus']
        self.fl_stay_bonus = exp['wrapper_fl_stay_bonus']

        # 統計
        self.total_games = 0
        self.fl_entries = 0
        self.fl_stays = 0
        self.fouls = 0

    def set_model(self, model):
        self.model = model

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
            self.total_games += 1
            res = self.env.engine.result()
            score = res.get_score(0)
            entered_fl = res.entered_fl(0)
            stayed_fl = res.stayed_fl(0)
            fouled = res.is_fouled(0)

            if entered_fl:
                self.fl_entries += 1
                reward += self.fl_entry_bonus
            if stayed_fl:
                self.fl_stays += 1
                reward += self.fl_stay_bonus
            if fouled:
                self.fouls += 1

            info = {
                'score': score,
                'royalty': res.get_royalty(0),
                'fouled': fouled,
                'entered_fl': entered_fl,
                'stayed_fl': stayed_fl,
                'win': score > 0,
            }
        return obs, reward, terminated, truncated, info

    def _play_opponents(self):
        max_iterations = 100
        for _ in range(max_iterations):
            if all(self.env.terminations.values()):
                break
            current = self.env.agent_selection
            if current == self.learning_agent:
                break
            if self.env.terminations.get(current, False) or self.env.truncations.get(current, False):
                self.env.step(None)
                continue
            obs = self.env.observe(current)
            mask = self.env.action_masks(current)
            if not hasattr(self, 'model') or self.model is None:
                valid = np.where(mask)[0]
                action = int(np.random.choice(valid)) if len(valid) > 0 else 0
            else:
                try:
                    action, _ = self.model.predict(obs, action_masks=mask, deterministic=False)
                    action = int(action)
                except:
                    valid = np.where(mask)[0]
                    action = int(np.random.choice(valid)) if len(valid) > 0 else 0
            self.env.step(action)

    def action_masks(self):
        return self.env.action_masks(self.learning_agent)


# ========================================
# コールバック
# ========================================

class Phase11Callback(BaseCallback):
    """Phase 11学習コールバック"""

    def __init__(self, model_save_path, prefix, notifier=None,
                 resume_step=0, total_target=50000000, verbose=0):
        super().__init__(verbose)
        self.model_save_path = model_save_path
        self.prefix = prefix
        self.notifier = notifier
        self.resume_step = resume_step
        self.total_target = total_target
        self.episode_rewards = deque(maxlen=1000)
        self.fl_entries = deque(maxlen=1000)
        self.fl_stays = deque(maxlen=1000)
        self.fouls = deque(maxlen=1000)
        self.start_time = None
        self.last_log_step = resume_step
        self.last_checkpoint_step = (resume_step // 1000000) * 1000000

    def _on_training_start(self):
        self.start_time = time.time()
        if hasattr(self.training_env, 'envs'):
            for env in self.training_env.envs:
                if hasattr(env, 'env') and hasattr(env.env, 'set_model'):
                    env.env.set_model(self.model)
                elif hasattr(env, 'set_model'):
                    env.set_model(self.model)

    def _on_step(self):
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'entered_fl' in info:
                self.fl_entries.append(1.0 if info['entered_fl'] else 0.0)
            if 'stayed_fl' in info:
                self.fl_stays.append(1.0 if info['stayed_fl'] else 0.0)
            if 'fouled' in info:
                self.fouls.append(1.0 if info['fouled'] else 0.0)
            if 'score' in info:
                self.episode_rewards.append(info['score'])
        return True

    def _on_rollout_end(self):
        steps = self.resume_step + self.num_timesteps

        if steps >= self.last_log_step + 50000 and len(self.fl_entries) > 0:
            self.last_log_step = steps
            elapsed = time.time() - self.start_time
            fps = steps / elapsed if elapsed > 0 else 0

            fl_entry_rate = np.mean(self.fl_entries) * 100
            fl_stay_rate = np.mean(self.fl_stays) * 100
            foul_rate = np.mean(self.fouls) * 100
            mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0

            print(f"\n[Step {steps:,}] Phase 11 [{self.prefix}]")
            print(f"  FPS: {fps:.0f}")
            print(f"  Foul Rate: {foul_rate:.1f}%")
            print(f"  FL Entry Rate: {fl_entry_rate:.1f}%")
            print(f"  FL Stay Rate: {fl_stay_rate:.1f}%")
            print(f"  Mean Reward: {mean_reward:+.2f}")

        # チェックポイント保存 (1Mステップごと)
        checkpoint_interval = 1000000
        if steps >= self.last_checkpoint_step + checkpoint_interval:
            self.last_checkpoint_step = (steps // checkpoint_interval) * checkpoint_interval
            path = f"{self.model_save_path}/{self.prefix}_{self.last_checkpoint_step}.zip"
            self.model.save(path)
            print(f"  Saved checkpoint: {path}")


# ========================================
# メイン
# ========================================

def make_env(env_id, experiment_id):
    def _init():
        env = Phase11SelfPlayEnv(env_id=env_id, experiment_id=experiment_id)
        env = ActionMasker(env, lambda e: e.action_masks())
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description='Phase 11 FL Entry Experiments')
    parser.add_argument('--experiment', type=int, required=True, choices=[0, 1, 2, 3, 4],
                        help='実験番号 (0=baseline, 1=reward, 2=obs, 3=opening, 4=foul)')
    parser.add_argument('--steps', type=int, default=50000000,
                        help='学習ステップ数')
    parser.add_argument('--num-envs', type=int, default=None,
                        help='並列環境数')
    parser.add_argument('--test-mode', action='store_true',
                        help='テストモード (10kステップ)')

    args = parser.parse_args()

    if args.test_mode:
        args.steps = min(args.steps, 10000)

    exp = EXPERIMENTS[args.experiment]
    exp_name = exp['name']
    num_envs = args.num_envs or NUM_ENVS

    print(f"=" * 60)
    print(f"Phase 11 Experiment {args.experiment}: {exp['description']}")
    print(f"=" * 60)
    print(f"  Name: {exp_name}")
    print(f"  Steps: {args.steps:,}")
    print(f"  Envs: {num_envs}")
    print(f"  Extended FL Obs: {exp['extended_fl_obs']}")
    print(f"  Reward Config: {exp['reward_config']}")
    print(f"  Wrapper FL Entry Bonus: {exp['wrapper_fl_entry_bonus']}")
    print(f"  Can Fine-tune: {exp['can_finetune']}")
    print()

    # 環境作成
    if num_envs > 1 and not args.test_mode:
        env = SubprocVecEnv(
            [make_env(i, args.experiment) for i in range(num_envs)],
            start_method='spawn'
        )
    else:
        env = DummyVecEnv([make_env(0, args.experiment)])

    # モデル保存先
    save_path = f'models/phase11/{exp_name}'
    os.makedirs(save_path, exist_ok=True)

    # 既存チェックポイントを探す
    prefix = f'p11_{exp_name}'
    checkpoints = glob.glob(f'{save_path}/{prefix}_*.zip')

    def _extract_step(path):
        m = re.search(rf'{prefix}_(\d+)\.zip', path)
        return int(m.group(1)) if m else 0
    checkpoints.sort(key=_extract_step)

    if checkpoints:
        latest = checkpoints[-1]
        resume_step = _extract_step(latest)
        print(f"Resuming from checkpoint: {latest} (step {resume_step:,})")
        model = MaskablePPO.load(latest, env=env)
        remaining_steps = args.steps - resume_step
    elif exp['can_finetune']:
        # Phase 10モデルから継続学習
        base_model = 'models/phase10_gcp/p10_fl_stay_150000000.zip'
        if os.path.exists(base_model):
            print(f"Fine-tuning from Phase 10: {base_model}")
            model = MaskablePPO.load(base_model, env=env)
        else:
            print(f"Phase 10 model not found, starting from Phase 9...")
            base_model = 'models/phase9/p9_fl_mastery_250000000.zip'
            model = MaskablePPO.load(base_model, env=env)
        resume_step = 0
        remaining_steps = args.steps
    else:
        # 観測空間が異なるため新規学習
        print("Creating new model (observation space changed)")
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            verbose=0,
        )
        resume_step = 0
        remaining_steps = args.steps

    # ハイパーパラメータ
    model.learning_rate = 0.0001
    model.ent_coef = 0.02

    # コールバック
    notifier = TrainingNotifier(
        discord_webhook=DISCORD_WEBHOOK_URL,
        project_name=f"OFC AI Phase 11 [{exp_name}]"
    )

    callbacks = [
        Phase11Callback(
            model_save_path=save_path,
            prefix=prefix,
            notifier=notifier,
            resume_step=resume_step,
            total_target=args.steps,
        ),
    ]

    # 学習
    print(f"\nStarting Phase 11 [{exp_name}] for {remaining_steps:,} steps...")
    print(f"Target: FL Entry Rate 17%+ (通常ゲーム)\n")

    model.learn(
        total_timesteps=remaining_steps,
        callback=callbacks,
        progress_bar=False,
    )

    # 最終モデル保存
    final_path = f'{save_path}/{prefix}_{args.steps}.zip'
    model.save(final_path)
    print(f"\nSaved final model to: {final_path}")


if __name__ == '__main__':
    main()
