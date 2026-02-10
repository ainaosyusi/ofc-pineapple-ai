#!/usr/bin/env python3
"""
Phase 10: FL Stay向上学習

Phase 9 (250M) をベースに、修正されたgreedy_fl_solverでFL Stayを向上

目標:
- FL Entry Rate: 22.8% → 27%+
- FL Stay Rate: 8% → 15%+

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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import gymnasium as gym

from ofc_3max_env import OFC3MaxEnv
from notifier import TrainingNotifier

NUM_ENVS = int(os.getenv("NUM_ENVS", "4"))
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")


class Phase10SelfPlayEnv(gym.Env):
    """
    Phase 10 Self-Play環境: FL Stay強化報酬

    FL中のQuads/Trips達成に追加報酬を付与
    """

    def __init__(self, env_id: int = 0):
        super().__init__()
        self.env = OFC3MaxEnv(
            enable_fl_turns=True,
            continuous_games=True,
            fl_solver_mode='greedy'  # 修正されたgreedy solver
        )
        self.env_id = env_id
        self.learning_agent = "player_0"

        self.observation_space = self.env.observation_space(self.learning_agent)
        self.action_space = self.env.action_space(self.learning_agent)
        self.current_reward = 0

        # FL報酬パラメータ
        self.fl_entry_bonus = 50.0
        self.fl_stay_bonus = 100.0  # Phase 9より増額

        # 統計
        self.total_games = 0
        self.fl_entries = 0
        self.fl_stays = 0
        self.fouls = 0

    def set_model(self, model):
        """推論用モデルを設定"""
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
        """対戦相手をプレイ（Self-Play: 同じモデル使用）"""
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


def make_env(env_id):
    def _init():
        env = Phase10SelfPlayEnv(env_id=env_id)
        env = ActionMasker(env, lambda e: e.action_masks())
        return env
    return _init


class Phase10Callback(BaseCallback):
    """Phase 10学習コールバック"""

    def __init__(self, model_save_path, notifier=None, resume_step=0, total_target=50000000, verbose=0):
        super().__init__(verbose)
        self.model_save_path = model_save_path
        self.notifier = notifier
        self.resume_step = resume_step
        self.total_target = total_target
        self.episode_rewards = deque(maxlen=1000)
        self.fl_entries = deque(maxlen=1000)
        self.fl_stays = deque(maxlen=1000)
        self.fouls = deque(maxlen=1000)
        self.start_time = None
        self.last_notify_step = resume_step
        self.last_log_step = resume_step
        self.last_checkpoint_step = (resume_step // 1000000) * 1000000

    def _on_training_start(self):
        self.start_time = time.time()
        # Set model for self-play (only works with DummyVecEnv)
        if hasattr(self.training_env, 'envs'):
            for env in self.training_env.envs:
                if hasattr(env, 'env') and hasattr(env.env, 'set_model'):
                    env.env.set_model(self.model)
                elif hasattr(env, 'set_model'):
                    env.set_model(self.model)

        # 開始通知
        if self.notifier:
            self.notifier.send_start({
                'phase': 'Phase 10 FL Stay',
                'timesteps': 50000000,
                'base_model': 'Phase 9 (250M)',
            })

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

        # 50kステップごとにログ出力（閾値ベース、modulo不使用）
        if steps >= self.last_log_step + 50000 and len(self.fl_entries) > 0:
            self.last_log_step = steps
            elapsed = time.time() - self.start_time
            fps = steps / elapsed if elapsed > 0 else 0

            fl_entry_rate = np.mean(self.fl_entries) * 100
            fl_stay_rate = np.mean(self.fl_stays) * 100
            foul_rate = np.mean(self.fouls) * 100
            mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0

            print(f"\n[Step {steps:,}] Phase 10 FL Stay Training")
            print(f"  FPS: {fps:.0f}")
            print(f"  Foul Rate: {foul_rate:.1f}%")
            print(f"  FL Entry Rate: {fl_entry_rate:.1f}%")
            print(f"  FL Stay Rate: {fl_stay_rate:.1f}%")
            print(f"  Mean Reward: {mean_reward:+.2f}")

            # Discord通知 (100k steps ごと)
            if self.notifier and steps - self.last_notify_step >= 100000:
                self.last_notify_step = steps
                self.notifier.send_progress(
                    step=steps,
                    total_steps=self.total_target,
                    metrics={
                        'foul_rate': foul_rate,
                        'fl_entry_rate': fl_entry_rate,
                        'fl_stay_rate': fl_stay_rate,
                        'mean_score': mean_reward,
                        'fps': fps,
                    }
                )

        # チェックポイント保存 (1Mステップごと、閾値ベース)
        checkpoint_interval = 1000000
        if steps >= self.last_checkpoint_step + checkpoint_interval:
            self.last_checkpoint_step = (steps // checkpoint_interval) * checkpoint_interval
            path = f"{self.model_save_path}/p10_fl_stay_{self.last_checkpoint_step}.zip"
            self.model.save(path)
            print(f"  Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser(description='Phase 10 FL Stay Training')
    parser.add_argument('--steps', type=int, default=200000000,
                        help='Total training steps')
    parser.add_argument('--num-envs', type=int, default=None,
                        help='Number of parallel environments')

    args = parser.parse_args()

    num_envs = args.num_envs or NUM_ENVS
    print(f"Creating {num_envs} parallel environments...")

    # 環境作成
    if num_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(num_envs)], start_method='spawn')
    else:
        env = DummyVecEnv([make_env(0)])

    # 既存のPhase 10チェックポイントを探す（再開用）
    save_path = 'models/phase10'
    os.makedirs(save_path, exist_ok=True)
    checkpoints = glob.glob(f'{save_path}/p10_fl_stay_*.zip')
    # ステップ数で正しくソート（アルファベット順だと 9M > 12M になるバグ修正）
    def _extract_step(path):
        m = re.search(r'p10_fl_stay_(\d+)\.zip', path)
        return int(m.group(1)) if m else 0
    checkpoints.sort(key=_extract_step)

    if checkpoints:
        # 最新チェックポイントから再開
        latest = checkpoints[-1]
        resume_step = _extract_step(latest)
        print(f"Resuming from checkpoint: {latest} (step {resume_step:,})")
        model = MaskablePPO.load(latest, env=env)
        remaining_steps = args.steps - resume_step
        print(f"Remaining steps: {remaining_steps:,}")
    else:
        # Phase 9 モデルから開始
        base_model = 'models/phase9/p9_fl_mastery_250000000.zip'
        print(f"Loading Phase 9 model: {base_model}")
        model = MaskablePPO.load(base_model, env=env)
        resume_step = 0
        remaining_steps = args.steps

    # ハイパーパラメータ更新（Fine-tuning）
    model.learning_rate = 0.0001
    model.ent_coef = 0.02

    # モデル保存先
    save_path = 'models/phase10'
    os.makedirs(save_path, exist_ok=True)

    # 通知設定
    notifier = TrainingNotifier(
        discord_webhook=DISCORD_WEBHOOK_URL,
        project_name="OFC AI Phase 10"
    )

    # コールバック
    callbacks = [
        Phase10Callback(
            model_save_path=save_path,
            notifier=notifier,
            resume_step=resume_step,
            total_target=args.steps
        ),
    ]

    # 学習
    if resume_step > 0:
        print(f"\nResuming Phase 10 training from {resume_step:,} steps...")
    else:
        print(f"\nStarting Phase 10 training for {args.steps:,} steps...")
    print(f"Target: FL Entry 27%+, FL Stay 30%+\n")

    model.learn(
        total_timesteps=remaining_steps,
        callback=callbacks,
        progress_bar=False,
    )

    # 最終モデル保存
    final_path = f'{save_path}/p10_fl_stay_{args.steps}.zip'
    model.save(final_path)
    print(f"\nSaved final model to: {final_path}")


if __name__ == '__main__':
    main()
