"""
OFC Pineapple AI - Enhanced Phase 3 Self-Play Training Script
確率特徴量と強化版モデルアーキテクチャを使用した学習スクリプト
"""

import io
import os
import sys
import time
import numpy as np
from datetime import datetime
from collections import deque
from typing import Optional, Dict

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from multi_ofc_env import OFCMultiAgentEnv
from notifier import init_notifier, get_notifier
from auto_curriculum import get_curriculum_manager

class EnhancedCallback(BaseCallback):
    """
    強化版Phase 3用コールバック
    """
    def __init__(self, save_path: str, save_freq: int = 200_000, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.episode_rewards = deque(maxlen=1000)
        self.wins = 0
        self.total_games = 0
        self.fouls = 0
        self.last_notify_step = 0
        self.start_time = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'final_score' in info:
                score = info['final_score']
                self.episode_rewards.append(score)
                self.total_games += 1
                if score > 0: self.wins += 1
                if info.get('fouled', False): self.fouls += 1
        
        if self.n_calls % self.save_freq == 0:
            path = f"{self.save_path}_{self.n_calls}"
            self.model.save(path)
            notifier = get_notifier()
            if notifier: notifier.send_checkpoint(f"{path}.zip", self.n_calls)
            
        if self.n_calls - self.last_notify_step >= 100_000:
            self._send_notification()
            self.last_notify_step = self.n_calls
            
        return True

    def _send_notification(self):
        notifier = get_notifier()
        if not notifier: return
        foul_rate = (self.fouls / max(1, self.total_games)) * 100
        win_rate = (self.wins / max(1, self.total_games)) * 100
        avg_score = np.mean(self.episode_rewards) if self.episode_rewards else 0
        notifier.send_progress(
            step=self.n_calls,
            total_steps=10_000_000,
            metrics={
                'foul_rate': foul_rate,
                'win_rate': win_rate,
                'avg_score': avg_score,
                'games': self.total_games
            }
        )

class EnhancedSelfPlayEnv(gym.Env):
    """
    確率特徴量対応のSelf-Play環境
    """
    def __init__(self, pool_ratio=0.8):
        super().__init__()
        self.env = OFCMultiAgentEnv()
        self.pool_ratio = pool_ratio
        self.opponent_pool = []
        self.latest_opponent = None
        self.active_opponent = None
        
        self.observation_space = self.env.observation_space("player_0")
        self.action_space = self.env.action_space("player_0")

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed)
        if self.opponent_pool and np.random.random() < self.pool_ratio:
            self.active_opponent = np.random.choice(self.opponent_pool)
        else:
            self.active_opponent = self.latest_opponent
        return self.env.observe("player_0"), {}

    def step(self, action):
        # 自分のターン
        self.env.step(action)
        
        # 相手のターンを自動進行
        while self.env.agent_selection == "player_1" and not all(self.env.terminations.values()):
            obs = self.env.observe("player_1")
            valid = self.env.get_valid_actions("player_1")
            mask = np.zeros(self.action_space.n, dtype=bool)
            for a in valid: mask[a] = True
            
            if self.active_opponent:
                opp_action, _ = self.active_opponent.predict(obs, action_masks=mask, deterministic=True)
            else:
                opp_action = np.random.choice(valid)
            self.env.step(opp_action)
            
        done = all(self.env.terminations.values())
        obs = self.env.observe("player_0")
        reward = self.env._cumulative_rewards["player_0"] if done else 0
        info = self.env.infos.get("player_0", {})
        
        return obs, reward, done, False, info

    def add_to_pool(self, model):
        # メモリ効率のため軽量化して保存したいが、ここでは単純化
        self.opponent_pool.append(model)
        if len(self.opponent_pool) > 10: self.opponent_pool.pop(0)
    
    def set_latest(self, model):
        self.latest_opponent = model

    def action_masks(self):
        valid = self.env.get_valid_actions("player_0")
        mask = np.zeros(self.action_space.n, dtype=bool)
        for action_id in valid:
            mask[action_id] = True
        if not mask.any():
            mask[0] = True
        return mask

def train(steps=10_000_000):
    notifier = init_notifier(project_name="OFC AI (Enhanced)")
    env = EnhancedSelfPlayEnv()
    
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        ent_coef=0.01,
    )
    
    env.set_latest(model)
    
    callback = EnhancedCallback(save_path="models/enhanced_ppo")
    
    try:
        steps_per_round = 100_000
        steps_done = 0
        while steps_done < steps:
            learn_steps = min(steps_per_round, steps - steps_done)
            model.learn(total_timesteps=learn_steps, callback=callback, reset_num_timesteps=False)
            steps_done += learn_steps
            
            # カリキュラム・フィードバック
            metrics = {
                'step': steps_done,
                'foul_rate': (callback.fouls / max(1, callback.total_games)) * 100,
                'win_rate': (callback.wins / max(1, callback.total_games)) * 100,
                'avg_score': np.mean(callback.episode_rewards) if callback.episode_rewards else 0
            }
            curriculum = get_curriculum_manager()
            curriculum.evaluate_and_progress(metrics)
            
            env.add_to_pool(model)
            env.set_latest(model)
    except KeyboardInterrupt:
        print("Training interrupted.")
    
    model.save("models/enhanced_ppo_final")
    if notifier: notifier.send_complete({"status": "completed"})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10_000_000)
    args = parser.parse_args()
    train(steps=args.steps)
