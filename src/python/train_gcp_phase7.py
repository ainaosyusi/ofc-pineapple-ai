"""
OFC Pineapple AI - Phase 7 Cloud Training (AWS/GCP両対応)
MCTSを教師、NNを生徒とした実力強化用学習スクリプト

環境変数:
    CLOUD_PROVIDER: "s3" or "gcs"
    GCS_BUCKET / S3_BUCKET: バケット名
    DISCORD_WEBHOOK_URL: Discord通知用Webhook
"""

import os
import sys
import time
import random
import numpy as np
import copy
from datetime import datetime
from collections import deque
from typing import Optional, List, Union
import torch
torch.distributions.Distribution.set_default_validate_args(False)

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym

from ofc_3max_env import OFC3MaxEnv
from mcts_agent import MCTSFLAgent, MCTSConfig
from notifier import TrainingNotifier
from cloud_storage import init_cloud_storage

def load_manual_weights(model: MaskablePPO, zip_path: str):
    """
    NumPy 2.xで保存されたSB3モデルを NumPy 1.x環境でロードするための特殊関数
    zipから直接 policy.pth を抽出して PyTorch でロードする
    """
    import zipfile
    import io
    import torch

    if not os.path.exists(zip_path):
        print(f"Warning: {zip_path} not found.")
        return False

    print(f"[Compatibility] Manual loading weights from {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with zip_ref.open('policy.pth') as f:
                state_dict = torch.load(io.BytesIO(f.read()), map_location='cpu')
                model.policy.load_state_dict(state_dict, strict=False)
        print("[Compatibility] Successfully loaded policy weights.")
        return True
    except Exception as e:
        print(f"[Compatibility] Failed to load weights: {e}")
        return False


# Discord Webhook (環境変数からも取得可能)
DISCORD_WEBHOOK_URL = os.getenv(
    "DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1461388510140301315/Ofc9ok3IDLgRhFR7Oa2iA4-CHHL9uFvFbIPT9G9tbA5X5r-wG-XMbnkg7ubocLJJQ5Pf"
)


class SuperhumanOpponentManager:
    """
    Self-Play + MCTS Teacher 管理クラス
    """
    def __init__(self, model_pool_size: int = 10):
        self.model_pool: List[dict] = []
        self.pool_size = model_pool_size
        self.mcts_teacher: Optional[MCTSFLAgent] = None

    def add_model(self, model: MaskablePPO):
        weights = {k: v.cpu().clone() for k, v in model.policy.state_dict().items()}
        self.model_pool.append(weights)
        if len(self.model_pool) > self.pool_size:
            self.model_pool.pop(0)

    def init_teacher(self, model):
        """MCTS教師エージェントを現在のモデルで初期化"""
        config = MCTSConfig(num_simulations=100, fl_weight=0.7)
        self.mcts_teacher = MCTSFLAgent(model=model, config=config)

    def get_opponent_action(self, agent_name, wrapper, current_model):
        """対戦相手を選択してアクションを決定 (80%最新, 10%過去, 10%MCTS)"""
        p = random.random()

        if p < 0.8:
            return self._predict_ppo(current_model, wrapper, agent_name)
        elif p < 0.9 and self.model_pool:
            weights = random.choice(self.model_pool)
            with torch.no_grad():
                original_weights = {k: v.clone() for k, v in current_model.policy.state_dict().items()}
                current_model.policy.load_state_dict(weights)
                action = self._predict_ppo(current_model, wrapper, agent_name)
                current_model.policy.load_state_dict(original_weights)
            return action
        elif self.mcts_teacher:
            action = self.mcts_teacher.select_action(
                wrapper.env.engine,
                wrapper.env.agent_name_mapping[agent_name],
                simulations=50
            )
            return action
        else:
            valid = wrapper.env.get_valid_actions(agent_name)
            return random.choice(valid) if valid else 0

    def _predict_ppo(self, model, wrapper, agent):
        obs = wrapper.env.observe(agent)
        action_masks = wrapper.env.action_masks(agent)

        for k, v in obs.items():
            if np.isnan(v).any():
                print(f"[Warning] NaN detected in observation {k} for {agent}")

        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        return action


class Phase7GymWrapper(gym.Env):
    """3人対戦環境のシングルエージェントラッパー"""
    def __init__(self, opponent_manager: SuperhumanOpponentManager):
        super().__init__()
        self.env = OFC3MaxEnv()
        self.opponent_manager = opponent_manager
        self.learning_agent = "player_0"
        self.current_model: Optional[MaskablePPO] = None

        self.observation_space = self.env.observation_space(self.learning_agent)
        self.action_space = self.env.action_space(self.learning_agent)
        self.current_reward = 0

    def set_current_model(self, model):
        self.current_model = model
        if not self.opponent_manager.mcts_teacher:
            self.opponent_manager.init_teacher(model)

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        self.current_reward = 0
        self._play_until_learning_agent()
        return self.env.observe(self.learning_agent), {}

    def step(self, action):
        self.env.step(action)
        self._play_until_learning_agent()

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
                'fouled': res.is_fouled(0),
                'entered_fl': res.entered_fl(0)
            }
        return obs, reward, terminated, truncated, info

    def _play_until_learning_agent(self):
        while not all(self.env.terminations.values()) and self.env.agent_selection != self.learning_agent:
            agent = self.env.agent_selection
            if self.env.terminations.get(agent, False):
                self.env.step(None)
                continue

            action = self.opponent_manager.get_opponent_action(agent, self, self.current_model)
            self.env.step(action)

    def action_masks(self):
        return self.env.action_masks(self.learning_agent)


class Phase7Callback(BaseCallback):
    def __init__(self, save_path, notifier, cloud_storage, opponent_manager, update_freq=200_000, report_freq=100_000):
        super().__init__()
        self.save_path = save_path
        self.notifier = notifier
        self.cloud_storage = cloud_storage
        self.opponent_manager = opponent_manager
        self.update_freq = update_freq
        self.report_freq = report_freq
        self.last_update = 0
        self.last_report = 0
        self.start_time = time.time()
        self.total_games = 0

        self.scores = deque(maxlen=100)
        self.fouls = deque(maxlen=100)
        self.fl_entries = deque(maxlen=100)

    def _on_training_start(self):
        self.last_report = (self.num_timesteps // self.report_freq) * self.report_freq
        self.last_update = (self.num_timesteps // self.update_freq) * self.update_freq
        self.start_time = time.time()
        print(f"[*] Callback initialized: last_report={self.last_report}, last_update={self.last_update}")

    def _on_step(self):
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'score' in info:
                    self.scores.append(info['score'])
                    self.fouls.append(1.0 if info['fouled'] else 0.0)
                    self.fl_entries.append(1.0 if info['entered_fl'] else 0.0)
                    self.total_games += 1

        # 定期レポート
        if self.num_timesteps - self.last_report >= self.report_freq:
            self.last_report = self.num_timesteps
            if self.notifier:
                metrics = {
                    'games': self.total_games,
                    'mean_score': np.mean(self.scores) if self.scores else 0,
                    'foul_rate': np.mean(self.fouls) * 100 if self.fouls else 0,
                    'fl_rate': np.mean(self.fl_entries) * 100 if self.fl_entries else 0,
                    'fps': self.n_calls / (time.time() - self.start_time) if (time.time() - self.start_time) > 0 else 0
                }
                self.notifier.send_progress(self.num_timesteps, 20_000_000, metrics)

        # チェックポイント保存 & 対戦相手プール更新
        if self.num_timesteps - self.last_update >= self.update_freq:
            self.last_update = self.num_timesteps
            self.opponent_manager.add_model(self.model)

            path = f"{self.save_path}_{self.num_timesteps}"
            self.model.save(path)

            if self.notifier:
                self.notifier.send_checkpoint(f"{path}.zip", self.num_timesteps)
            if self.cloud_storage and self.cloud_storage.enabled:
                self.cloud_storage.upload_checkpoint(f"{path}.zip", step=self.num_timesteps)

            self._cleanup_old_checkpoints()

        return True

    def _cleanup_old_checkpoints(self):
        """古いチェックポイントを削除してディスク容量を節約"""
        import glob
        checkpoint_dir = os.path.dirname(self.save_path)
        if not checkpoint_dir:
            checkpoint_dir = "."

        files = glob.glob(os.path.join(checkpoint_dir, "p7_mcts_*.zip"))
        if not files:
            return

        def get_step(f):
            try:
                return int(f.split('_')[-1].replace('.zip', ''))
            except:
                return 0

        sorted_files = sorted(files, key=get_step)

        for f in sorted_files[:-2]:
            step = get_step(f)
            if step % 1_000_000 == 0:
                continue

            try:
                os.remove(f)
            except Exception as e:
                print(f"[Cleanup] Error removing {f}: {e}")


def train_phase7():
    """
    Phase 7 学習メイン関数
    AWS/GCP両対応
    """
    print("=" * 60)
    print("OFC Pineapple AI - Phase 7: Cloud Training")
    print("=" * 60)

    # 1. 準備
    base_model_path = "models/phase4/ofc_phase4_joker_20260115_190744_10500000_steps.zip"
    opponent_manager = SuperhumanOpponentManager()
    env_wrapper = Phase7GymWrapper(opponent_manager)

    # クラウドストレージ初期化（環境変数から自動検出）
    cloud_storage = init_cloud_storage()
    print(f"[*] Cloud Storage Provider: {cloud_storage.provider or 'None'}")
    print(f"[*] Cloud Storage Enabled: {cloud_storage.enabled}")

    # Notifier
    notifier = TrainingNotifier(
        discord_webhook=DISCORD_WEBHOOK_URL,
        project_name="Phase 7: Cloud Distillation"
    )

    # 2. モデルの初期化
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    import glob
    latest_checkpoint = None
    checkpoints = glob.glob(os.path.join(save_dir, "p7_mcts_*.zip"))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda f: int(f.split('_')[-1].replace('.zip', '')))
        print(f"[*] Found existing checkpoint: {latest_checkpoint}. Resuming...")
        model = MaskablePPO.load(latest_checkpoint, env=env_wrapper)
    else:
        print("[*] No existing checkpoint found. Initializing fresh model.")
        model = MaskablePPO(
            "MultiInputPolicy",
            env_wrapper,
            verbose=1,
            learning_rate=1e-4,
            gamma=0.99,
            n_steps=2048,
            batch_size=128,
            tensorboard_log="./logs/phase7_cloud/"
        )
        if os.path.exists(base_model_path):
            success = load_manual_weights(model, base_model_path)
            if not success:
                print("Starting from random weights due to base model load failure.")
        else:
            print("Base model not found. Starting from scratch.")

    env_wrapper.set_current_model(model)
    callback = Phase7Callback(
        "models/p7_mcts",
        notifier,
        cloud_storage,
        opponent_manager
    )

    # 開始通知
    if notifier:
        notifier.send_start({
            'timesteps': 20_000_000,
            'lr': 1e-4,
            'opponent_update': 200_000,
            'cloud_provider': cloud_storage.provider or "local"
        })

    # 3. 学習
    total_steps = 20_000_000
    if latest_checkpoint:
        try:
            current_steps = int(latest_checkpoint.split('_')[-1].replace('.zip', ''))
        except:
            current_steps = 0
        remaining_steps = total_steps - current_steps
        reset_num_timesteps = False
        print(f"[*] Resuming from step {current_steps:,}. Remaining: {remaining_steps:,} steps.")
    else:
        remaining_steps = total_steps
        reset_num_timesteps = True
        print(f"[*] Starting fresh training for {total_steps:,} steps.")

    print(f"\nStarting Distillation Training (Goal: {total_steps:,} steps)")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=remaining_steps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps
        )

        # 完了通知
        if notifier:
            notifier.send_complete({
                'total_steps': total_steps,
                'total_games': callback.total_games,
                'foul_rate': np.mean(callback.fouls) * 100 if callback.fouls else 0,
                'model_path': f"models/p7_mcts_{total_steps}.zip"
            })

    except Exception as e:
        import traceback
        error_msg = str(e)
        tb = traceback.format_exc()
        print(f"[ERROR] Training failed: {error_msg}")
        if notifier:
            notifier.send_error(error_msg, tb)
        raise


if __name__ == "__main__":
    train_phase7()
