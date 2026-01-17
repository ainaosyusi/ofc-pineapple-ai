"""
OFC Pineapple AI - Phase 7 Parallel Training (AWS/GCP両対応)
SubprocVecEnvを使用した並列環境学習

環境変数:
    NUM_ENVS: 並列環境数 (デフォルト: 4)
    CLOUD_PROVIDER: "s3" or "gcs"
    GCS_BUCKET / S3_BUCKET: バケット名
    DISCORD_WEBHOOK_URL: Discord通知用Webhook
"""

import os
import sys
import time
import random
import numpy as np
from datetime import datetime
from collections import deque
from typing import Optional, List, Callable
import torch
torch.distributions.Distribution.set_default_validate_args(False)

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import gymnasium as gym

from ofc_3max_env import OFC3MaxEnv
from notifier import TrainingNotifier
from cloud_storage import init_cloud_storage

# 並列環境数
NUM_ENVS = int(os.getenv("NUM_ENVS", "4"))

# Discord Webhook
DISCORD_WEBHOOK_URL = os.getenv(
    "DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1461388510140301315/Ofc9ok3IDLgRhFR7Oa2iA4-CHHL9uFvFbIPT9G9tbA5X5r-wG-XMbnkg7ubocLJJQ5Pf"
)


def load_manual_weights(model: MaskablePPO, zip_path: str):
    """NumPy互換性のための手動重みロード"""
    import zipfile
    import io

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


class SimpleOpponentManager:
    """シンプルな対戦相手管理（過去モデルプール）"""
    def __init__(self, pool_size: int = 5):
        self.model_pool: List[dict] = []
        self.pool_size = pool_size

    def add_model(self, model: MaskablePPO):
        weights = {k: v.cpu().clone() for k, v in model.policy.state_dict().items()}
        self.model_pool.append(weights)
        if len(self.model_pool) > self.pool_size:
            self.model_pool.pop(0)

    def get_random_weights(self) -> Optional[dict]:
        if self.model_pool:
            return random.choice(self.model_pool)
        return None


class ParallelOFCEnv(gym.Env):
    """並列実行用のOFC環境ラッパー"""
    def __init__(self, env_id: int = 0):
        super().__init__()
        self.env = OFC3MaxEnv()
        self.env_id = env_id
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
                'fouled': res.is_fouled(0),
                'entered_fl': res.entered_fl(0)
            }
        return obs, reward, terminated, truncated, info

    def _play_opponents(self):
        """相手プレイヤーをランダムにプレイ"""
        while not all(self.env.terminations.values()) and self.env.agent_selection != self.learning_agent:
            agent = self.env.agent_selection
            if self.env.terminations.get(agent, False):
                self.env.step(None)
                continue

            valid_actions = self.env.get_valid_actions(agent)
            if valid_actions:
                action = random.choice(valid_actions)
            else:
                action = 0
            self.env.step(action)

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks(self.learning_agent)


def mask_fn(env: ParallelOFCEnv) -> np.ndarray:
    """ActionMasker用のマスク関数"""
    return env.action_masks()


def make_env(env_id: int) -> Callable:
    """環境作成関数"""
    def _init():
        env = ParallelOFCEnv(env_id=env_id)
        env = ActionMasker(env, mask_fn)
        return env
    return _init


class ParallelCallback(BaseCallback):
    """並列学習用コールバック"""
    def __init__(self, save_path, notifier, cloud_storage, opponent_manager,
                 update_freq=200_000, report_freq=100_000):
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

        self.scores = deque(maxlen=500)
        self.fouls = deque(maxlen=500)
        self.fl_entries = deque(maxlen=500)
        self.total_games = 0

    def _on_training_start(self):
        self.last_report = (self.num_timesteps // self.report_freq) * self.report_freq
        self.last_update = (self.num_timesteps // self.update_freq) * self.update_freq
        self.start_time = time.time()
        print(f"[*] Callback: last_report={self.last_report}, last_update={self.last_update}")

    def _on_step(self):
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if isinstance(info, dict) and 'score' in info:
                    self.scores.append(info['score'])
                    self.fouls.append(1.0 if info.get('fouled', False) else 0.0)
                    self.fl_entries.append(1.0 if info.get('entered_fl', False) else 0.0)
                    self.total_games += 1

        # 定期レポート
        if self.num_timesteps - self.last_report >= self.report_freq:
            self.last_report = self.num_timesteps
            elapsed = time.time() - self.start_time
            fps = self.num_timesteps / elapsed if elapsed > 0 else 0

            if self.notifier:
                metrics = {
                    'games': self.total_games,
                    'mean_score': np.mean(self.scores) if self.scores else 0,
                    'foul_rate': np.mean(self.fouls) * 100 if self.fouls else 0,
                    'fl_rate': np.mean(self.fl_entries) * 100 if self.fl_entries else 0,
                    'fps': fps
                }
                self.notifier.send_progress(self.num_timesteps, 20_000_000, metrics)

        # チェックポイント保存
        if self.num_timesteps - self.last_update >= self.update_freq:
            self.last_update = self.num_timesteps

            if self.opponent_manager:
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
        """古いチェックポイントを削除"""
        import glob
        checkpoint_dir = os.path.dirname(self.save_path)
        if not checkpoint_dir:
            checkpoint_dir = "."

        files = glob.glob(os.path.join(checkpoint_dir, "p7_parallel_*.zip"))
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
                print(f"[Cleanup] Error: {e}")


def train_parallel():
    """並列学習メイン関数"""
    print("=" * 60)
    print(f"OFC Pineapple AI - Phase 7: Parallel Training ({NUM_ENVS} envs)")
    print("=" * 60)

    # クラウドストレージ
    cloud_storage = init_cloud_storage()
    print(f"[*] Cloud Storage: {cloud_storage.provider or 'None'}")

    # Notifier
    notifier = TrainingNotifier(
        discord_webhook=DISCORD_WEBHOOK_URL,
        project_name=f"Phase 7: Parallel ({NUM_ENVS} envs)"
    )

    # 対戦相手管理
    opponent_manager = SimpleOpponentManager()

    # 並列環境作成
    print(f"[*] Creating {NUM_ENVS} parallel environments...")
    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])

    # モデル初期化/レジューム
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    import glob
    # まず並列用チェックポイントを探す
    checkpoints = glob.glob(os.path.join(save_dir, "p7_parallel_*.zip"))
    if not checkpoints:
        # なければ通常のPhase7チェックポイントを探す
        checkpoints = glob.glob(os.path.join(save_dir, "p7_mcts_*.zip"))

    latest_checkpoint = None
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda f: int(f.split('_')[-1].replace('.zip', '')))
        print(f"[*] Found checkpoint: {latest_checkpoint}. Resuming...")
        model = MaskablePPO.load(latest_checkpoint, env=env)
    else:
        print("[*] No checkpoint found. Starting fresh.")
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,
            gamma=0.99,
            n_steps=2048,
            batch_size=256,  # 並列環境用に増加
            tensorboard_log="./logs/phase7_parallel/"
        )

        # ベースモデルからの重み読み込み
        base_model_path = "models/phase4/ofc_phase4_joker_20260115_190744_10500000_steps.zip"
        if os.path.exists(base_model_path):
            load_manual_weights(model, base_model_path)

    callback = ParallelCallback(
        "models/p7_parallel",
        notifier,
        cloud_storage,
        opponent_manager
    )

    # 開始通知
    if notifier:
        notifier.send_start({
            'timesteps': 20_000_000,
            'lr': 1e-4,
            'num_envs': NUM_ENVS,
            'cloud_provider': cloud_storage.provider or "local"
        })

    # 学習
    total_steps = 20_000_000
    if latest_checkpoint:
        try:
            current_steps = int(latest_checkpoint.split('_')[-1].replace('.zip', ''))
        except:
            current_steps = 0
        remaining_steps = max(0, total_steps - current_steps)
        reset_num_timesteps = False
        print(f"[*] Resuming from {current_steps:,}. Remaining: {remaining_steps:,}")
    else:
        remaining_steps = total_steps
        reset_num_timesteps = True

    print(f"\nStarting Parallel Training (Goal: {total_steps:,} steps)")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=remaining_steps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps
        )

        if notifier:
            notifier.send_complete({
                'total_steps': total_steps,
                'total_games': callback.total_games,
                'foul_rate': np.mean(callback.fouls) * 100 if callback.fouls else 0,
                'model_path': f"models/p7_parallel_{total_steps}.zip"
            })

    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        if notifier:
            notifier.send_error(str(e), traceback.format_exc())
        raise
    finally:
        env.close()


if __name__ == "__main__":
    train_parallel()
