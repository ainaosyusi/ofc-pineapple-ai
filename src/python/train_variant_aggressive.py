"""
OFC Pineapple AI - Aggressive Variant Training
攻撃的プレイスタイルの学習

特徴:
- Fantasyland突入に高いボーナス報酬
- ロイヤリティ獲得に追加報酬
- リスクを取る戦略を学習

Phase 7を基盤として、より攻撃的なプレイスタイルを獲得する。
"""

import os
import sys
import time
import random
import argparse
import numpy as np
from datetime import datetime
from collections import deque
from typing import Optional, Callable, Dict, Any
import torch
torch.distributions.Distribution.set_default_validate_args(False)

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

NUM_ENVS = int(os.getenv("NUM_ENVS", "4"))
DISCORD_WEBHOOK_URL = os.getenv(
    "DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1462113917571694632/iQ26kCzDmJ-DLA9TD_IRYNh4_TWc3UOxnVEa2-890B66mEYvrm7jufLOsMs_ZaATq2bb"
)


def load_manual_weights(model: MaskablePPO, zip_path: str) -> bool:
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


class AggressiveOFCEnv(gym.Env):
    """
    攻撃的プレイスタイル用の報酬修正環境

    報酬設計:
    - 基本報酬: ゲームスコア
    - FL突入ボーナス: +25 (通常+15から増加)
    - ロイヤリティボーナス: royalty × 1.5
    - 高役ボーナス: SF/Quadsに追加報酬
    """

    def __init__(self, env_id: int = 0):
        super().__init__()
        self.env = OFC3MaxEnv()
        self.env_id = env_id
        self.learning_agent = "player_0"

        self.observation_space = self.env.observation_space(self.learning_agent)
        self.action_space = self.env.action_space(self.learning_agent)

        # 攻撃的報酬パラメータ
        self.fl_bonus = 25.0  # FL突入ボーナス (通常15)
        self.royalty_multiplier = 1.5  # ロイヤリティ倍率
        self.high_hand_bonus = 10.0  # SF/Quads追加報酬

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.env.reset(seed=seed)
        self._play_opponents()

        obs = self.env.observe(self.learning_agent)
        return obs, {}

    def step(self, action):
        self.env.step(action)
        self._play_opponents()

        obs = self.env.observe(self.learning_agent)
        terminated = self.env.terminations.get(self.learning_agent, False)
        truncated = False

        reward = 0.0
        info = {}

        if terminated:
            result = self.env.engine.result()
            base_score = result.get_score(0)
            royalty = result.get_royalty(0)
            fouled = result.is_fouled(0)
            entered_fl = result.entered_fl(0)

            # 攻撃的報酬設計
            reward = base_score

            if not fouled:
                # ロイヤリティボーナス (1.5倍)
                reward += royalty * self.royalty_multiplier

                # FL突入ボーナス (強化)
                if entered_fl:
                    reward += self.fl_bonus

            info = {
                'score': base_score,
                'royalty': royalty,
                'fouled': fouled,
                'entered_fl': entered_fl,
                'variant': 'aggressive',
                'win': base_score > 0,
                'loss': base_score < 0,
                'draw': base_score == 0
            }

        return obs, reward, terminated, truncated, info

    def _play_opponents(self):
        """相手のターンをランダムで処理"""
        while not all(self.env.terminations.values()) and \
              self.env.agent_selection != self.learning_agent:
            agent = self.env.agent_selection
            if self.env.terminations.get(agent, False):
                self.env.step(None)
                continue
            valid_actions = self.env.get_valid_actions(agent)
            action = random.choice(valid_actions) if valid_actions else 0
            self.env.step(action)

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks(self.learning_agent)


def mask_fn(env):
    return env.action_masks()


def make_env(env_id: int):
    def _init():
        env = AggressiveOFCEnv(env_id=env_id)
        env = ActionMasker(env, mask_fn)
        return env
    return _init


class AggressiveCallback(BaseCallback):
    """攻撃的バリアント用コールバック"""

    def __init__(
        self,
        save_path: str,
        notifier: Optional[TrainingNotifier],
        cloud_storage,
        total_timesteps: int = 20_000_000,
        update_freq: int = 200_000,
        report_freq: int = 100_000
    ):
        super().__init__()
        self.save_path = save_path
        self.notifier = notifier
        self.cloud_storage = cloud_storage
        self.total_timesteps = total_timesteps
        self.update_freq = update_freq
        self.report_freq = report_freq
        self.last_update = 0
        self.last_report = 0
        self.start_time = time.time()

        self.scores = deque(maxlen=500)
        self.royalties = deque(maxlen=500)
        self.fouls = deque(maxlen=500)
        self.fl_entries = deque(maxlen=500)
        self.total_games = 0
        self.wins = 0
        self.losses = 0

    def _on_training_start(self):
        self.last_report = (self.num_timesteps // self.report_freq) * self.report_freq
        self.last_update = (self.num_timesteps // self.update_freq) * self.update_freq
        self.start_time = time.time()

    def _on_step(self):
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if isinstance(info, dict) and 'score' in info:
                    self._record_game(info)

        if self.num_timesteps - self.last_report >= self.report_freq:
            self.last_report = self.num_timesteps
            self._send_report()

        if self.num_timesteps - self.last_update >= self.update_freq:
            self.last_update = self.num_timesteps
            self._save_checkpoint()

        return True

    def _record_game(self, info: dict):
        self.scores.append(info['score'])
        self.royalties.append(info.get('royalty', 0))
        self.fouls.append(1.0 if info.get('fouled', False) else 0.0)
        self.fl_entries.append(1.0 if info.get('entered_fl', False) else 0.0)
        self.total_games += 1
        if info.get('win'):
            self.wins += 1
        elif info.get('loss'):
            self.losses += 1

    def _send_report(self):
        elapsed = time.time() - self.start_time
        fps = self.num_timesteps / elapsed if elapsed > 0 else 0

        foul_rate = np.mean(self.fouls) * 100 if self.fouls else 0
        mean_royalty = np.mean(self.royalties) if self.royalties else 0
        mean_score = np.mean(self.scores) if self.scores else 0
        fl_rate = np.mean(self.fl_entries) * 100 if self.fl_entries else 0
        win_rate = self.wins / self.total_games * 100 if self.total_games > 0 else 0

        print(f"\n[Step {self.num_timesteps:,}] AGGRESSIVE VARIANT")
        print(f"  Games: {self.total_games:,}")
        print(f"  Foul Rate: {foul_rate:.1f}%")
        print(f"  Mean Score: {mean_score:+.2f}")
        print(f"  Mean Royalty: {mean_royalty:.2f}")
        print(f"  FL Entry Rate: {fl_rate:.1f}%")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  FPS: {fps:.0f}")

        if self.notifier:
            metrics = {
                'games': self.total_games,
                'foul_rate': foul_rate,
                'mean_score': mean_score,
                'mean_royalty': mean_royalty,
                'fl_rate': fl_rate,
                'win_rate': win_rate,
                'fps': fps,
                'variant': 'AGGRESSIVE'
            }
            self.notifier.send_progress(self.num_timesteps, self.total_timesteps, metrics)

    def _save_checkpoint(self):
        path = f"{self.save_path}_{self.num_timesteps}"
        self.model.save(path)

        if self.notifier:
            self.notifier.send_checkpoint(f"{path}.zip", self.num_timesteps)
        if self.cloud_storage and self.cloud_storage.enabled:
            self.cloud_storage.upload_checkpoint(f"{path}.zip", step=self.num_timesteps)

        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        import glob
        checkpoints = sorted(glob.glob(f"{self.save_path}_*.zip"))
        milestones = {1_000_000, 5_000_000, 10_000_000, 15_000_000, 20_000_000}

        for cp in checkpoints[:-2]:
            step = int(cp.split('_')[-1].replace('.zip', ''))
            if step not in milestones:
                try:
                    os.remove(cp)
                except:
                    pass


def main():
    parser = argparse.ArgumentParser(description="Aggressive Variant Training")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--steps", type=int, default=20_000_000)
    args = parser.parse_args()

    total_timesteps = args.steps if args.test_mode else 20_000_000
    save_path = "models/aggressive"
    base_model_path = "models/p7_parallel_20000000.zip"

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs/aggressive", exist_ok=True)

    cloud_storage = init_cloud_storage()
    notifier = TrainingNotifier(
        discord_webhook=DISCORD_WEBHOOK_URL,
        project_name="OFC AI - AGGRESSIVE"
    )

    # 環境作成
    num_envs = 2 if args.test_mode else NUM_ENVS
    if num_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    else:
        env = DummyVecEnv([make_env(0)])

    # モデル作成
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./logs/aggressive"
    )

    # Phase 7の重みをロード
    if os.path.exists(base_model_path):
        load_manual_weights(model, base_model_path)
        print(f"[*] Loaded base model: {base_model_path}")

    # レジューム確認
    import glob
    checkpoints = sorted(glob.glob(f"{save_path}_*.zip"))
    remaining_steps = total_timesteps
    reset_num_timesteps = True

    if checkpoints:
        latest = checkpoints[-1]
        step = int(latest.split('_')[-1].replace('.zip', ''))
        if step < total_timesteps:
            load_manual_weights(model, latest)
            remaining_steps = total_timesteps - step
            reset_num_timesteps = False
            print(f"[*] Resuming from {latest}, remaining: {remaining_steps:,}")

    callback = AggressiveCallback(
        save_path=save_path,
        notifier=notifier,
        cloud_storage=cloud_storage,
        total_timesteps=total_timesteps
    )

    print(f"\nStarting AGGRESSIVE Variant Training (Goal: {total_timesteps:,} steps)")
    print("=" * 60)

    notifier.send_start({
        'timesteps': total_timesteps,
        'variant': 'AGGRESSIVE',
        'fl_bonus': 25,
        'royalty_multiplier': 1.5,
        'lr': 1e-4
    })

    model.learn(
        total_timesteps=remaining_steps,
        callback=callback,
        reset_num_timesteps=reset_num_timesteps,
        progress_bar=True
    )

    final_path = f"{save_path}_final"
    model.save(final_path)
    print(f"\n[*] Training complete! Final model: {final_path}.zip")


if __name__ == "__main__":
    main()
