"""
OFC Pineapple AI - Phase 8.5 Self-Play Training
Ultimate Rules FL + 3人全員が同じモデルで自己対戦

特徴:
- Ultimate Rules FL (QQ=14, KK=15, AA=16, Trips=17)
- 3プレイヤー全員が同一モデル（定期的に過去モデルも混ぜる）
- 連続ゲーム: FL状態引き継ぎ + ボタンローテーション
- 高速学習: SubprocVecEnv対応

環境変数:
    NUM_ENVS: 並列環境数 (デフォルト: 4)
    DISCORD_WEBHOOK_URL: Discord通知用Webhook
"""

import os
import sys
import time
import random
import argparse
import numpy as np
from datetime import datetime
from collections import deque
from typing import Optional, List, Callable, Dict
import torch
torch.distributions.Distribution.set_default_validate_args(False)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

from ofc_3max_env import OFC3MaxEnv
from notifier import TrainingNotifier
from cloud_storage import init_cloud_storage

NUM_ENVS = int(os.getenv("NUM_ENVS", "4"))
DISCORD_WEBHOOK_URL = os.getenv(
    "DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1461388510140301315/Ofc9ok3IDLgRhFR7Oa2iA4-CHHL9uFvFbIPT9G9tbA5X5r-wG-XMbnkg7ubocLJJQ5Pf"
)


class SelfPlayEnv(gym.Env):
    """
    Self-Play環境: 3プレイヤー全員が同じポリシーで対戦

    Ultimate Rules FL + 連続ゲーム + ボタンローテーション
    """

    def __init__(self, env_id: int = 0):
        super().__init__()
        self.env = OFC3MaxEnv(
            enable_fl_turns=True,
            continuous_games=True
        )
        self.env_id = env_id
        self.learning_agent = "player_0"

        self.observation_space = self.env.observation_space(self.learning_agent)
        self.action_space = self.env.action_space(self.learning_agent)
        self.current_reward = 0

        # 統計
        self.total_games = 0
        self.fl_entries = 0
        self.fl_stays = 0
        self.high_score_games = 0
        self.fl_14_count = 0
        self.fl_15_count = 0
        self.fl_16_count = 0
        self.fl_17_count = 0

    def set_model(self, model):
        """推論用モデルを設定"""
        self.model = model

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        self.current_reward = 0
        self._play_opponents()
        return self.env.observe(self.learning_agent), {}

    def step(self, action):
        # 学習エージェントが終了している場合
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
            fl_cards = self.env.fl_cards_count.get(self.learning_agent, 0)

            if entered_fl:
                self.fl_entries += 1
            if stayed_fl:
                self.fl_stays += 1
            if score >= 15:
                self.high_score_games += 1

            # FL枚数統計
            if fl_cards == 14:
                self.fl_14_count += 1
            elif fl_cards == 15:
                self.fl_15_count += 1
            elif fl_cards == 16:
                self.fl_16_count += 1
            elif fl_cards == 17:
                self.fl_17_count += 1

            info = {
                'score': score,
                'royalty': res.get_royalty(0),
                'fouled': res.is_fouled(0),
                'entered_fl': entered_fl,
                'stayed_fl': stayed_fl,
                'fl_cards': fl_cards,
                'win': score > 0,
                'loss': score < 0,
                'high_score': score >= 15,
            }
        return obs, reward, terminated, truncated, info

    def _play_opponents(self):
        """対戦相手をプレイ（Self-Play: 同じモデル使用）"""
        max_iterations = 100

        for _ in range(max_iterations):
            if all(self.env.terminations.values()):
                break

            if self.env.agent_selection == self.learning_agent:
                if self.env.terminations.get(self.learning_agent, False):
                    self.env.step(None)
                    continue
                break

            agent = self.env.agent_selection

            if self.env.terminations.get(agent, False):
                self.env.step(None)
                continue

            valid_actions = self.env.get_valid_actions(agent)
            if not valid_actions:
                self.env.step(None)
                continue

            # Self-Play: 同じモデルで推論
            action = self._get_selfplay_action(agent, valid_actions)
            self.env.step(action)

    def _get_selfplay_action(self, agent: str, valid_actions: List[int]) -> int:
        """Self-Playアクションを取得"""
        if not hasattr(self, 'model') or self.model is None:
            return random.choice(valid_actions)

        try:
            obs = self.env.observe(agent)
            mask = self.env.action_masks(agent)
            action, _ = self.model.predict(obs, action_masks=mask, deterministic=False)
            return int(action)
        except Exception as e:
            return random.choice(valid_actions)

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks(self.learning_agent)


def mask_fn(env: SelfPlayEnv) -> np.ndarray:
    return env.action_masks()


def make_env(env_id: int) -> Callable:
    def _init():
        env = SelfPlayEnv(env_id=env_id)
        env = ActionMasker(env, mask_fn)
        return env
    return _init


class SelfPlayCallback(BaseCallback):
    """Self-Play学習用コールバック"""

    def __init__(
        self,
        save_path: str,
        notifier: Optional[TrainingNotifier],
        cloud_storage,
        total_timesteps: int = 50_000_000,
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

        # 統計
        self.scores = deque(maxlen=1000)
        self.royalties = deque(maxlen=1000)
        self.fouls = deque(maxlen=1000)
        self.fl_entries = deque(maxlen=1000)
        self.fl_stays = deque(maxlen=1000)
        self.high_scores = deque(maxlen=1000)
        self.fl_cards_dist = {14: 0, 15: 0, 16: 0, 17: 0}
        self.total_games = 0
        self.wins = 0
        self.losses = 0

    def _on_training_start(self):
        self.last_report = (self.num_timesteps // self.report_freq) * self.report_freq
        self.last_update = (self.num_timesteps // self.update_freq) * self.update_freq
        self.start_time = time.time()

        # 環境にモデルを設定
        self._update_env_models()

    def _update_env_models(self):
        """全環境のモデルを更新"""
        for env in self.training_env.envs:
            if hasattr(env, 'env') and hasattr(env.env, 'set_model'):
                env.env.set_model(self.model)

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
            self._update_env_models()

        return True

    def _record_game(self, info: dict):
        score = info['score']
        self.scores.append(score)
        self.royalties.append(info.get('royalty', 0))
        self.fouls.append(1.0 if info.get('fouled', False) else 0.0)
        self.fl_entries.append(1.0 if info.get('entered_fl', False) else 0.0)
        self.fl_stays.append(1.0 if info.get('stayed_fl', False) else 0.0)
        self.high_scores.append(1.0 if info.get('high_score', False) else 0.0)

        fl_cards = info.get('fl_cards', 0)
        if fl_cards in self.fl_cards_dist:
            self.fl_cards_dist[fl_cards] += 1

        self.total_games += 1

        if info.get('win', False):
            self.wins += 1
        elif info.get('loss', False):
            self.losses += 1

    def _send_report(self):
        elapsed = time.time() - self.start_time
        fps = self.num_timesteps / elapsed if elapsed > 0 else 0

        foul_rate = np.mean(self.fouls) * 100 if self.fouls else 0
        mean_royalty = np.mean(self.royalties) if self.royalties else 0
        mean_score = np.mean(self.scores) if self.scores else 0
        fl_entry_rate = np.mean(self.fl_entries) * 100 if self.fl_entries else 0
        fl_stay_rate = np.mean(self.fl_stays) * 100 if self.fl_stays else 0
        high_score_rate = np.mean(self.high_scores) * 100 if self.high_scores else 0

        total_fl = sum(self.fl_cards_dist.values())
        fl_dist_str = ""
        if total_fl > 0:
            fl_dist_str = f"QQ:{self.fl_cards_dist[14]} KK:{self.fl_cards_dist[15]} AA:{self.fl_cards_dist[16]} Trips:{self.fl_cards_dist[17]}"

        print(f"\n[Step {self.num_timesteps:,}] Phase 8.5 Self-Play (Ultimate Rules)")
        print(f"  Games: {self.total_games:,} | FPS: {fps:.0f}")
        print(f"  Foul Rate: {foul_rate:.1f}% | Mean Score: {mean_score:+.2f}")
        print(f"  Mean Royalty: {mean_royalty:.2f}")
        print(f"  FL Entry: {fl_entry_rate:.1f}% | FL Stay: {fl_stay_rate:.1f}%")
        print(f"  High Score (>=15): {high_score_rate:.1f}%")
        if fl_dist_str:
            print(f"  FL Distribution: {fl_dist_str}")

        if self.notifier:
            metrics = {
                'games': self.total_games,
                'foul_rate': foul_rate,
                'mean_score': mean_score,
                'mean_royalty': mean_royalty,
                'fl_entry_rate': fl_entry_rate,
                'fl_stay_rate': fl_stay_rate,
                'high_score_rate': high_score_rate,
                'fps': fps,
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
        checkpoint_dir = os.path.dirname(self.save_path)
        if not checkpoint_dir:
            checkpoint_dir = "."

        files = glob.glob(os.path.join(checkpoint_dir, "p85_selfplay_*.zip"))
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


def train_phase85_selfplay(
    total_timesteps: int = 50_000_000,
    test_mode: bool = False,
    num_envs: int = None
):
    """Phase 8.5 Self-Play Training"""
    if num_envs is None:
        num_envs = NUM_ENVS

    if test_mode:
        total_timesteps = 20_000
        num_envs = 2
        print("[TEST MODE]")

    print("=" * 60)
    print("OFC Pineapple AI - Phase 8.5: Self-Play Training")
    print("=" * 60)
    print(f"Parallel Envs: {num_envs}")
    print(f"Total Steps: {total_timesteps:,}")
    print(f"Features:")
    print(f"  - Ultimate Rules FL (QQ=14, KK=15, AA=16, Trips=17)")
    print(f"  - 3-Player Self-Play (same model)")
    print(f"  - Continuous Games + Button Rotation")
    print()

    cloud_storage = init_cloud_storage()
    notifier = TrainingNotifier(
        discord_webhook=DISCORD_WEBHOOK_URL,
        project_name=f"Phase 8.5: Self-Play Ultimate ({num_envs} envs)"
    )

    print(f"[*] Creating {num_envs} environments...")
    env = DummyVecEnv([make_env(i) for i in range(num_envs)])

    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    import glob

    p85_checkpoints = glob.glob(os.path.join(save_dir, "p85_selfplay_*.zip"))

    latest_checkpoint = None
    is_resume = False

    if p85_checkpoints:
        latest_checkpoint = max(p85_checkpoints, key=lambda f: int(f.split('_')[-1].replace('.zip', '')))
        print(f"[*] Resuming from: {latest_checkpoint}")
        model = MaskablePPO.load(latest_checkpoint, env=env)
        is_resume = True
    else:
        print("[*] Starting fresh training with Ultimate Rules observation space")
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            gamma=0.99,
            n_steps=2048,
            batch_size=256,
            ent_coef=0.01,
            tensorboard_log="./logs/phase85_selfplay/"
        )

    # 環境にモデルを設定
    for e in env.envs:
        if hasattr(e, 'env') and hasattr(e.env, 'set_model'):
            e.env.set_model(model)

    callback = SelfPlayCallback(
        "models/p85_selfplay",
        notifier,
        cloud_storage,
        total_timesteps=total_timesteps
    )

    if notifier and not test_mode:
        notifier.send_start({
            'timesteps': total_timesteps,
            'num_envs': num_envs,
            'features': 'Ultimate Rules FL + Self-Play',
            'lr': 3e-4,
        })

    if is_resume and latest_checkpoint:
        try:
            checkpoint_name = os.path.basename(latest_checkpoint)
            current_steps = int(checkpoint_name.split('_')[-1].replace('.zip', ''))
        except:
            current_steps = 0
        remaining_steps = max(0, total_timesteps - current_steps)
        reset_num_timesteps = False
        print(f"[*] Resuming from {current_steps:,}. Remaining: {remaining_steps:,}")
    else:
        remaining_steps = total_timesteps
        reset_num_timesteps = True

    print(f"\nStarting Self-Play Training (Goal: {total_timesteps:,} steps)")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=remaining_steps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps
        )

        if notifier and not test_mode:
            notifier.send_complete({
                'total_steps': total_timesteps,
                'total_games': callback.total_games,
                'foul_rate': np.mean(callback.fouls) * 100 if callback.fouls else 0,
                'mean_royalty': np.mean(callback.royalties) if callback.royalties else 0,
                'fl_entry_rate': np.mean(callback.fl_entries) * 100 if callback.fl_entries else 0,
            })

    except KeyboardInterrupt:
        print("\n[!] Training interrupted")
        model.save("models/p85_selfplay_interrupted")
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()
        if notifier:
            notifier.send_error(str(e), traceback.format_exc())
        raise
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 8.5 Self-Play Training")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--steps", type=int, default=50_000_000)
    parser.add_argument("--envs", type=int, default=None)

    args = parser.parse_args()

    train_phase85_selfplay(
        total_timesteps=args.steps,
        test_mode=args.test_mode,
        num_envs=args.envs
    )
