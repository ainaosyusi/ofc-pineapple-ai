"""
OFC Pineapple AI - AWS Phase 3 Self-Play Training Script
EC2本番用のSelf-Play学習スクリプト

推奨設定:
- 総ステップ: 1000万 (10M)
- モデル保存: 20万ステップ毎
- Opponent Strategy: Latest vs Pool (80% Pool, 20% Latest)
- 報酬: Foul -30, Royalty直接加算, 勝利ボーナス +10
"""

import io
import os
import sys
import time
import numpy as np
from datetime import datetime
from collections import deque

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

import gymnasium as gym
from multi_ofc_env import OFCMultiAgentEnv
from notifier import TrainingNotifier, init_notifier, get_notifier


class Phase3Callback(BaseCallback):
    """
    Phase 3 Self-Play用コールバック
    定期的な統計記録、モデル保存、通知
    """
    
    def __init__(
        self, 
        save_path: str,
        save_freq: int = 200_000,
        log_freq: int = 10_000,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.log_freq = log_freq
        
        # 統計
        self.episode_rewards = deque(maxlen=1000)
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.fouls = 0
        self.total_games = 0
        self.total_timesteps = 10_000_000  # デフォルト
        self.notify_freq = 100_000  # 通知間隔
        self.last_notify_step = 0
        self.start_time = time.time()
        
    def _on_step(self) -> bool:
        # エピソード終了時の処理
        for info in self.locals.get('infos', []):
            if 'final_score' in info:
                score = info['final_score']
                self.episode_rewards.append(score)
                self.total_games += 1
                
                if score > 0:
                    self.wins += 1
                elif score < 0:
                    self.losses += 1
                else:
                    self.draws += 1
                
                if info.get('fouled', False):
                    self.fouls += 1
        
        # ログ出力
        if self.n_calls % self.log_freq == 0 and len(self.episode_rewards) > 0:
            self._log_stats()
        
        # モデル保存
        if self.n_calls % self.save_freq == 0:
            self._save_model()
        
        # Discord/Slack通知
        if self.n_calls - self.last_notify_step >= self.notify_freq:
            self._send_progress_notification()
            self.last_notify_step = self.n_calls
        
        return True
    
    def _log_stats(self):
        recent = list(self.episode_rewards)[-100:]
        mean_reward = np.mean(recent)
        foul_rate = self.fouls / max(1, self.total_games) * 100
        win_rate = self.wins / max(1, self.total_games) * 100
        
        print(f"\n[Step {self.n_calls}]")
        print(f"  Games: {self.total_games}")
        print(f"  Foul Rate (last 100): {foul_rate:.1f}%")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Mean Score (last 100): {mean_reward:.2f}")
        print("-" * 30)
    
    def _save_model(self):
        path = f"{self.save_path}_{self.n_calls}_steps"
        self.model.save(path)
        print(f"[Checkpoint] Model saved to {path}.zip")
        
        # チェックポイント通知
        notifier = get_notifier()
        if notifier and notifier.enabled:
            notifier.send_checkpoint(f"{path}.zip", self.n_calls)
    
    def _send_progress_notification(self):
        """Discord/Slackへ進捗通知"""
        notifier = get_notifier()
        if not notifier or not notifier.enabled:
            return
        
        elapsed = time.time() - self.start_time
        fps = self.n_calls / max(1, elapsed)
        foul_rate = self.fouls / max(1, self.total_games) * 100
        win_rate = self.wins / max(1, self.total_games) * 100
        recent = list(self.episode_rewards)[-100:]
        mean_score = np.mean(recent) if recent else 0
        
        notifier.send_progress(
            step=self.n_calls,
            total_steps=self.total_timesteps,
            metrics={
                'games': self.total_games,
                'foul_rate': foul_rate,
                'win_rate': win_rate,
                'mean_score': mean_score,
                'fps': fps
            }
        )


class SelfPlayEnvPhase3(gym.Env):
    """
    Phase 3 Self-Play用環境
    - Latest vs Pool (80/20) 戦略
    - 勝利ボーナス報酬
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, seed=None, pool_ratio: float = 0.8, win_bonus: float = 10.0):
        super().__init__()
        self.env = OFCMultiAgentEnv()
        self._seed = seed
        self.pool_ratio = pool_ratio  # Pool vs Latest の確率
        self.win_bonus = win_bonus
        
        self.opponent_pool = []
        self.latest_opponent = None
        self.active_opponent = None
        
        # Gym互換のスペース
        self.observation_space = self.env.observation_space("player_0")
        self.action_space = self.env.action_space("player_0")
    
    def add_to_pool(self, model):
        """モデルをプールに追加"""
        buffer = io.BytesIO()
        model.save(buffer)
        buffer.seek(0)
        cloned_model = MaskablePPO.load(buffer)
        self.opponent_pool.append(cloned_model)
        
        # プールサイズ制限
        if len(self.opponent_pool) > 15:
            self.opponent_pool.pop(0)
    
    def set_latest(self, model):
        """最新モデルを設定"""
        buffer = io.BytesIO()
        model.save(buffer)
        buffer.seek(0)
        self.latest_opponent = MaskablePPO.load(buffer)
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed = seed
        elif self._seed is None:
            self._seed = np.random.randint(0, 2**32)
        
        self.env.reset(seed=self._seed)
        self._seed = (self._seed + 1) % (2**32)
        
        # Latest vs Pool (80/20) で対戦相手を選択
        if self.opponent_pool and np.random.random() < self.pool_ratio:
            self.active_opponent = np.random.choice(self.opponent_pool)
        elif self.latest_opponent is not None:
            self.active_opponent = self.latest_opponent
        elif self.opponent_pool:
            self.active_opponent = np.random.choice(self.opponent_pool)
        else:
            self.active_opponent = None
        
        obs = self.env.observe("player_0")
        return obs, {}
    
    def step(self, action):
        info = {}
        
        if self.env.agent_selection != "player_0":
            self._run_opponent_turn()
        
        if all(self.env.terminations.values()):
            return self._terminal_step()
        
        self.env.step(action)
        
        if all(self.env.terminations.values()):
            return self._terminal_step()
        
        while self.env.agent_selection == "player_1" and not all(self.env.terminations.values()):
            self._run_opponent_turn()
        
        done = all(self.env.terminations.values())
        obs = self.env.observe("player_0")
        
        if done:
            return self._terminal_step()
        
        return obs, 0, False, False, {}
    
    def _terminal_step(self):
        obs = self.env.observe("player_0")
        base_reward = self.env._cumulative_rewards["player_0"]
        info = self.env.infos.get("player_0", {})
        
        # 勝利ボーナスを追加
        final_reward = base_reward
        if base_reward > 0:
            final_reward += self.win_bonus
            info['won'] = True
        elif base_reward < 0:
            info['won'] = False
        
        return obs, final_reward, True, False, info
    
    def _run_opponent_turn(self):
        if self.env.agent_selection != "player_1":
            return
        
        valid_actions = self.env.get_valid_actions("player_1")
        
        if not valid_actions:
            self.env.step(0)
            return
        
        if self.active_opponent is not None:
            opponent_obs = self.env.observe("player_1")
            mask = np.zeros(self.action_space.n, dtype=bool)
            for a in valid_actions:
                mask[a] = True
            
            opponent_action, _ = self.active_opponent.predict(
                opponent_obs, action_masks=mask, deterministic=True
            )
        else:
            opponent_action = np.random.choice(valid_actions)
        
        self.env.step(opponent_action)
    
    def action_masks(self):
        valid = self.env.get_valid_actions("player_0")
        mask = np.zeros(self.action_space.n, dtype=bool)
        for action_id in valid:
            mask[action_id] = True
        
        if not mask.any():
            mask[0] = True
        return mask


def train_phase3_aws(
    total_timesteps: int = 10_000_000,
    opponent_update_freq: int = 100_000,
    save_freq: int = 200_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 128,
    seed: int = 42,
    save_path: str = "models/ofc_phase3",
):
    """
    Phase 3 AWS本番学習
    """
    print("=" * 60)
    print("OFC Pineapple AI - Phase 3 Self-Play Training (AWS)")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Opponent update freq: {opponent_update_freq:,}")
    print(f"Save freq: {save_freq:,}")
    print(f"Strategy: Latest vs Pool (80% Pool, 20% Latest)")
    print()
    
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    
    # 通知システム初期化
    notifier = init_notifier()
    if notifier.enabled:
        print(f"[Notifier] Discord/Slack notifications enabled")
        notifier.send_start({
            'timesteps': total_timesteps,
            'opponent_update': opponent_update_freq,
            'lr': learning_rate,
            'strategy': 'Latest vs Pool (80/20)'
        })
    
    # 環境作成
    print("Creating environment...")
    env = SelfPlayEnvPhase3(seed=seed, pool_ratio=0.8, win_bonus=10.0)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()
    
    # モデル作成
    print("Creating MaskablePPO model...")
    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        verbose=1,
        seed=seed,
    )
    
    # 初期対戦相手
    env.add_to_pool(model)
    env.set_latest(model)
    
    # コールバック
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    callback = Phase3Callback(
        save_path=f"{save_path}_{timestamp}",
        save_freq=save_freq,
        log_freq=10_000,
        verbose=1
    )
    callback.total_timesteps = total_timesteps
    callback.notify_freq = min(100_000, save_freq)  # 保存頻度より短い間隔で通知
    
    # 学習開始
    print("\nStarting training...")
    start_time = time.time()
    
    try:
        steps_done = 0
        while steps_done < total_timesteps:
            learn_steps = min(opponent_update_freq, total_timesteps - steps_done)
            model.learn(
                total_timesteps=learn_steps,
                callback=callback,
                reset_num_timesteps=False,
            )
            steps_done += learn_steps
            
            # 対戦相手更新
            if steps_done < total_timesteps:
                print(f"\n[Step {steps_done}] Updating opponent pool...")
                env.add_to_pool(model)
                env.set_latest(model)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    elapsed = time.time() - start_time
    
    # 結果表示
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Elapsed time: {elapsed:.1f}s ({elapsed/3600:.2f}h)")
    print(f"Total games: {callback.total_games}")
    print(f"Pool size: {len(env.opponent_pool)}")
    
    if callback.total_games > 0:
        print(f"Final win rate: {callback.wins / callback.total_games * 100:.1f}%")
        print(f"Final foul rate: {callback.fouls / callback.total_games * 100:.1f}%")
    
    # 最終モデル保存
    final_path = f"{save_path}_{timestamp}_final"
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}.zip")
    
    # 完了通知
    if notifier.enabled:
        hours = elapsed / 3600
        notifier.send_complete({
            'total_steps': callback.n_calls,
            'total_games': callback.total_games,
            'win_rate': callback.wins / max(1, callback.total_games) * 100,
            'foul_rate': callback.fouls / max(1, callback.total_games) * 100,
            'elapsed_time': f"{hours:.2f}h",
            'model_path': f"{final_path}.zip"
        })
    
    return model, callback


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 3 Self-Play Training for AWS")
    parser.add_argument("--steps", type=int, default=10_000_000, help="Total training timesteps")
    parser.add_argument("--opponent-update", type=int, default=100_000, help="Opponent update frequency")
    parser.add_argument("--save-freq", type=int, default=200_000, help="Model save frequency")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    train_phase3_aws(
        total_timesteps=args.steps,
        opponent_update_freq=args.opponent_update,
        save_freq=args.save_freq,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )
