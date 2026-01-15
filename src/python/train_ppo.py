"""
OFC Pineapple AI - PPO Training Script
Stable-Baselines3を使用したPPO学習
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from ofc_env import OFCPineappleEnv


class OFCTrainingCallback(BaseCallback):
    """
    OFC学習用カスタムコールバック
    エピソードごとの統計を記録
    """
    
    def __init__(self, log_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.foul_count = 0
        self.total_episodes = 0
        
    def _on_step(self) -> bool:
        # エピソード終了時の処理
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
                self.total_episodes += 1
                
                if info.get('final_reward', 0) < -10:
                    self.foul_count += 1
        
        # ログ出力
        if self.n_calls % self.log_freq == 0:
            if len(self.episode_rewards) > 0:
                recent_rewards = self.episode_rewards[-100:]
                mean_reward = np.mean(recent_rewards)
                foul_rate = self.foul_count / max(1, self.total_episodes) * 100
                
                if self.verbose:
                    print(f"\n[Step {self.n_calls}]")
                    print(f"  Episodes: {self.total_episodes}")
                    print(f"  Mean Reward (last 100): {mean_reward:.2f}")
                    print(f"  Foul Rate: {foul_rate:.1f}%")
        
        return True


def create_env(seed: int = None) -> OFCPineappleEnv:
    """環境を作成"""
    env = OFCPineappleEnv()
    env = Monitor(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def train_ppo(
    total_timesteps: int = 100_000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    seed: int = 42,
    save_path: str = "models/ofc_ppo",
    log_path: str = "logs/",
):
    """
    PPO学習を実行
    
    Args:
        total_timesteps: 総学習ステップ数
        n_envs: 並列環境数
        learning_rate: 学習率
        n_steps: 各更新で収集するステップ数
        batch_size: バッチサイズ
        n_epochs: 各更新のエポック数
        gamma: 割引率
        seed: 乱数シード
        save_path: モデル保存先
        log_path: ログ保存先
    """
    print("=" * 50)
    print("OFC Pineapple AI - PPO Training")
    print("=" * 50)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs: {n_envs}")
    print(f"Learning rate: {learning_rate}")
    print()
    
    # ディレクトリ作成
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # 環境作成
    print("Creating environments...")
    
    def make_env(rank: int):
        def _init():
            env = OFCPineappleEnv()
            env = Monitor(env)
            return env
        return _init
    
    # DummyVecEnvで並列化（macOSでSubprocVecEnvは問題がある場合がある）
    env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()
    
    # PPOモデル作成
    print("Creating PPO model...")
    model = PPO(
        policy="MultiInputPolicy",  # Dict observation用
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=1,
        seed=seed,
        tensorboard_log=None,  # TensorBoardオプション（インストール時に有効化）
    )
    
    # コールバック
    callback = OFCTrainingCallback(log_freq=5000, verbose=1)
    
    # 学習開始
    print("\nStarting training...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,  # tqdmが必要なので無効化
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    elapsed = time.time() - start_time
    
    # 結果表示
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Elapsed time: {elapsed:.1f}s")
    print(f"Total episodes: {callback.total_episodes}")
    
    if len(callback.episode_rewards) > 0:
        print(f"Mean reward: {np.mean(callback.episode_rewards):.2f}")
        print(f"Final foul rate: {callback.foul_count / max(1, callback.total_episodes) * 100:.1f}%")
    
    # モデル保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{save_path}_{timestamp}"
    model.save(model_path)
    print(f"\nModel saved to: {model_path}.zip")
    
    return model, callback


def evaluate_model(model, n_episodes: int = 100):
    """
    モデルを評価
    """
    print("\n" + "=" * 50)
    print("Evaluating Model")
    print("=" * 50)
    
    env = OFCPineappleEnv()
    
    rewards = []
    fouls = 0
    fl_entries = 0
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # 有効なアクションでマスク（簡易版）
            valid_actions = env.get_valid_actions()
            
            # モデルで予測
            action, _ = model.predict(obs, deterministic=True)
            
            # 無効なアクションの場合、有効なアクションからランダム選択
            if action not in valid_actions and len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
            
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        
        rewards.append(total_reward)
        
        # 結果チェック
        if total_reward < -10:
            fouls += 1
        if info.get('final_reward', 0) > 5:
            fl_entries += 1
    
    print(f"Episodes: {n_episodes}")
    print(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Foul rate: {fouls / n_episodes * 100:.1f}%")
    print(f"FL entry rate: {fl_entries / n_episodes * 100:.1f}%")
    
    return rewards


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train OFC Pineapple AI with PPO")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Total training timesteps")
    parser.add_argument("--envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval", action="store_true", help="Evaluate after training")
    
    args = parser.parse_args()
    
    # 学習実行
    model, callback = train_ppo(
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        learning_rate=args.lr,
        seed=args.seed,
    )
    
    # 評価
    if args.eval:
        evaluate_model(model, n_episodes=100)
