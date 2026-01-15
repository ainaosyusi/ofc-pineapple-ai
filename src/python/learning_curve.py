"""
OFC Pineapple AI - 学習曲線可視化
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from ofc_env import OFCPineappleEnv


def evaluate_foul_rate(model, n_episodes: int = 100) -> dict:
    """ファウル率を評価"""
    env = OFCPineappleEnv()
    
    rewards = []
    fouls = 0
    fl_entries = 0
    royalties = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        step_count = 0
        max_steps = 50  # 無限ループ防止
        
        while not done and step_count < max_steps:
            valid_actions = env.get_valid_actions()
            
            if len(valid_actions) == 0:
                break
            
            # モデルで予測
            action, _ = model.predict(obs, deterministic=True)
            
            # 無効なアクションの場合、有効なアクションからランダム選択
            if action not in valid_actions:
                action = np.random.choice(valid_actions)
            
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
        
        rewards.append(total_reward)
        
        # 終了時の判定
        board = env.engine.player(0).board
        if board.is_complete():
            if board.is_foul():
                fouls += 1
            else:
                royalties.append(board.calculate_royalties())
                if board.qualifies_for_fl():
                    fl_entries += 1
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'foul_rate': fouls / n_episodes * 100,
        'fl_rate': fl_entries / n_episodes * 100,
        'mean_royalty': np.mean(royalties) if royalties else 0,
    }


def run_learning_curve_experiment(
    timesteps_per_checkpoint: int = 10000,
    num_checkpoints: int = 5,
    n_eval_episodes: int = 50,
    n_envs: int = 4,
):
    """学習曲線を生成"""
    print("=" * 50)
    print("OFC AI - Learning Curve Experiment")
    print("=" * 50)
    
    # 結果を格納
    results = {
        'timesteps': [],
        'foul_rate': [],
        'fl_rate': [],
        'mean_reward': [],
        'mean_royalty': [],
    }
    
    # 環境作成
    def make_env(rank: int):
        def _init():
            env = OFCPineappleEnv()
            env = Monitor(env)
            return env
        return _init
    
    env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    
    # モデル作成
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0,
        seed=42,
    )
    
    # 初期評価
    print("\n[Timestep 0] Initial evaluation...")
    eval_result = evaluate_foul_rate(model, n_eval_episodes)
    results['timesteps'].append(0)
    results['foul_rate'].append(eval_result['foul_rate'])
    results['fl_rate'].append(eval_result['fl_rate'])
    results['mean_reward'].append(eval_result['mean_reward'])
    results['mean_royalty'].append(eval_result['mean_royalty'])
    
    print(f"  Foul Rate: {eval_result['foul_rate']:.1f}%")
    print(f"  Mean Reward: {eval_result['mean_reward']:.2f}")
    
    # 学習＆評価ループ
    for i in range(num_checkpoints):
        print(f"\n[Checkpoint {i+1}/{num_checkpoints}] Training {timesteps_per_checkpoint} steps...")
        
        model.learn(total_timesteps=timesteps_per_checkpoint, reset_num_timesteps=False)
        
        total_steps = (i + 1) * timesteps_per_checkpoint
        
        print(f"[Timestep {total_steps}] Evaluating...")
        eval_result = evaluate_foul_rate(model, n_eval_episodes)
        
        results['timesteps'].append(total_steps)
        results['foul_rate'].append(eval_result['foul_rate'])
        results['fl_rate'].append(eval_result['fl_rate'])
        results['mean_reward'].append(eval_result['mean_reward'])
        results['mean_royalty'].append(eval_result['mean_royalty'])
        
        print(f"  Foul Rate: {eval_result['foul_rate']:.1f}%")
        print(f"  Mean Reward: {eval_result['mean_reward']:.2f}")
        print(f"  FL Entry Rate: {eval_result['fl_rate']:.1f}%")
    
    # 結果表示
    print("\n" + "=" * 50)
    print("Learning Curve Summary")
    print("=" * 50)
    print(f"{'Timesteps':>10} | {'Foul%':>8} | {'Reward':>10} | {'FL%':>6}")
    print("-" * 45)
    for i in range(len(results['timesteps'])):
        print(f"{results['timesteps'][i]:>10} | {results['foul_rate'][i]:>7.1f}% | {results['mean_reward'][i]:>10.1f} | {results['fl_rate'][i]:>5.1f}%")
    
    # 保存
    model.save("models/ofc_ppo_learning_curve")
    print(f"\nModel saved: models/ofc_ppo_learning_curve.zip")
    
    return results


if __name__ == "__main__":
    results = run_learning_curve_experiment(
        timesteps_per_checkpoint=10000,
        num_checkpoints=5,
        n_eval_episodes=30,
        n_envs=4,
    )
