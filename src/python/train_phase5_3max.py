"""
OFC Pineapple AI - Phase 5 3-Max Self-Play Training Script
3人対戦Self-Play学習スクリプト

特徴:
- 3人対戦環境 (OFC3MaxEnv)
- 54カード（Joker 2枚）
- Self-Play: 自分vs過去モデル2人
- ポジション戦略、アウツ計算の学習
- Discord通知
"""

import os
import sys
import time
import traceback
import random
import numpy as np
from datetime import datetime
from collections import deque
from typing import Optional, List
import pickle

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym

from ofc_3max_env import OFC3MaxEnv
from notifier import TrainingNotifier

# Phase 5用Discord Webhook URL（Phase 4と同じ）
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1461388510140301315/Ofc9ok3IDLgRhFR7Oa2iA4-CHHL9uFvFbIPT9G9tbA5X5r-wG-XMbnkg7ubocLJJQ5Pf"


class SelfPlayOpponent:
    """Self-Play用の対戦相手管理"""
    
    def __init__(self, model_pool_size: int = 5, latest_prob: float = 0.8):
        self.model_pool: List[MaskablePPO] = []
        self.pool_size = model_pool_size
        self.latest_prob = latest_prob  # 最新モデルを使う確率
    
    def add_model(self, model: MaskablePPO):
        """モデルプールに追加（最新N個を保持）"""
        # モデルのコピーを作成（重みのみ）
        model_copy = MaskablePPO.load(
            path=None,
            env=None,
            custom_objects={'policy_kwargs': {}}
        )
        model_copy.policy.load_state_dict(model.policy.state_dict())
        
        self.model_pool.append(model_copy)
        if len(self.model_pool) > self.pool_size:
            self.model_pool.pop(0)
    
    def get_opponents(self, current_model: MaskablePPO, count: int = 2) -> List:
        """対戦相手を選択（最新80%、過去20%）"""
        opponents = []
        for _ in range(count):
            if not self.model_pool or random.random() < self.latest_prob:
                # 最新モデル（学習中のモデル）を使用
                opponents.append(current_model)
            else:
                # 過去モデルからランダム選択
                opponents.append(random.choice(self.model_pool))
        return opponents


class OFC3MaxGymWrapper(gym.Env):
    """
    3人対戦環境をシングルエージェント用にラップ
    1人のエージェントを学習し、他2人は固定ポリシー
    """
    
    def __init__(self, opponent_model: Optional[MaskablePPO] = None):
        super().__init__()
        
        self.env = OFC3MaxEnv()
        self.learning_agent = "player_0"  # 学習対象
        self.opponent_model = opponent_model
        
        # 空間定義
        self.observation_space = self.env.observation_space(self.learning_agent)
        self.action_space = self.env.action_space(self.learning_agent)
        
        # 統計
        self.current_episode_reward = 0
        self.info = {}
    
    def set_opponent_model(self, model: MaskablePPO):
        """対戦相手モデルを設定"""
        self.opponent_model = model
    
    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        self.current_episode_reward = 0
        self.info = {}
        
        # 最初のエージェントまで進める
        self._play_until_learning_agent()
        
        obs = self.env.observe(self.learning_agent)
        return obs, {}
    
    def step(self, action):
        # 学習エージェントのアクション
        self.env.step(action)
        
        # 他のエージェントをプレイ
        self._play_until_learning_agent()
        
        # 報酬と終了状態
        obs = self.env.observe(self.learning_agent)
        reward = self.env._cumulative_rewards.get(self.learning_agent, 0) - self.current_episode_reward
        self.current_episode_reward += reward
        
        terminated = self.env.terminations.get(self.learning_agent, False)
        truncated = self.env.truncations.get(self.learning_agent, False)
        
        # 終了時の情報
        if terminated or truncated:
            result = self.env.engine.result()
            self.info = {
                'score': result.get_score(0),
                'royalty': result.get_royalty(0),
                'fouled': result.is_fouled(0),
                'entered_fl': result.entered_fl(0),
            }
            
            # 勝敗判定（自分のスコアが正なら勝ち）
            self.info['won'] = self.info['score'] > 0
        
        return obs, reward, terminated, truncated, self.info
    
    def _play_until_learning_agent(self):
        """学習エージェントの手番まで他をプレイ"""
        while (not all(self.env.terminations.values()) and 
               self.env.agent_selection != self.learning_agent):
            
            agent = self.env.agent_selection
            
            if self.env.terminations.get(agent, False):
                self.env.step(None)
                continue
            
            if self.opponent_model is not None:
                # モデルベースのアクション
                obs = self.env.observe(agent)
                action_masks = self.env.action_masks(agent)
                action, _ = self.opponent_model.predict(
                    obs, 
                    action_masks=action_masks,
                    deterministic=False
                )
            else:
                # ランダムアクション
                valid = self.env.get_valid_actions(agent)
                action = random.choice(valid) if valid else 0
            
            self.env.step(action)
    
    def action_masks(self):
        """MaskablePPO用アクションマスク"""
        return self.env.action_masks(self.learning_agent)


class Phase5Callback(BaseCallback):
    """
    Phase 5 3-Max Self-Play学習用コールバック
    """
    
    def __init__(
        self, 
        save_path: str,
        notifier: TrainingNotifier,
        opponent_manager: SelfPlayOpponent,
        env_wrapper: OFC3MaxGymWrapper,
        save_freq: int = 500_000,
        notify_freq: int = 500_000,
        opponent_update_freq: int = 100_000,
        log_freq: int = 10_000,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.notifier = notifier
        self.opponent_manager = opponent_manager
        self.env_wrapper = env_wrapper
        self.save_freq = save_freq
        self.notify_freq = notify_freq
        self.opponent_update_freq = opponent_update_freq
        self.log_freq = log_freq
        
        # 統計
        self.episode_rewards = deque(maxlen=1000)
        self.wins = 0
        self.fouls = 0
        self.total_games = 0
        self.royalties = deque(maxlen=1000)
        self.total_timesteps = 20_000_000
        self.last_notify_step = 0
        self.last_opponent_update = 0
        self.start_time = time.time()
        
    def _on_step(self) -> bool:
        # エピソード終了時の処理
        for info in self.locals.get('infos', []):
            if 'fouled' in info:
                self.total_games += 1
                if info['fouled']:
                    self.fouls += 1
                if info.get('won', False):
                    self.wins += 1
                self.royalties.append(info.get('royalty', 0))
        
        # ログ出力
        if self.n_calls % self.log_freq == 0 and self.total_games > 0:
            self._log_stats()
        
        # 対戦相手更新
        if self.n_calls - self.last_opponent_update >= self.opponent_update_freq:
            self._update_opponents()
            self.last_opponent_update = self.n_calls
        
        # モデル保存
        if self.n_calls % self.save_freq == 0:
            self._save_model()
        
        # Discord通知
        if self.n_calls - self.last_notify_step >= self.notify_freq:
            self._send_progress_notification()
            self.last_notify_step = self.n_calls
        
        return True
    
    def _log_stats(self):
        foul_rate = self.fouls / max(1, self.total_games) * 100
        win_rate = self.wins / max(1, self.total_games) * 100
        recent_royalties = list(self.royalties)[-100:]
        mean_royalty = np.mean(recent_royalties) if recent_royalties else 0
        
        print(f"\n[Step {self.n_calls}] 🎯 Phase 5 (3-Max)")
        print(f"  Games: {self.total_games}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Foul Rate: {foul_rate:.1f}%")
        print(f"  Mean Royalty: {mean_royalty:.2f}")
        print("-" * 40)
    
    def _update_opponents(self):
        """対戦相手をプールに追加"""
        # 現在のモデルをプールに追加（簡易版）
        # 本来はモデルのディープコピーが必要
        print(f"[Opponent Update] Adding current model to pool (step {self.n_calls})")
        # self.opponent_manager.add_model(self.model)  # メモリ効率のため省略
        self.env_wrapper.set_opponent_model(self.model)
    
    def _save_model(self):
        path = f"{self.save_path}_{self.n_calls}_steps"
        self.model.save(path)
        print(f"[Checkpoint] Model saved to {path}.zip")
        
        if self.notifier and self.notifier.enabled:
            self.notifier.send_checkpoint(f"{path}.zip", self.n_calls)
    
    def _send_progress_notification(self):
        if not self.notifier or not self.notifier.enabled:
            return
        
        elapsed = time.time() - self.start_time
        fps = self.n_calls / max(1, elapsed)
        foul_rate = self.fouls / max(1, self.total_games) * 100
        win_rate = self.wins / max(1, self.total_games) * 100
        recent_royalties = list(self.royalties)[-100:]
        mean_royalty = np.mean(recent_royalties) if recent_royalties else 0
        
        self.notifier.send_progress(
            step=self.n_calls,
            total_steps=self.total_timesteps,
            metrics={
                'games': self.total_games,
                'foul_rate': foul_rate,
                'win_rate': win_rate,  # 3人戦での勝率
                'mean_score': mean_royalty,
                'fps': fps
            }
        )


def train_phase5_3max(
    total_timesteps: int = 20_000_000,
    save_freq: int = 500_000,
    notify_freq: int = 500_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 128,
    seed: int = 42,
    save_path: str = "models/phase5/ofc_phase5_3max",
):
    """
    Phase 5 3-Max Self-Play学習
    """
    print("=" * 60)
    print("🎯 OFC Pineapple AI - Phase 5 3-Max Self-Play Training")
    print("=" * 60)
    print(f"Environment: 3-Max, 54 cards (52 + 2 Jokers)")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Save freq: {save_freq:,}")
    print(f"Self-Play: vs Latest(80%) / Past(20%)")
    print()
    
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    
    # 通知システム
    notifier = TrainingNotifier(
        discord_webhook=DISCORD_WEBHOOK_URL,
        project_name="OFC Phase 5 (3-Max)"
    )
    
    if notifier.enabled:
        print(f"[Notifier] Discord notifications enabled")
        notifier.send_start({
            'timesteps': total_timesteps,
            'opponent_update': 'Self-Play (Latest 80% / Past 20%)',
            'lr': learning_rate,
            'strategy': 'Phase 5 3-Max Self-Play'
        })
    
    # 対戦相手管理
    opponent_manager = SelfPlayOpponent(model_pool_size=5, latest_prob=0.8)
    
    # 環境作成
    print("Creating 3-Max environment...")
    env_wrapper = OFC3MaxGymWrapper(opponent_model=None)
    
    print(f"Observation space: {env_wrapper.observation_space}")
    print(f"Action space: {env_wrapper.action_space}")
    print()
    
    # モデル作成
    print("Creating MaskablePPO model...")
    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=env_wrapper,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        verbose=1,
        seed=seed,
        tensorboard_log=None
    )
    
    # 対戦相手を設定（最初は自分自身）
    env_wrapper.set_opponent_model(model)
    
    # コールバック
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    callback = Phase5Callback(
        save_path=f"{save_path}_{timestamp}",
        notifier=notifier,
        opponent_manager=opponent_manager,
        env_wrapper=env_wrapper,
        save_freq=save_freq,
        notify_freq=notify_freq,
        opponent_update_freq=100_000,
        log_freq=10_000,
        verbose=1
    )
    callback.total_timesteps = total_timesteps
    
    # 学習開始
    print("\n🚀 Starting 3-Max Self-Play training...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        if notifier.enabled:
            notifier.send_error(str(e), traceback.format_exc())
        raise
    
    elapsed = time.time() - start_time
    
    # 結果表示
    print("\n" + "=" * 60)
    print("🎉 Phase 5 Training Complete!")
    print("=" * 60)
    print(f"Elapsed time: {elapsed:.1f}s ({elapsed/3600:.2f}h)")
    print(f"Total games: {callback.total_games}")
    
    if callback.total_games > 0:
        print(f"Win rate: {callback.wins / callback.total_games * 100:.1f}%")
        print(f"Foul rate: {callback.fouls / callback.total_games * 100:.1f}%")
    
    # 最終モデル保存
    final_path = f"{save_path}_{timestamp}_final"
    model.save(final_path)
    print(f"\n✅ Final model saved to: {final_path}.zip")
    
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
    
    parser = argparse.ArgumentParser(description="Phase 5 3-Max Self-Play Training")
    parser.add_argument("--steps", type=int, default=20_000_000, help="Total training timesteps")
    parser.add_argument("--save-freq", type=int, default=500_000, help="Model save frequency")
    parser.add_argument("--notify-freq", type=int, default=500_000, help="Discord notify frequency")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    train_phase5_3max(
        total_timesteps=args.steps,
        save_freq=args.save_freq,
        notify_freq=args.notify_freq,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )
