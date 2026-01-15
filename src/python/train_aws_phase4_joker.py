"""
OFC Pineapple AI - AWS Phase 4 Joker Training Script
EC2本番用のジョーカー対応学習スクリプト

特徴:
- 54カード環境（Joker 2枚を含む）
- Phase 2ベースの報酬設計（ファウル回避優先 + ロイヤリティボーナス）
- 1周完了で停止
- Discord通知のみ（Feedback機能なし）
"""

import os
import sys
import time
import traceback
import numpy as np
from datetime import datetime
from collections import deque

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from ofc_phase1_env import OFCPhase1Env
from notifier import TrainingNotifier

# Phase 4用の新しいDiscord Webhook URL
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1461388510140301315/Ofc9ok3IDLgRhFR7Oa2iA4-CHHL9uFvFbIPT9G9tbA5X5r-wG-XMbnkg7ubocLJJQ5Pf"


class Phase4Callback(BaseCallback):
    """
    Phase 4 ジョーカー対応学習用コールバック
    通知のみ（Feedback機能なし）
    """
    
    def __init__(
        self, 
        save_path: str,
        notifier: TrainingNotifier,
        save_freq: int = 200_000,
        notify_freq: int = 200_000,
        log_freq: int = 10_000,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.notifier = notifier
        self.save_freq = save_freq
        self.notify_freq = notify_freq
        self.log_freq = log_freq
        
        # 統計
        self.episode_rewards = deque(maxlen=1000)
        self.fouls = 0
        self.total_games = 0
        self.royalties = deque(maxlen=1000)
        self.total_timesteps = 10_000_000  # デフォルト
        self.last_notify_step = 0
        self.start_time = time.time()
        
    def _on_step(self) -> bool:
        # エピソード終了時の処理
        for info in self.locals.get('infos', []):
            if 'fouled' in info:
                self.total_games += 1
                if info['fouled']:
                    self.fouls += 1
                self.royalties.append(info.get('royalty', 0))
        
        # ログ出力
        if self.n_calls % self.log_freq == 0 and self.total_games > 0:
            self._log_stats()
        
        # モデル保存
        if self.n_calls % self.save_freq == 0:
            self._save_model()
        
        # Discord通知（シンプル版）
        if self.n_calls - self.last_notify_step >= self.notify_freq:
            self._send_progress_notification()
            self.last_notify_step = self.n_calls
        
        return True
    
    def _log_stats(self):
        foul_rate = self.fouls / max(1, self.total_games) * 100
        recent_royalties = list(self.royalties)[-100:]
        mean_royalty = np.mean(recent_royalties) if recent_royalties else 0
        
        print(f"\n[Step {self.n_calls}] 🃏 Phase 4 (Joker)")
        print(f"  Games: {self.total_games}")
        print(f"  Foul Rate (overall): {foul_rate:.1f}%")
        print(f"  Mean Royalty (last 100): {mean_royalty:.1f}")
        print("-" * 40)
    
    def _save_model(self):
        path = f"{self.save_path}_{self.n_calls}_steps"
        self.model.save(path)
        print(f"[Checkpoint] Model saved to {path}.zip")
        
        # チェックポイント通知
        if self.notifier and self.notifier.enabled:
            self.notifier.send_checkpoint(f"{path}.zip", self.n_calls)
    
    def _send_progress_notification(self):
        """Discord進捗通知（シンプル版）"""
        if not self.notifier or not self.notifier.enabled:
            return
        
        elapsed = time.time() - self.start_time
        fps = self.n_calls / max(1, elapsed)
        foul_rate = self.fouls / max(1, self.total_games) * 100
        recent_royalties = list(self.royalties)[-100:]
        mean_royalty = np.mean(recent_royalties) if recent_royalties else 0
        
        self.notifier.send_progress(
            step=self.n_calls,
            total_steps=self.total_timesteps,
            metrics={
                'games': self.total_games,
                'foul_rate': foul_rate,
                'win_rate': 0,  # Phase 4ではSelf-Playではないので不使用
                'mean_score': mean_royalty,
                'fps': fps
            }
        )


def train_phase4_aws(
    total_timesteps: int = 10_000_000,
    save_freq: int = 200_000,
    notify_freq: int = 200_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 128,
    seed: int = 42,
    save_path: str = "models/phase4/ofc_phase4_joker",
    reward_royalties: bool = True,  # Phase 2ベース
):
    """
    Phase 4 AWS本番学習（ジョーカー対応）
    1周完了で停止
    """
    print("=" * 60)
    print("🃏 OFC Pineapple AI - Phase 4 Joker Training (AWS)")
    print("=" * 60)
    print(f"Environment: 54 cards (52 + 2 Jokers)")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Save freq: {save_freq:,}")
    print(f"Reward mode: Phase 2 (Foul + Royalty)")
    print()
    
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    
    # 通知システム初期化（新しいWebhook URL使用）
    notifier = TrainingNotifier(
        discord_webhook=DISCORD_WEBHOOK_URL,
        project_name="OFC Phase 4 (Joker)"
    )
    
    if notifier.enabled:
        print(f"[Notifier] Discord notifications enabled")
        notifier.send_start({
            'timesteps': total_timesteps,
            'opponent_update': 'N/A (Phase 4)',
            'lr': learning_rate,
            'strategy': 'Phase 4 Joker (54 cards)'
        })
    
    # 環境作成（Phase 2ベース: ファウル回避 + ロイヤリティ）
    print("Creating 54-card environment...")
    env = OFCPhase1Env(reward_royalties=reward_royalties)
    
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
        tensorboard_log=None  # tensorboardなしで実行
    )
    
    # コールバック
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    callback = Phase4Callback(
        save_path=f"{save_path}_{timestamp}",
        notifier=notifier,
        save_freq=save_freq,
        notify_freq=notify_freq,
        log_freq=10_000,
        verbose=1
    )
    callback.total_timesteps = total_timesteps
    
    # 学習開始
    print("\n🚀 Starting training...")
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
    print("🎉 Training Complete!")
    print("=" * 60)
    print(f"Elapsed time: {elapsed:.1f}s ({elapsed/3600:.2f}h)")
    print(f"Total games: {callback.total_games}")
    
    if callback.total_games > 0:
        print(f"Final foul rate: {callback.fouls / callback.total_games * 100:.1f}%")
    
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
            'win_rate': 0,  # Phase 4では不使用
            'foul_rate': callback.fouls / max(1, callback.total_games) * 100,
            'elapsed_time': f"{hours:.2f}h",
            'model_path': f"{final_path}.zip"
        })
    
    return model, callback


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 4 Joker Training for AWS")
    parser.add_argument("--steps", type=int, default=10_000_000, help="Total training timesteps")
    parser.add_argument("--save-freq", type=int, default=200_000, help="Model save frequency")
    parser.add_argument("--notify-freq", type=int, default=200_000, help="Discord notify frequency")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-royalty", action="store_true", help="Disable royalty reward (Pure Phase 1)")
    
    args = parser.parse_args()
    
    train_phase4_aws(
        total_timesteps=args.steps,
        save_freq=args.save_freq,
        notify_freq=args.notify_freq,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        reward_royalties=not args.no_royalty,
    )
