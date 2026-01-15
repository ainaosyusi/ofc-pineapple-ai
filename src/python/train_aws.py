"""
OFC Pineapple AI - AWS Training Script
通知・S3統合込みのSelf-Play学習
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

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from multi_ofc_env import OFCMultiAgentEnv
from train_selfplay import SelfPlayEnv
from notifier import TrainingNotifier, init_notifier, get_notifier
from s3_utils import S3Manager, init_s3, get_s3


class AWSTrainingCallback(BaseCallback):
    """
    AWS環境用コールバック
    通知とS3アップロードを統合
    """
    
    def __init__(
        self,
        total_timesteps: int,
        notification_interval: int = 50000,
        checkpoint_interval: int = 100000,
        log_freq: int = 5000,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.notification_interval = notification_interval
        self.checkpoint_interval = checkpoint_interval
        self.log_freq = log_freq
        
        # 統計
        self.episode_rewards = deque(maxlen=1000)
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.fouls = 0
        self.total_games = 0
        
        # タイミング
        self.start_time = time.time()
        self.last_notification = 0
        self.last_checkpoint = 0
        
        # 通知とS3
        self.notifier = get_notifier()
        self.s3 = get_s3()
    
    def _on_training_start(self):
        """学習開始時"""
        self.start_time = time.time()
        
        if self.notifier and self.notifier.enabled:
            self.notifier.send_start({
                "timesteps": self.total_timesteps,
                "opponent_update": os.getenv("OPPONENT_UPDATE_FREQ", "N/A"),
                "lr": self.model.learning_rate
            })
    
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
        
        current_step = self.num_timesteps
        
        # ログ出力
        if current_step % self.log_freq == 0 and len(self.episode_rewards) > 0:
            self._log_progress(current_step)
        
        # 通知
        if current_step - self.last_notification >= self.notification_interval:
            self._send_notification(current_step)
            self.last_notification = current_step
        
        # チェックポイント保存
        if current_step - self.last_checkpoint >= self.checkpoint_interval:
            self._save_checkpoint(current_step)
            self.last_checkpoint = current_step
        
        return True
    
    def _log_progress(self, step: int):
        """コンソールにログ出力"""
        if len(self.episode_rewards) == 0:
            return
        
        recent = list(self.episode_rewards)[-100:]
        elapsed = time.time() - self.start_time
        fps = step / elapsed if elapsed > 0 else 0
        
        print(f"\n[Step {step:,} / {self.total_timesteps:,}]")
        print(f"  Games: {self.total_games}")
        print(f"  Mean Score: {np.mean(recent):.2f}")
        print(f"  Win Rate: {self.wins / max(1, self.total_games) * 100:.1f}%")
        print(f"  Foul Rate: {self.fouls / max(1, self.total_games) * 100:.1f}%")
        print(f"  FPS: {fps:.0f}")
    
    def _send_notification(self, step: int):
        """進捗通知を送信"""
        if not self.notifier or not self.notifier.enabled:
            return
        
        if len(self.episode_rewards) == 0:
            return
        
        recent = list(self.episode_rewards)[-100:]
        elapsed = time.time() - self.start_time
        
        self.notifier.send_progress(
            step=step,
            total_steps=self.total_timesteps,
            metrics={
                "games": self.total_games,
                "win_rate": self.wins / max(1, self.total_games) * 100,
                "foul_rate": self.fouls / max(1, self.total_games) * 100,
                "mean_score": np.mean(recent),
                "fps": step / elapsed if elapsed > 0 else 0
            }
        )
    
    def _save_checkpoint(self, step: int):
        """チェックポイントを保存しS3にアップロード"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_step{step:08d}_{timestamp}.zip"
        local_path = os.path.join("models", filename)
        
        # ローカル保存
        os.makedirs("models", exist_ok=True)
        self.model.save(local_path.replace(".zip", ""))
        
        print(f"\n[Checkpoint] Saved: {local_path}")
        
        # S3アップロード
        if self.s3 and self.s3.enabled:
            metadata = {
                "step": step,
                "games": self.total_games,
                "foul_rate": self.fouls / max(1, self.total_games)
            }
            s3_uri = self.s3.upload_checkpoint(local_path, step=step, metadata=metadata)
            
            if s3_uri and self.notifier and self.notifier.enabled:
                self.notifier.send_checkpoint(s3_uri, step)
    
    def get_summary(self) -> dict:
        """学習サマリーを取得"""
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        
        return {
            "total_steps": self.num_timesteps,
            "total_games": self.total_games,
            "win_rate": self.wins / max(1, self.total_games) * 100,
            "foul_rate": self.fouls / max(1, self.total_games) * 100,
            "elapsed_time": f"{hours}h {minutes}m"
        }


def train_aws(
    total_timesteps: int = 1000000,
    opponent_update_freq: int = 50000,
    notification_interval: int = 50000,
    checkpoint_interval: int = 100000,
    learning_rate: float = 3e-4,
    seed: int = 42,
    resume_from: str = None,
    save_path: str = "models/ofc_aws",
):
    """
    AWS向けSelf-Play学習
    """
    print("=" * 50)
    print("OFC Pineapple AI - AWS Training")
    print("=" * 50)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Opponent update freq: {opponent_update_freq:,}")
    print(f"Notification interval: {notification_interval:,}")
    print()
    
    # 通知・S3を初期化
    notifier = init_notifier()
    s3 = init_s3()
    
    # ディレクトリ作成
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 環境作成
    print("Creating environment...")
    env = SelfPlayEnv(seed=seed)
    
    # モデル作成または読み込み
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from: {resume_from}")
        model = PPO.load(resume_from, env=env)
    else:
        print("Creating new PPO model...")
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            verbose=1,
            seed=seed,
        )
    
    # 対戦相手を設定
    env.set_opponent(model)
    
    # コールバック
    callback = AWSTrainingCallback(
        total_timesteps=total_timesteps,
        notification_interval=notification_interval,
        checkpoint_interval=checkpoint_interval,
        log_freq=5000,
        verbose=1
    )
    
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
            
            # 対戦相手を更新
            if steps_done < total_timesteps:
                print(f"\n[Updating opponent at step {steps_done:,}]")
                env.set_opponent(model)
        
        # 学習完了
        summary = callback.get_summary()
        
        # 最終モデル保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = f"{save_path}_{timestamp}"
        model.save(final_path)
        summary["model_path"] = f"{final_path}.zip"
        
        print("\n" + "=" * 50)
        print("Training Complete!")
        print("=" * 50)
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # 完了通知
        if notifier and notifier.enabled:
            notifier.send_complete(summary)
        
        # S3にアップロード
        if s3 and s3.enabled:
            s3.upload_checkpoint(f"{final_path}.zip", step=steps_done)
            s3.upload_logs()
        
        return model, callback
        
    except Exception as e:
        # エラー通知
        error_msg = str(e)
        tb = traceback.format_exc()
        
        print(f"\n[ERROR] {error_msg}")
        print(tb)
        
        if notifier and notifier.enabled:
            notifier.send_error(error_msg, tb)
        
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train OFC Pineapple AI on AWS")
    parser.add_argument("--timesteps", type=int, 
                        default=int(os.getenv("TOTAL_TIMESTEPS", 1000000)))
    parser.add_argument("--opponent-update", type=int,
                        default=int(os.getenv("OPPONENT_UPDATE_FREQ", 50000)))
    parser.add_argument("--notification-interval", type=int,
                        default=int(os.getenv("NOTIFICATION_INTERVAL", 50000)))
    parser.add_argument("--checkpoint-interval", type=int,
                        default=int(os.getenv("CHECKPOINT_INTERVAL", 100000)))
    parser.add_argument("--lr", type=float,
                        default=float(os.getenv("LEARNING_RATE", 3e-4)))
    parser.add_argument("--seed", type=int,
                        default=int(os.getenv("SEED", 42)))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval", action="store_true")
    
    args = parser.parse_args()
    
    model, callback = train_aws(
        total_timesteps=args.timesteps,
        opponent_update_freq=args.opponent_update,
        notification_interval=args.notification_interval,
        checkpoint_interval=args.checkpoint_interval,
        learning_rate=args.lr,
        seed=args.seed,
        resume_from=args.resume,
    )
