"""
OFC Pineapple AI - Phase 8 Superhuman Strategy Training
AWS上での大規模スケールアップ学習スクリプト

特徴:
- 3人対戦環境 (OFC3MaxEnv) + Observation Masking
- FL突入ボーナス (+15.0) による戦略強化
- Self-Play (最新モデル vs 過去の傑作モデル)
- AWS EC2/S3 連携 (チェックポイントの自動保存・バックアップ)
"""

import os
import sys
import time
import random
import numpy as np
from datetime import datetime
from collections import deque
from typing import Optional, List

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym

from ofc_3max_env import OFC3MaxEnv
from notifier import TrainingNotifier
from s3_utils import S3Manager, init_s3, get_s3

# Discord Webhook (Phase 8用)
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1461388510140301315/Ofc9ok3IDLgRhFR7Oa2iA4-CHHL9uFvFbIPT9G9tbA5X5r-wG-XMbnkg7ubocLJJQ5Pf"

class SuperhumanCallback(BaseCallback):
    """
    Phase 8 大規模学習用コールバック
    - チェックポイント保存
    - Discord通知
    - S3バックアップ
    """
    def __init__(self, save_path, notifier, s3_manager=None, save_freq=1_000_000):
        super().__init__()
        self.save_path = save_path
        self.notifier = notifier
        self.s3_manager = s3_manager
        self.save_freq = save_freq
        self.last_save = 0
        
    def _on_step(self):
        if self.n_calls - self.last_save >= self.save_freq:
            self.last_save = self.n_calls
            path = f"{self.save_path}_{self.n_calls}_steps"
            self.model.save(path)
            
            # Discord通知
            if self.notifier:
                self.notifier.send_checkpoint(f"{path}.zip", self.n_calls)
            
            # S3アップロード
            if self.s3_manager and self.s3_manager.enabled:
                self.s3_manager.upload_checkpoint(f"{path}.zip", step=self.n_calls)
                
        return True

def train_phase8():
    # 1. 環境セットアップ
    env = OFC3MaxEnv()
    # TODO: Wrapper for multi-agent self-play if needed, or use the one from Phase 5
    
    # 簡単のため、まずはシングルエージェント版のラッパー（自分vs自分）を使用
    from train_phase5_3max import OFC3MaxGymWrapper, SelfPlayOpponent
    
    opponent_manager = SelfPlayOpponent(model_pool_size=10)
    env_wrapper = OFC3MaxGymWrapper()
    
    # 2. AWS S3 初期化
    s3_manager = init_s3()
    if s3_manager.enabled:
        print(f"[S3] Checkpoints will be backed up to {s3_manager.bucket_name}")

    # 3. モデル初期化 (既存のPhase 5モデルから継続)
    model_path = "models/ofc_phase5_latest.zip" # 暫定
    if os.path.exists(model_path):
        print(f"Loading existing model: {model_path}")
        model = MaskablePPO.load(model_path, env=env_wrapper)
    else:
        print("Creating new model")
        model = MaskablePPO(
            "MultiInputPolicy",
            env_wrapper,
            verbose=1,
            learning_rate=2e-4,
            n_steps=2048,
            batch_size=256,
            ent_coef=0.01,
        )

    # 4. コールバック
    notifier = TrainingNotifier(DISCORD_WEBHOOK_URL, "Phase 8: Superhuman")
    callback = SuperhumanCallback(
        save_path="models/p8_superhuman", 
        notifier=notifier,
        s3_manager=s3_manager,
        save_freq=1_000_000
    )

    # 4. 学習開始
    total_timesteps = 20_000_000  # 20M steps
    print(f"Starting Phase 8 Training: {total_timesteps} steps")
    
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    except Exception as e:
        notifier.send_error(str(e))
        raise

if __name__ == "__main__":
    train_phase8()
