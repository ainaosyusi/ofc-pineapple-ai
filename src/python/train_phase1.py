"""
OFC Pineapple AI - Phase 1 Training
MaskablePPOを使用したファウル回避学習
"""

import os
import argparse
from datetime import datetime
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from src.python.ofc_phase1_env import OFCPhase1Env

class Phase1Callback(BaseCallback):
    """
    学習の進捗（ファウル率、ロイヤリティ）を記録するコールバック
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.game_results = []
        
    def _on_step(self) -> bool:
        # infoからゲーム結果を取得
        for info in self.locals.get("infos", []):
            if "fouled" in info:
                self.game_results.append({
                    "fouled": info["fouled"],
                    "royalty": info.get("royalty", 0)
                })
        
        # 定期的な出力
        if self.n_calls % 10000 == 0 and self.game_results:
            recent = self.game_results[-100:]
            foul_rate = sum(1 for r in recent if r["fouled"]) / len(recent)
            mean_royalty = sum(r["royalty"] for r in recent) / len(recent)
            
            print(f"\n[Step {self.num_timesteps}]")
            print(f"  Games: {len(self.game_results)}")
            print(f"  Foul Rate (last 100): {foul_rate:.1%}")
            print(f"  Mean Royalty (last 100): {mean_royalty:.1f}")
            print("-" * 30)
            
        return True

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=5000000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--save-name", type=str, default=None)
    args = parser.parse_args()
    
    # モデル保存名
    if args.save_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_name = f"ofc_phase2_{timestamp}"
    
    # 環境作成 (Phase 2: ロイヤリティ報酬有効)
    env = OFCPhase1Env(reward_royalties=True)
    
    # チェックポイント保存用
    os.makedirs("models", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./models/",
        name_prefix=args.save_name
    )
    
    phase1_callback = Phase1Callback()
    
    # モデル作成 (MaskablePPO)
    print(f"Starting Phase 2 Training: {args.timesteps} steps")
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./logs/phase1/"
    )
    
    # 学習開始
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, phase1_callback]
    )
    
    # 最終モデル保存
    model.save(f"models/{args.save_name}_final")
    print(f"Model saved to models/{args.save_name}_final")

if __name__ == "__main__":
    train()
