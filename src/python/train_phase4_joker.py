"""
OFC Pineapple AI - Phase 4-1 Training (Joker Edition)
ジョーカーを含む54カード環境でのPhase 1ファウル回避学習

特徴:
- 54カード環境（Joker 2枚を含む）
- ジョーカーにより、ファウル率が52枚版より早く低下する見込み
"""

import os
import argparse
from datetime import datetime
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from src.python.ofc_phase1_env import OFCPhase1Env

class Phase4Callback(BaseCallback):
    """
    ジョーカー対応学習の進捗を記録するコールバック
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.game_results = []
        self.log_file = None
        
    def _on_training_start(self):
        # ログファイルを開く
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("logs/phase4", exist_ok=True)
        self.log_file = open(f"logs/phase4/phase4_log_{timestamp}.txt", "w")
        self.log_file.write("step,games,foul_rate,mean_royalty\n")
        
    def _on_step(self) -> bool:
        # infoからゲーム結果を取得
        for info in self.locals.get("infos", []):
            if "fouled" in info:
                self.game_results.append({
                    "fouled": info["fouled"],
                    "royalty": info.get("royalty", 0)
                })
        
        # 定期的な出力（5000ステップ毎）
        if self.n_calls % 5000 == 0 and self.game_results:
            recent = self.game_results[-100:]
            foul_rate = sum(1 for r in recent if r["fouled"]) / len(recent)
            mean_royalty = sum(r["royalty"] for r in recent) / len(recent)
            
            print(f"\n[Step {self.num_timesteps}] 🃏 Phase 4-1 (Joker)")
            print(f"  Games: {len(self.game_results)}")
            print(f"  Foul Rate (last 100): {foul_rate:.1%}")
            print(f"  Mean Royalty (last 100): {mean_royalty:.1f}")
            print("-" * 40)
            
            # ログファイルに記録
            if self.log_file:
                self.log_file.write(f"{self.num_timesteps},{len(self.game_results)},{foul_rate:.4f},{mean_royalty:.1f}\n")
                self.log_file.flush()
            
        return True
    
    def _on_training_end(self):
        if self.log_file:
            self.log_file.close()

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500000,
                        help="学習ステップ数（デフォルト: 500000）")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="学習率")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="バッチサイズ")
    parser.add_argument("--save-name", type=str, default=None,
                        help="モデル保存名")
    args = parser.parse_args()
    
    # モデル保存名
    if args.save_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_name = f"ofc_phase4_joker_{timestamp}"
    
    # 環境作成（Phase 4: 54カード環境）
    print("=" * 60)
    print("🃏 OFC Pineapple AI - Phase 4-1 Training (Joker Edition)")
    print("=" * 60)
    print(f"  Environment: 54 cards (52 + 2 Jokers)")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Batch Size: {args.batch_size}")
    print("=" * 60)
    
    env = OFCPhase1Env(reward_royalties=False)  # Phase 1: ファウル回避フォーカス
    
    # チェックポイント保存用
    os.makedirs("models/phase4", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/phase4/",
        name_prefix=args.save_name
    )
    
    phase4_callback = Phase4Callback()
    
    # モデル作成 (MaskablePPO)
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./logs/phase4/"
    )
    
    print("\n🚀 Starting training...")
    
    # 学習開始
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, phase4_callback]
    )
    
    # 最終モデル保存
    model.save(f"models/phase4/{args.save_name}_final")
    print(f"\n✅ Model saved to models/phase4/{args.save_name}_final")

if __name__ == "__main__":
    train()
