"""
OFC Pineapple AI - Self-Play Training Script
AI vs AI のSelf-Play学習を実行
"""

import io
import os
import sys
import time
import copy
import numpy as np
from datetime import datetime
from collections import deque

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from multi_ofc_env import OFCMultiAgentEnv


class SelfPlayCallback(BaseCallback):
    """
    Self-Play用コールバック
    定期的に対戦相手を更新し、統計を記録
    """
    
    def __init__(
        self, 
        opponent_update_freq: int = 10_000,
        log_freq: int = 2000,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.opponent_update_freq = opponent_update_freq
        self.log_freq = log_freq
        
        # 統計
        self.episode_rewards = deque(maxlen=1000)
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.fouls = 0
        self.total_games = 0
        
        # 対戦相手履歴
        self.opponent_versions = []
    
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
            recent = list(self.episode_rewards)[-100:]
            mean_reward = np.mean(recent)
            win_rate = self.wins / max(1, self.total_games) * 100
            foul_rate = self.fouls / max(1, self.total_games) * 100
            
            if self.verbose:
                print(f"\n[Step {self.n_calls}]")
                print(f"  Games: {self.total_games}")
                print(f"  Mean Score (last 100): {mean_reward:.2f}")
                print(f"  Win Rate: {win_rate:.1f}%")
                print(f"  Foul Rate: {foul_rate:.1f}%")
        
        return True

import gymnasium as gym


class SelfPlayEnv(gym.Env):
    """
    Self-Play用の環境ラッパー
    エージェントと対戦相手を管理
    Gymnasium互換
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, seed=None):
        super().__init__()
        self.env = OFCMultiAgentEnv()
        self._seed = seed
        self.opponent_pool = []  # 対戦相手の候補（ポリシーのリスト）
        self.active_opponent = None
        self.current_obs = None
        
        # Gym互換のスペース
        self.observation_space = self.env.observation_space("player_0")
        self.action_space = self.env.action_space("player_0")
    
    def add_opponent(self, model):
        """対戦相手プールにモデルを追加"""
        # C++エンジンを含むためdeepcopyは不可。一旦バッファに保存してロード
        buffer = io.BytesIO()
        model.save(buffer)
        buffer.seek(0)
        
        # ロードしてポリシーのみを保存 (環境は不要)
        # Note: MaskablePPO.load(buffer) は新しいモデルを生成する
        cloned_model = MaskablePPO.load(buffer)
        self.opponent_pool.append(cloned_model)
        
        # プールが大きすぎる場合は古いものを削除
        if len(self.opponent_pool) > 10:
            self.opponent_pool.pop(0)

    def set_opponent(self, model):
        """単一の対戦相手を直接設定（初期化用）"""
        self.add_opponent(model)
        self.active_opponent = self.opponent_pool[-1]
    
    def reset(self, seed=None, options=None):
        """環境をリセット"""
        if seed is not None:
            self._seed = seed
        elif self._seed is None:
            self._seed = np.random.randint(0, 2**32)
        
        self.env.reset(seed=self._seed)
        self._seed = (self._seed + 1) % (2**32)  # 次回用にインクリメント
        
        # プールから対戦相手をランダムに選択（Latest vs Pool）
        if self.opponent_pool:
            self.active_opponent = np.random.choice(self.opponent_pool)
        
        # 初期観測を取得
        obs = self.env.observe("player_0")
        return obs, {}
    
    def step(self, action):
        """
        メインエージェント（player_0）のアクションを実行し、
        対戦相手（player_1）も自動的にアクション
        """
        info = {}
        
        # まずplayer_0のアクションを実行
        if self.env.agent_selection != "player_0":
            # player_0のターンでない場合は対戦相手を先に処理
            self._run_opponent_turn()
        
        if all(self.env.terminations.values()):
            obs = self.env.observe("player_0")
            reward = self.env._cumulative_rewards["player_0"]
            info = self.env.infos.get("player_0", {})
            return obs, reward, True, False, info
        
        # player_0のアクションを実行 (action_masksは外部で処理される)
        self.env.step(action)
        
        if all(self.env.terminations.values()):
            obs = self.env.observe("player_0")
            reward = self.env._cumulative_rewards["player_0"]
            info = self.env.infos.get("player_0", {})
            return obs, reward, True, False, info
        
        # 対戦相手のターンを処理
        while self.env.agent_selection == "player_1" and not all(self.env.terminations.values()):
            self._run_opponent_turn()
        
        # 終了判定
        done = all(self.env.terminations.values())
        obs = self.env.observe("player_0")
        reward = self.env._cumulative_rewards["player_0"] if done else 0
        info = self.env.infos.get("player_0", {}) if done else {}
        
        return obs, reward, done, False, info
    
    def _run_opponent_turn(self):
        """対戦相手（player_1）のターンを実行"""
        if self.env.agent_selection != "player_1":
            return
        
        valid_actions = self.env.get_valid_actions("player_1")
        
        if not valid_actions:
            self.env.step(0)  # ダミーアクション
            return
        
        if self.active_opponent is not None:
            opponent_obs = self.env.observe("player_1")
            # 対戦相手もaction_masksを使用
            opp_mask = self.env.get_valid_actions("player_1")
            mask = np.zeros(self.action_space.n, dtype=bool)
            for a in opp_mask:
                mask[a] = True
            
            opponent_action, _ = self.active_opponent.predict(
                opponent_obs, action_masks=mask, deterministic=True
            )
        else:
            opponent_action = np.random.choice(valid_actions)
        
        self.env.step(opponent_action)
    
    def render(self):
        self.env.render()
    
    def action_masks(self):
        """MaskablePPO用アクションマスク (player_0用)"""
        valid = self.env.get_valid_actions("player_0")
        mask = np.zeros(self.action_space.n, dtype=bool)
        for action_id in valid:
            mask[action_id] = True
        
        # 安全対策: マスクがすべてFalseの場合はアクション0を有効にする
        if not mask.any():
            mask[0] = True
        return mask


def train_selfplay(
    total_timesteps: int = 200_000,
    opponent_update_freq: int = 20_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    seed: int = 42,
    save_path: str = "models/ofc_selfplay",
    resume_from: str = None,
):
    """
    Self-Play学習を実行
    """
    print("=" * 50)
    print("OFC Pineapple AI - Self-Play Training")
    print("=" * 50)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Opponent update freq: {opponent_update_freq:,}")
    print()
    
    # ディレクトリ作成
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 環境作成
    print("Creating environment...")
    env = SelfPlayEnv(seed=seed)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()
    
    # MaskablePPOモデル作成またはロード
    if resume_from:
        print(f"Loading pre-trained model from: {resume_from}")
        model = MaskablePPO.load(resume_from, env=env)
        model.learning_rate = learning_rate
    else:
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
    
    # 初期は自分自身を対戦相手に
    env.set_opponent(model)
    
    # コールバック
    callback = SelfPlayCallback(
        opponent_update_freq=opponent_update_freq,
        log_freq=5000,
        verbose=1
    )
    
    # 学習開始
    print("\nStarting training...")
    start_time = time.time()
    
    try:
        # 段階的に学習
        steps_done = 0
        while steps_done < total_timesteps:
            # 一定ステップ学習
            learn_steps = min(opponent_update_freq, total_timesteps - steps_done)
            model.learn(
                total_timesteps=learn_steps,
                callback=callback,
                reset_num_timesteps=False,
            )
            steps_done += learn_steps
            
            # 対戦相手プールを更新
            if steps_done < total_timesteps:
                print(f"\n[Adding current model to opponent pool at step {steps_done}]")
                env.add_opponent(model)
                callback.opponent_versions.append(steps_done)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    elapsed = time.time() - start_time
    
    # 結果表示
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Elapsed time: {elapsed:.1f}s")
    print(f"Total games: {callback.total_games}")
    print(f"Opponent updates: {len(callback.opponent_versions)}")
    
    if callback.total_games > 0:
        print(f"Win rate: {callback.wins / callback.total_games * 100:.1f}%")
        print(f"Foul rate: {callback.fouls / callback.total_games * 100:.1f}%")
    
    # モデル保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{save_path}_{timestamp}"
    model.save(model_path)
    print(f"\nModel saved to: {model_path}.zip")
    
    return model, callback


def evaluate_selfplay(model, n_games: int = 100, seed: int = 12345):
    """
    Self-Playモデルを評価
    """
    print("\n" + "=" * 50)
    print("Evaluating Self-Play Model")
    print("=" * 50)
    
    env = SelfPlayEnv(seed=seed)
    env.set_opponent(model)  # 自分自身と対戦
    
    wins = 0
    losses = 0
    draws = 0
    fouls = 0
    total_scores = []
    
    for game in range(n_games):
        obs, _ = env.reset()
        done = False
        
        while not done:
            # MaskablePPOのpredictを使用
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            
            obs, reward, done, _, info = env.step(action)
        
        score = info.get('final_score', 0)
        total_scores.append(score)
        
        if score > 0:
            wins += 1
        elif score < 0:
            losses += 1
        else:
            draws += 1
        
        if info.get('fouled', False):
            fouls += 1
    
    print(f"Games: {n_games}")
    print(f"Mean score: {np.mean(total_scores):.2f} ± {np.std(total_scores):.2f}")
    print(f"Win/Draw/Loss: {wins}/{draws}/{losses}")
    print(f"Foul rate: {fouls / n_games * 100:.1f}%")
    
    return total_scores


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train OFC Pineapple AI with Self-Play")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--opponent-update", type=int, default=20_000, help="Opponent update frequency")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval", action="store_true", help="Evaluate after training")
    parser.add_argument("--resume", type=str, default=None, help="Path to pre-trained model to resume from")
    
    args = parser.parse_args()
    
    # 学習実行
    model, callback = train_selfplay(
        total_timesteps=args.timesteps,
        opponent_update_freq=args.opponent_update,
        learning_rate=args.lr,
        seed=args.seed,
        resume_from=args.resume,
    )
    
    # 評価
    if args.eval:
        evaluate_selfplay(model, n_games=100)
