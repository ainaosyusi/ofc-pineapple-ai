"""
OFC Pineapple AI - Phase 8.5 Battle Training
3種類のPhase 8モデルを対戦させながら学習

特徴:
- Ultimate Rules FL (QQ=14, KK=15, AA=16, Trips=17)
- 3モデル対戦: Self-Play vs Aggressive vs Learner
- 連続ゲーム: FL状態引き継ぎ + ボタンローテーション
- 各対戦相手は定期的に最新の学習モデルに更新

モデル配置:
- Player 0: 学習エージェント（メイン学習対象）
- Player 1: Self-Playモデル（対戦相手1）
- Player 2: Aggressiveモデル（対戦相手2）

環境変数:
    NUM_ENVS: 並列環境数 (デフォルト: 4)
    DISCORD_WEBHOOK_URL: Discord通知用Webhook
"""

import os
import sys
import time
import random
import argparse
import numpy as np
from datetime import datetime
from collections import deque
from typing import Optional, List, Callable, Tuple, Dict
import torch
torch.distributions.Distribution.set_default_validate_args(False)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import gymnasium as gym

from ofc_3max_env import OFC3MaxEnv
from notifier import TrainingNotifier
from cloud_storage import init_cloud_storage

NUM_ENVS = int(os.getenv("NUM_ENVS", "4"))
DISCORD_WEBHOOK_URL = os.getenv(
    "DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1461388510140301315/Ofc9ok3IDLgRhFR7Oa2iA4-CHHL9uFvFbIPT9G9tbA5X5r-wG-XMbnkg7ubocLJJQ5Pf"
)

# Phase 8 モデルパス
SELFPLAY_MODEL = "models/phase8/p8_selfplay_5000000.zip"
AGGRESSIVE_MODEL = "models/phase8/aggressive_1000000.zip"


def load_manual_weights(model: MaskablePPO, zip_path: str) -> bool:
    """NumPy互換性のための手動重みロード"""
    import zipfile
    import io

    if not os.path.exists(zip_path):
        print(f"Warning: {zip_path} not found.")
        return False

    print(f"[Compatibility] Manual loading weights from {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with zip_ref.open('policy.pth') as f:
                state_dict = torch.load(io.BytesIO(f.read()), map_location='cpu')
                model.policy.load_state_dict(state_dict, strict=False)
        print("[Compatibility] Successfully loaded policy weights.")
        return True
    except Exception as e:
        print(f"[Compatibility] Failed to load weights: {e}")
        return False


def load_opponent_weights(zip_path: str) -> Optional[Dict[str, torch.Tensor]]:
    """対戦相手の重みをロード"""
    import zipfile
    import io

    if not os.path.exists(zip_path):
        print(f"Warning: {zip_path} not found.")
        return None

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with zip_ref.open('policy.pth') as f:
                state_dict = torch.load(io.BytesIO(f.read()), map_location='cpu')
                return {k: v.clone() for k, v in state_dict.items()}
    except Exception as e:
        print(f"Failed to load opponent weights from {zip_path}: {e}")
        return None


class BattleOpponentManager:
    """
    3モデル対戦用の対戦相手管理

    - opponent1: Self-Playモデル
    - opponent2: Aggressiveモデル
    - 定期的に学習モデルの重みで更新可能
    """

    def __init__(
        self,
        selfplay_path: str = SELFPLAY_MODEL,
        aggressive_path: str = AGGRESSIVE_MODEL,
        update_freq: int = 500_000
    ):
        self.selfplay_weights = load_opponent_weights(selfplay_path)
        self.aggressive_weights = load_opponent_weights(aggressive_path)
        self.learner_weights: Optional[Dict[str, torch.Tensor]] = None
        self.update_freq = update_freq
        self.last_update_step = 0

        # 対戦統計
        self.vs_selfplay_games = 0
        self.vs_selfplay_wins = 0
        self.vs_aggressive_games = 0
        self.vs_aggressive_wins = 0

        print(f"[Battle] Loaded Self-Play model: {selfplay_path}")
        print(f"[Battle] Loaded Aggressive model: {aggressive_path}")

    def get_opponent_weights(self, opponent_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        対戦相手の重みを取得

        Args:
            opponent_idx: 1 = Self-Play, 2 = Aggressive
        """
        if opponent_idx == 1:
            return self.selfplay_weights
        elif opponent_idx == 2:
            return self.aggressive_weights
        return None

    def update_learner_weights(self, model: MaskablePPO) -> None:
        """学習モデルの重みを保存（対戦相手更新用）"""
        self.learner_weights = {k: v.cpu().clone() for k, v in model.policy.state_dict().items()}

    def should_update(self, current_step: int) -> bool:
        """対戦相手の更新が必要かチェック"""
        return current_step - self.last_update_step >= self.update_freq

    def update_opponents_with_learner(self, current_step: int, mix_ratio: float = 0.3) -> None:
        """
        対戦相手を学習モデルの重みで部分的に更新

        Args:
            mix_ratio: 学習モデルの重みを混ぜる比率 (0.0-1.0)
        """
        if self.learner_weights is None:
            return

        # Self-Playモデルを更新（学習モデルに近づける）
        if self.selfplay_weights:
            for key in self.selfplay_weights:
                if key in self.learner_weights:
                    self.selfplay_weights[key] = (
                        (1 - mix_ratio) * self.selfplay_weights[key] +
                        mix_ratio * self.learner_weights[key]
                    )

        self.last_update_step = current_step
        print(f"[Battle] Updated opponents with learner weights (mix_ratio={mix_ratio})")

    def record_result(self, opponent_idx: int, won: bool) -> None:
        """対戦結果を記録"""
        if opponent_idx == 1:
            self.vs_selfplay_games += 1
            if won:
                self.vs_selfplay_wins += 1
        elif opponent_idx == 2:
            self.vs_aggressive_games += 1
            if won:
                self.vs_aggressive_wins += 1

    def get_stats(self) -> Dict[str, float]:
        """対戦統計を取得"""
        vs_selfplay_winrate = (
            self.vs_selfplay_wins / self.vs_selfplay_games * 100
            if self.vs_selfplay_games > 0 else 0
        )
        vs_aggressive_winrate = (
            self.vs_aggressive_wins / self.vs_aggressive_games * 100
            if self.vs_aggressive_games > 0 else 0
        )
        return {
            'vs_selfplay_winrate': vs_selfplay_winrate,
            'vs_selfplay_games': self.vs_selfplay_games,
            'vs_aggressive_winrate': vs_aggressive_winrate,
            'vs_aggressive_games': self.vs_aggressive_games,
        }


# グローバル変数
_global_battle_manager: Optional[BattleOpponentManager] = None
_global_inference_model: Optional[MaskablePPO] = None


def set_global_battle_manager(manager: BattleOpponentManager, model: MaskablePPO) -> None:
    global _global_battle_manager, _global_inference_model
    _global_battle_manager = manager
    _global_inference_model = model


class BattleEnv(gym.Env):
    """
    3モデル対戦用環境ラッパー

    Ultimate Rules FL + 連続ゲーム + ボタンローテーション
    """

    def __init__(self, env_id: int = 0):
        super().__init__()
        self.env = OFC3MaxEnv(
            enable_fl_turns=True,
            continuous_games=True
        )
        self.env_id = env_id
        self.learning_agent = "player_0"

        self.observation_space = self.env.observation_space(self.learning_agent)
        self.action_space = self.env.action_space(self.learning_agent)
        self.current_reward = 0

        # FL統計
        self.fl_entries = 0
        self.fl_stays = 0
        self.total_games = 0

        # 高得点統計（FL効果）
        self.high_score_games = 0

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        self.current_reward = 0
        self._play_opponents()
        return self.env.observe(self.learning_agent), {}

    def step(self, action):
        # 学習エージェントが既に終了している場合はNoneでステップ
        if self.env.terminations.get(self.learning_agent, False):
            self.env.step(None)
        else:
            self.env.step(action)
        self._play_opponents()

        obs = self.env.observe(self.learning_agent)
        reward = self.env._cumulative_rewards.get(self.learning_agent, 0) - self.current_reward
        self.current_reward += reward

        terminated = self.env.terminations.get(self.learning_agent, False)
        truncated = self.env.truncations.get(self.learning_agent, False)

        info = {}
        if terminated or truncated:
            self.total_games += 1
            res = self.env.engine.result()
            score = res.get_score(0)
            entered_fl = res.entered_fl(0)
            stayed_fl = res.stayed_fl(0)

            if entered_fl:
                self.fl_entries += 1
            if stayed_fl:
                self.fl_stays += 1
            if score >= 15:
                self.high_score_games += 1

            # 対戦相手ごとの勝敗を記録
            global _global_battle_manager
            if _global_battle_manager:
                # Player 1 (Self-Play)
                score_vs_p1 = self._calculate_heads_up_score(res, 0, 1)
                _global_battle_manager.record_result(1, score_vs_p1 > 0)
                # Player 2 (Aggressive)
                score_vs_p2 = self._calculate_heads_up_score(res, 0, 2)
                _global_battle_manager.record_result(2, score_vs_p2 > 0)

            info = {
                'score': score,
                'royalty': res.get_royalty(0),
                'fouled': res.is_fouled(0),
                'entered_fl': entered_fl,
                'stayed_fl': stayed_fl,
                'fl_cards_next': self.env.fl_cards_count[self.learning_agent],
                'win': score > 0,
                'loss': score < 0,
                'high_score': score >= 15,
            }
        return obs, reward, terminated, truncated, info

    def _calculate_heads_up_score(self, result, p1: int, p2: int) -> int:
        """2プレイヤー間のスコアを計算"""
        # 簡易計算: 全体スコアの差
        return result.get_score(p1) - result.get_score(p2)

    def _play_opponents(self):
        """対戦相手をプレイ"""
        max_iterations = 100  # 無限ループ防止

        for _ in range(max_iterations):
            # ゲーム終了チェック
            if all(self.env.terminations.values()):
                break

            # 学習エージェントの番になったら終了
            if self.env.agent_selection == self.learning_agent:
                # 学習エージェントが終了している場合はスキップ
                if self.env.terminations.get(self.learning_agent, False):
                    self.env.step(None)
                    continue
                break

            agent = self.env.agent_selection

            # 終了したエージェントはNoneでステップ
            if self.env.terminations.get(agent, False):
                self.env.step(None)
                continue

            # 有効なアクションを取得
            valid_actions = self.env.get_valid_actions(agent)
            if not valid_actions:
                # 有効なアクションがない場合はNone
                self.env.step(None)
                continue

            action = self._get_opponent_action(agent, valid_actions)
            self.env.step(action)

    def _get_opponent_action(self, agent: str, valid_actions: List[int]) -> int:
        """対戦相手のアクションを取得

        Note: Phase 8モデルは観測空間が異なるため、直接使用できない。
        現在はランダム選択にフォールバック。
        将来的には適切な観測空間変換を実装予定。
        """
        # TODO: Phase 8モデルの観測空間変換を実装
        # 現在はランダム選択（対戦相手としては弱い）
        return random.choice(valid_actions)

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks(self.learning_agent)


def mask_fn(env: BattleEnv) -> np.ndarray:
    return env.action_masks()


def make_env(env_id: int) -> Callable:
    def _init():
        env = BattleEnv(env_id=env_id)
        env = ActionMasker(env, mask_fn)
        return env
    return _init


class BattleCallback(BaseCallback):
    """3モデル対戦学習用コールバック"""

    def __init__(
        self,
        save_path: str,
        notifier: Optional[TrainingNotifier],
        cloud_storage,
        battle_manager: BattleOpponentManager,
        total_timesteps: int = 20_000_000,
        update_freq: int = 200_000,
        report_freq: int = 100_000
    ):
        super().__init__()
        self.save_path = save_path
        self.notifier = notifier
        self.cloud_storage = cloud_storage
        self.battle_manager = battle_manager
        self.total_timesteps = total_timesteps
        self.update_freq = update_freq
        self.report_freq = report_freq
        self.last_update = 0
        self.last_report = 0
        self.start_time = time.time()

        # 統計
        self.scores = deque(maxlen=500)
        self.royalties = deque(maxlen=500)
        self.fouls = deque(maxlen=500)
        self.fl_entries = deque(maxlen=500)
        self.fl_stays = deque(maxlen=500)
        self.high_scores = deque(maxlen=500)
        self.total_games = 0
        self.wins = 0
        self.losses = 0

    def _on_training_start(self):
        self.last_report = (self.num_timesteps // self.report_freq) * self.report_freq
        self.last_update = (self.num_timesteps // self.update_freq) * self.update_freq
        self.start_time = time.time()
        self.battle_manager.update_learner_weights(self.model)

    def _on_step(self):
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if isinstance(info, dict) and 'score' in info:
                    self._record_game(info)

        if self.num_timesteps - self.last_report >= self.report_freq:
            self.last_report = self.num_timesteps
            self._send_report()

        if self.num_timesteps - self.last_update >= self.update_freq:
            self.last_update = self.num_timesteps
            self._save_checkpoint()

        return True

    def _record_game(self, info: dict):
        score = info['score']
        self.scores.append(score)
        self.royalties.append(info.get('royalty', 0))
        self.fouls.append(1.0 if info.get('fouled', False) else 0.0)
        self.fl_entries.append(1.0 if info.get('entered_fl', False) else 0.0)
        self.fl_stays.append(1.0 if info.get('stayed_fl', False) else 0.0)
        self.high_scores.append(1.0 if info.get('high_score', False) else 0.0)
        self.total_games += 1

        if info.get('win', False):
            self.wins += 1
        elif info.get('loss', False):
            self.losses += 1

    def _send_report(self):
        elapsed = time.time() - self.start_time
        fps = self.num_timesteps / elapsed if elapsed > 0 else 0

        foul_rate = np.mean(self.fouls) * 100 if self.fouls else 0
        mean_royalty = np.mean(self.royalties) if self.royalties else 0
        mean_score = np.mean(self.scores) if self.scores else 0
        fl_entry_rate = np.mean(self.fl_entries) * 100 if self.fl_entries else 0
        fl_stay_rate = np.mean(self.fl_stays) * 100 if self.fl_stays else 0
        high_score_rate = np.mean(self.high_scores) * 100 if self.high_scores else 0

        battle_stats = self.battle_manager.get_stats()

        print(f"\n[Step {self.num_timesteps:,}] Phase 8.5 Battle")
        print(f"  Games: {self.total_games:,}")
        print(f"  Foul Rate: {foul_rate:.1f}%")
        print(f"  Mean Score: {mean_score:+.2f}")
        print(f"  Mean Royalty: {mean_royalty:.2f}")
        print(f"  FL Entry: {fl_entry_rate:.1f}%, FL Stay: {fl_stay_rate:.1f}%")
        print(f"  High Score (>=15): {high_score_rate:.1f}%")
        print(f"  vs Self-Play: {battle_stats['vs_selfplay_winrate']:.1f}% ({battle_stats['vs_selfplay_games']} games)")
        print(f"  vs Aggressive: {battle_stats['vs_aggressive_winrate']:.1f}% ({battle_stats['vs_aggressive_games']} games)")
        print(f"  FPS: {fps:.0f}")

        if self.notifier:
            metrics = {
                'games': self.total_games,
                'foul_rate': foul_rate,
                'mean_score': mean_score,
                'mean_royalty': mean_royalty,
                'fl_entry_rate': fl_entry_rate,
                'fl_stay_rate': fl_stay_rate,
                'high_score_rate': high_score_rate,
                'vs_selfplay_winrate': battle_stats['vs_selfplay_winrate'],
                'vs_aggressive_winrate': battle_stats['vs_aggressive_winrate'],
                'fps': fps,
            }
            self.notifier.send_progress(self.num_timesteps, self.total_timesteps, metrics)

    def _save_checkpoint(self):
        # 学習モデルの重みを更新
        self.battle_manager.update_learner_weights(self.model)

        # 対戦相手を部分的に更新（学習モデルに近づける）
        if self.battle_manager.should_update(self.num_timesteps):
            self.battle_manager.update_opponents_with_learner(self.num_timesteps, mix_ratio=0.2)

        # チェックポイント保存
        path = f"{self.save_path}_{self.num_timesteps}"
        self.model.save(path)

        if self.notifier:
            self.notifier.send_checkpoint(f"{path}.zip", self.num_timesteps)
        if self.cloud_storage and self.cloud_storage.enabled:
            self.cloud_storage.upload_checkpoint(f"{path}.zip", step=self.num_timesteps)

        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        import glob
        checkpoint_dir = os.path.dirname(self.save_path)
        if not checkpoint_dir:
            checkpoint_dir = "."

        files = glob.glob(os.path.join(checkpoint_dir, "p85_battle_*.zip"))
        if not files:
            return

        def get_step(f):
            try:
                return int(f.split('_')[-1].replace('.zip', ''))
            except:
                return 0

        sorted_files = sorted(files, key=get_step)

        for f in sorted_files[:-2]:
            step = get_step(f)
            if step % 1_000_000 == 0:
                continue
            try:
                os.remove(f)
            except Exception as e:
                print(f"[Cleanup] Error: {e}")


def train_phase85_battle(
    total_timesteps: int = 20_000_000,
    test_mode: bool = False,
    num_envs: int = None
):
    """Phase 8.5 Battle Training"""
    if num_envs is None:
        num_envs = NUM_ENVS

    if test_mode:
        total_timesteps = 10_000
        num_envs = 2
        print("[TEST MODE]")

    print("=" * 60)
    print("OFC Pineapple AI - Phase 8.5: Battle Training")
    print("=" * 60)
    print(f"Parallel Envs: {num_envs}")
    print(f"Total Steps: {total_timesteps:,}")
    print(f"Features:")
    print(f"  - Ultimate Rules FL (QQ=14, KK=15, AA=16, Trips=17)")
    print(f"  - 3-Model Battle: Learner vs Self-Play vs Aggressive")
    print(f"  - Continuous Games + Button Rotation")
    print()

    cloud_storage = init_cloud_storage()
    notifier = TrainingNotifier(
        discord_webhook=DISCORD_WEBHOOK_URL,
        project_name=f"Phase 8.5: Battle ({num_envs} envs)"
    )

    # 対戦相手管理
    battle_manager = BattleOpponentManager(
        selfplay_path=SELFPLAY_MODEL,
        aggressive_path=AGGRESSIVE_MODEL,
        update_freq=500_000
    )

    # DummyVecEnvを使用（SubprocVecEnvはマルチプロセスで複雑な状態管理が必要）
    print(f"[*] Creating {num_envs} environments (DummyVecEnv for stability)...")
    env = DummyVecEnv([make_env(i) for i in range(num_envs)])

    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    import glob

    # Phase 8.5 Battleチェックポイントを探す
    p85_checkpoints = glob.glob(os.path.join(save_dir, "p85_battle_*.zip"))

    latest_checkpoint = None
    is_resume = False

    if p85_checkpoints:
        latest_checkpoint = max(p85_checkpoints, key=lambda f: int(f.split('_')[-1].replace('.zip', '')))
        print(f"[*] Found checkpoint: {latest_checkpoint}. Resuming...")
        model = MaskablePPO.load(latest_checkpoint, env=env)
        is_resume = True
    else:
        print("[*] Starting new training (fresh model with Ultimate Rules observation space)...")
        print("[*] Note: Phase 8 models used as opponents only (different observation space)")
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,
            gamma=0.99,
            n_steps=2048,
            batch_size=256,
            tensorboard_log="./logs/phase85_battle/"
        )
        # Phase 8モデルは観測空間が異なるため、対戦相手としてのみ使用
        # 学習モデルは新規で開始

    set_global_battle_manager(battle_manager, model)

    callback = BattleCallback(
        "models/p85_battle",
        notifier,
        cloud_storage,
        battle_manager,
        total_timesteps=total_timesteps
    )

    if notifier and not test_mode:
        notifier.send_start({
            'timesteps': total_timesteps,
            'num_envs': num_envs,
            'opponents': 'Self-Play + Aggressive',
            'features': 'Ultimate Rules FL + Battle'
        })

    if is_resume and latest_checkpoint:
        try:
            checkpoint_name = os.path.basename(latest_checkpoint)
            current_steps = int(checkpoint_name.split('_')[-1].replace('.zip', ''))
        except:
            current_steps = 0
        remaining_steps = max(0, total_timesteps - current_steps)
        reset_num_timesteps = False
    else:
        remaining_steps = total_timesteps
        reset_num_timesteps = True

    print(f"\nStarting Battle Training (Goal: {total_timesteps:,} steps)")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=remaining_steps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps
        )

        if notifier and not test_mode:
            battle_stats = battle_manager.get_stats()
            notifier.send_complete({
                'total_steps': total_timesteps,
                'total_games': callback.total_games,
                'foul_rate': np.mean(callback.fouls) * 100 if callback.fouls else 0,
                'mean_royalty': np.mean(callback.royalties) if callback.royalties else 0,
                'vs_selfplay_winrate': battle_stats['vs_selfplay_winrate'],
                'vs_aggressive_winrate': battle_stats['vs_aggressive_winrate'],
            })

    except KeyboardInterrupt:
        print("\n[!] Training interrupted")
        model.save("models/p85_battle_interrupted")
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        if notifier:
            notifier.send_error(str(e), traceback.format_exc())
        raise
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 8.5 Battle Training")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--steps", type=int, default=20_000_000)
    parser.add_argument("--envs", type=int, default=None)

    args = parser.parse_args()

    train_phase85_battle(
        total_timesteps=args.steps,
        test_mode=args.test_mode,
        num_envs=args.envs
    )
