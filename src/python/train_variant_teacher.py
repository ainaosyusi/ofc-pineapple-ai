"""
OFC Pineapple AI - Teacher Learning Variant
ルールベース教師からの模倣学習

特徴:
- ルールベースの「教師」エージェントを参照
- 教師の行動との一致にボーナス報酬
- 徐々に自律的な判断へ移行

Phase 7を基盤として、教師の知識を取り込む。
"""

import os
import sys
import time
import random
import argparse
import numpy as np
from datetime import datetime
from collections import deque, Counter
from typing import Optional, Callable, Dict, Any, List
import torch
torch.distributions.Distribution.set_default_validate_args(False)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import gymnasium as gym

import ofc_engine as ofc
from ofc_3max_env import OFC3MaxEnv
from notifier import TrainingNotifier
from cloud_storage import init_cloud_storage

NUM_ENVS = int(os.getenv("NUM_ENVS", "4"))
DISCORD_WEBHOOK_URL = os.getenv(
    "DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1462114012085882892/qvX-MEc9aO9sHuBAeLb8wAgbl84oFn_z3vekxBQXQSHNXnOoujwOGV7G0_nJNR_rvnaY"
)


def load_manual_weights(model: MaskablePPO, zip_path: str) -> bool:
    """NumPy互換性のための手動重みロード"""
    import zipfile
    import io

    if not os.path.exists(zip_path):
        print(f"Warning: {zip_path} not found.")
        return False

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with zip_ref.open('policy.pth') as f:
                state_dict = torch.load(io.BytesIO(f.read()), map_location='cpu')
                model.policy.load_state_dict(state_dict, strict=False)
        print(f"[*] Loaded weights from {zip_path}")
        return True
    except Exception as e:
        print(f"[!] Failed to load weights: {e}")
        return False


class RuleBasedTeacher:
    """
    ルールベースの教師エージェント

    戦略:
    1. ファウル回避を最優先
    2. 高い役を狙いつつ安全策を取る
    3. FL突入のチャンスがあれば狙う
    """

    def __init__(self):
        pass

    def evaluate_action(self, engine, player_idx: int, action: int) -> float:
        """
        アクションの評価スコアを計算

        Returns:
            score: アクションの評価値（高いほど良い）
        """
        ps = engine.player(player_idx)
        board = ps.board

        # アクションをデコード
        card1_idx = action // (54 * 4)
        remainder = action % (54 * 4)
        card2_idx = remainder // 4
        placement = remainder % 4

        # 配置先を決定
        if placement == 0:
            row1, row2 = "top", "mid"
        elif placement == 1:
            row1, row2 = "top", "bot"
        elif placement == 2:
            row1, row2 = "mid", "bot"
        else:  # placement == 3 (discard)
            row1, row2 = "mid", "bot"

        score = 0.0

        # ファウルリスク評価（シミュレーション）
        # 実際にはボードの状態から推定
        top_count = bin(board.top_mask()).count('1')
        mid_count = bin(board.mid_mask()).count('1')
        bot_count = bin(board.bot_mask()).count('1')

        # 基本スコア
        score += 10.0  # ベースライン

        # 配置バランス（均等に配置することを好む）
        if top_count < 3 and "top" in [row1, row2]:
            score += 2.0
        if mid_count < 5 and "mid" in [row1, row2]:
            score += 1.5
        if bot_count < 5 and "bot" in [row1, row2]:
            score += 1.0

        # ペア/トリップス形成の可能性
        # (簡易評価)

        return score

    def get_best_action(self, engine, player_idx: int, valid_actions: List[int]) -> int:
        """
        最適なアクションを選択

        ヒューリスティック:
        1. 各アクションを評価
        2. 最高評価のアクションを選択
        """
        if not valid_actions:
            return 0

        # 簡易版: ランダムより少しマシな選択
        # 完全なルールベースは複雑すぎるので、
        # 「ファウルしにくい配置」を優先

        ps = engine.player(player_idx)
        board = ps.board

        top_count = bin(board.top_mask()).count('1')
        mid_count = bin(board.mid_mask()).count('1')
        bot_count = bin(board.bot_mask()).count('1')

        best_action = valid_actions[0]
        best_score = -float('inf')

        for action in valid_actions[:min(len(valid_actions), 50)]:  # 上位50件を評価
            placement = (action % (54 * 4)) % 4

            score = 0.0

            # 配置バランスを重視
            if placement == 0:  # top, mid
                if top_count < 3:
                    score += 3.0
                if mid_count < 5:
                    score += 2.0
            elif placement == 1:  # top, bot
                if top_count < 3:
                    score += 3.0
                if bot_count < 5:
                    score += 1.5
            elif placement == 2:  # mid, bot
                if mid_count < 5:
                    score += 2.0
                if bot_count < 5:
                    score += 1.5
            else:  # discard
                score += 0.5  # 捨てる場合は低評価

            # ランダム性を少し加える
            score += random.random() * 0.5

            if score > best_score:
                best_score = score
                best_action = action

        return best_action


class TeacherLearningEnv(gym.Env):
    """
    教師学習用環境（FL対応）

    報酬設計:
    - 基本報酬: ゲームスコア
    - 教師一致ボーナス: 教師の選択と一致した場合 +bonus
    - 教師ボーナスは学習が進むと減衰
    - FL突入/継続ボーナス

    FL対応:
    - enable_fl_turns=Trueで相手のFLターンを実装
    - FLプレイヤーはFantasySolverで最適配置
    """

    def __init__(self, env_id: int = 0, teacher_bonus: float = 5.0, enable_fl_turns: bool = True):
        super().__init__()
        self.env = OFC3MaxEnv(enable_fl_turns=enable_fl_turns, continuous_games=False)
        self.env_id = env_id
        self.learning_agent = "player_0"
        self.teacher = RuleBasedTeacher()

        self.observation_space = self.env.observation_space(self.learning_agent)
        self.action_space = self.env.action_space(self.learning_agent)

        # 教師学習パラメータ
        self.teacher_bonus = teacher_bonus
        self.fl_bonus = 15.0  # FL突入ボーナス
        self.fl_stay_bonus = 10.0  # FL継続ボーナス
        self.teacher_matches = 0
        self.total_decisions = 0

        # 最後の教師推奨アクション
        self.last_teacher_action = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.env.reset(seed=seed)
        self._play_opponents()

        # 教師の推奨アクションを計算
        valid_actions = self.env.get_valid_actions(self.learning_agent)
        if valid_actions:
            self.last_teacher_action = self.teacher.get_best_action(
                self.env.engine, 0, valid_actions
            )
        else:
            self.last_teacher_action = None

        obs = self.env.observe(self.learning_agent)
        return obs, {}

    def step(self, action):
        # 教師との一致チェック
        teacher_match = (action == self.last_teacher_action)
        if teacher_match:
            self.teacher_matches += 1
        self.total_decisions += 1

        self.env.step(action)
        self._play_opponents()

        obs = self.env.observe(self.learning_agent)
        terminated = self.env.terminations.get(self.learning_agent, False)
        truncated = False

        reward = 0.0
        info = {}

        if not terminated:
            # 中間報酬: 教師一致ボーナス
            if teacher_match:
                reward += self.teacher_bonus * 0.1  # 中間は小さめ

            # 次の教師推奨を計算
            valid_actions = self.env.get_valid_actions(self.learning_agent)
            if valid_actions:
                self.last_teacher_action = self.teacher.get_best_action(
                    self.env.engine, 0, valid_actions
                )
        else:
            result = self.env.engine.result()
            base_score = result.get_score(0)
            royalty = result.get_royalty(0)
            fouled = result.is_fouled(0)
            entered_fl = result.entered_fl(0)
            stayed_fl = result.stayed_fl(0)

            # 基本報酬
            reward = base_score

            # FL突入ボーナス
            if entered_fl:
                reward += self.fl_bonus

            # FL継続ボーナス
            if stayed_fl:
                reward += self.fl_stay_bonus

            # 教師一致率に基づくボーナス
            match_rate = self.teacher_matches / max(1, self.total_decisions)
            reward += match_rate * self.teacher_bonus

            info = {
                'score': base_score,
                'royalty': royalty,
                'fouled': fouled,
                'entered_fl': entered_fl,
                'stayed_fl': stayed_fl,
                'variant': 'teacher',
                'teacher_match_rate': match_rate * 100,
                'win': base_score > 0,
                'loss': base_score < 0,
                'draw': base_score == 0
            }

            # リセット
            self.teacher_matches = 0
            self.total_decisions = 0

        return obs, reward, terminated, truncated, info

    def _play_opponents(self):
        while not all(self.env.terminations.values()) and \
              self.env.agent_selection != self.learning_agent:
            agent = self.env.agent_selection
            if self.env.terminations.get(agent, False):
                self.env.step(None)
                continue
            valid_actions = self.env.get_valid_actions(agent)
            action = random.choice(valid_actions) if valid_actions else 0
            self.env.step(action)

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks(self.learning_agent)


def mask_fn(env):
    return env.action_masks()


def make_env(env_id: int, teacher_bonus: float = 5.0):
    def _init():
        env = TeacherLearningEnv(env_id=env_id, teacher_bonus=teacher_bonus)
        env = ActionMasker(env, mask_fn)
        return env
    return _init


class TeacherCallback(BaseCallback):
    """教師学習バリアント用コールバック"""

    def __init__(
        self,
        save_path: str,
        notifier: Optional[TrainingNotifier],
        cloud_storage,
        total_timesteps: int = 20_000_000,
        update_freq: int = 200_000,
        report_freq: int = 100_000
    ):
        super().__init__()
        self.save_path = save_path
        self.notifier = notifier
        self.cloud_storage = cloud_storage
        self.total_timesteps = total_timesteps
        self.update_freq = update_freq
        self.report_freq = report_freq
        self.last_update = 0
        self.last_report = 0
        self.start_time = time.time()

        self.scores = deque(maxlen=500)
        self.royalties = deque(maxlen=500)
        self.fouls = deque(maxlen=500)
        self.fl_entries = deque(maxlen=500)
        self.teacher_match_rates = deque(maxlen=500)
        self.total_games = 0
        self.wins = 0
        self.losses = 0

    def _on_training_start(self):
        self.last_report = (self.num_timesteps // self.report_freq) * self.report_freq
        self.last_update = (self.num_timesteps // self.update_freq) * self.update_freq
        self.start_time = time.time()

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
        self.scores.append(info['score'])
        self.royalties.append(info.get('royalty', 0))
        self.fouls.append(1.0 if info.get('fouled', False) else 0.0)
        self.fl_entries.append(1.0 if info.get('entered_fl', False) else 0.0)
        self.teacher_match_rates.append(info.get('teacher_match_rate', 0))
        self.total_games += 1
        if info.get('win'):
            self.wins += 1
        elif info.get('loss'):
            self.losses += 1

    def _send_report(self):
        elapsed = time.time() - self.start_time
        fps = self.num_timesteps / elapsed if elapsed > 0 else 0

        foul_rate = np.mean(self.fouls) * 100 if self.fouls else 0
        mean_royalty = np.mean(self.royalties) if self.royalties else 0
        mean_score = np.mean(self.scores) if self.scores else 0
        fl_rate = np.mean(self.fl_entries) * 100 if self.fl_entries else 0
        win_rate = self.wins / self.total_games * 100 if self.total_games > 0 else 0
        teacher_match = np.mean(self.teacher_match_rates) if self.teacher_match_rates else 0

        print(f"\n[Step {self.num_timesteps:,}] TEACHER LEARNING VARIANT")
        print(f"  Games: {self.total_games:,}")
        print(f"  Foul Rate: {foul_rate:.1f}%")
        print(f"  Mean Score: {mean_score:+.2f}")
        print(f"  Mean Royalty: {mean_royalty:.2f}")
        print(f"  FL Entry Rate: {fl_rate:.1f}%")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Teacher Match: {teacher_match:.1f}%")
        print(f"  FPS: {fps:.0f}")

        if self.notifier:
            metrics = {
                'games': self.total_games,
                'foul_rate': foul_rate,
                'mean_score': mean_score,
                'mean_royalty': mean_royalty,
                'fl_rate': fl_rate,
                'win_rate': win_rate,
                'teacher_match': teacher_match,
                'fps': fps,
                'variant': 'TEACHER'
            }
            self.notifier.send_progress(self.num_timesteps, self.total_timesteps, metrics)

    def _save_checkpoint(self):
        path = f"{self.save_path}_{self.num_timesteps}"
        self.model.save(path)

        if self.notifier:
            self.notifier.send_checkpoint(f"{path}.zip", self.num_timesteps)
        if self.cloud_storage and self.cloud_storage.enabled:
            self.cloud_storage.upload_checkpoint(f"{path}.zip", step=self.num_timesteps)

        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        import glob
        checkpoints = sorted(glob.glob(f"{self.save_path}_*.zip"))
        milestones = {1_000_000, 5_000_000, 10_000_000, 15_000_000, 20_000_000}

        for cp in checkpoints[:-2]:
            step = int(cp.split('_')[-1].replace('.zip', ''))
            if step not in milestones:
                try:
                    os.remove(cp)
                except:
                    pass


def main():
    parser = argparse.ArgumentParser(description="Teacher Learning Variant Training")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--steps", type=int, default=20_000_000)
    parser.add_argument("--teacher-bonus", type=float, default=5.0)
    args = parser.parse_args()

    total_timesteps = args.steps if args.test_mode else 20_000_000
    save_path = "models/teacher"
    base_model_path = "models/p7_parallel_20000000.zip"

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs/teacher", exist_ok=True)

    cloud_storage = init_cloud_storage()
    notifier = TrainingNotifier(
        discord_webhook=DISCORD_WEBHOOK_URL,
        project_name="OFC AI - TEACHER"
    )

    # 環境作成
    num_envs = 2 if args.test_mode else NUM_ENVS
    if num_envs > 1:
        env = SubprocVecEnv([make_env(i, args.teacher_bonus) for i in range(num_envs)])
    else:
        env = DummyVecEnv([make_env(0, args.teacher_bonus)])

    # モデル作成
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./logs/teacher"
    )

    # Phase 7の重みをロード
    if os.path.exists(base_model_path):
        load_manual_weights(model, base_model_path)
        print(f"[*] Loaded base model: {base_model_path}")

    # レジューム確認
    import glob
    checkpoints = sorted(glob.glob(f"{save_path}_*.zip"))
    remaining_steps = total_timesteps
    reset_num_timesteps = True

    if checkpoints:
        latest = checkpoints[-1]
        step = int(latest.split('_')[-1].replace('.zip', ''))
        if step < total_timesteps:
            load_manual_weights(model, latest)
            remaining_steps = total_timesteps - step
            reset_num_timesteps = False
            print(f"[*] Resuming from {latest}, remaining: {remaining_steps:,}")

    callback = TeacherCallback(
        save_path=save_path,
        notifier=notifier,
        cloud_storage=cloud_storage,
        total_timesteps=total_timesteps
    )

    print(f"\nStarting TEACHER LEARNING Variant Training (Goal: {total_timesteps:,} steps)")
    print("=" * 60)

    notifier.send_start({
        'timesteps': total_timesteps,
        'variant': 'TEACHER',
        'teacher_bonus': args.teacher_bonus,
        'lr': 1e-4
    })

    model.learn(
        total_timesteps=remaining_steps,
        callback=callback,
        reset_num_timesteps=reset_num_timesteps,
        progress_bar=True
    )

    final_path = f"{save_path}_final"
    model.save(final_path)
    print(f"\n[*] Training complete! Final model: {final_path}.zip")


if __name__ == "__main__":
    main()
