"""
OFC Pineapple AI - FL Specialist Training
Fantasy Land特化型AIの学習

戦略:
- Top: A/K/Q ペア or トリップス狙い
- Middle/Bottom: 2ペア以上、ストレート/フラッシュドロー
- ファウル率は気にしない（FL継続重視）

報酬設計:
- FL突入: +50
- FL継続: +40 (トリップス), +35 (AA), +30 (KK), +25 (QQ)
- 通常スコア: x0.5 (抑制)
- ファウル: -5 (軽い)
"""

import os
import sys
import time
import random
import argparse
import multiprocessing as mp
import numpy as np
from datetime import datetime
from collections import deque
from typing import Optional, Dict, List
import torch
torch.distributions.Distribution.set_default_validate_args(False)

# SubprocVecEnv用: spawnメソッドを使用
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn')

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import gymnasium as gym

from ofc_3max_env import OFC3MaxEnv
from notifier import TrainingNotifier

# 設定
NUM_ENVS = int(os.getenv("NUM_ENVS", "4"))
DISCORD_WEBHOOK_URL = os.getenv(
    "DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1462113917571694632/iQ26kCzDmJ-DLA9TD_IRYNh4_TWc3UOxnVEa2-890B66mEYvrm7jufLOsMs_ZaATq2bb"
)


class FLSpecialistMetrics:
    """FL特化メトリクス追跡"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_games = 0
        self.total_fouls = 0
        self.total_fl_entries = 0
        self.total_score = 0.0

        # FL継続追跡（Topハンド別）
        self.fl_continuations = {
            'trips': {'attempts': 0, 'stays': 0},
            'aa': {'attempts': 0, 'stays': 0},
            'kk': {'attempts': 0, 'stays': 0},
            'qq': {'attempts': 0, 'stays': 0},
        }

    def record_game(self, fouled: bool, entered_fl: bool, score: float):
        self.total_games += 1
        if fouled:
            self.total_fouls += 1
        if entered_fl:
            self.total_fl_entries += 1
        self.total_score += score

    def record_fl_attempt(self, fl_type: str, stayed: bool):
        """FL継続試行を記録"""
        if fl_type in self.fl_continuations:
            self.fl_continuations[fl_type]['attempts'] += 1
            if stayed:
                self.fl_continuations[fl_type]['stays'] += 1

    @property
    def foul_rate(self) -> float:
        return self.total_fouls / max(1, self.total_games) * 100

    @property
    def fl_entry_rate(self) -> float:
        return self.total_fl_entries / max(1, self.total_games) * 100

    @property
    def mean_score(self) -> float:
        return self.total_score / max(1, self.total_games)

    def fl_stay_rate(self, fl_type: str) -> float:
        """FL継続率"""
        data = self.fl_continuations.get(fl_type, {'attempts': 0, 'stays': 0})
        return data['stays'] / max(1, data['attempts']) * 100

    def get_summary(self) -> Dict:
        return {
            'games': self.total_games,
            'foul_rate': self.foul_rate,
            'fl_entry_rate': self.fl_entry_rate,
            'mean_score': self.mean_score,
            'trips_stay_rate': self.fl_stay_rate('trips'),
            'aa_stay_rate': self.fl_stay_rate('aa'),
            'kk_stay_rate': self.fl_stay_rate('kk'),
            'qq_stay_rate': self.fl_stay_rate('qq'),
        }


class FLSpecialistEnv(gym.Env):
    """FL特化型環境ラッパー

    報酬設計（FL継続を絶対優先）:
    - FL継続: +100 (最優先)
    - FL突入: +50
    - FL中にファウル: -50 (継続失敗は大きなペナルティ)
    - 通常ファウル: -5 (軽い)
    - 点数: FL継続時のみ加算、それ以外は抑制
    """

    # FL報酬（継続を絶対優先）
    FL_ENTRY_BONUS = 50.0
    FL_STAY_BONUS = 100.0  # 継続は最優先
    FL_FAIL_PENALTY = -50.0  # FL中のファウル/継続失敗

    # タイプ別追加ボーナス（継続した場合）
    FL_TYPE_BONUS = {
        'trips': 20.0,  # 17枚で継続 = 最高評価
        'aa': 15.0,
        'kk': 10.0,
        'qq': 5.0,
    }

    FOUL_PENALTY = -5.0  # 通常時のファウル
    SCORE_MULTIPLIER_FL_STAY = 1.0  # FL継続時は点数もフル加算
    SCORE_MULTIPLIER_NORMAL = 0.3  # 通常時は点数抑制

    def __init__(self, env_id: int = 0):
        super().__init__()
        # FL継続を有効化、ただしFL自動プレイは無効（ソルバーが遅すぎるため）
        # AIがFLハンドも自分でプレイする
        self.env = OFC3MaxEnv(
            enable_fl_turns=False,  # ソルバーによる自動配置を無効化
            continuous_games=True
        )
        self.env_id = env_id
        self.learning_agent = "player_0"

        # メトリクス
        self.metrics = FLSpecialistMetrics()

        # 空間定義
        self.observation_space = self.env.observation_space(self.learning_agent)
        self.action_space = self.env.action_space(self.learning_agent)

        # FL状態追跡
        self.was_in_fl = False
        self.fl_type = None  # 'trips', 'aa', 'kk', 'qq'

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed)

        # FL状態を引き継ぎ（reset後にチェック - solverで自動処理済み）
        player_idx = self.env.agent_name_mapping[self.learning_agent]
        ps = self.env.engine.player(player_idx)
        self.was_in_fl = ps.in_fantasy_land
        if self.was_in_fl:
            # FL中のカード枚数は14-17（Topハンドに依存）
            fl_cards = self.env.fl_cards_count.get(self.learning_agent, 14)
            self.fl_type = self._get_fl_type(fl_cards)
        else:
            self.fl_type = None

        # FL中のプレイヤーは solver で自動処理されるので、
        # 他のプレイヤーをプレイ（学習エージェントがFL中の場合はゲーム終了まで進む）
        self._play_until_my_turn_or_end()

        return self.env.observe(self.learning_agent), {}

    def _play_until_my_turn_or_end(self):
        """自分のターンまで、またはゲーム終了まで進める"""
        max_iterations = 1000  # 安全のための上限
        iterations = 0

        while iterations < max_iterations:
            # ゲーム終了チェック
            if all(self.env.terminations.get(a, False) for a in self.env.possible_agents):
                break

            current_agent = self.env.agent_selection

            # 自分のターンならループを抜ける
            if current_agent == self.learning_agent:
                if not self.env.terminations.get(self.learning_agent, False):
                    break

            # 相手をプレイ
            if self.env.terminations.get(current_agent, False):
                self.env.step(None)
            else:
                # ボードが完成していないか確認
                player_idx = self.env.agent_name_mapping[current_agent]
                ps = self.env.engine.player(player_idx)
                if ps.board.total_placed() >= 13:
                    # 既に完成 → terminated扱いでスキップ
                    self.env.step(None)
                else:
                    mask = self.env.action_masks(current_agent)
                    valid = np.where(mask == 1)[0]
                    action = np.random.choice(valid) if len(valid) > 0 else 0
                    self.env.step(action)

            iterations += 1

    def _get_fl_type(self, fl_cards: int) -> str:
        """FL枚数からタイプを判定"""
        if fl_cards >= 17:
            return 'trips'
        elif fl_cards >= 16:
            return 'aa'
        elif fl_cards >= 15:
            return 'kk'
        else:
            return 'qq'

    def step(self, action):
        # 自分がterminated済みの場合は何もしない
        if self.env.terminations.get(self.learning_agent, False):
            self.env.step(None)
        else:
            self.env.step(action)

        # 相手のターンを進める（自分のターンまたはゲーム終了まで）
        self._play_until_my_turn_or_end()

        # ゲーム終了チェック（全員terminated）
        all_terminated = all(self.env.terminations.get(a, False) for a in self.env.possible_agents)

        obs = self.env.observe(self.learning_agent)
        terminated = all_terminated  # ゲーム終了時にterminatedをTrueに
        truncated = self.env.truncations.get(self.learning_agent, False)
        info = self.env.infos.get(self.learning_agent, {}).copy()

        # FL特化報酬計算（ゲーム終了時のみ）
        reward = 0.0
        if all_terminated:
            reward = self._calculate_fl_reward(info)

            # コールバック用にFL情報を追加
            if self.was_in_fl:
                info['fl_type'] = self.fl_type
                info['fl_stayed'] = info.get('stayed_fl', False)

            # 次ゲームのFL状態を記録（reset時に使用）
            self._record_next_fl_state()

        return obs, reward, terminated, truncated, info

    def _record_next_fl_state(self):
        """次のゲームのFL状態を記録"""
        # infoにFL継続情報がある場合、次のゲームのFL枚数を記録
        info = self.env.infos.get(self.learning_agent, {})
        if info.get('entered_fl') or info.get('stayed_fl'):
            # FL突入/継続した場合、次ゲームのFL枚数を保存
            self._next_fl_cards = info.get('next_fl_cards', 14)
        else:
            self._next_fl_cards = 0

    def _update_fl_state_for_next_game(self):
        """次のゲームのためにFL状態を更新"""
        # 次のゲームでFL中かどうかを確認
        self.was_in_fl = self.env.fl_status.get(self.learning_agent, False)
        if self.was_in_fl:
            fl_cards = self.env.fl_cards_count.get(self.learning_agent, 14)
            self.fl_type = self._get_fl_type(fl_cards)
        else:
            self.fl_type = None

    def _calculate_fl_reward(self, info: Dict) -> float:
        """FL特化報酬を計算（FL継続を絶対優先）

        思考手順:
        1. FL中の場合、まず継続可能かを判定
        2. 継続成功 → 大ボーナス + 点数フル加算
        3. 継続失敗 → 大ペナルティ
        4. FL外の場合 → FL突入を狙う、点数は抑制
        """
        base_score = info.get('score', 0)
        entered_fl = info.get('entered_fl', False)
        stayed_fl = info.get('stayed_fl', False)
        fouled = info.get('fouled', False)

        # 記録
        self.metrics.record_game(fouled, entered_fl, base_score)

        reward = 0.0

        # FL中だった場合（継続判定）
        if self.was_in_fl and self.fl_type:
            self.metrics.record_fl_attempt(self.fl_type, stayed_fl)

            if stayed_fl:
                # FL継続成功！最優先で報酬
                reward += self.FL_STAY_BONUS
                reward += self.FL_TYPE_BONUS.get(self.fl_type, 5.0)
                reward += base_score * self.SCORE_MULTIPLIER_FL_STAY
            else:
                # FL継続失敗 → 大きなペナルティ
                reward += self.FL_FAIL_PENALTY
                # ファウルだった場合は追加ペナルティなし（FL_FAIL_PENALTYに含む）

        else:
            # FL外からのプレイ
            # FL突入ボーナス
            if entered_fl:
                reward += self.FL_ENTRY_BONUS

            # 点数は抑制（FL突入を優先させるため）
            reward += base_score * self.SCORE_MULTIPLIER_NORMAL

            # 通常時のファウルは軽いペナルティ
            if fouled:
                reward += self.FOUL_PENALTY

        return reward


    def action_masks(self) -> np.ndarray:
        if self.env.terminations.get(self.learning_agent, False):
            mask = np.zeros(243, dtype=np.int8)
            mask[0] = 1
            return mask
        return self.env.action_masks(self.learning_agent)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class FLSpecialistCallback(BaseCallback):
    """FL特化学習用コールバック"""

    def __init__(
        self,
        save_dir: str,
        save_freq: int = 100_000,
        notifier: Optional[TrainingNotifier] = None,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.notifier = notifier

        self.last_report = 0
        self.metrics = FLSpecialistMetrics()
        self.best_fl_rate = 0.0

    def _on_step(self) -> bool:
        # info辞書からメトリクスを収集（SubprocVecEnv対応）
        infos = self.locals.get('infos', [])
        for info in infos:
            if info:
                # ゲーム終了時のみ記録
                if info.get('entered_fl') is not None or info.get('fouled') is not None:
                    self.metrics.record_game(
                        fouled=info.get('fouled', False),
                        entered_fl=info.get('entered_fl', False),
                        score=info.get('score', 0)
                    )
                # FL継続情報
                if info.get('fl_type') and info.get('fl_stayed') is not None:
                    self.metrics.record_fl_attempt(
                        info.get('fl_type'),
                        info.get('fl_stayed', False)
                    )

        # 定期レポート
        current_steps = self.num_timesteps
        if current_steps - self.last_report >= self.save_freq:
            self._report_and_save(current_steps)
            self.last_report = current_steps

        return True

    def _merge_metrics(self, other: FLSpecialistMetrics):
        """メトリクスを統合"""
        self.metrics.total_games += other.total_games
        self.metrics.total_fouls += other.total_fouls
        self.metrics.total_fl_entries += other.total_fl_entries
        self.metrics.total_score += other.total_score

        for fl_type in self.metrics.fl_continuations:
            self.metrics.fl_continuations[fl_type]['attempts'] += \
                other.fl_continuations[fl_type]['attempts']
            self.metrics.fl_continuations[fl_type]['stays'] += \
                other.fl_continuations[fl_type]['stays']

    def _report_and_save(self, steps: int):
        """レポート出力とモデル保存"""
        summary = self.metrics.get_summary()

        print("\n" + "=" * 60)
        print(f"FL Specialist Training - {steps:,} steps")
        print("=" * 60)
        print(f"  Games:        {summary['games']:,}")
        print(f"  Foul Rate:    {summary['foul_rate']:.1f}%")
        print(f"  FL Entry:     {summary['fl_entry_rate']:.1f}%")
        print(f"  Mean Score:   {summary['mean_score']:.2f}")
        print("-" * 40)
        print("  FL Continuation Rates:")
        print(f"    Trips (17):  {summary['trips_stay_rate']:.1f}%")
        print(f"    AA (16):     {summary['aa_stay_rate']:.1f}%")
        print(f"    KK (15):     {summary['kk_stay_rate']:.1f}%")
        print(f"    QQ (14):     {summary['qq_stay_rate']:.1f}%")
        print("=" * 60 + "\n")

        # モデル保存
        save_path = os.path.join(self.save_dir, f"fl_specialist_{steps}.zip")
        self.model.save(save_path)
        print(f"[*] Saved: {save_path}")

        # ベスト更新チェック
        if summary['fl_entry_rate'] > self.best_fl_rate:
            self.best_fl_rate = summary['fl_entry_rate']
            best_path = os.path.join(self.save_dir, "fl_specialist_best.zip")
            self.model.save(best_path)
            print(f"[*] New best FL rate: {self.best_fl_rate:.1f}%")

        # Discord通知
        if self.notifier:
            self.notifier.send_progress(
                step=steps,
                total_steps=50_000_000,  # TODO: make configurable
                metrics={
                    'games': summary['games'],
                    'foul_rate': summary['foul_rate'],
                    'mean_score': summary['mean_score'],
                    'fl_entry_rate': summary['fl_entry_rate'],
                    'trips_stay': summary['trips_stay_rate'],
                    'aa_stay': summary['aa_stay_rate'],
                    'kk_stay': summary['kk_stay_rate'],
                    'qq_stay': summary['qq_stay_rate'],
                }
            )

        # メトリクスリセット
        self.metrics.reset()


def make_env(env_id: int):
    """環境ファクトリ"""
    def _init():
        env = FLSpecialistEnv(env_id=env_id)
        env = ActionMasker(env, lambda e: e.action_masks())
        return env
    return _init


def train_fl_specialist(
    total_timesteps: int = 20_000_000,
    num_envs: Optional[int] = None,
    base_model_path: Optional[str] = None,
):
    """FL特化型AI学習"""

    num_envs = num_envs or NUM_ENVS

    print("=" * 60)
    print("OFC Pineapple AI - FL Specialist Training")
    print("=" * 60)
    print(f"Parallel Envs: {num_envs}")
    print(f"Total Steps: {total_timesteps:,}")
    print(f"Base Model: {base_model_path or 'None (fresh start)'}")
    print("=" * 60)

    notifier = TrainingNotifier(
        discord_webhook=DISCORD_WEBHOOK_URL,
        project_name=f"FL Specialist ({num_envs} envs)"
    )

    # 環境作成（SubprocVecEnvで高速化）
    use_dummy = os.getenv("USE_DUMMY_VEC_ENV", "0").lower() in ("1", "true", "yes")
    if use_dummy:
        print(f"[*] Creating {num_envs} environments with DummyVecEnv...")
        env = DummyVecEnv([make_env(i) for i in range(num_envs)])
    else:
        print(f"[*] Creating {num_envs} parallel environments with SubprocVecEnv (spawn)...")
        env = SubprocVecEnv([make_env(i) for i in range(num_envs)], start_method='spawn')

    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    # モデル初期化
    if base_model_path and os.path.exists(base_model_path):
        print(f"[*] Loading base model: {base_model_path}")
        model = MaskablePPO.load(base_model_path, env=env, verbose=1)
    else:
        print("[*] Creating new model...")
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,
            gamma=0.99,
            n_steps=2048,
            batch_size=256,
            tensorboard_log="./logs/fl_specialist/"
        )

    callback = FLSpecialistCallback(
        save_dir=save_dir,
        save_freq=100_000,
        notifier=notifier,
        verbose=1
    )

    print(f"\nStarting FL Specialist Training (Goal: {total_timesteps:,} steps)")
    print("=" * 60)

    print("[*] Starting model.learn()...")
    sys.stdout.flush()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,  # Disabled for better logging
            reset_num_timesteps=True
        )
    except KeyboardInterrupt:
        print("\n[!] Training interrupted")
    finally:
        env.close()
        final_path = os.path.join(save_dir, "fl_specialist_final.zip")
        model.save(final_path)
        print(f"[*] Final model saved: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Specialist Training")
    parser.add_argument("--steps", type=int, default=20_000_000, help="Total training steps")
    parser.add_argument("--envs", type=int, default=None, help="Number of parallel environments")
    parser.add_argument("--base-model", type=str, default=None, help="Base model to fine-tune from")

    args = parser.parse_args()

    train_fl_specialist(
        total_timesteps=args.steps,
        num_envs=args.envs,
        base_model_path=args.base_model
    )
