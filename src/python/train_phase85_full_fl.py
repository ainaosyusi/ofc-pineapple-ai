"""
OFC Pineapple AI - Phase 9 FL Mastery Training
完全なFL（Fantasy Land）ターンを含む実践形式の学習

Phase 8との違い:
- continuous_games=True: FLステータスを次ゲームに引き継ぎ
- ボタンローテーション: ポジションが毎ゲーム変化
- 実際のゲームと同じ形式での学習

環境変数:
    NUM_ENVS: 並列環境数 (デフォルト: 4)
    CLOUD_PROVIDER: "s3" or "gcs"
    GCS_BUCKET / S3_BUCKET: バケット名
    DISCORD_WEBHOOK_URL: Discord通知用Webhook
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
from typing import Optional, List, Callable, Tuple, Dict
import torch
torch.distributions.Distribution.set_default_validate_args(False)

# SubprocVecEnv用: spawnメソッドを使用（forkserverより安定）
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
from cloud_storage import init_cloud_storage

try:
    from endgame_solver import EndgameSolver
    import ofc_engine as ofc
    HAS_SOLVER = True
except ImportError:
    HAS_SOLVER = False

# 設定
NUM_ENVS = int(os.getenv("NUM_ENVS", "4"))
DISCORD_WEBHOOK_URL = os.getenv(
    "DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1461388510140301315/Ofc9ok3IDLgRhFR7Oa2iA4-CHHL9uFvFbIPT9G9tbA5X5r-wG-XMbnkg7ubocLJJQ5Pf"
)


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


class SelfPlayOpponentManager:
    """Self-Play用の対戦相手管理"""

    def __init__(
        self,
        pool_size: int = 5,
        latest_prob: float = 0.8,
        update_freq: int = 200_000
    ):
        self.model_pool: List[Dict[str, torch.Tensor]] = []
        self.pool_size = pool_size
        self.latest_prob = latest_prob
        self.update_freq = update_freq
        self.current_weights: Optional[Dict[str, torch.Tensor]] = None
        self.last_update_step = 0
        self.latest_selections = 0
        self.past_selections = 0

    def add_model(self, model: MaskablePPO) -> None:
        weights = {k: v.cpu().clone() for k, v in model.policy.state_dict().items()}
        self.model_pool.append(weights)
        if len(self.model_pool) > self.pool_size:
            self.model_pool.pop(0)
        print(f"[SelfPlay] Added model to pool. Pool size: {len(self.model_pool)}")

    def update_current(self, model: MaskablePPO) -> None:
        self.current_weights = {k: v.cpu().clone() for k, v in model.policy.state_dict().items()}

    def select_opponent(self) -> Tuple[Optional[Dict[str, torch.Tensor]], str]:
        if not self.model_pool or random.random() < self.latest_prob:
            self.latest_selections += 1
            return self.current_weights, "latest"
        else:
            self.past_selections += 1
            return random.choice(self.model_pool), "past"

    def should_update(self, current_step: int) -> bool:
        return current_step - self.last_update_step >= self.update_freq

    def mark_updated(self, current_step: int) -> None:
        self.last_update_step = current_step

    def get_stats(self) -> Dict[str, float]:
        total = self.latest_selections + self.past_selections
        if total == 0:
            return {'latest_ratio': 0.0, 'past_ratio': 0.0}
        return {
            'latest_ratio': self.latest_selections / total * 100,
            'past_ratio': self.past_selections / total * 100
        }


_global_opponent_manager: Optional[SelfPlayOpponentManager] = None
_global_inference_model: Optional[MaskablePPO] = None


def set_global_opponent_manager(manager: SelfPlayOpponentManager, model: MaskablePPO) -> None:
    global _global_opponent_manager, _global_inference_model
    _global_opponent_manager = manager
    _global_inference_model = model


class ParallelOFCEnv(gym.Env):
    """
    並列実行用のOFC環境ラッパー（Full FL対応）

    Phase 8.5の特徴:
    - continuous_games=True: FL状態を次ゲームに引き継ぎ
    - ボタンローテーション: ポジションが毎ゲーム変化
    - 実践形式の学習
    """

    def __init__(self, env_id: int = 0, use_selfplay: bool = True):
        super().__init__()
        # Phase 8.5b: continuous_games=True でFL継続学習を有効化
        self.env = OFC3MaxEnv(
            enable_fl_turns=True,
            continuous_games=True,  # FL状態引き継ぎ + ボタンローテーション有効
            fl_solver_mode='greedy'  # 学習用高速solver (brute-forceは遅すぎる)
        )
        self.env_id = env_id
        self.learning_agent = "player_0"
        self.use_selfplay = use_selfplay

        self.observation_space = self.env.observation_space(self.learning_agent)
        self.action_space = self.env.action_space(self.learning_agent)
        self.current_reward = 0

        # Self-Play統計
        self.opponent_type = "random"

        # FL統計（拡張）
        self.fl_entries = 0
        self.fl_stays = 0
        self.fl_games_played = 0  # FL中にプレイしたゲーム数
        self.games_started_in_fl = 0  # FL状態でゲーム開始した回数
        self.games_with_fl_opponent = 0

        # ボタン統計
        self.button_positions = {0: 0, 1: 0, 2: 0}

        # C3: EndgameSolver統合（残り3スロット以下で発動）
        self.endgame_solver = EndgameSolver(max_remaining=3) if HAS_SOLVER else None
        self.solver_overrides = 0

    def reset(self, seed=None, options=None):
        # リセット前にFL状態をチェック
        player_state = self.env.engine.player(0) if hasattr(self.env, 'engine') else None
        if player_state and player_state.in_fantasy_land:
            self.games_started_in_fl += 1

        self.env.reset(seed=seed, options=options)
        self.current_reward = 0
        self.opponent_type = "random"

        # ボタン位置を記録
        btn_pos = self.env.button_position if hasattr(self.env, 'button_position') else 0
        self.button_positions[btn_pos] = self.button_positions.get(btn_pos, 0) + 1

        # 相手がFLに入っているかチェック
        for i in range(1, 3):
            ps = self.env.engine.player(i)
            if ps.in_fantasy_land:
                self.games_with_fl_opponent += 1
                break

        self._play_opponents()
        return self.env.observe(self.learning_agent), {}

    def step(self, action):
        # learning_agentまたはagent_selectionがterminatedの場合はNoneを渡す
        # (PettingZooの_was_dead_stepはNoneのみ受け付ける)
        is_terminated = (
            self.env.terminations.get(self.learning_agent, False) or
            self.env.terminations.get(self.env.agent_selection, False) or
            all(self.env.terminations.values())  # 全員terminated
        )
        if is_terminated:
            self.env.step(None)
        else:
            # C3: EndgameSolver override for learning agent's endgame
            if (self.endgame_solver is not None
                    and self.env.engine.phase() == ofc.GamePhase.TURN
                    and self.endgame_solver.can_solve(self.env.engine, 0)):
                solver_action, _ = self.endgame_solver.solve(self.env.engine, 0)
                mask = self.env.action_masks(self.learning_agent)
                if solver_action < len(mask) and mask[solver_action] == 1:
                    action = solver_action
                    self.solver_overrides += 1
            self.env.step(action)
        self._play_opponents()

        obs = self.env.observe(self.learning_agent)
        reward = self.env._cumulative_rewards.get(self.learning_agent, 0) - self.current_reward
        self.current_reward += reward

        terminated = self.env.terminations.get(self.learning_agent, False)
        truncated = self.env.truncations.get(self.learning_agent, False)

        info = {}
        if terminated or truncated:
            res = self.env.engine.result()
            score = res.get_score(0)
            entered_fl = res.entered_fl(0)
            stayed_fl = res.stayed_fl(0)

            # FL統計を更新
            if entered_fl:
                self.fl_entries += 1
            if stayed_fl:
                self.fl_stays += 1

            # FL中にゲームをプレイしたかチェック
            player_state = self.env.engine.player(0)
            if player_state.in_fantasy_land:
                self.fl_games_played += 1

            info = {
                'score': score,
                'royalty': res.get_royalty(0),
                'fouled': res.is_fouled(0),
                'entered_fl': entered_fl,
                'stayed_fl': stayed_fl,
                'opponent_type': self.opponent_type,
                # 勝敗判定
                'win': score > 0,
                'loss': score < 0,
                'draw': score == 0,
                # FL関連（拡張）
                'fl_entries_total': self.fl_entries,
                'fl_stays_total': self.fl_stays,
                'fl_games_played': self.fl_games_played,
                'games_started_in_fl': self.games_started_in_fl,
                'games_with_fl_opponent': self.games_with_fl_opponent,
                # ボタン位置
                'button_position': self.env.button_position if hasattr(self.env, 'button_position') else 0
            }
        return obs, reward, terminated, truncated, info

    def _play_opponents(self):
        """相手プレイヤーをプレイ（Self-Play対応、無限ループ防止付き）"""
        global _global_opponent_manager, _global_inference_model

        max_iterations = 500
        iterations = 0
        while not all(self.env.terminations.values()) and self.env.agent_selection != self.learning_agent:
            iterations += 1
            if iterations > max_iterations:
                # 安全弁: 強制終了
                for a in self.env.possible_agents:
                    self.env.terminations[a] = True
                break

            agent = self.env.agent_selection
            if self.env.terminations.get(agent, False):
                self.env.step(None)
                continue

            # ボード満杯プレイヤーはスキップ
            pidx = self.env.agent_name_mapping[agent]
            ps = self.env.engine.player(pidx)
            if ps.board.total_placed() >= 13 or ps.in_fantasy_land:
                self.env.step(None)
                continue

            valid_actions = self.env.get_valid_actions(agent)
            action = self._get_opponent_action(agent, valid_actions)
            self.env.step(action)

    def _get_opponent_action(self, agent: str, valid_actions: List[int]) -> int:
        """対戦相手のアクションを取得"""
        global _global_opponent_manager, _global_inference_model

        if self.use_selfplay and _global_opponent_manager is not None and _global_inference_model is not None:
            try:
                opponent_weights, self.opponent_type = _global_opponent_manager.select_opponent()

                if opponent_weights is not None:
                    obs = self.env.observe(agent)
                    mask = self.env.action_masks(agent)

                    original_weights = _global_inference_model.policy.state_dict()
                    _global_inference_model.policy.load_state_dict(opponent_weights)

                    action, _ = _global_inference_model.predict(obs, action_masks=mask, deterministic=False)

                    _global_inference_model.policy.load_state_dict(original_weights)

                    return int(action)
            except Exception as e:
                print(f"[SelfPlay] Error in opponent action: {e}")

        self.opponent_type = "random"
        return random.choice(valid_actions)

    def action_masks(self) -> np.ndarray:
        # terminatedの場合はダミーマスクを返す（最初のアクションのみ有効）
        if self.env.terminations.get(self.learning_agent, False):
            mask = np.zeros(243, dtype=np.int8)
            mask[0] = 1
            return mask
        return self.env.action_masks(self.learning_agent)


def mask_fn(env: ParallelOFCEnv) -> np.ndarray:
    return env.action_masks()


def make_env(env_id: int, use_selfplay: bool = True) -> Callable:
    """環境作成関数（Phase 8.5: Full FL）"""
    def _init():
        env = ParallelOFCEnv(env_id=env_id, use_selfplay=use_selfplay)
        env = ActionMasker(env, mask_fn)
        return env
    return _init


class Phase85Callback(BaseCallback):
    """
    Phase 9 FL Mastery学習用コールバック

    拡張統計:
    - FL Entry / Stay率
    - FL中のゲーム数
    - ボタンポジション分布
    - 高得点ゲーム（FL効果）の追跡
    """

    def __init__(
        self,
        save_path: str,
        notifier: Optional[TrainingNotifier],
        cloud_storage,
        opponent_manager: SelfPlayOpponentManager,
        total_timesteps: int = 20_000_000,
        update_freq: int = 200_000,
        report_freq: int = 100_000
    ):
        super().__init__()
        self.save_path = save_path
        self.notifier = notifier
        self.cloud_storage = cloud_storage
        self.opponent_manager = opponent_manager
        self.total_timesteps = total_timesteps
        self.update_freq = update_freq
        self.report_freq = report_freq
        self.last_update = 0
        self.last_report = 0
        self.start_time = time.time()

        # 基本統計
        self.scores = deque(maxlen=500)
        self.royalties = deque(maxlen=500)
        self.fouls = deque(maxlen=500)
        self.fl_entries = deque(maxlen=500)
        self.fl_stays = deque(maxlen=500)
        self.total_games = 0

        # 勝敗統計
        self.wins = 0
        self.losses = 0
        self.draws = 0

        # 対戦相手タイプ別統計
        self.vs_latest_games = 0
        self.vs_latest_wins = 0
        self.vs_past_games = 0
        self.vs_past_wins = 0

        # FL拡張統計
        self.total_fl_entries = 0
        self.total_fl_stays = 0
        self.high_score_games = 0  # score >= 15 のゲーム数

        # ボタンポジション統計
        self.button_dist = {0: 0, 1: 0, 2: 0}

    def _on_training_start(self):
        self.last_report = (self.num_timesteps // self.report_freq) * self.report_freq
        self.last_update = (self.num_timesteps // self.update_freq) * self.update_freq
        self.start_time = time.time()
        print(f"[*] Phase85Callback: last_report={self.last_report}, last_update={self.last_update}")
        self.opponent_manager.update_current(self.model)

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
        """ゲーム結果を記録"""
        score = info['score']
        self.scores.append(score)
        self.royalties.append(info.get('royalty', 0))
        self.fouls.append(1.0 if info.get('fouled', False) else 0.0)

        entered_fl = info.get('entered_fl', False)
        stayed_fl = info.get('stayed_fl', False)
        self.fl_entries.append(1.0 if entered_fl else 0.0)
        self.fl_stays.append(1.0 if stayed_fl else 0.0)

        self.total_games += 1

        # FL累計
        if entered_fl:
            self.total_fl_entries += 1
        if stayed_fl:
            self.total_fl_stays += 1

        # 高得点ゲーム（FL効果の指標）
        if score >= 15:
            self.high_score_games += 1

        # 勝敗記録
        if info.get('win', False):
            self.wins += 1
        elif info.get('loss', False):
            self.losses += 1
        else:
            self.draws += 1

        # 対戦相手タイプ別記録
        opponent_type = info.get('opponent_type', 'random')
        if opponent_type == 'latest':
            self.vs_latest_games += 1
            if info.get('win', False):
                self.vs_latest_wins += 1
        elif opponent_type == 'past':
            self.vs_past_games += 1
            if info.get('win', False):
                self.vs_past_wins += 1

        # ボタンポジション
        btn_pos = info.get('button_position', 0)
        self.button_dist[btn_pos] = self.button_dist.get(btn_pos, 0) + 1

    def _send_report(self):
        """進捗レポートを送信"""
        elapsed = time.time() - self.start_time
        fps = self.num_timesteps / elapsed if elapsed > 0 else 0

        foul_rate = np.mean(self.fouls) * 100 if self.fouls else 0
        mean_royalty = np.mean(self.royalties) if self.royalties else 0
        mean_score = np.mean(self.scores) if self.scores else 0
        fl_entry_rate = np.mean(self.fl_entries) * 100 if self.fl_entries else 0
        fl_stay_rate = np.mean(self.fl_stays) * 100 if self.fl_stays else 0

        total_games = self.wins + self.losses + self.draws
        win_rate = self.wins / total_games * 100 if total_games > 0 else 0

        high_score_rate = self.high_score_games / self.total_games * 100 if self.total_games > 0 else 0

        vs_latest_winrate = self.vs_latest_wins / self.vs_latest_games * 100 if self.vs_latest_games > 0 else 0
        vs_past_winrate = self.vs_past_wins / self.vs_past_games * 100 if self.vs_past_games > 0 else 0

        print(f"\n[Step {self.num_timesteps:,}] Phase 9 FL Mastery")
        print(f"  Games: {self.total_games:,}")
        print(f"  Foul Rate: {foul_rate:.1f}%")
        print(f"  Mean Score: {mean_score:+.2f}")
        print(f"  Mean Royalty: {mean_royalty:.2f}")
        print(f"  FL Entry Rate: {fl_entry_rate:.1f}%")
        print(f"  FL Stay Rate: {fl_stay_rate:.1f}%")
        print(f"  High Score (>=15) Rate: {high_score_rate:.1f}%")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  vs Latest: {vs_latest_winrate:.1f}% ({self.vs_latest_games} games)")
        print(f"  vs Past: {vs_past_winrate:.1f}% ({self.vs_past_games} games)")
        print(f"  Button Dist: {self.button_dist}")
        print(f"  FPS: {fps:.0f}")
        print(f"  Pool Size: {len(self.opponent_manager.model_pool)}")

        if self.notifier:
            metrics = {
                'games': self.total_games,
                'foul_rate': foul_rate,
                'mean_score': mean_score,
                'mean_royalty': mean_royalty,
                'fl_entry_rate': fl_entry_rate,
                'fl_stay_rate': fl_stay_rate,
                'high_score_rate': high_score_rate,
                'win_rate': win_rate,
                'vs_selfplay_winrate': vs_past_winrate,
                'fps': fps,
                'pool_size': len(self.opponent_manager.model_pool)
            }
            self.notifier.send_progress(self.num_timesteps, self.total_timesteps, metrics)

    def _save_checkpoint(self):
        """チェックポイント保存 & モデルプール更新"""
        self.opponent_manager.add_model(self.model)
        self.opponent_manager.update_current(self.model)
        self.opponent_manager.mark_updated(self.num_timesteps)

        path = f"{self.save_path}_{self.num_timesteps}"
        self.model.save(path)

        if self.notifier:
            self.notifier.send_checkpoint(f"{path}.zip", self.num_timesteps)
        if self.cloud_storage and self.cloud_storage.enabled:
            self.cloud_storage.upload_checkpoint(f"{path}.zip", step=self.num_timesteps)

        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """古いチェックポイントを削除"""
        import glob
        checkpoint_dir = os.path.dirname(self.save_path)
        if not checkpoint_dir:
            checkpoint_dir = "."

        files = glob.glob(os.path.join(checkpoint_dir, "p9_fl_mastery_*.zip"))
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


def train_phase85_full_fl(
    total_timesteps: int = 20_000_000,
    test_mode: bool = False,
    num_envs: int = None,
    pretrained_path: str = None,
):
    """Phase 9 FL Mastery学習メイン関数"""
    if num_envs is None:
        num_envs = NUM_ENVS

    if test_mode:
        total_timesteps = 10_000
        num_envs = 2
        print("[TEST MODE] Running with limited settings")

    print("=" * 60)
    print(f"OFC Pineapple AI - Phase 9: FL Mastery Training")
    print("=" * 60)
    print(f"Parallel Envs: {num_envs}")
    print(f"Total Steps: {total_timesteps:,}")
    print(f"Features:")
    print(f"  - FL Turns: Enabled (continuous)")
    print(f"  - Button Rotation: Enabled")
    print(f"  - FL Status Preserved: Yes")
    print()

    cloud_storage = init_cloud_storage()
    print(f"[*] Cloud Storage: {cloud_storage.provider or 'None'}")

    notifier = TrainingNotifier(
        discord_webhook=DISCORD_WEBHOOK_URL,
        project_name=f"Phase 8.5: Full FL ({num_envs} envs)"
    )

    opponent_manager = SelfPlayOpponentManager(
        pool_size=5,
        latest_prob=0.8,
        update_freq=200_000
    )

    # SubprocVecEnv (spawn) または DummyVecEnv
    use_dummy = os.getenv("USE_DUMMY_VEC_ENV", "0").lower() in ("1", "true", "yes")
    if use_dummy:
        print(f"[*] Creating {num_envs} environments with DummyVecEnv...")
        env = DummyVecEnv([make_env(i, use_selfplay=False) for i in range(num_envs)])
    else:
        print(f"[*] Creating {num_envs} parallel environments with SubprocVecEnv (spawn)...")
        env = SubprocVecEnv([make_env(i, use_selfplay=False) for i in range(num_envs)], start_method='spawn')

    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    import glob

    # Phase 9 FL Mastery チェックポイントを探す
    p9_checkpoints = glob.glob(os.path.join(save_dir, "p9_fl_mastery_*.zip"))

    latest_checkpoint = None
    is_resume = False

    if p9_checkpoints:
        latest_checkpoint = max(p9_checkpoints, key=lambda f: int(f.split('_')[-1].replace('.zip', '')))
        print(f"[*] Found Phase 9 checkpoint: {latest_checkpoint}. Resuming...")
        model = MaskablePPO.load(latest_checkpoint, env=env)
        is_resume = True
    else:
        print("[*] Phase 9: FL Mastery - Starting NEW training with expanded network...")

        # Phase 9: 大容量ネットワーク (881次元入力を処理)
        policy_kwargs = dict(
            net_arch=dict(
                pi=[512, 256, 128],   # Policy: 3層・大容量
                vf=[512, 256, 128],   # Value: 3層・独立
            ),
        )

        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=3e-4,       # 大きめの初期学習率
            gamma=0.998,              # 超長期視野（FL連鎖の価値を最大化）
            n_steps=8192,             # 大量データで安定更新
            batch_size=1024,          # 大バッチ
            n_epochs=5,               # 更新回数を減らして安定性確保
            ent_coef=0.03,            # 高エントロピーでFL探索を促進
            clip_range=0.2,
            max_grad_norm=0.5,
            tensorboard_log="./logs/phase9_fl_mastery/"
        )

    set_global_opponent_manager(opponent_manager, model)

    callback = Phase85Callback(
        "models/p9_fl_mastery",
        notifier,
        cloud_storage,
        opponent_manager,
        total_timesteps=total_timesteps
    )

    if notifier and not test_mode:
        notifier.send_start({
            'timesteps': total_timesteps,
            'lr': 1e-4,
            'num_envs': num_envs,
            'selfplay_pool_size': 5,
            'latest_prob': 0.8,
            'cloud_provider': cloud_storage.provider or "local",
            'features': 'Full FL + Button Rotation'
        })

    if is_resume and latest_checkpoint:
        try:
            checkpoint_name = os.path.basename(latest_checkpoint)
            current_steps = int(checkpoint_name.split('_')[-1].replace('.zip', ''))
        except:
            current_steps = 0
        remaining_steps = max(0, total_timesteps - current_steps)
        reset_num_timesteps = False
        print(f"[*] Resuming Phase 8.5 from {current_steps:,}. Remaining: {remaining_steps:,}")
    else:
        remaining_steps = total_timesteps
        reset_num_timesteps = True
        print(f"[*] Starting Phase 8.5 from step 0.")

    print(f"\nStarting Phase 9 FL Mastery Training (Goal: {total_timesteps:,} steps)")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=remaining_steps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps
        )

        if notifier and not test_mode:
            notifier.send_complete({
                'total_steps': total_timesteps,
                'total_games': callback.total_games,
                'foul_rate': np.mean(callback.fouls) * 100 if callback.fouls else 0,
                'mean_royalty': np.mean(callback.royalties) if callback.royalties else 0,
                'fl_entry_rate': np.mean(callback.fl_entries) * 100 if callback.fl_entries else 0,
                'fl_stay_rate': np.mean(callback.fl_stays) * 100 if callback.fl_stays else 0,
                'win_rate': callback.wins / (callback.wins + callback.losses + callback.draws) * 100 if callback.total_games > 0 else 0,
                'model_path': f"models/p9_fl_mastery_{total_timesteps}.zip"
            })

    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user")
        model.save("models/p9_fl_mastery_interrupted")
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        if notifier:
            notifier.send_error(str(e), traceback.format_exc())
        raise
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 9 FL Mastery Training")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode (10k steps)")
    parser.add_argument("--steps", type=int, default=20_000_000, help="Total training steps")
    parser.add_argument("--envs", type=int, default=None, help="Number of parallel environments")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pre-trained model (from pretrain_imitation.py)")

    args = parser.parse_args()

    train_phase85_full_fl(
        total_timesteps=args.steps,
        test_mode=args.test_mode,
        num_envs=args.envs,
        pretrained_path=args.pretrained,
    )
