"""
OFC Pineapple AI - V2 Training Script
Phase 1 アブレーション実験用

V1からの主な変更点:
- 報酬ハッキング対策（FL接近報酬を1/10に縮小）
- extended_fl_obs=True（FL関連追加特徴量）
- current_street を one-hot 化（891次元観測空間）
- 拡張監視指標（FL追求フォール率、ポジション別スコア、FPS推移）
- 報酬設定 A/B/C の切り替え対応

使い方:
    # 設定A（最小報酬）で1Mステップ
    python v2/train_v2.py --reward-config A --steps 1000000

    # 設定B（ゼロ報酬）で1Mステップ
    python v2/train_v2.py --reward-config B --steps 1000000

    # 設定C（条件付き報酬）で1Mステップ
    python v2/train_v2.py --reward-config C --steps 1000000

    # テストモード
    python v2/train_v2.py --test-mode --steps 10000

環境変数:
    NUM_ENVS: 並列環境数 (デフォルト: 4)
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

# SubprocVecEnv用: spawnメソッドを使用
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn')

# パス設定
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src', 'python'))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import gymnasium as gym

from ofc_3max_env import OFC3MaxEnv
from notifier import TrainingNotifier
from v2.rule_based_agent import SafeAgent, AggressiveAgent, RandomAgent

try:
    import ofc_engine as ofc
except ImportError:
    raise ImportError("ofc_engine not found. Run 'python setup.py build_ext --inplace' first.")

# === 報酬設定プリセット ===
REWARD_CONFIGS = {
    'A': {
        'name': 'Minimal',
        'fl_stage1_reward': 0.5,
        'fl_stage2_reward': 1.0,
        'fl_stage3_reward': 0.8,
        'fl_entry_bonus': 30.0,
        'fl_stay_bonus': 60.0,
        'fl_aa_bonus': 5.0,
        'fl_trips_bonus': 5.0,
        'foul_penalty_fl': 0.2,
        'foul_penalty_normal': 0.5,
    },
    'B': {
        'name': 'Zero-shaping',
        'fl_stage1_reward': 0.0,
        'fl_stage2_reward': 0.0,
        'fl_stage3_reward': 0.0,
        'fl_entry_bonus': 50.0,
        'fl_stay_bonus': 100.0,
        'fl_aa_bonus': 10.0,
        'fl_trips_bonus': 10.0,
        'foul_penalty_fl': 0.5,
        'foul_penalty_normal': 0.5,
    },
    'C': {
        'name': 'Conditional',
        'fl_stage1_reward': 0.5,
        'fl_stage2_reward': 1.0,
        'fl_stage3_reward': 0.8,
        'fl_entry_bonus': 30.0,
        'fl_stay_bonus': 60.0,
        'fl_aa_bonus': 5.0,
        'fl_trips_bonus': 5.0,
        'foul_penalty_fl': 0.2,
        'foul_penalty_normal': 0.5,
        'conditional_shaping': True,  # ゲーム終了時に非フォールの場合のみ付与
    },
}

NUM_ENVS = int(os.getenv("NUM_ENVS", "4"))
DISCORD_WEBHOOK_URL = os.getenv(
    "DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1461388510140301315/Ofc9ok3IDLgRhFR7Oa2iA4-CHHL9uFvFbIPT9G9tbA5X5r-wG-XMbnkg7ubocLJJQ5Pf"
)


class SelfPlayOpponentManager:
    """Self-Play用の対戦相手管理（V2: pool_size/latest_prob/rule_based_prob 設定可能）"""

    def __init__(self, pool_size: int = 5, latest_prob: float = 0.8,
                 update_freq: int = 200_000, rule_based_prob: float = 0.0):
        self.model_pool: List[Dict[str, torch.Tensor]] = []
        self.pool_size = pool_size
        self.latest_prob = latest_prob
        self.rule_based_prob = rule_based_prob
        self.update_freq = update_freq
        self.current_weights: Optional[Dict[str, torch.Tensor]] = None
        self.last_update_step = 0
        self.latest_selections = 0
        self.past_selections = 0
        self.rule_based_selections = 0
        # ルールベースエージェントのプール
        self._rule_based_agents = [SafeAgent(), AggressiveAgent()]

    def add_model(self, model: MaskablePPO) -> None:
        weights = {k: v.cpu().clone() for k, v in model.policy.state_dict().items()}
        self.model_pool.append(weights)
        if len(self.model_pool) > self.pool_size:
            self.model_pool.pop(0)

    def update_current(self, model: MaskablePPO) -> None:
        self.current_weights = {k: v.cpu().clone() for k, v in model.policy.state_dict().items()}

    def select_opponent(self) -> Tuple[Optional[Dict[str, torch.Tensor]], str]:
        r = random.random()
        # ルールベース混合判定
        if self.rule_based_prob > 0 and r < self.rule_based_prob:
            self.rule_based_selections += 1
            agent = random.choice(self._rule_based_agents)
            return None, f"rule_based:{type(agent).__name__}"

        # latest vs past の判定
        if not self.model_pool or random.random() < self.latest_prob:
            self.latest_selections += 1
            return self.current_weights, "latest"
        else:
            self.past_selections += 1
            return random.choice(self.model_pool), "past"

    def get_rule_based_agent(self, opponent_type: str):
        """opponent_type から対応するルールベースエージェントを返す"""
        name = opponent_type.split(":")[-1] if ":" in opponent_type else ""
        for agent in self._rule_based_agents:
            if type(agent).__name__ == name:
                return agent
        return random.choice(self._rule_based_agents)

    def should_update(self, current_step: int) -> bool:
        return current_step - self.last_update_step >= self.update_freq

    def mark_updated(self, current_step: int) -> None:
        self.last_update_step = current_step


_global_opponent_manager: Optional[SelfPlayOpponentManager] = None
_global_inference_model: Optional[MaskablePPO] = None


def set_global_opponent_manager(manager: SelfPlayOpponentManager, model: MaskablePPO) -> None:
    global _global_opponent_manager, _global_inference_model
    _global_opponent_manager = manager
    _global_inference_model = model


class ParallelOFCEnv(gym.Env):
    """V2 並列実行用の OFC 環境ラッパー"""

    def __init__(self, env_id: int = 0, reward_config: Optional[Dict] = None, use_selfplay: bool = True):
        super().__init__()
        self.env = OFC3MaxEnv(
            enable_fl_turns=True,
            continuous_games=True,
            fl_solver_mode='greedy',
            reward_config=reward_config,
            extended_fl_obs=True,
        )
        self.env_id = env_id
        self.learning_agent = "player_0"
        self.use_selfplay = use_selfplay
        self.reward_config = reward_config or {}

        self.observation_space = self.env.observation_space(self.learning_agent)
        self.action_space = self.env.action_space(self.learning_agent)
        self.current_reward = 0
        self.opponent_type = "random"

        # V2 拡張統計
        self.fl_entries = 0
        self.fl_stays = 0
        self.fl_games_played = 0
        self.games_started_in_fl = 0

        # ポジション別スコア追跡（V2新規）
        self.position_scores = {0: [], 1: [], 2: []}

        # FL追求フォール追跡（V2新規: 報酬ハッキング検出）
        self.fl_pursuing_fouls = 0
        self.fl_pursuing_games = 0
        self.normal_fouls = 0
        self.normal_games = 0

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        self.current_reward = 0
        self.opponent_type = "random"
        self._play_opponents()
        return self.env.observe(self.learning_agent), {}

    def step(self, action):
        is_terminated = (
            self.env.terminations.get(self.learning_agent, False) or
            self.env.terminations.get(self.env.agent_selection, False) or
            all(self.env.terminations.values())
        )
        if is_terminated:
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
            res = self.env.engine.result()
            score = res.get_score(0)
            entered_fl = res.entered_fl(0)
            stayed_fl = res.stayed_fl(0)
            fouled = res.is_fouled(0)

            if entered_fl:
                self.fl_entries += 1
            if stayed_fl:
                self.fl_stays += 1

            # ポジション別スコア記録（V2新規）
            btn_pos = self.env.button_position if hasattr(self.env, 'button_position') else 0
            rel_pos = (0 - btn_pos) % 3  # 0=BTN, 1=SB, 2=BB
            self.position_scores[rel_pos].append(score)

            # FL追求フォール追跡（V2新規: 報酬ハッキング検出）
            fl_stage = self.env._fl_approach_stage.get(self.learning_agent, 0)
            if fl_stage >= 1:
                self.fl_pursuing_games += 1
                if fouled:
                    self.fl_pursuing_fouls += 1
            else:
                self.normal_games += 1
                if fouled:
                    self.normal_fouls += 1

            info = {
                'score': score,
                'royalty': res.get_royalty(0),
                'fouled': fouled,
                'entered_fl': entered_fl,
                'stayed_fl': stayed_fl,
                'opponent_type': self.opponent_type,
                'win': score > 0,
                'loss': score < 0,
                'draw': score == 0,
                'button_position': btn_pos,
                # V2 拡張
                'fl_stage': fl_stage,
                'position': rel_pos,
            }
        return obs, reward, terminated, truncated, info

    def _play_opponents(self):
        """相手プレイヤーをプレイ"""
        global _global_opponent_manager, _global_inference_model

        max_iterations = 500
        iterations = 0
        while not all(self.env.terminations.values()) and self.env.agent_selection != self.learning_agent:
            iterations += 1
            if iterations > max_iterations:
                for a in self.env.possible_agents:
                    self.env.terminations[a] = True
                break

            agent = self.env.agent_selection
            if self.env.terminations.get(agent, False):
                self.env.step(None)
                continue

            pidx = self.env.agent_name_mapping[agent]
            ps = self.env.engine.player(pidx)
            if ps.board.total_placed() >= 13 or ps.in_fantasy_land:
                self.env.step(None)
                continue

            valid_actions = self.env.get_valid_actions(agent)
            action = self._get_opponent_action(agent, valid_actions)
            self.env.step(action)

    def _get_opponent_action(self, agent: str, valid_actions: List[int]) -> int:
        global _global_opponent_manager, _global_inference_model

        if self.use_selfplay and _global_opponent_manager is not None and _global_inference_model is not None:
            try:
                opponent_weights, self.opponent_type = _global_opponent_manager.select_opponent()

                # ルールベースエージェントの場合
                if self.opponent_type.startswith("rule_based:"):
                    rb_agent = _global_opponent_manager.get_rule_based_agent(self.opponent_type)
                    return rb_agent.select_action(self.env, agent)

                # モデルベースの場合
                if opponent_weights is not None:
                    obs = self.env.observe(agent)
                    mask = self.env.action_masks(agent)
                    original_weights = _global_inference_model.policy.state_dict()
                    _global_inference_model.policy.load_state_dict(opponent_weights)
                    action, _ = _global_inference_model.predict(obs, action_masks=mask, deterministic=False)
                    _global_inference_model.policy.load_state_dict(original_weights)
                    return int(action)
            except Exception as e:
                print(f"[SelfPlay] Error: {e}")

        self.opponent_type = "random"
        return random.choice(valid_actions)

    def action_masks(self) -> np.ndarray:
        if self.env.terminations.get(self.learning_agent, False):
            mask = np.zeros(243, dtype=np.int8)
            mask[0] = 1
            return mask
        return self.env.action_masks(self.learning_agent)


def mask_fn(env: ParallelOFCEnv) -> np.ndarray:
    return env.action_masks()


# reward_config をグローバルに保持（make_env のクロージャ用）
_global_reward_config: Optional[Dict] = None


def make_env(env_id: int, use_selfplay: bool = True) -> Callable:
    def _init():
        env = ParallelOFCEnv(
            env_id=env_id,
            reward_config=_global_reward_config,
            use_selfplay=use_selfplay,
        )
        env = ActionMasker(env, mask_fn)
        return env
    return _init


class V2Callback(BaseCallback):
    """V2 学習コールバック（拡張監視指標付き）"""

    def __init__(
        self,
        save_path: str,
        notifier: Optional[TrainingNotifier],
        opponent_manager: SelfPlayOpponentManager,
        reward_config_name: str = "A",
        total_timesteps: int = 1_000_000,
        update_freq: int = 200_000,
        report_freq: int = 100_000,
    ):
        super().__init__()
        self.save_path = save_path
        self.notifier = notifier
        self.opponent_manager = opponent_manager
        self.reward_config_name = reward_config_name
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

        # 勝敗
        self.wins = 0
        self.losses = 0
        self.draws = 0

        # V2 拡張: ポジション別スコア
        self.position_scores = {0: deque(maxlen=200), 1: deque(maxlen=200), 2: deque(maxlen=200)}

        # V2 拡張: FL追求フォール率（報酬ハッキング検出）
        self.fl_pursuing_fouls = 0
        self.fl_pursuing_games = 0
        self.normal_fouls = 0
        self.normal_games = 0

        # V2 拡張: FPS推移（サーマルスロットリング計測）
        self.fps_history = []
        self.fps_check_interval = 600  # 10分ごとに記録

    def _on_training_start(self):
        self.last_report = (self.num_timesteps // self.report_freq) * self.report_freq
        self.last_update = (self.num_timesteps // self.update_freq) * self.update_freq
        self.start_time = time.time()
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

        # FPS推移を記録
        elapsed = time.time() - self.start_time
        if elapsed > 0 and int(elapsed) % self.fps_check_interval < 2:
            current_fps = self.num_timesteps / elapsed
            if not self.fps_history or abs(elapsed - len(self.fps_history) * self.fps_check_interval) < 10:
                self.fps_history.append((elapsed, current_fps))

        return True

    def _record_game(self, info: dict):
        score = info['score']
        self.scores.append(score)
        self.royalties.append(info.get('royalty', 0))
        self.fouls.append(1.0 if info.get('fouled', False) else 0.0)
        self.fl_entries.append(1.0 if info.get('entered_fl', False) else 0.0)
        self.fl_stays.append(1.0 if info.get('stayed_fl', False) else 0.0)
        self.total_games += 1

        if info.get('win', False):
            self.wins += 1
        elif info.get('loss', False):
            self.losses += 1
        else:
            self.draws += 1

        # ポジション別スコア
        pos = info.get('position', 0)
        self.position_scores[pos].append(score)

        # FL追求フォール追跡
        fl_stage = info.get('fl_stage', 0)
        fouled = info.get('fouled', False)
        if fl_stage >= 1:
            self.fl_pursuing_games += 1
            if fouled:
                self.fl_pursuing_fouls += 1
        else:
            self.normal_games += 1
            if fouled:
                self.normal_fouls += 1

    def _send_report(self):
        elapsed = time.time() - self.start_time
        fps = self.num_timesteps / elapsed if elapsed > 0 else 0

        foul_rate = np.mean(self.fouls) * 100 if self.fouls else 0
        mean_royalty = np.mean(self.royalties) if self.royalties else 0
        mean_score = np.mean(self.scores) if self.scores else 0
        fl_entry_rate = np.mean(self.fl_entries) * 100 if self.fl_entries else 0
        fl_stay_rate = np.mean(self.fl_stays) * 100 if self.fl_stays else 0

        total_wld = self.wins + self.losses + self.draws
        win_rate = self.wins / total_wld * 100 if total_wld > 0 else 0
        net_score = sum(self.scores) / len(self.scores) if self.scores else 0

        # V2: FL追求フォール率
        fl_foul_rate = (self.fl_pursuing_fouls / self.fl_pursuing_games * 100
                        if self.fl_pursuing_games > 0 else 0)
        normal_foul_rate = (self.normal_fouls / self.normal_games * 100
                            if self.normal_games > 0 else 0)

        # V2: ポジション別スコア
        pos_scores = {}
        pos_names = {0: 'BTN', 1: 'SB', 2: 'BB'}
        for p in range(3):
            if self.position_scores[p]:
                pos_scores[pos_names[p]] = np.mean(self.position_scores[p])
            else:
                pos_scores[pos_names[p]] = 0.0

        # コンソール出力
        print(f"\n[Step {self.num_timesteps:,}] V2 Config-{self.reward_config_name}")
        print(f"  Games: {self.total_games:,}")
        print(f"  Foul Rate: {foul_rate:.1f}%")
        print(f"  Mean Score: {mean_score:+.2f} | Net Score: {net_score:+.2f}")
        print(f"  Mean Royalty: {mean_royalty:.2f}")
        print(f"  FL Entry: {fl_entry_rate:.1f}% | FL Stay: {fl_stay_rate:.1f}%")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  --- Reward Hacking Check ---")
        print(f"  FL-Pursuit Foul: {fl_foul_rate:.1f}% ({self.fl_pursuing_games} games)")
        print(f"  Normal Foul:     {normal_foul_rate:.1f}% ({self.normal_games} games)")
        foul_diff = fl_foul_rate - normal_foul_rate
        hack_status = "WARNING" if foul_diff > 5 else "OK"
        print(f"  Difference: {foul_diff:+.1f}% [{hack_status}]")
        print(f"  --- Position Scores ---")
        for name, sc in pos_scores.items():
            print(f"  {name}: {sc:+.2f}")
        print(f"  FPS: {fps:.0f}")

        # Discord通知
        if self.notifier:
            metrics = {
                'games': self.total_games,
                'foul_rate': foul_rate,
                'mean_score': mean_score,
                'mean_royalty': mean_royalty,
                'fl_entry_rate': fl_entry_rate,
                'fl_stay_rate': fl_stay_rate,
                'win_rate': win_rate,
                'fps': fps,
            }
            self.notifier.send_progress(self.num_timesteps, self.total_timesteps, metrics)

    def _save_checkpoint(self):
        self.opponent_manager.add_model(self.model)
        self.opponent_manager.update_current(self.model)
        self.opponent_manager.mark_updated(self.num_timesteps)

        path = f"{self.save_path}_{self.num_timesteps}"
        self.model.save(path)

        if self.notifier:
            self.notifier.send_checkpoint(f"{path}.zip", self.num_timesteps)

        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        import glob as g
        checkpoint_dir = os.path.dirname(self.save_path)
        if not checkpoint_dir:
            checkpoint_dir = "."
        prefix = os.path.basename(self.save_path)
        files = g.glob(os.path.join(checkpoint_dir, f"{prefix}_*.zip"))
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


def train_v2(
    reward_config_key: str = 'A',
    total_timesteps: int = 1_000_000,
    test_mode: bool = False,
    num_envs: int = None,
    pool_size: int = 5,
    latest_prob: float = 0.8,
    rule_based_prob: float = 0.0,
    run_name: str = None,
):
    """V2 学習メイン関数"""
    global _global_reward_config

    if num_envs is None:
        num_envs = NUM_ENVS
    if test_mode:
        total_timesteps = 10_000
        num_envs = 2

    # 報酬設定の選択
    reward_cfg = REWARD_CONFIGS.get(reward_config_key, REWARD_CONFIGS['A'])
    reward_config_name = f"{reward_config_key} ({reward_cfg['name']})"
    _global_reward_config = reward_cfg

    # run_name が指定されていればモデルディレクトリ名として使う
    display_name = run_name or f"Config-{reward_config_key}"

    print("=" * 60)
    print(f"OFC Pineapple AI - V2 Training [{display_name}]")
    print(f"Reward Config: {reward_config_name}")
    print("=" * 60)
    print(f"Parallel Envs: {num_envs}")
    print(f"Total Steps: {total_timesteps:,}")
    print(f"Self-Play Pool: size={pool_size}, latest_prob={latest_prob}, rule_based={rule_based_prob:.0%}")
    print(f"Observation: 891 dims (extended_fl_obs + street one-hot)")
    print(f"Reward Settings:")
    print(f"  FL Stage1: {reward_cfg.get('fl_stage1_reward', 0)}")
    print(f"  FL Stage2: {reward_cfg.get('fl_stage2_reward', 0)}")
    print(f"  FL Stage3: {reward_cfg.get('fl_stage3_reward', 0)}")
    print(f"  FL Entry Bonus: {reward_cfg.get('fl_entry_bonus', 0)}")
    print(f"  FL Stay Bonus: {reward_cfg.get('fl_stay_bonus', 0)}")
    if reward_cfg.get('conditional_shaping'):
        print(f"  Conditional Shaping: ON (non-foul only)")
    print()

    # Discord通知
    notifier = TrainingNotifier(
        discord_webhook=DISCORD_WEBHOOK_URL,
        project_name=f"V2 {display_name}: {reward_cfg['name']}"
    )

    opponent_manager = SelfPlayOpponentManager(
        pool_size=pool_size,
        latest_prob=latest_prob,
        update_freq=200_000,
        rule_based_prob=rule_based_prob,
    )

    # 環境作成
    use_dummy = os.getenv("USE_DUMMY_VEC_ENV", "0").lower() in ("1", "true", "yes")
    if use_dummy or test_mode:
        print(f"[*] Creating {num_envs} environments with DummyVecEnv...")
        env = DummyVecEnv([make_env(i, use_selfplay=False) for i in range(num_envs)])
    else:
        print(f"[*] Creating {num_envs} environments with SubprocVecEnv (spawn)...")
        env = SubprocVecEnv([make_env(i, use_selfplay=False) for i in range(num_envs)], start_method='spawn')

    # モデルディレクトリ（run_name があればそれを使う）
    if run_name:
        save_dir = os.path.join(project_root, "models", f"v2_{run_name}")
        save_prefix = f"v2_{run_name}"
    else:
        save_dir = os.path.join(project_root, "models", f"v2_config{reward_config_key}")
        save_prefix = f"v2_{reward_config_key.lower()}"
    os.makedirs(save_dir, exist_ok=True)

    # チェックポイント探索
    import glob
    checkpoints = glob.glob(os.path.join(save_dir, f"{save_prefix}_*.zip"))

    latest_checkpoint = None
    is_resume = False

    if checkpoints:
        def extract_step(f):
            try:
                return int(os.path.basename(f).split('_')[-1].replace('.zip', ''))
            except:
                return 0
        latest_checkpoint = max(checkpoints, key=extract_step)
        print(f"[*] Found checkpoint: {latest_checkpoint}. Resuming...")
        model = MaskablePPO.load(latest_checkpoint, env=env)
        is_resume = True
    else:
        print("[*] Starting NEW V2 training...")
        policy_kwargs = dict(
            net_arch=dict(
                pi=[512, 256, 128],
                vf=[512, 256, 128],
            ),
        )
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=3e-4,
            gamma=0.998,
            n_steps=8192,
            batch_size=1024,
            n_epochs=5,
            ent_coef=0.03,
            clip_range=0.2,
            max_grad_norm=0.5,
            tensorboard_log=os.path.join(project_root, "logs", f"v2_config{reward_config_key}"),
        )

    set_global_opponent_manager(opponent_manager, model)

    callback = V2Callback(
        save_path=os.path.join(save_dir, save_prefix),
        notifier=notifier,
        opponent_manager=opponent_manager,
        reward_config_name=reward_config_key,
        total_timesteps=total_timesteps,
    )

    if notifier and not test_mode:
        notifier.send_start({
            'timesteps': total_timesteps,
            'lr': 3e-4,
            'opponent_update': 200_000,
            'reward_config': reward_config_name,
            'num_envs': num_envs,
        })

    if is_resume and latest_checkpoint:
        try:
            cp_name = os.path.basename(latest_checkpoint)
            current_steps = int(cp_name.split('_')[-1].replace('.zip', ''))
        except:
            current_steps = 0
        remaining_steps = max(0, total_timesteps - current_steps)
        reset_num_timesteps = False
        print(f"[*] Resuming from {current_steps:,}. Remaining: {remaining_steps:,}")
    else:
        remaining_steps = total_timesteps
        reset_num_timesteps = True

    print(f"\nStarting V2 Training (Goal: {total_timesteps:,} steps)")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=remaining_steps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
        )

        # FPS推移レポート（サーマルスロットリング計測）
        if callback.fps_history:
            print("\n--- FPS History (Thermal Throttling Check) ---")
            for elapsed, fps in callback.fps_history:
                print(f"  {elapsed/60:.0f}min: {fps:.0f} FPS")
            if len(callback.fps_history) >= 2:
                initial_fps = callback.fps_history[0][1]
                final_fps = callback.fps_history[-1][1]
                drop_rate = (initial_fps - final_fps) / initial_fps * 100
                print(f"  FPS Drop Rate: {drop_rate:.1f}%")

        if notifier and not test_mode:
            elapsed = time.time() - callback.start_time
            hours = int(elapsed // 3600)
            mins = int((elapsed % 3600) // 60)
            notifier.send_complete({
                'total_steps': total_timesteps,
                'total_games': callback.total_games,
                'foul_rate': np.mean(callback.fouls) * 100 if callback.fouls else 0,
                'win_rate': callback.wins / max(1, callback.wins + callback.losses + callback.draws) * 100,
                'elapsed_time': f"{hours}h{mins}m",
                'model_path': os.path.join(save_dir, f"{save_prefix}_{total_timesteps}.zip"),
            })

    except KeyboardInterrupt:
        print("\n[!] Training interrupted")
        model.save(os.path.join(save_dir, f"{save_prefix}_interrupted"))
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        if notifier:
            notifier.send_error(str(e), traceback.format_exc())
        raise
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V2 Training - Phase 1")
    parser.add_argument("--reward-config", type=str, default="A", choices=["A", "B", "C"],
                        help="Reward config preset (A=Minimal, B=Zero-shaping, C=Conditional)")
    parser.add_argument("--test-mode", action="store_true", help="Test mode (10k steps)")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total training steps")
    parser.add_argument("--envs", type=int, default=None, help="Number of parallel environments")
    parser.add_argument("--pool-size", type=int, default=5, help="Self-play opponent pool size")
    parser.add_argument("--latest-prob", type=float, default=0.8, help="Probability of using latest model")
    parser.add_argument("--rule-based-prob", type=float, default=0.0,
                        help="Probability of rule-based opponent (0.0-1.0)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name (used for model dir and Discord). Overrides default naming.")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint to resume from (copies to new run's dir)")

    args = parser.parse_args()

    # --resume-from: 指定チェックポイントを新しい run のディレクトリにコピー
    if args.resume_from and args.run_name:
        import shutil
        src_path = args.resume_from
        if not os.path.exists(src_path):
            print(f"[ERROR] Checkpoint not found: {src_path}")
            sys.exit(1)
        dest_dir = os.path.join(project_root, "models", f"v2_{args.run_name}")
        os.makedirs(dest_dir, exist_ok=True)
        # チェックポイントからステップ数を抽出して新名前でコピー
        try:
            basename = os.path.basename(src_path)
            step_str = basename.split('_')[-1]  # e.g. "5000000.zip"
            dest_name = f"v2_{args.run_name}_{step_str}"
            dest_path = os.path.join(dest_dir, dest_name)
            if not os.path.exists(dest_path):
                shutil.copy2(src_path, dest_path)
                print(f"[*] Copied checkpoint: {src_path} -> {dest_path}")
            else:
                print(f"[*] Checkpoint already exists: {dest_path}")
        except Exception as e:
            print(f"[ERROR] Failed to copy checkpoint: {e}")
            sys.exit(1)

    train_v2(
        reward_config_key=args.reward_config,
        total_timesteps=args.steps,
        test_mode=args.test_mode,
        num_envs=args.envs,
        pool_size=args.pool_size,
        latest_prob=args.latest_prob,
        rule_based_prob=args.rule_based_prob,
        run_name=args.run_name,
    )
