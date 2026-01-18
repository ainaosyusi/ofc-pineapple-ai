"""
OFC Pineapple AI - Teacher-Student Training Pipeline
Teacherヒューリスティックエージェントを用いたカリキュラム学習

学習フロー:
1. Phase 1: Imitation Learning - Teacherの行動を模倣
2. Phase 2: RL vs Teacher - Teacherを対戦相手として強化学習
3. Phase 3: Self-Play - 自己対戦で最適化

使用例:
    python train_with_teacher.py --phase 2 --steps 1000000
"""

import os
import sys
import time
import argparse
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List

# パス設定
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import gymnasium as gym
from gymnasium import spaces

try:
    import ofc_engine as ofc
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False
    print("[Train] Warning: ofc_engine not available")

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("[Train] Warning: sb3_contrib not available")

from ofc_3max_env import OFC3MaxEnv
from teacher_agent import TeacherHeuristicAgent, RandomAgent


class TeacherGymWrapper(gym.Env):
    """
    Teacher対戦用Gymラッパー

    learning_agent (player_0) が学習対象
    opponents (player_1, player_2) はTeacher/Random/NNから選択
    """

    def __init__(
        self,
        teacher_prob: float = 0.5,
        random_prob: float = 0.2,
        model_prob: float = 0.3,
        model: Optional[Any] = None,
        teacher_strategy: str = 'balanced',
        render_mode: str = None,
    ):
        """
        Args:
            teacher_prob: Teacher対戦確率
            random_prob: Random対戦確率
            model_prob: 学習モデル(旧バージョン)対戦確率
            model: 過去の学習モデル (Optional)
            teacher_strategy: Teacherの戦略 ('aggressive'/'balanced'/'conservative')
        """
        super().__init__()

        self.env = OFC3MaxEnv(render_mode=render_mode)
        self.learning_agent = "player_0"

        # 対戦相手設定
        self.teacher = TeacherHeuristicAgent(strategy=teacher_strategy)
        self.random_agent = RandomAgent()
        self.model = model

        # 対戦確率
        self.teacher_prob = teacher_prob
        self.random_prob = random_prob
        self.model_prob = model_prob

        # 現在の対戦相手
        self.current_opponents = [None, None]  # player_1, player_2

        # 観測・アクション空間
        self.observation_space = self.env.observation_space(self.learning_agent)
        self.action_space = self.env.action_space(self.learning_agent)

        # 統計
        self.episode_count = 0
        self.stats = {
            'vs_teacher_wins': 0,
            'vs_teacher_games': 0,
            'vs_random_wins': 0,
            'vs_random_games': 0,
        }

    def _select_opponents(self):
        """対戦相手をランダムに選択"""
        opponents = []
        for _ in range(2):
            r = np.random.random()
            if r < self.teacher_prob:
                opponents.append(('teacher', self.teacher))
            elif r < self.teacher_prob + self.random_prob:
                opponents.append(('random', self.random_agent))
            else:
                if self.model is not None:
                    opponents.append(('model', self.model))
                else:
                    opponents.append(('teacher', self.teacher))
        return opponents

    def reset(self, seed=None, options=None):
        """環境リセット"""
        self.env.reset(seed=seed, options=options)
        self.current_opponents = self._select_opponents()
        self.episode_count += 1

        # 学習エージェントのターンまで進める
        self._advance_to_learning_agent()

        obs = self.env.observe(self.learning_agent)
        info = {'opponents': [o[0] for o in self.current_opponents]}
        return obs, info

    def step(self, action):
        """アクション実行"""
        # 学習エージェントのアクション
        self.env.step(action)

        # 対戦相手のターンを処理
        self._advance_to_learning_agent()

        # 結果取得
        obs = self.env.observe(self.learning_agent)
        reward = self.env.rewards.get(self.learning_agent, 0)
        terminated = self.env.terminations.get(self.learning_agent, False)
        truncated = self.env.truncations.get(self.learning_agent, False)
        info = self.env.infos.get(self.learning_agent, {})

        # 統計更新
        if terminated or truncated:
            self._update_stats(reward)

        return obs, reward, terminated, truncated, info

    def _advance_to_learning_agent(self):
        """学習エージェントのターンまで進める"""
        while True:
            current_agent = self.env.agent_selection

            if current_agent == self.learning_agent:
                break

            if self.env.terminations.get(current_agent, False):
                self.env.step(None)
                continue

            # 対戦相手のアクション
            player_idx = self.env.agent_name_mapping[current_agent]
            opponent_idx = player_idx - 1  # player_1->0, player_2->1
            opponent_type, opponent_agent = self.current_opponents[opponent_idx]

            action = self._get_opponent_action(opponent_agent, player_idx)
            self.env.step(action)

            # ゲーム終了チェック
            if all(self.env.terminations.values()):
                break

    def _get_opponent_action(self, opponent, player_idx: int) -> int:
        """対戦相手のアクションを取得"""
        engine = self.env.engine
        agent_name = self.env.possible_agents[player_idx]

        if hasattr(opponent, 'select_action'):
            # TeacherHeuristicAgent or RandomAgent
            return opponent.select_action(engine, player_idx)
        elif hasattr(opponent, 'predict'):
            # MaskablePPO model
            obs = self.env.observe(agent_name)
            mask = self.env.action_masks(agent_name)
            action, _ = opponent.predict(
                {k: np.expand_dims(v, 0) for k, v in obs.items()},
                action_masks=mask.reshape(1, -1),
                deterministic=True
            )
            return int(action[0]) if hasattr(action, '__len__') else int(action)
        else:
            # Fallback: random
            valid = self.env.get_valid_actions(agent_name)
            return np.random.choice(valid)

    def _update_stats(self, reward):
        """統計更新"""
        opponent_type = self.current_opponents[0][0]
        if opponent_type == 'teacher':
            self.stats['vs_teacher_games'] += 1
            if reward > 0:
                self.stats['vs_teacher_wins'] += 1
        elif opponent_type == 'random':
            self.stats['vs_random_games'] += 1
            if reward > 0:
                self.stats['vs_random_wins'] += 1

    def action_masks(self) -> np.ndarray:
        """MaskablePPO用アクションマスク"""
        mask = self.env.action_masks(self.learning_agent)
        # 安全チェック: 少なくとも1つのアクションが有効であることを保証
        if not mask.any():
            mask[0] = 1  # フォールバック
        return mask

    def get_stats(self) -> Dict[str, Any]:
        """統計情報"""
        stats = self.stats.copy()
        if stats['vs_teacher_games'] > 0:
            stats['vs_teacher_winrate'] = stats['vs_teacher_wins'] / stats['vs_teacher_games']
        if stats['vs_random_games'] > 0:
            stats['vs_random_winrate'] = stats['vs_random_wins'] / stats['vs_random_games']
        return stats


def mask_fn(env: gym.Env) -> np.ndarray:
    """ActionMasker用のマスク関数"""
    return env.action_masks()


class TeacherTrainingCallback(BaseCallback):
    """
    Teacher学習用コールバック

    - 定期的な評価
    - 統計ログ
    - 対戦相手の切り替え
    """

    def __init__(
        self,
        eval_freq: int = 10000,
        log_freq: int = 1000,
        teacher_switch_threshold: float = 0.55,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.log_freq = log_freq
        self.teacher_switch_threshold = teacher_switch_threshold

        self.episode_rewards = []
        self.episode_count = 0
        self.best_mean_reward = -float('inf')

    def _on_step(self) -> bool:
        # エピソード終了時
        if self.locals.get('dones', [False])[0]:
            reward = self.locals.get('rewards', [0])[0]
            self.episode_rewards.append(reward)
            self.episode_count += 1

        # ログ出力
        if self.n_calls % self.log_freq == 0 and len(self.episode_rewards) > 0:
            recent = self.episode_rewards[-100:]
            mean_reward = np.mean(recent)
            std_reward = np.std(recent)

            if self.verbose > 0:
                print(f"[Step {self.n_calls}] Episodes: {self.episode_count}, "
                      f"Mean Reward: {mean_reward:.2f} (+/- {std_reward:.2f})")

        return True

    def _on_training_end(self):
        if self.verbose > 0:
            print(f"\nTraining completed. Total episodes: {self.episode_count}")
            if self.episode_rewards:
                print(f"Final mean reward: {np.mean(self.episode_rewards[-100:]):.2f}")


def create_env(
    teacher_prob: float = 0.5,
    random_prob: float = 0.2,
    model_prob: float = 0.3,
    model: Optional[Any] = None,
    teacher_strategy: str = 'balanced',
) -> gym.Env:
    """環境作成関数"""
    env = TeacherGymWrapper(
        teacher_prob=teacher_prob,
        random_prob=random_prob,
        model_prob=model_prob,
        model=model,
        teacher_strategy=teacher_strategy,
    )
    env = ActionMasker(env, mask_fn)
    return env


def train_phase1_imitation(
    model_path: Optional[str] = None,
    total_steps: int = 100000,
    save_path: str = "models/teacher_phase1",
):
    """
    Phase 1: 模倣学習

    Teacherの行動を模倣することで基礎を学習
    """
    print("=" * 60)
    print("Phase 1: Imitation Learning from Teacher")
    print("=" * 60)

    # Teacherのみと対戦
    env = create_env(
        teacher_prob=1.0,
        random_prob=0.0,
        model_prob=0.0,
        teacher_strategy='balanced',
    )

    # モデル作成
    if model_path and os.path.exists(model_path):
        model = MaskablePPO.load(model_path, env=env)
        print(f"Loaded model from {model_path}")
    else:
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
        )

    # コールバック
    callback = TeacherTrainingCallback(eval_freq=10000, log_freq=5000)

    # 学習
    model.learn(total_timesteps=total_steps, callback=callback, progress_bar=True)

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")

    return model


def train_phase2_vs_teacher(
    model_path: Optional[str] = None,
    total_steps: int = 500000,
    save_path: str = "models/teacher_phase2",
    teacher_prob: float = 0.6,
):
    """
    Phase 2: Teacher対戦学習

    Teacherを主な対戦相手として強化学習
    """
    print("=" * 60)
    print("Phase 2: Reinforcement Learning vs Teacher")
    print("=" * 60)

    # Teacher中心の対戦
    env = create_env(
        teacher_prob=teacher_prob,
        random_prob=0.2,
        model_prob=0.2,
        teacher_strategy='balanced',
    )

    # モデル作成/読み込み
    if model_path and os.path.exists(model_path):
        model = MaskablePPO.load(model_path, env=env)
        # 学習率を設定 (ロード後に設定)
        model.learning_rate = 5e-5
        print(f"Loaded model from {model_path}")
    else:
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            learning_rate=5e-5,
            n_steps=1024,
            batch_size=64,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.15,
            ent_coef=0.01,
            max_grad_norm=0.5,
            verbose=1,
        )

    # コールバック
    callback = TeacherTrainingCallback(eval_freq=20000, log_freq=10000)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="models/checkpoints/",
        name_prefix="teacher_phase2"
    )

    # 学習
    model.learn(
        total_timesteps=total_steps,
        callback=[callback, checkpoint_callback],
        progress_bar=True
    )

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")

    return model


def evaluate_vs_teacher(
    model_path: str,
    num_games: int = 100,
    teacher_strategy: str = 'balanced',
) -> Dict[str, float]:
    """
    Teacherに対するモデル評価
    """
    print("=" * 60)
    print(f"Evaluating model vs Teacher ({teacher_strategy})")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return {}

    model = MaskablePPO.load(model_path)
    teacher = TeacherHeuristicAgent(strategy=teacher_strategy)

    wins = 0
    total_score = 0
    model_fouls = 0
    teacher_fouls = 0

    for game_idx in range(num_games):
        engine = ofc.GameEngine(2)
        engine.start_new_game(game_idx)

        agents = [model, teacher]  # model=player_0, teacher=player_1

        while engine.phase() not in [ofc.GamePhase.SHOWDOWN, ofc.GamePhase.COMPLETE]:
            for player_idx in range(2):
                if engine.phase() in [ofc.GamePhase.SHOWDOWN, ofc.GamePhase.COMPLETE]:
                    break

                ps = engine.player(player_idx)
                hand = ps.get_hand()
                phase = engine.phase()

                if player_idx == 0:
                    # Model action
                    obs = _create_observation(engine, player_idx)
                    mask = _get_action_mask(engine, player_idx, phase)
                    action, _ = model.predict(
                        {k: np.expand_dims(v, 0) for k, v in obs.items()},
                        action_masks=mask.reshape(1, -1),
                        deterministic=True
                    )
                    action = int(action[0]) if hasattr(action, '__len__') else int(action)
                else:
                    # Teacher action
                    action = teacher.select_action(engine, player_idx)

                # Apply action
                _apply_action(engine, player_idx, action, hand, phase)

        # Results
        result = engine.result()
        score = result.get_score(0)
        total_score += score
        if score > 0:
            wins += 1
        if result.is_fouled(0):
            model_fouls += 1
        if result.is_fouled(1):
            teacher_fouls += 1

        if (game_idx + 1) % 20 == 0:
            print(f"Game {game_idx + 1}: Score={score:.1f}, "
                  f"Win Rate={wins/(game_idx+1)*100:.1f}%")

    results = {
        'win_rate': wins / num_games,
        'avg_score': total_score / num_games,
        'model_foul_rate': model_fouls / num_games,
        'teacher_foul_rate': teacher_fouls / num_games,
    }

    print(f"\n{'='*60}")
    print(f"Evaluation Results ({num_games} games):")
    print(f"Win Rate: {results['win_rate']*100:.1f}%")
    print(f"Avg Score: {results['avg_score']:.2f}")
    print(f"Model Foul Rate: {results['model_foul_rate']*100:.1f}%")
    print(f"Teacher Foul Rate: {results['teacher_foul_rate']*100:.1f}%")
    print(f"{'='*60}")

    return results


def _create_observation(engine, player_idx: int) -> Dict[str, np.ndarray]:
    """観測データ作成"""
    NUM_CARDS = 54
    ps = engine.player(player_idx)
    num_players = engine.num_players()

    # 自分のボード
    my_board = np.zeros(3 * NUM_CARDS, dtype=np.float32)
    masks = [ps.board.top_mask(), ps.board.mid_mask(), ps.board.bot_mask()]
    for row_idx, mask in enumerate(masks):
        for i in range(NUM_CARDS):
            if (mask >> i) & 1:
                my_board[row_idx * NUM_CARDS + i] = 1

    # 手札
    hand = ps.get_hand()
    my_hand = np.zeros(5 * NUM_CARDS, dtype=np.float32)
    for i, card in enumerate(hand[:5]):
        my_hand[i * NUM_CARDS + card.index] = 1

    # 相手ボード
    next_idx = (player_idx + 1) % num_players
    prev_idx = (player_idx - 1) % num_players

    next_board = np.zeros(3 * NUM_CARDS, dtype=np.float32)
    prev_board = np.zeros(3 * NUM_CARDS, dtype=np.float32)

    for idx, arr in [(next_idx, next_board), (prev_idx, prev_board)]:
        opp = engine.player(idx)
        masks = [opp.board.top_mask(), opp.board.mid_mask(), opp.board.bot_mask()]
        for row_idx, mask in enumerate(masks):
            for i in range(NUM_CARDS):
                if (mask >> i) & 1:
                    arr[row_idx * NUM_CARDS + i] = 1

    # 捨て札, 確率, ポジション, ゲーム状態
    my_discards = np.zeros(NUM_CARDS, dtype=np.float32)
    unseen_prob = np.ones(NUM_CARDS, dtype=np.float32) / NUM_CARDS
    position_info = np.zeros(3, dtype=np.float32)
    position_info[player_idx % 3] = 1

    game_state = np.array([
        engine.current_turn(),
        ps.board.count(ofc.TOP),
        ps.board.count(ofc.MIDDLE),
        ps.board.count(ofc.BOTTOM),
        engine.player(next_idx).board.count(ofc.TOP),
        engine.player(next_idx).board.count(ofc.MIDDLE),
        engine.player(next_idx).board.count(ofc.BOTTOM),
        engine.player(prev_idx).board.count(ofc.TOP),
        engine.player(prev_idx).board.count(ofc.MIDDLE),
        engine.player(prev_idx).board.count(ofc.BOTTOM),
        1.0 if ps.in_fantasy_land else 0.0
    ], dtype=np.float32)

    return {
        'my_board': my_board,
        'my_hand': my_hand,
        'next_opponent_board': next_board,
        'prev_opponent_board': prev_board,
        'my_discards': my_discards,
        'unseen_probability': unseen_prob,
        'position_info': position_info,
        'game_state': game_state,
    }


def _get_action_mask(engine, player_idx: int, phase) -> np.ndarray:
    """アクションマスク取得"""
    ps = engine.player(player_idx)
    hand = ps.get_hand()
    board = ps.board

    mask = np.zeros(243, dtype=bool)

    if phase == ofc.GamePhase.INITIAL_DEAL:
        for action in range(243):
            if _is_valid_initial(board, action):
                mask[action] = True
    elif phase == ofc.GamePhase.TURN:
        for action in range(27):
            if _is_valid_turn(board, len(hand), action):
                mask[action] = True

    if not mask.any():
        mask[0] = True

    return mask


def _is_valid_initial(board, action: int) -> bool:
    top_count = board.count(ofc.TOP)
    mid_count = board.count(ofc.MIDDLE)
    bot_count = board.count(ofc.BOTTOM)

    temp = action
    for _ in range(5):
        row = temp % 3
        temp //= 3

        if row == 0 and top_count >= 3:
            return False
        elif row == 1 and mid_count >= 5:
            return False
        elif row == 2 and bot_count >= 5:
            return False

        if row == 0:
            top_count += 1
        elif row == 1:
            mid_count += 1
        else:
            bot_count += 1

    return True


def _is_valid_turn(board, hand_size: int, action: int) -> bool:
    if hand_size < 3:
        return False

    row1 = action % 3
    row2 = (action // 3) % 3

    top_slots = 3 - board.count(ofc.TOP)
    mid_slots = 5 - board.count(ofc.MIDDLE)
    bot_slots = 5 - board.count(ofc.BOTTOM)

    row_counts = [0, 0, 0]
    row_counts[row1] += 1
    row_counts[row2] += 1

    return (row_counts[0] <= top_slots and
            row_counts[1] <= mid_slots and
            row_counts[2] <= bot_slots)


def _apply_action(engine, player_idx: int, action: int, hand, phase):
    """アクション適用"""
    if phase == ofc.GamePhase.INITIAL_DEAL:
        initial_action = ofc.InitialAction()
        placements = []
        temp = action
        for _ in range(5):
            placements.append(temp % 3)
            temp //= 3

        rows = [ofc.TOP, ofc.MIDDLE, ofc.BOTTOM]
        for i, row_idx in enumerate(placements):
            if i < len(hand):
                initial_action.set_placement(i, hand[i], rows[row_idx])
        engine.apply_initial_action(player_idx, initial_action)

    elif phase == ofc.GamePhase.TURN:
        turn_action = ofc.TurnAction()
        row1 = action % 3
        row2 = (action // 3) % 3
        discard_idx = (action // 9) % 3

        rows = [ofc.TOP, ofc.MIDDLE, ofc.BOTTOM]
        play_indices = [i for i in range(len(hand)) if i != discard_idx][:2]

        for i, pi in enumerate(play_indices):
            turn_action.set_placement(i, hand[pi], rows[[row1, row2][i]])
        turn_action.discard = hand[discard_idx]
        engine.apply_turn_action(player_idx, turn_action)


def main():
    parser = argparse.ArgumentParser(description='OFC Teacher-Student Training')
    parser.add_argument('--phase', type=int, default=2, choices=[1, 2],
                        help='Training phase (1=imitation, 2=RL vs teacher)')
    parser.add_argument('--steps', type=int, default=500000,
                        help='Total training steps')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to pretrained model')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save trained model')
    parser.add_argument('--eval', action='store_true',
                        help='Run evaluation only')
    parser.add_argument('--eval-games', type=int, default=100,
                        help='Number of evaluation games')

    args = parser.parse_args()

    if not HAS_SB3:
        print("sb3_contrib not available. Please install: pip install sb3-contrib")
        return

    if args.eval:
        if args.model is None:
            print("Please specify --model for evaluation")
            return
        evaluate_vs_teacher(args.model, num_games=args.eval_games)
        return

    if args.phase == 1:
        save_path = args.save or "models/teacher_phase1"
        train_phase1_imitation(
            model_path=args.model,
            total_steps=args.steps,
            save_path=save_path,
        )
    elif args.phase == 2:
        save_path = args.save or "models/teacher_phase2"
        train_phase2_vs_teacher(
            model_path=args.model,
            total_steps=args.steps,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
