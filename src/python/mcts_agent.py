"""
OFC Pineapple AI - MCTS Agent (Phase 7)
Monte Carlo Tree Search + Fantasy Land Solver 統合エンジン

Phase 7の要件:
- MCTSのロールアウト末尾でFL突入価値を考慮
- Policy Network (MaskablePPO) を事前確率として使用
- C++ FantasySolver との連携

使用例:
    from mcts_agent import MCTSFLAgent

    agent = MCTSFLAgent(model_path="models/ofc_phase5.zip")
    action = agent.select_action(engine, player_idx, simulations=200)
"""

import os
import sys
import time
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
from dataclasses import dataclass, field

# パス設定
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import ofc_engine as ofc
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False
    print("[MCTSAgent] Warning: ofc_engine not available")

try:
    from sb3_contrib import MaskablePPO
    import torch
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False
    print("[MCTSAgent] Warning: MaskablePPO not available")


@dataclass
class MCTSConfig:
    """MCTS設定パラメータ"""
    num_simulations: int = 100      # シミュレーション回数（速度重視）
    exploration_weight: float = 1.4  # UCB探索係数
    fl_weight: float = 0.6           # FL価値の重み (0-1)
    fl_approach_bonus: float = 0.0   # FL接近状態ボーナス（TopにQ/K/Aがある時）
    policy_temperature: float = 1.0  # Policy温度パラメータ
    max_time_ms: int = 500           # 最大思考時間(ms)
    use_policy_prior: bool = True    # Policy事前確率を使用
    use_fl_solver: bool = True       # FLソルバーを使用
    use_nn_value: bool = True        # NN Value関数をrollout backupに使用
    nn_value_weight: float = 0.5     # NN Value vs MC rolloutの混合比率
    rollout_depth: int = 10          # ロールアウト深さ
    parallel_rollouts: int = 4       # 並列ロールアウト数


@dataclass
class MCTSNode:
    """MCTSツリーノード"""
    state_hash: int = 0
    parent: Optional['MCTSNode'] = None
    action: int = -1                 # このノードに至ったアクション
    children: Dict[int, 'MCTSNode'] = field(default_factory=dict)

    visit_count: int = 0
    total_value: float = 0.0
    prior_prob: float = 0.0          # Policy事前確率

    fl_potential: float = 0.0        # FL突入ポテンシャル
    is_terminal: bool = False

    def ucb_score(self, exploration_weight: float, fl_weight: float) -> float:
        """UCB1スコアを計算"""
        if self.visit_count == 0:
            return float('inf')

        exploitation = self.total_value / self.visit_count
        exploration = exploration_weight * self.prior_prob * math.sqrt(
            math.log(self.parent.visit_count + 1) / (self.visit_count + 1)
        )
        fl_bonus = fl_weight * self.fl_potential

        return exploitation + exploration + fl_bonus


class MCTSFLAgent:
    """
    MCTS + Fantasy Land Solver 統合エージェント

    Phase 7の核心実装:
    - Policy Network から候補手の事前確率を取得
    - MCTSで各候補をシミュレーション
    - ロールアウト終了時にFL突入価値を加算
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[Any] = None,
        config: Optional[MCTSConfig] = None
    ):
        self.config = config or MCTSConfig()

        # モデル読み込み
        if model is not None:
            self.model = model
        elif model_path and HAS_MODEL:
            self.model = MaskablePPO.load(model_path)
        else:
            self.model = None

        # 統計
        self.stats = {
            'total_searches': 0,
            'avg_simulations': 0,
            'avg_time_ms': 0,
            'fl_triggered': 0,
            'policy_agreement_rate': 0,
        }

        # キャッシュ
        self._policy_cache = {}

    def select_action(
        self,
        engine: Any,
        player_idx: int,
        simulations: Optional[int] = None,
        max_time_ms: Optional[int] = None
    ) -> int:
        """
        MCTSを実行して最適なアクションを選択

        Args:
            engine: C++ GameEngine インスタンス
            player_idx: プレイヤーインデックス
            simulations: シミュレーション回数 (Noneでconfig値を使用)
            max_time_ms: 最大思考時間 (Noneでconfig値を使用)

        Returns:
            選択されたアクションID
        """
        start_time = time.time()
        num_sims = simulations or self.config.num_simulations
        time_limit = (max_time_ms or self.config.max_time_ms) / 1000.0

        self.stats['total_searches'] += 1

        # 有効アクションを取得
        valid_actions = self._get_valid_actions(engine, player_idx)

        if len(valid_actions) == 0:
            return 0
        if len(valid_actions) == 1:
            return valid_actions[0]

        # Policy事前確率を取得
        policy_probs = self._get_policy_priors(engine, player_idx, valid_actions)

        # MCTSルートノード作成
        root = MCTSNode()
        root.visit_count = 1

        # 子ノード初期化
        for action in valid_actions:
            child = MCTSNode(
                parent=root,
                action=action,
                prior_prob=policy_probs.get(action, 1.0 / len(valid_actions))
            )
            # FL潜在価値を計算
            child.fl_potential = self._calculate_fl_potential(
                engine, player_idx, action
            )
            root.children[action] = child

        # MCTSシミュレーション実行
        sim_count = 0
        while sim_count < num_sims:
            if time.time() - start_time > time_limit:
                break

            # Selection: UCBで最良ノードを選択
            node = self._select_node(root)

            # Expansion & Simulation
            value = self._simulate(engine, player_idx, node.action)

            # Backpropagation
            self._backpropagate(node, value)

            sim_count += 1

        # 最良アクションを選択 (訪問回数最大)
        best_action = max(
            root.children.keys(),
            key=lambda a: root.children[a].visit_count
        )

        # 統計更新
        elapsed_ms = (time.time() - start_time) * 1000
        self.stats['avg_simulations'] = (
            self.stats['avg_simulations'] * 0.95 + sim_count * 0.05
        )
        self.stats['avg_time_ms'] = (
            self.stats['avg_time_ms'] * 0.95 + elapsed_ms * 0.05
        )

        # Policy一致率
        policy_best = max(policy_probs.keys(), key=lambda a: policy_probs[a])
        if best_action == policy_best:
            self.stats['policy_agreement_rate'] = (
                self.stats['policy_agreement_rate'] * 0.95 + 1.0 * 0.05
            )
        else:
            self.stats['policy_agreement_rate'] *= 0.95

        return best_action

    def _select_node(self, root: MCTSNode) -> MCTSNode:
        """UCB1で最良ノードを選択"""
        best_score = -float('inf')
        best_node = None

        for action, node in root.children.items():
            score = node.ucb_score(
                self.config.exploration_weight,
                self.config.fl_weight
            )
            if score > best_score:
                best_score = score
                best_node = node

        return best_node

    def _simulate(
        self,
        engine: Any,
        player_idx: int,
        action: int
    ) -> float:
        """
        シミュレーション（ロールアウト）を実行

        AlphaGoスタイル: C++ MC評価 + NN Value関数の混合

        Returns:
            評価値（スコア + FL価値）
        """
        if not HAS_ENGINE:
            return np.random.normal(0, 5)

        # エンジン状態をクローン
        cloned = engine.clone()

        # アクションを適用
        success = self._apply_action(cloned, player_idx, action)
        if not success:
            return -50.0  # 無効アクション

        # C++ MCTS評価を使用
        ps = cloned.player(player_idx)
        board = ps.board

        # 使用済みカードマスク
        used_mask = board.all_mask()
        for i in range(cloned.num_players()):
            if i != player_idx:
                used_mask |= cloned.player(i).board.all_mask()

        remaining_deck = 0xFFFFFFFFFFFFFFFF & ~used_mask
        remaining_deck &= ((1 << 54) - 1)

        # 残りターン数
        remaining_turns = max(0, 5 - cloned.current_turn())

        # C++ MCTSノード評価（ヒューリスティック + FL期待値）
        eval_result = ofc.evaluate_mcts_node(
            board,
            remaining_deck,
            remaining_turns,
            self.config.fl_weight
        )

        # NN Value関数を使用（AlphaGoスタイル）
        nn_value = 0.0
        if self.config.use_nn_value and self.model is not None:
            try:
                obs = self._create_observation(cloned, player_idx)
                obs_tensor = {
                    key: torch.tensor(np.expand_dims(val, 0), dtype=torch.float32)
                    for key, val in obs.items()
                }
                with torch.no_grad():
                    features = self.model.policy.extract_features(obs_tensor)
                    if hasattr(self.model.policy, 'mlp_extractor'):
                        _, latent_vf = self.model.policy.mlp_extractor(features)
                    else:
                        latent_vf = features
                    nn_value = float(self.model.policy.value_net(latent_vf).squeeze())
            except Exception:
                nn_value = 0.0

        # モンテカルロロールアウト
        mc_value = eval_result.total_value
        if remaining_turns > 0 and self.config.rollout_depth > 0:
            try:
                mc_score = ofc.monte_carlo_evaluation(
                    board,
                    used_mask,
                    self.config.parallel_rollouts,
                    int(time.time() * 1000) % 1000000
                )
                mc_value = 0.3 * eval_result.total_value + 0.7 * mc_score
            except Exception:
                pass

        # FL接近ボーナス: TopにQ/K/A がある状態を評価
        fl_bonus = 0.0
        if self.config.fl_approach_bonus > 0:
            fl_bonus = self._evaluate_fl_approach(board)

        # MC評価 + NN Value関数の混合
        if self.config.use_nn_value and self.model is not None:
            w = self.config.nn_value_weight
            return w * nn_value + (1.0 - w) * mc_value + fl_bonus
        return mc_value + fl_bonus

    def _evaluate_fl_approach(self, board: Any) -> float:
        """
        TopのFL接近度を評価してボーナスを返す (ビットマスクベース)

        Rank体系: ACE=0, TWO=1, ..., QUEEN=11, KING=12
        FL条件: QQ+(QUEEN/KING/ACE pair) or Trips on Top

        - TopにQ/K/Aが1枚: +approach_bonus * 0.4
        - TopにQ/K/Aペア(FL確定): +approach_bonus * 1.0
        - TopにTrips(任意): +approach_bonus * 1.5
        """
        bonus = self.config.fl_approach_bonus
        try:
            top_mask = board.top_mask()
            if top_mask == 0:
                return 0.0

            # top_maskからランクを抽出 (4 suits per rank, +2 jokers at idx 52,53)
            # card index = rank * 4 + suit (0-51), jokers at 52,53
            ranks = []
            for i in range(54):
                if (top_mask >> i) & 1:
                    if i < 52:
                        ranks.append(i // 4)  # ACE=0..KING=12
                    else:
                        ranks.append(13)  # Joker

            if not ranks:
                return 0.0

            # FL対象ランク: ACE(0), QUEEN(11), KING(12)
            fl_ranks = [r for r in ranks if r in (0, 11, 12)]

            # ペア/Trips判定
            from collections import Counter
            rank_counts = Counter(ranks)
            max_count = max(rank_counts.values())

            if max_count >= 3:
                return bonus * 1.5  # Trips on top (any rank)
            elif max_count >= 2:
                # ペアのランクがFL対象か確認
                pair_rank = [r for r, c in rank_counts.items() if c >= 2][0]
                if pair_rank in (0, 11, 12):  # ACE, QUEEN, KING
                    return bonus * 1.0  # FL-qualifying pair
                return 0.0  # Non-FL pair (JJ etc.)
            elif len(fl_ranks) >= 1:
                return bonus * 0.4  # Single FL-rank card approaching
            return 0.0
        except Exception:
            return 0.0

    def _backpropagate(self, node: MCTSNode, value: float):
        """値をルートまで逆伝播"""
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_value += value
            current = current.parent

    def _calculate_fl_potential(
        self,
        engine: Any,
        player_idx: int,
        action: int
    ) -> float:
        """アクション後のFL突入ポテンシャルを計算"""
        if not HAS_ENGINE:
            return 0.0

        # エンジン状態をクローン
        cloned = engine.clone()

        # アクションを適用
        success = self._apply_action(cloned, player_idx, action)
        if not success:
            return 0.0

        ps = cloned.player(player_idx)
        board = ps.board

        # 使用済みカードマスク
        used_mask = board.all_mask()
        for i in range(cloned.num_players()):
            if i != player_idx:
                used_mask |= cloned.player(i).board.all_mask()

        remaining_deck = 0xFFFFFFFFFFFFFFFF & ~used_mask
        remaining_deck &= ((1 << 54) - 1)

        remaining_turns = max(0, 5 - cloned.current_turn())

        # C++ FL確率計算
        fl_prob = ofc.calculate_fl_probability(
            board,
            remaining_deck,
            remaining_turns
        )

        return fl_prob.expected_ev

    def _get_valid_actions(self, engine: Any, player_idx: int) -> List[int]:
        """有効なアクションのリストを取得"""
        if not HAS_ENGINE:
            return list(range(10))

        ps = engine.player(player_idx)
        hand = ps.get_hand()
        board = ps.board

        valid_actions = []
        phase = engine.phase()

        if phase == ofc.GamePhase.INITIAL_DEAL:
            # 初期配置: 5枚を各行に配置
            # アクションID: 3^5 = 243通り (各カードの配置先を3進数で表現)
            for action_id in range(243):
                if self._is_valid_initial_action(board, hand, action_id):
                    valid_actions.append(action_id)

        elif phase == ofc.GamePhase.TURN:
            # ターン中: 3枚から2枚配置、1枚捨て
            # アクションID: 3 * 3 * 3 = 27通り (card0_row, card1_row, which_to_discard)
            for action_id in range(27):
                if self._is_valid_turn_action(board, hand, action_id):
                    valid_actions.append(action_id)

        if not valid_actions:
            valid_actions = [0]

        return valid_actions

    def _is_valid_initial_action(
        self,
        board: Any,
        hand: List[Any],
        action_id: int
    ) -> bool:
        """初期配置アクションが有効かチェック"""
        top_count = board.count(ofc.TOP)
        mid_count = board.count(ofc.MIDDLE)
        bot_count = board.count(ofc.BOTTOM)

        temp_action = action_id
        for i in range(5):
            row = temp_action % 3
            temp_action //= 3

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

    def _is_valid_turn_action(
        self,
        board: Any,
        hand: List[Any],
        action_id: int
    ) -> bool:
        """ターンアクションが有効かチェック"""
        if len(hand) < 3:
            return False

        card0_row = action_id % 3
        card1_row = (action_id // 3) % 3
        # discard_idx = (action_id // 9) % 3  # 捨てるカードのインデックス

        top_slots = 3 - board.count(ofc.TOP)
        mid_slots = 5 - board.count(ofc.MIDDLE)
        bot_slots = 5 - board.count(ofc.BOTTOM)

        # 各行への配置枚数をカウント
        row_counts = [0, 0, 0]
        row_counts[card0_row] += 1
        row_counts[card1_row] += 1

        if row_counts[0] > top_slots:
            return False
        if row_counts[1] > mid_slots:
            return False
        if row_counts[2] > bot_slots:
            return False

        return True

    def _apply_action(
        self,
        engine: Any,
        player_idx: int,
        action_id: int
    ) -> bool:
        """アクションをエンジンに適用"""
        ps = engine.player(player_idx)
        hand = ps.get_hand()
        phase = engine.phase()

        if phase == ofc.GamePhase.INITIAL_DEAL:
            action = ofc.InitialAction()
            temp_action = action_id
            for i in range(5):
                row = temp_action % 3
                temp_action //= 3
                action.set_placement(i, hand[i], ofc.Row(row))
            return engine.apply_initial_action(player_idx, action)

        elif phase == ofc.GamePhase.TURN:
            action = ofc.TurnAction()
            card0_row = action_id % 3
            card1_row = (action_id // 3) % 3
            discard_idx = (action_id // 9) % 3

            place_indices = [i for i in range(3) if i != discard_idx]
            rows = [card0_row, card1_row]

            for i, (pi, row) in enumerate(zip(place_indices, rows)):
                action.set_placement(i, hand[pi], ofc.Row(row))
            action.discard = hand[discard_idx]

            return engine.apply_turn_action(player_idx, action)

        return False

    def _get_policy_priors(
        self,
        engine: Any,
        player_idx: int,
        valid_actions: List[int]
    ) -> Dict[int, float]:
        """
        Policy Networkから事前確率を取得

        Returns:
            {action_id: probability} の辞書
        """
        if self.model is None or not self.config.use_policy_prior:
            # 均一分布
            prob = 1.0 / len(valid_actions)
            return {a: prob for a in valid_actions}

        # 観測データを作成
        obs = self._create_observation(engine, player_idx)

        # アクションマスク
        action_mask = np.zeros(243, dtype=bool)
        for a in valid_actions:
            action_mask[a] = True

        # モデルからlogitsを取得
        try:
            obs_tensor = {
                key: torch.tensor(np.expand_dims(val, 0), dtype=torch.float32)
                for key, val in obs.items()
            }

            with torch.no_grad():
                # MaskablePPOの内部ポリシーにアクセス
                policy = self.model.policy

                # 特徴抽出
                features = policy.extract_features(obs_tensor)
                if hasattr(policy, 'mlp_extractor'):
                    latent_pi, _ = policy.mlp_extractor(features)
                else:
                    latent_pi = features

                # アクション分布のlogits
                action_logits = policy.action_net(latent_pi)

                # マスク適用
                mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)
                action_logits = action_logits.masked_fill(~mask_tensor, -float('inf'))

                # Softmax
                probs = torch.softmax(
                    action_logits / self.config.policy_temperature,
                    dim=-1
                ).squeeze(0).numpy()

        except Exception as e:
            # フォールバック: 均一分布
            print(f"[MCTSAgent] Policy extraction failed: {e}. Keys: {list(obs.keys())}")
            prob = 1.0 / len(valid_actions)
            return {a: prob for a in valid_actions}

        return {a: float(probs[a]) for a in valid_actions}

    def _create_observation(
        self,
        engine: Any,
        player_idx: int
    ) -> Dict[str, np.ndarray]:
        """エンジン状態からモデル用の観測データを生成 (Phase 7: 3-Max対応)"""
        NUM_CARDS = 54
        ps = engine.player(player_idx)

        # 自分のボード (162)
        my_board = self._board_to_array(ps.board, NUM_CARDS)

        # 相手のボード (3-Max対応)
        num_players = engine.num_players()
        next_idx = (player_idx + 1) % num_players
        prev_idx = (player_idx - 1) % num_players

        next_board = self._board_to_array(engine.player(next_idx).board, NUM_CARDS)
        prev_board = self._board_to_array(engine.player(prev_idx).board, NUM_CARDS)

        # 手札 (270)
        hand = ps.get_hand()
        hand_obs = np.zeros(5 * NUM_CARDS, dtype=np.float32)
        for i, card in enumerate(hand[:5]):
            hand_obs[i * NUM_CARDS + card.index] = 1

        # 使用済みカード / 確率情報
        all_mask = 0
        for i in range(num_players):
            all_mask |= engine.player(i).board.all_mask()

        # 自分の捨て札 (54)
        my_discards = np.zeros(NUM_CARDS, dtype=np.float32) # 簡略化

        # 未知カード確率 (54)
        visible_mask = all_mask
        for card in hand: visible_mask |= (1 << card.index)
        remaining_count = NUM_CARDS - bin(visible_mask & ((1 << NUM_CARDS) - 1)).count('1')
        unseen_prob = np.zeros(NUM_CARDS, dtype=np.float32)
        for i in range(NUM_CARDS):
            if not (visible_mask & (1 << i)):
                unseen_prob[i] = 1.0 / max(1, remaining_count)

        # ゲーム状態 (14次元 — ofc_3max_env.pyと同一)
        next_ps = engine.player(next_idx)
        prev_ps = engine.player(prev_idx)
        fl_hand_count = len(hand) if ps.in_fantasy_land else 0
        game_state = np.array([
            engine.current_turn(),
            ps.board.count(ofc.TOP),
            ps.board.count(ofc.MIDDLE),
            ps.board.count(ofc.BOTTOM),
            next_ps.board.count(ofc.TOP),
            next_ps.board.count(ofc.MIDDLE),
            next_ps.board.count(ofc.BOTTOM),
            prev_ps.board.count(ofc.TOP),
            prev_ps.board.count(ofc.MIDDLE),
            prev_ps.board.count(ofc.BOTTOM),
            1.0 if ps.in_fantasy_land else 0.0,
            float(fl_hand_count),
            1.0 if next_ps.in_fantasy_land else 0.0,
            1.0 if prev_ps.in_fantasy_land else 0.0,
        ], dtype=np.float32)

        # ポジション情報 (3次元 one-hot)
        position_info = np.zeros(3, dtype=np.float32)
        position_info[player_idx % 3] = 1.0

        return {
            'my_board': my_board,
            'my_hand': hand_obs,
            'next_opponent_board': next_board,
            'prev_opponent_board': prev_board,
            'my_discards': my_discards,
            'unseen_probability': unseen_prob,
            'position_info': position_info,
            'game_state': game_state,
        }

    def _board_to_array(self, board, num_cards):
        arr = np.zeros(3 * num_cards, dtype=np.float32)
        masks = [board.top_mask(), board.mid_mask(), board.bot_mask()]
        for row_idx, mask in enumerate(masks):
            for i in range(num_cards):
                if (mask >> i) & 1:
                    arr[row_idx * num_cards + i] = 1
        return arr

    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        return self.stats.copy()


class NNAgent:
    """
    純粋なPolicy Networkエージェント（比較用ベースライン）
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[Any] = None,
        temperature: float = 1.0
    ):
        if model is not None:
            self.model = model
        elif model_path and HAS_MODEL:
            self.model = MaskablePPO.load(model_path)
        else:
            self.model = None

        self.temperature = temperature
        self.stats = {'total_predictions': 0}

    def select_action(
        self,
        engine: Any,
        player_idx: int,
        deterministic: bool = True
    ) -> int:
        """Policy Networkからアクションを選択"""
        self.stats['total_predictions'] += 1

        if self.model is None:
            return self._random_action(engine, player_idx)

        # MCTSFLAgentと同じ観測データ作成ロジックを使用
        mcts_agent = MCTSFLAgent.__new__(MCTSFLAgent)
        obs = mcts_agent._create_observation(engine, player_idx)
        valid_actions = mcts_agent._get_valid_actions(engine, player_idx)

        # アクションマスク
        action_mask = np.zeros(243, dtype=bool)
        for a in valid_actions:
            action_mask[a] = True

        obs_tensor = {
            key: np.expand_dims(val, 0) for key, val in obs.items()
        }

        action, _ = self.model.predict(
            obs_tensor,
            action_masks=action_mask.reshape(1, -1),
            deterministic=deterministic
        )

        return int(action[0]) if hasattr(action, '__len__') else int(action)

    def _random_action(self, engine: Any, player_idx: int) -> int:
        """ランダムアクション（フォールバック）"""
        mcts_agent = MCTSFLAgent.__new__(MCTSFLAgent)
        valid_actions = mcts_agent._get_valid_actions(engine, player_idx)
        return np.random.choice(valid_actions)


def test_mcts_agent():
    """MCTSエージェントの基本テスト"""
    print("=" * 60)
    print("MCTS FL Agent - Basic Test")
    print("=" * 60)

    if not HAS_ENGINE:
        print("ofc_engine not available. Skipping test.")
        return

    # エンジン初期化
    engine = ofc.GameEngine(2)
    engine.start_new_game(12345)

    # MCTSエージェント作成
    config = MCTSConfig(
        num_simulations=50,
        fl_weight=0.5,
        max_time_ms=500
    )
    agent = MCTSFLAgent(config=config)

    print("\nTesting MCTS action selection...")
    action = agent.select_action(engine, 0, simulations=50)
    print(f"Selected action: {action}")
    print(f"Stats: {agent.get_stats()}")

    # FL確率計算テスト
    print("\nTesting FL probability calculation...")
    ps = engine.player(0)
    remaining_deck = 0xFFFFFFFFFFFFFFFF & ~ps.board.all_mask()
    remaining_deck &= ((1 << 54) - 1)

    fl_prob = ofc.calculate_fl_probability(ps.board, remaining_deck, 5)
    print(f"FL Probability: {fl_prob.total_prob:.2%}")
    print(f"FL Expected EV: {fl_prob.expected_ev:.2f}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")


if __name__ == "__main__":
    test_mcts_agent()
