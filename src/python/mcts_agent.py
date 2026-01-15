"""
OFC Pineapple AI - MCTS Agent
Monte Carlo Tree Search による推論時探索エンジン

Policy Network の「直感」に「思考（探索）」を組み合わせることで、
推論時の精度を向上させる。

使用例:
    from mcts_agent import MCTSAgent

    agent = MCTSAgent(model_path="models/ofc_phase3.zip")
    action = agent.predict_with_search(obs, simulations=100)
"""

import os
import sys
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

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
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False

try:
    from endgame_solver import EndgameSolver
    HAS_SOLVER = True
except ImportError:
    HAS_SOLVER = False


class MCTSAgent:
    """
    MCTS を用いた推論時探索エージェント
    
    戦略:
    1. Policy Network から候補手（Top-K）を取得
    2. 各候補手に対してランダムロールアウトを実行
    3. 「直感（Policy）」と「シミュレーション結果」を統合して最終決定
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[object] = None,
        top_k: int = 5,
        exploration_weight: float = 1.4,
        policy_weight: float = 0.3,  # Policy vs Simulation のバランス
        use_policy_rollout: bool = True,  # ロールアウトにもPolicyを使用
        use_endgame_solver: bool = True,  # 終盤ソルバーを使用
    ):
        """
        Args:
            model_path: MaskablePPO モデルのパス
            model: 既に読み込み済みのモデル
            top_k: 探索する候補手の数
            exploration_weight: UCB1 の探索係数
            policy_weight: 最終決定における Policy の重み (0-1)
        """
        self.top_k = top_k
        self.exploration_weight = exploration_weight
        self.policy_weight = policy_weight
        self.use_policy_rollout = use_policy_rollout
        self.use_endgame_solver = use_endgame_solver
        
        # 終盤ソルバー初期化
        if use_endgame_solver and HAS_SOLVER:
            self.solver = EndgameSolver(max_remaining=5)
        else:
            self.solver = None
        
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
            'policy_chosen_rate': 0,
        }
    
    def get_policy_actions(
        self,
        obs: Dict,
        action_mask: np.ndarray,
        k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Policy Network から確率の高い上位 K 個のアクションを取得
        
        Returns:
            [(action_id, probability), ...] のリスト
        """
        if self.model is None:
            # モデルがない場合はランダム
            valid_actions = np.where(action_mask)[0]
            probs = np.ones(len(valid_actions)) / len(valid_actions)
            return list(zip(valid_actions[:k], probs[:k]))
        
        # モデルから確率分布を取得
        # Note: MaskablePPO の内部ポリシーにアクセス
        obs_tensor = {
            key: np.expand_dims(val, 0) for key, val in obs.items()
        }
        
        # action_distribution を取得するため低レベルアクセス
        with np.errstate(divide='ignore', invalid='ignore'):
            action, _ = self.model.predict(
                obs_tensor,
                action_masks=action_mask.reshape(1, -1),
                deterministic=False
            )
        
        # 簡易的に確率を推定（実際はモデル内部を見る必要あり）
        # ここでは有効なアクションを均等に扱い、選ばれたアクションに重みを置く
        valid_actions = np.where(action_mask)[0]
        n_valid = len(valid_actions)
        
        if n_valid == 0:
            return [(0, 1.0)]
        
        # 選択されたアクションをトップに
        probs = np.ones(n_valid) / n_valid
        predicted_action = int(action[0]) if hasattr(action, '__len__') else int(action)
        
        if predicted_action in valid_actions:
            idx = np.where(valid_actions == predicted_action)[0][0]
            probs[idx] = 0.5  # より高い確率を付与
            probs /= probs.sum()
        
        # ソートして上位K個を返す
        indices = np.argsort(-probs)[:k]
        return [(valid_actions[i], probs[i]) for i in indices]
    
    def simulate_rollout(
        self,
        engine: Optional[object],
        action: int,
        n_simulations: int = 10,
        max_depth: int = 5
    ) -> float:
        """
        指定アクションから先をプレイし、期待スコアを返す
        
        Args:
            engine: C++ GameEngine インスタンス
            action: 評価するアクション
            n_simulations: シミュレーション回数
            max_depth: ロールアウトの最大深さ
            
        Returns:
            期待スコア
        """
        if not engine or self.model is None:
            return self._estimate_policy_value(action)
        
        total_score = 0.0
        
        for _ in range(n_simulations):
            cloned = engine.clone()
            
            # 最初のアクションを適用
            # 注意: 手札のカードを取得する必要がある
            player_idx = 0 # 探索対象を0番プレイヤーと想定
            ps = cloned.player(player_idx)
            hand = ps.get_hand()
            
            # アクション解釈（単純化）
            # InitialAction/TurnAction の適用が必要
            # ここでは簡単のため、最初のアクション適用後の状態から開始するように
            # predict_with_search 側で制御する
            
            score = self._run_policy_rollout(cloned, player_idx, max_depth)
            total_score += score
            
        return total_score / n_simulations

    def _run_policy_rollout(self, engine, player_idx, max_depth):
        """モデルを使ってプレイし、最終的なスコア（またはヒューリスティック）を返す"""
        depth = 0
        while engine.phase() != ofc.GamePhase.COMPLETE and depth < max_depth:
            obs = self._get_obs_from_engine(engine, player_idx)
            
            # 無効アクションマスク取得（ランダムまたは全許可）
            # TODO: 正密なマスクが必要
            mask = np.ones(243, dtype=np.int8) 
            
            action, _ = self.model.predict(obs, action_masks=mask, deterministic=False)
            
            # アクション適用
            if engine.phase() == ofc.GamePhase.INITIAL_DEAL:
                # InitialAction
                ia = ofc.InitialAction()
                hand = engine.player(player_idx).get_hand()
                temp_action = action
                for i in range(5):
                    row = temp_action % 3
                    temp_action //= 3
                    ia.set_placement(i, hand[i], ofc.Row(row))
                engine.apply_initial_action(player_idx, ia)
            else:
                # TurnAction (簡略化: 最初の有効な組合せを使用)
                ta = ofc.TurnAction()
                hand = engine.player(player_idx).get_hand()
                if len(hand) >= 3:
                    ta.set_placement(0, hand[0], ofc.Row(action % 3))
                    ta.set_placement(1, hand[1], ofc.Row((action // 3) % 3))
                    ta.discard = hand[2]
                    engine.apply_turn_action(player_idx, ta)
            
            depth += 1
            
        # 最終スコア評価
        ps = engine.player(player_idx)
        if ps.board.is_foul():
            return -50.0 # ファウルペナルティ強め
        return float(ps.board.calculate_royalties())

    def _get_obs_from_engine(self, engine, player_idx):
        """エンジン状態からモデル用の観測データを生成"""
        # multi_ofc_env.py の _get_observation と同等のロジック
        NUM_CARDS = 52
        opponent_idx = 1 - player_idx
        
        ps = engine.player(player_idx)
        opp_ps = engine.player(opponent_idx)
        
        # 自分のボード
        my_board = np.zeros(3 * NUM_CARDS, dtype=np.int8)
        my_masks = [ps.board.top_mask(), ps.board.mid_mask(), ps.board.bot_mask()]
        for row_idx, mask in enumerate(my_masks):
            for i in range(NUM_CARDS):
                if (mask >> i) & 1:
                    my_board[row_idx * NUM_CARDS + i] = 1

        # 相手のボード
        opp_board = np.zeros(3 * NUM_CARDS, dtype=np.int8)
        opp_masks = [opp_ps.board.top_mask(), opp_ps.board.mid_mask(), opp_ps.board.bot_mask()]
        for row_idx, mask in enumerate(opp_masks):
            for i in range(NUM_CARDS):
                if (mask >> i) & 1:
                    opp_board[row_idx * NUM_CARDS + i] = 1
        
        # 手札
        hand = ps.get_hand()
        hand_obs = np.zeros(5 * NUM_CARDS, dtype=np.int8)
        for i, card in enumerate(hand[:5]):
            hand_obs[i * NUM_CARDS + card.index] = 1
            
        # 使用済み
        all_mask = ps.board.all_mask() | opp_ps.board.all_mask()
        used_obs = np.zeros(NUM_CARDS, dtype=np.int8)
        for i in range(NUM_CARDS):
            if (all_mask >> i) & 1:
                used_obs[i] = 1
                
        # ゲーム状態
        game_state = np.array([
            engine.current_turn(),
            ps.board.count(ofc.Row.TOP),
            ps.board.count(ofc.Row.MIDDLE),
            ps.board.count(ofc.Row.BOTTOM),
            1.0 if ps.in_fantasy_land else 0.0,
            opp_ps.board.count(ofc.Row.TOP),
            opp_ps.board.count(ofc.Row.MIDDLE),
            opp_ps.board.count(ofc.Row.BOTTOM),
        ], dtype=np.float32)
        
        # 確率
        hand_mask = 0
        for card in hand:
            hand_mask |= (1 << card.index)
        visible_mask = all_mask | hand_mask
        remaining_mask = 0xFFFFFFFFFFFFFFFF & ~visible_mask
        remaining_mask &= ((1 << NUM_CARDS) - 1)
        remaining_count = NUM_CARDS - bin(visible_mask).count('1')
        
        my_probs = [
            ofc.flush_probability(ps.board.mid_mask(), ps.board.count(ofc.Row.MIDDLE), 5, remaining_mask, remaining_count),
            ofc.straight_probability(ps.board.mid_mask(), ps.board.count(ofc.Row.MIDDLE), remaining_mask, remaining_count),
            ofc.flush_probability(ps.board.bot_mask(), ps.board.count(ofc.Row.BOTTOM), 5, remaining_mask, remaining_count),
            ofc.straight_probability(ps.board.bot_mask(), ps.board.count(ofc.Row.BOTTOM), remaining_mask, remaining_count),
        ]
        opp_probs = [
            ofc.flush_probability(opp_ps.board.mid_mask(), opp_ps.board.count(ofc.Row.MIDDLE), 5, remaining_mask, remaining_count),
            ofc.straight_probability(opp_ps.board.mid_mask(), opp_ps.board.count(ofc.Row.MIDDLE), remaining_mask, remaining_count),
            ofc.flush_probability(opp_ps.board.bot_mask(), opp_ps.board.count(ofc.Row.BOTTOM), 5, remaining_mask, remaining_count),
            ofc.straight_probability(opp_ps.board.bot_mask(), opp_ps.board.count(ofc.Row.BOTTOM), remaining_mask, remaining_count),
        ]
        probs_obs = np.array(my_probs + opp_probs, dtype=np.float32)

        obs = {
            'my_board': np.expand_dims(my_board, 0),
            'opponent_board': np.expand_dims(opp_board, 0),
            'hand': np.expand_dims(hand_obs, 0),
            'used_cards': np.expand_dims(used_obs, 0),
            'game_state': np.expand_dims(game_state, 0),
            'probabilities': np.expand_dims(probs_obs, 0),
        }
        return obs
    
    def _estimate_policy_value(self, action: int) -> float:
        """
        Policyを基にアクションの価値を推定
        
        より正確な実装:
        - C++エンジンの状態をコピーして実際にロールアウト
        - 各ステップでModel.predictを使用
        
        現在の簡易実装:
        - アクションIDからヒューリスティックな価値を推定
        """
        # アクションID: card_idx * 3 + row_idx
        # row: 0=TOP, 1=MID, 2=BOT
        row = action % 3
        card_idx = action // 3
        
        # ヒューリスティック:
        # - TOP (3枚) はリスク高いがボーナス大
        # - BOT (5枚) は安全だがボーナス少
        # - MID (5枚) はバランス
        row_scores = {0: 0.0, 1: 2.0, 2: 5.0}  # 安全度スコア
        
        base_score = row_scores.get(row, 0)
        noise = np.random.normal(0, 3)  # 不確実性
        
        return base_score + noise
    
    def predict_with_search(
        self,
        obs: Dict,
        action_mask: np.ndarray,
        engine: Optional[object] = None,
        simulations: int = 100
    ) -> int:
        """
        MCTS 探索を行い、最適なアクションを返す
        
        Args:
            obs: 観測データ
            action_mask: 有効アクションマスク
            engine: C++ GameEngine インスタンス（終盤ソルバー用）
            simulations: 総シミュレーション回数
            
        Returns:
            選択されたアクション ID
        """
        self.stats['total_searches'] += 1
        
        # 終盤ソルバーでの最適解チェック
        if self.solver is not None and engine is not None:
            if self.solver.can_solve(engine, player=0):
                best_action, expected_score = self.solver.solve(engine, player=0)
                self.stats['endgame_solves'] = self.stats.get('endgame_solves', 0) + 1
                return best_action
        
        # 1. Policy から候補手を取得
        candidates = self.get_policy_actions(obs, action_mask, k=self.top_k)
        
        if len(candidates) == 1:
            return candidates[0][0]
        
        # 2. 各候補手のシミュレーション
        sims_per_action = max(1, simulations // len(candidates))
        
        scores = {}
        for action, policy_prob in candidates:
            if self.use_policy_rollout:
                sim_score = self.simulate_rollout(engine, action)
            else:
                sim_score = 0
            
            # Policy と Simulation を組み合わせ
            # policy_weight が高いほど直感を重視
            combined_score = (
                self.policy_weight * policy_prob * 100 +
                (1 - self.policy_weight) * sim_score
            )
            scores[action] = combined_score
        
        # 3. 最高スコアのアクションを選択
        best_action = max(scores.keys(), key=lambda a: scores[a])
        
        # 統計更新
        policy_action = candidates[0][0]
        if best_action == policy_action:
            self.stats['policy_chosen_rate'] = (
                self.stats['policy_chosen_rate'] * 0.99 + 0.01
            )
        else:
            self.stats['policy_chosen_rate'] *= 0.99
        
        return best_action
    
    def evaluate_vs_policy(
        self,
        env,
        n_games: int = 100,
        simulations: int = 50
    ) -> Dict:
        """
        MCTS エージェント vs Policy-only エージェントの対戦評価
        
        Returns:
            評価結果の辞書
        """
        results = {
            'mcts_wins': 0,
            'policy_wins': 0,
            'draws': 0,
            'mcts_avg_score': 0,
            'policy_avg_score': 0,
        }
        
        print(f"Running {n_games} games: MCTS vs Policy-only...")
        
        for i in range(n_games):
            # MCTS がPlayer 0、Policy がPlayer 1
            obs, _ = env.reset()
            done = False
            
            while not done:
                current_agent = env.env.agent_selection
                
                if current_agent == "player_0":
                    # MCTS プレイヤー
                    action_mask = env.action_masks()
                    action = self.predict_with_search(
                        obs, action_mask, simulations=simulations
                    )
                else:
                    # Policy-only プレイヤー
                    if self.model:
                        action_mask = np.zeros(env.action_space.n, dtype=bool)
                        valid = env.env.get_valid_actions("player_1")
                        for a in valid:
                            action_mask[a] = True
                        action, _ = self.model.predict(
                            obs, action_masks=action_mask
                        )
                    else:
                        valid = env.env.get_valid_actions("player_1")
                        action = np.random.choice(valid) if valid else 0
                
                obs, reward, done, truncated, info = env.step(action)
            
            # 結果集計
            mcts_score = reward
            if mcts_score > 0:
                results['mcts_wins'] += 1
            elif mcts_score < 0:
                results['policy_wins'] += 1
            else:
                results['draws'] += 1
            
            results['mcts_avg_score'] += mcts_score
            
            if (i + 1) % 10 == 0:
                print(f"  Game {i+1}/{n_games}: MCTS {results['mcts_wins']} - {results['policy_wins']} Policy")
        
        results['mcts_avg_score'] /= n_games
        results['mcts_win_rate'] = results['mcts_wins'] / n_games * 100
        
        return results


def simple_rollout_test():
    """簡易ロールアウトテスト（エンジンなし）"""
    print("=" * 50)
    print("MCTS Agent - Simple Test")
    print("=" * 50)
    
    agent = MCTSAgent(top_k=3, policy_weight=0.5)
    
    # ダミー観測
    obs = {
        'game_state': np.zeros(8),
        'hand': np.zeros(260),
        'my_board': np.zeros(156),
        'opponent_board': np.zeros(156),
        'used_cards': np.zeros(52),
    }
    
    # ダミーアクションマスク（10個のアクションが有効）
    action_mask = np.zeros(243, dtype=bool)
    action_mask[:10] = True
    
    # 探索実行
    print("\nRunning MCTS search (random simulation)...")
    action = agent.predict_with_search(obs, action_mask, simulations=50)
    print(f"Selected action: {action}")
    print(f"Stats: {agent.stats}")


if __name__ == "__main__":
    simple_rollout_test()
