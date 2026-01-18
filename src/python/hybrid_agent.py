"""
OFC Pineapple AI - Hybrid Inference Agent
推論時の知性を最大化するハイブリッドエージェント

Phase 8.5+ の中核コンポーネント:
- 終盤（残り≤2ターン）: EndgameSolver で完璧な解
- 重要局面（勝率拮抗/高スコア機会）: MCTS で深読み
- 通常: Policy Network で即決

使用例:
    from hybrid_agent import HybridInferenceAgent

    agent = HybridInferenceAgent(model_path="models/ofc_phase7.zip")
    action = agent.act(engine, player_idx)
"""

import os
import sys
import time
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import ofc_engine as ofc
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False
    print("[HybridAgent] Warning: ofc_engine not available")

try:
    from sb3_contrib import MaskablePPO
    import torch
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False

# 既存モジュールのインポート
from mcts_agent import MCTSFLAgent, MCTSConfig, NNAgent
from endgame_solver import EndgameSolver


@dataclass
class HybridConfig:
    """ハイブリッドエージェント設定

    Note: 現時点ではEndgame SolverとMCTSはPure NNより性能が悪いため、
    デフォルトでは無効化されています。将来の改善後に有効化してください。
    """
    # 終盤ソルバー設定 (デフォルト無効化)
    endgame_threshold: int = 2          # 残りターン数がこれ以下で終盤ソルバー
    endgame_max_remaining: int = 0      # 0=無効。残り配置スロット数

    # MCTS設定 (デフォルト無効化)
    mcts_simulations: int = 200         # MCTSシミュレーション数
    mcts_time_limit_ms: int = 1000      # MCTS時間制限
    mcts_fl_weight: float = 0.6         # FL重み

    # 重要局面判定 (デフォルト無効化 - 閾値を高く設定)
    critical_fl_threshold: float = 1.0  # FL確率がこれ以上で重要 (1.0=無効)
    critical_royalty_threshold: int = 100 # 現在ロイヤリティがこれ以上で重要
    critical_foul_risk: float = 1.0     # ファウルリスクがこれ以上で重要 (1.0=無効)

    # NN設定
    nn_temperature: float = 0.8         # 低めで確定的に

    # 一般設定
    verbose: bool = False               # デバッグ出力


class HybridInferenceAgent:
    """
    状況に応じて最適な推論方法を選択するハイブリッドエージェント

    推論フロー:
    1. 終盤判定 → EndgameSolver（完璧な解）
    2. 重要局面判定 → MCTS（深読み）
    3. 通常 → PolicyNetwork（即決）
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[Any] = None,
        config: Optional[HybridConfig] = None
    ):
        self.config = config or HybridConfig()

        # サブエージェント初期化
        self._init_subagents(model_path, model)

        # 統計
        self.stats = {
            'total_actions': 0,
            'endgame_solves': 0,
            'mcts_searches': 0,
            'nn_predictions': 0,
            'avg_time_ms': 0.0,
            'fl_critical_count': 0,
            'foul_avoidance_count': 0,
        }

    def _init_subagents(self, model_path: Optional[str], model: Optional[Any]):
        """サブエージェントを初期化"""
        # EndgameSolver
        self.endgame_solver = EndgameSolver(
            max_remaining=self.config.endgame_max_remaining
        )

        # MCTSエージェント
        mcts_config = MCTSConfig(
            num_simulations=self.config.mcts_simulations,
            max_time_ms=self.config.mcts_time_limit_ms,
            fl_weight=self.config.mcts_fl_weight,
            use_policy_prior=True,
            use_fl_solver=True,
        )
        self.mcts_agent = MCTSFLAgent(
            model_path=model_path,
            model=model,
            config=mcts_config
        )

        # NNエージェント
        self.nn_agent = NNAgent(
            model_path=model_path,
            model=model,
            temperature=self.config.nn_temperature
        )

        # モデル参照を保持
        self.model = self.mcts_agent.model

    def act(
        self,
        engine: Any,
        player_idx: int,
        force_mode: Optional[str] = None  # 'endgame', 'mcts', 'nn'
    ) -> int:
        """
        最適なアクションを選択

        Args:
            engine: C++ GameEngine インスタンス
            player_idx: プレイヤーインデックス
            force_mode: 強制的に特定のモードを使用（デバッグ用）

        Returns:
            選択されたアクションID
        """
        start_time = time.time()
        self.stats['total_actions'] += 1

        # 決定モードを判定
        if force_mode:
            mode = force_mode
        else:
            mode = self._determine_mode(engine, player_idx)

        # 各モードで推論
        if mode == 'endgame':
            action = self._act_endgame(engine, player_idx)
            self.stats['endgame_solves'] += 1
        elif mode == 'mcts':
            action = self._act_mcts(engine, player_idx)
            self.stats['mcts_searches'] += 1
        else:  # 'nn'
            action = self._act_nn(engine, player_idx)
            self.stats['nn_predictions'] += 1

        # 時間統計
        elapsed_ms = (time.time() - start_time) * 1000
        self.stats['avg_time_ms'] = (
            self.stats['avg_time_ms'] * 0.95 + elapsed_ms * 0.05
        )

        if self.config.verbose:
            print(f"[Hybrid] Mode={mode}, Action={action}, Time={elapsed_ms:.1f}ms")

        return action

    def _determine_mode(self, engine: Any, player_idx: int) -> str:
        """
        現在の局面に最適な推論モードを判定

        Returns:
            'endgame' | 'mcts' | 'nn'
        """
        if not HAS_ENGINE:
            return 'nn'

        # 1. 終盤判定: 残り配置スロットが少ない場合のみ
        # endgame_solverのcan_solve()と同じ条件を使用
        remaining_slots = engine.remaining_cards_in_board(player_idx)
        if remaining_slots <= self.config.endgame_max_remaining:
            if self.config.verbose:
                print(f"[Hybrid] Endgame detected: {remaining_slots} slots remaining")
            return 'endgame'

        # 2. 重要局面判定
        if self._is_critical_moment(engine, player_idx):
            return 'mcts'

        # 3. デフォルト: NN
        return 'nn'

    def _get_remaining_turns(self, engine: Any) -> int:
        """残りターン数を取得（最大5ターン）"""
        current_turn = engine.current_turn()
        return max(0, 5 - current_turn)

    def _is_critical_moment(self, engine: Any, player_idx: int) -> bool:
        """
        重要局面かどうかを判定

        重要局面の条件:
        - FL突入の可能性が高い（フラグ回収チャンス）
        - 高ロイヤリティ構築中
        - ファウルリスクが高い
        """
        ps = engine.player(player_idx)
        board = ps.board

        # 条件1: FL突入の可能性
        fl_prob = self._estimate_fl_probability(engine, player_idx)
        if fl_prob >= self.config.critical_fl_threshold:
            self.stats['fl_critical_count'] += 1
            if self.config.verbose:
                print(f"[Hybrid] FL opportunity: {fl_prob:.1%}")
            return True

        # 条件2: 高ロイヤリティ構築中
        current_royalty = board.calculate_royalties()
        if current_royalty >= self.config.critical_royalty_threshold:
            if self.config.verbose:
                print(f"[Hybrid] High royalty: {current_royalty}")
            return True

        # 条件3: ファウルリスク
        foul_risk = self._estimate_foul_risk(engine, player_idx)
        if foul_risk >= self.config.critical_foul_risk:
            self.stats['foul_avoidance_count'] += 1
            if self.config.verbose:
                print(f"[Hybrid] Foul risk: {foul_risk:.1%}")
            return True

        return False

    def _estimate_fl_probability(self, engine: Any, player_idx: int) -> float:
        """FL突入確率を推定（C++ calculate_fl_probability使用）"""
        if not HAS_ENGINE:
            return 0.0

        ps = engine.player(player_idx)
        board = ps.board

        # 既にFL資格を持っているか
        if board.qualifies_for_fl():
            return 1.0

        # 使用済みカードマスク
        used_mask = board.all_mask()
        for i in range(engine.num_players()):
            if i != player_idx:
                used_mask |= engine.player(i).board.all_mask()

        remaining_deck = 0xFFFFFFFFFFFFFFFF & ~used_mask
        remaining_deck &= ((1 << 54) - 1)

        remaining_turns = self._get_remaining_turns(engine)

        try:
            fl_prob = ofc.calculate_fl_probability(
                board, remaining_deck, remaining_turns
            )
            return fl_prob.total_prob
        except Exception:
            # フォールバック: 簡易推定
            top_count = board.count(ofc.TOP)
            if top_count <= 2 and remaining_turns >= 1:
                base_prob = 0.15 * (3 - top_count) * remaining_turns
                return min(0.5, base_prob)
            return 0.0

    def _estimate_foul_risk(self, engine: Any, player_idx: int) -> float:
        """
        ファウルリスクを推定

        ボードの現在の強さの順序（Bottom > Middle > Top）が
        崩れそうかどうかを簡易判定
        """
        if not HAS_ENGINE:
            return 0.0

        ps = engine.player(player_idx)
        board = ps.board

        # 既にファウル
        if board.is_foul():
            return 1.0

        top_count = board.count(ofc.TOP)
        mid_count = board.count(ofc.MIDDLE)
        bot_count = board.count(ofc.BOTTOM)

        risk = 0.0

        # 行が一定数完成している場合、ハンド評価で比較
        if mid_count >= 5 and top_count >= 3:
            # Middle と Top を評価
            top_value = board.evaluate_top()
            mid_value = board.evaluate_mid()
            if mid_value < top_value:
                return 1.0  # 確実にファウル

        if bot_count >= 5 and mid_count >= 5:
            # Bottom と Middle を評価
            bot_value = board.evaluate_bot()
            mid_value = board.evaluate_mid()
            if bot_value < mid_value:
                return 1.0  # 確実にファウル

        # 残りスロットが少ない + 危険な状態
        remaining_slots = 13 - (top_count + mid_count + bot_count)
        if remaining_slots <= 3:
            risk += 0.3  # 終盤は慎重に

        # Topが強くなりすぎている兆候
        if top_count >= 2 and mid_count >= 3:
            top_value = board.evaluate_top()
            mid_value = board.evaluate_mid()
            # Topが既にMiddleより強い場合
            if mid_value < top_value:
                risk += 0.5

        return min(1.0, risk)

    def _act_endgame(self, engine: Any, player_idx: int) -> int:
        """終盤ソルバーでアクションを選択

        改善版: NNの選択を検証し、明らかな問題がある場合のみSolverで上書き
        """
        # まずNNの選択を取得
        nn_action = self._act_nn(engine, player_idx)

        if not self.endgame_solver.can_solve(engine, player_idx):
            return nn_action

        # Solverの選択を取得
        solver_action, solver_score = self.endgame_solver.solve(engine, player_idx)

        # NNの選択をシミュレート
        nn_score = self._simulate_action_score(engine, player_idx, nn_action)

        if self.config.verbose:
            print(f"[Endgame] NN={nn_action}(score={nn_score:.1f}), Solver={solver_action}(score={solver_score:.1f})")

        # 判断ロジック:
        # 1. NNの選択がファウルを引き起こす場合はSolverを使用
        # 2. Solverの方がはるかに良い場合はSolverを使用
        # 3. それ以外はNNを信頼
        if nn_score < -50.0 and solver_score > nn_score + 20.0:
            # NNがファウルを選び、Solverがより良い選択肢を持つ
            if self.config.verbose:
                print(f"[Endgame] Using Solver (NN would foul)")
            return solver_action

        if solver_score > nn_score + 30.0:
            # Solverがはるかに良いスコア
            if self.config.verbose:
                print(f"[Endgame] Using Solver (much better score)")
            return solver_action

        # NNを信頼
        return nn_action

    def _simulate_action_score(self, engine: Any, player_idx: int, action: int) -> float:
        """アクションをシミュレートしてスコアを評価"""
        cloned = engine.clone()
        phase = cloned.phase()

        try:
            ps = cloned.player(player_idx)
            hand = [ps.get_hand()[i] for i in range(ps.hand_count)]

            if phase == ofc.GamePhase.INITIAL_DEAL:
                # 初期配置
                ia = ofc.InitialAction()
                for i in range(len(hand)):
                    row_val = (action // (3 ** i)) % 3
                    row = [ofc.Row.TOP, ofc.Row.MIDDLE, ofc.Row.BOTTOM][row_val]
                    ia.set_placement(i, hand[i], row)
                cloned.apply_initial_action(player_idx, ia)
            else:
                # ターンアクション
                discard_idx = action // 9
                row1_val = (action % 9) // 3
                row2_val = action % 3

                place_indices = [i for i in range(3) if i != discard_idx]
                rows = [ofc.Row.TOP, ofc.Row.MIDDLE, ofc.Row.BOTTOM]

                ta = ofc.TurnAction()
                ta.set_placement(0, hand[place_indices[0]], rows[row1_val])
                ta.set_placement(1, hand[place_indices[1]], rows[row2_val])
                ta.discard = hand[discard_idx]
                cloned.apply_turn_action(player_idx, ta)

            # スコアを評価（EndgameSolverの評価関数を再利用）
            return self.endgame_solver._evaluate_final_score(cloned, player_idx)

        except Exception as e:
            if self.config.verbose:
                print(f"[Endgame] Simulation error: {e}")
            return -100.0

    def _act_mcts(self, engine: Any, player_idx: int) -> int:
        """MCTSでアクションを選択"""
        try:
            return self.mcts_agent.select_action(
                engine, player_idx,
                simulations=self.config.mcts_simulations,
                max_time_ms=self.config.mcts_time_limit_ms
            )
        except Exception as e:
            # MCTSが失敗した場合はNNにフォールバック
            if self.config.verbose:
                print(f"[Hybrid] MCTS failed ({e}), falling back to NN")
            return self._act_nn(engine, player_idx)

    def _act_nn(self, engine: Any, player_idx: int) -> int:
        """NNでアクションを選択"""
        try:
            return self.nn_agent.select_action(
                engine, player_idx,
                deterministic=True
            )
        except Exception as e:
            # NNが失敗した場合はランダムにフォールバック
            if self.config.verbose:
                print(f"[Hybrid] NN failed ({e}), using random action")
            return self._random_action(engine, player_idx)

    def _random_action(self, engine: Any, player_idx: int) -> int:
        """ランダムアクション（最終フォールバック）"""
        if not HAS_ENGINE:
            return 0

        ps = engine.player(player_idx)
        board = ps.board
        phase = engine.phase()

        # 有効なアクションを見つける
        valid_actions = []

        if phase == ofc.GamePhase.INITIAL_DEAL:
            # 初期配置: 各カードをどの行に置くか（3^5 = 243通り）
            for action_id in range(243):
                if self._is_valid_initial_action(board, action_id):
                    valid_actions.append(action_id)
        else:
            # ターン中: 3カードから2配置1捨て（27通り）
            for action_id in range(27):
                if self._is_valid_turn_action(board, action_id):
                    valid_actions.append(action_id)

        if not valid_actions:
            return 0

        return np.random.choice(valid_actions)

    def _is_valid_initial_action(self, board: Any, action_id: int) -> bool:
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

    def _is_valid_turn_action(self, board: Any, action_id: int) -> bool:
        """ターンアクションが有効かチェック"""
        card0_row = action_id % 3
        card1_row = (action_id // 3) % 3

        top_slots = 3 - board.count(ofc.TOP)
        mid_slots = 5 - board.count(ofc.MIDDLE)
        bot_slots = 5 - board.count(ofc.BOTTOM)

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

    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        stats = self.stats.copy()
        if stats['total_actions'] > 0:
            stats['endgame_rate'] = stats['endgame_solves'] / stats['total_actions']
            stats['mcts_rate'] = stats['mcts_searches'] / stats['total_actions']
            stats['nn_rate'] = stats['nn_predictions'] / stats['total_actions']
        return stats

    def reset_stats(self):
        """統計をリセット"""
        self.stats = {k: 0 if isinstance(v, int) else 0.0
                      for k, v in self.stats.items()}


class TournamentHybridAgent(HybridInferenceAgent):
    """
    トーナメント用の最適化されたハイブリッドエージェント

    より慎重なパラメータ設定:
    - 終盤ソルバーを早めに発動
    - MCTSシミュレーション数を増加
    - ファウル回避を最優先
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[Any] = None
    ):
        config = HybridConfig(
            # 終盤をより早く発動
            endgame_threshold=3,
            endgame_max_remaining=8,

            # MCTSをより深く
            mcts_simulations=400,
            mcts_time_limit_ms=2000,
            mcts_fl_weight=0.7,

            # 重要局面の閾値を下げる（より頻繁にMCTS）
            critical_fl_threshold=0.2,
            critical_royalty_threshold=5,
            critical_foul_risk=0.3,

            # 確定的に
            nn_temperature=0.5,
        )
        super().__init__(model_path, model, config)


def evaluate_hybrid_agent(
    model_path: str,
    num_games: int = 100,
    compare_nn: bool = True
):
    """
    ハイブリッドエージェントを評価

    Args:
        model_path: モデルファイルのパス
        num_games: 評価ゲーム数
        compare_nn: NN-onlyと比較するか
    """
    if not HAS_ENGINE:
        print("ofc_engine not available")
        return

    print("=" * 60)
    print("Hybrid Agent Evaluation")
    print("=" * 60)

    # エージェント作成
    hybrid_agent = HybridInferenceAgent(
        model_path=model_path,
        config=HybridConfig(verbose=False)
    )

    nn_agent = NNAgent(model_path=model_path) if compare_nn else None

    def run_games(agent, agent_name: str):
        """ゲームを実行して統計を取得"""
        fouls = 0
        total_royalty = 0
        fl_entries = 0
        total_time_ms = 0

        for game_idx in range(num_games):
            engine = ofc.GameEngine(2)
            engine.start_new_game(game_idx)

            game_start = time.time()

            while engine.phase() != ofc.GamePhase.COMPLETE:
                current_player = engine.current_player()

                if current_player == 0:
                    # 評価対象のエージェント
                    action = agent.act(engine, 0) if hasattr(agent, 'act') else \
                             agent.select_action(engine, 0)
                else:
                    # 相手はNN
                    action = nn_agent.select_action(engine, current_player) if nn_agent else 0

                # アクション適用（簡略化）
                # 実際にはエンジンのメソッドを呼ぶ
                try:
                    engine.step(current_player, action)
                except Exception:
                    break

            total_time_ms += (time.time() - game_start) * 1000

            # 結果集計
            ps = engine.player(0)
            if ps.board.is_foul():
                fouls += 1
            total_royalty += ps.board.calculate_royalties()
            if ps.in_fantasy_land:
                fl_entries += 1

        return {
            'foul_rate': fouls / num_games,
            'avg_royalty': total_royalty / num_games,
            'fl_rate': fl_entries / num_games,
            'avg_time_ms': total_time_ms / num_games,
        }

    # 評価実行
    print(f"\nEvaluating {num_games} games...")

    hybrid_results = run_games(hybrid_agent, "Hybrid")
    print(f"\nHybrid Agent:")
    print(f"  Foul Rate: {hybrid_results['foul_rate']:.1%}")
    print(f"  Avg Royalty: {hybrid_results['avg_royalty']:.2f}")
    print(f"  FL Rate: {hybrid_results['fl_rate']:.1%}")
    print(f"  Avg Time: {hybrid_results['avg_time_ms']:.1f}ms")
    print(f"  Mode distribution: {hybrid_agent.get_stats()}")

    if compare_nn:
        nn_results = run_games(nn_agent, "NN-only")
        print(f"\nNN-only Agent:")
        print(f"  Foul Rate: {nn_results['foul_rate']:.1%}")
        print(f"  Avg Royalty: {nn_results['avg_royalty']:.2f}")
        print(f"  FL Rate: {nn_results['fl_rate']:.1%}")
        print(f"  Avg Time: {nn_results['avg_time_ms']:.1f}ms")

        print(f"\n--- Comparison ---")
        foul_diff = hybrid_results['foul_rate'] - nn_results['foul_rate']
        royalty_diff = hybrid_results['avg_royalty'] - nn_results['avg_royalty']
        print(f"  Foul Rate: {foul_diff:+.1%} (Hybrid {'better' if foul_diff < 0 else 'worse'})")
        print(f"  Royalty: {royalty_diff:+.2f} (Hybrid {'better' if royalty_diff > 0 else 'worse'})")


def test_hybrid_agent():
    """ハイブリッドエージェントの基本テスト"""
    print("=" * 60)
    print("Hybrid Agent - Basic Test")
    print("=" * 60)

    if not HAS_ENGINE:
        print("ofc_engine not available. Skipping test.")
        return

    # エージェント作成（モデルなし = ランダム）
    config = HybridConfig(verbose=True)
    agent = HybridInferenceAgent(config=config)

    # ゲーム初期化
    engine = ofc.GameEngine(2)
    engine.start_new_game(42)

    print(f"\nGame phase: {engine.phase()}")
    print(f"Current turn: {engine.current_turn()}")

    # アクション選択テスト
    print("\n--- Testing action selection ---")
    for i in range(3):
        action = agent.act(engine, 0)
        print(f"Turn {i+1}: Action={action}")

    print(f"\nFinal stats: {agent.get_stats()}")
    print("\nTest completed!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run basic test")
    parser.add_argument("--eval", type=str, help="Evaluate model at path")
    parser.add_argument("--games", type=int, default=100, help="Number of games for evaluation")
    args = parser.parse_args()

    if args.test:
        test_hybrid_agent()
    elif args.eval:
        evaluate_hybrid_agent(args.eval, args.games)
    else:
        test_hybrid_agent()
