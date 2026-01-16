"""
OFC Pineapple AI - MCTS Evaluation (Phase 7)
MCTS + FL Solver Agent vs Pure NN Agent の EV 検証シミュレーション

Phase 7 検証プラン:
- MCTS有効時の方が「無理にでもFLを狙い、かつ成功させる」動きが強まるか
- 1手あたりの計算時間が1秒以内に収まっているか
- チップ期待値(EV)の向上を数値で証明

使用例:
    python eval_mcts.py --hands 100 --simulations 200 --model models/ofc_phase5.zip
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# パス設定
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import ofc_engine as ofc
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False
    print("[Eval] Error: ofc_engine not available")
    sys.exit(1)

from mcts_agent import MCTSFLAgent, NNAgent, MCTSConfig


@dataclass
class MatchResult:
    """対戦結果"""
    hands_played: int = 0
    mcts_total_score: float = 0.0
    nn_total_score: float = 0.0
    mcts_wins: int = 0
    nn_wins: int = 0
    draws: int = 0

    # FL統計
    mcts_fl_entries: int = 0
    mcts_fl_stays: int = 0
    nn_fl_entries: int = 0
    nn_fl_stays: int = 0

    # ファウル統計
    mcts_fouls: int = 0
    nn_fouls: int = 0

    # 時間統計
    mcts_avg_time_ms: float = 0.0
    nn_avg_time_ms: float = 0.0

    def summary(self) -> str:
        """結果サマリーを生成"""
        lines = [
            "=" * 60,
            "Match Summary",
            "=" * 60,
            f"Hands Played: {self.hands_played}",
            "",
            "--- Score ---",
            f"MCTS+FL Agent: {self.mcts_total_score:+.1f} pts (avg: {self.mcts_total_score / max(1, self.hands_played):.2f})",
            f"NN Agent:      {self.nn_total_score:+.1f} pts (avg: {self.nn_total_score / max(1, self.hands_played):.2f})",
            f"EV Difference: {(self.mcts_total_score - self.nn_total_score) / max(1, self.hands_played):.2f} pts/hand",
            "",
            "--- Win Rate ---",
            f"MCTS Wins: {self.mcts_wins} ({100 * self.mcts_wins / max(1, self.hands_played):.1f}%)",
            f"NN Wins:   {self.nn_wins} ({100 * self.nn_wins / max(1, self.hands_played):.1f}%)",
            f"Draws:     {self.draws}",
            "",
            "--- Fantasy Land ---",
            f"MCTS FL Entries: {self.mcts_fl_entries} ({100 * self.mcts_fl_entries / max(1, self.hands_played):.1f}%)",
            f"MCTS FL Stays:   {self.mcts_fl_stays}",
            f"NN FL Entries:   {self.nn_fl_entries} ({100 * self.nn_fl_entries / max(1, self.hands_played):.1f}%)",
            f"NN FL Stays:     {self.nn_fl_stays}",
            "",
            "--- Fouls ---",
            f"MCTS Fouls: {self.mcts_fouls} ({100 * self.mcts_fouls / max(1, self.hands_played):.1f}%)",
            f"NN Fouls:   {self.nn_fouls} ({100 * self.nn_fouls / max(1, self.hands_played):.1f}%)",
            "",
            "--- Time Performance ---",
            f"MCTS Avg Time: {self.mcts_avg_time_ms:.1f} ms/action",
            f"NN Avg Time:   {self.nn_avg_time_ms:.1f} ms/action",
            f"Time Limit Check: {'PASS' if self.mcts_avg_time_ms < 1000 else 'FAIL'} (< 1000ms)",
            "=" * 60,
        ]
        return "\n".join(lines)


class MCTSEvaluator:
    """
    MCTS + FL Solver Agent vs Pure NN Agent の対戦評価

    Phase 7 の検証要件:
    1. FL突入を積極的に狙う動きが強まるか
    2. 1手あたり1秒以内の思考時間
    3. チップ期待値(EV)の向上
    """

    def __init__(
        self,
        mcts_agent: MCTSFLAgent,
        nn_agent: NNAgent,
        verbose: bool = True
    ):
        self.mcts_agent = mcts_agent
        self.nn_agent = nn_agent
        self.verbose = verbose

        # 結果記録
        self.results = MatchResult()
        self.hand_history: List[Dict[str, Any]] = []

    def run_match(
        self,
        num_hands: int,
        seed: int = 12345,
        swap_positions: bool = True
    ) -> MatchResult:
        """
        対戦を実行

        Args:
            num_hands: ハンド数
            seed: 乱数シード
            swap_positions: 位置を交互に入れ替える

        Returns:
            対戦結果
        """
        print(f"\n{'=' * 60}")
        print(f"Running {num_hands} hands: MCTS+FL vs NN")
        print(f"{'=' * 60}\n")

        self.results = MatchResult()
        self.hand_history = []

        for hand_idx in range(num_hands):
            hand_seed = seed + hand_idx

            # 位置を決定（交互に入れ替え）
            if swap_positions:
                mcts_position = hand_idx % 2
            else:
                mcts_position = 0

            nn_position = 1 - mcts_position

            # ハンドを実行
            hand_result = self._play_hand(hand_seed, mcts_position, nn_position)
            self._record_hand_result(hand_result, mcts_position)

            # 進捗表示
            if self.verbose and (hand_idx + 1) % 10 == 0:
                ev_diff = (self.results.mcts_total_score - self.results.nn_total_score) / (hand_idx + 1)
                fl_rate = self.results.mcts_fl_entries / (hand_idx + 1) * 100
                print(f"Hand {hand_idx + 1:3}/{num_hands} | "
                      f"MCTS: {self.results.mcts_total_score:+6.1f} | "
                      f"NN: {self.results.nn_total_score:+6.1f} | "
                      f"EV Diff: {ev_diff:+.2f}/hand | "
                      f"FL: {fl_rate:.0f}%")

        return self.results

    def _play_hand(
        self,
        seed: int,
        mcts_position: int,
        nn_position: int
    ) -> Dict[str, Any]:
        """1ハンドをプレイ"""
        engine = ofc.GameEngine(2)
        engine.start_new_game(seed)

        mcts_time_total = 0.0
        nn_time_total = 0.0
        mcts_actions = 0
        nn_actions = 0

        # ゲームループ
        while engine.phase() not in [ofc.GamePhase.SHOWDOWN, ofc.GamePhase.COMPLETE]:
            for player_idx in range(2):
                ps = engine.player(player_idx)

                # FL中はソルバーで処理
                if ps.in_fantasy_land:
                    self._apply_fl_action(engine, player_idx)
                    continue

                phase = engine.phase()
                if phase == ofc.GamePhase.INITIAL_DEAL:
                    if ps.board.total_placed() == 0:
                        if player_idx == mcts_position:
                            start_time = time.time()
                            action = self.mcts_agent.select_action(engine, player_idx)
                            mcts_time_total += (time.time() - start_time) * 1000
                            mcts_actions += 1
                        else:
                            start_time = time.time()
                            action = self.nn_agent.select_action(engine, player_idx)
                            nn_time_total += (time.time() - start_time) * 1000
                            nn_actions += 1

                        self._apply_action(engine, player_idx, action)

                elif phase == ofc.GamePhase.TURN:
                    if ps.hand_count > 0:
                        if player_idx == mcts_position:
                            start_time = time.time()
                            action = self.mcts_agent.select_action(engine, player_idx)
                            mcts_time_total += (time.time() - start_time) * 1000
                            mcts_actions += 1
                        else:
                            start_time = time.time()
                            action = self.nn_agent.select_action(engine, player_idx)
                            nn_time_total += (time.time() - start_time) * 1000
                            nn_actions += 1

                        self._apply_action(engine, player_idx, action)

        # ショーダウン
        if engine.phase() == ofc.GamePhase.SHOWDOWN:
            engine.calculate_scores()

        result = engine.result()

        return {
            'scores': [result.get_score(i) for i in range(2)],
            'royalties': [result.get_royalty(i) for i in range(2)],
            'fouled': [result.is_fouled(i) for i in range(2)],
            'entered_fl': [result.entered_fl(i) for i in range(2)],
            'stayed_fl': [result.stayed_fl(i) for i in range(2)],
            'mcts_time_ms': mcts_time_total / max(1, mcts_actions),
            'nn_time_ms': nn_time_total / max(1, nn_actions),
            'mcts_position': mcts_position,
        }

    def _apply_fl_action(self, engine: Any, player_idx: int):
        """FLアクションを適用（ソルバー使用）"""
        ps = engine.player(player_idx)
        fl_cards = ps.get_hand()

        solution = ofc.solve_fantasy_land(fl_cards, True)

        action = ofc.FLAction()

        # Bottom: 5 cards
        for j, card in enumerate(solution.bot):
            action.set_placement(j, card, ofc.BOTTOM)

        # Middle: 5 cards
        for j, card in enumerate(solution.mid):
            action.set_placement(j + 5, card, ofc.MIDDLE)

        # Top: 3 cards
        for j, card in enumerate(solution.top):
            action.set_placement(j + 10, card, ofc.TOP)

        action.discards = solution.discards

        engine.apply_fl_action(player_idx, action)

    def _apply_action(self, engine: Any, player_idx: int, action_id: int):
        """アクションを適用"""
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
            engine.apply_initial_action(player_idx, action)

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

            engine.apply_turn_action(player_idx, action)

    def _record_hand_result(self, hand_result: Dict[str, Any], mcts_position: int):
        """ハンド結果を記録"""
        nn_position = 1 - mcts_position

        mcts_score = hand_result['scores'][mcts_position]
        nn_score = hand_result['scores'][nn_position]

        self.results.hands_played += 1
        self.results.mcts_total_score += mcts_score
        self.results.nn_total_score += nn_score

        if mcts_score > nn_score:
            self.results.mcts_wins += 1
        elif nn_score > mcts_score:
            self.results.nn_wins += 1
        else:
            self.results.draws += 1

        # FL統計
        if hand_result['entered_fl'][mcts_position]:
            self.results.mcts_fl_entries += 1
        if hand_result['stayed_fl'][mcts_position]:
            self.results.mcts_fl_stays += 1
        if hand_result['entered_fl'][nn_position]:
            self.results.nn_fl_entries += 1
        if hand_result['stayed_fl'][nn_position]:
            self.results.nn_fl_stays += 1

        # ファウル統計
        if hand_result['fouled'][mcts_position]:
            self.results.mcts_fouls += 1
        if hand_result['fouled'][nn_position]:
            self.results.nn_fouls += 1

        # 時間統計（移動平均）
        self.results.mcts_avg_time_ms = (
            self.results.mcts_avg_time_ms * 0.9 +
            hand_result['mcts_time_ms'] * 0.1
        )
        self.results.nn_avg_time_ms = (
            self.results.nn_avg_time_ms * 0.9 +
            hand_result['nn_time_ms'] * 0.1
        )

        self.hand_history.append(hand_result)


def run_3player_evaluation(
    mcts_agent: MCTSFLAgent,
    nn_agents: List[NNAgent],
    num_hands: int = 50,
    seed: int = 12345,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    3人対戦評価 (Phase 7 Task 3)

    プレイヤー構成:
    - Player 0: MCTS + FL Solver Agent
    - Player 1, 2: Pure NN Agent (Phase 5 Model)

    Returns:
        3人対戦の結果
    """
    print(f"\n{'=' * 60}")
    print(f"Running {num_hands} hands: 3-Player Match")
    print(f"Player 0: MCTS+FL Agent")
    print(f"Player 1, 2: NN Agent")
    print(f"{'=' * 60}\n")

    results = {
        'hands_played': 0,
        'scores': [0.0, 0.0, 0.0],
        'fl_entries': [0, 0, 0],
        'fl_stays': [0, 0, 0],
        'fouls': [0, 0, 0],
        'avg_times_ms': [0.0, 0.0, 0.0],
    }

    for hand_idx in range(num_hands):
        hand_seed = seed + hand_idx

        # エンジン初期化
        engine = ofc.GameEngine(3)
        engine.start_new_game(hand_seed)

        action_counts = [0, 0, 0]
        time_totals = [0.0, 0.0, 0.0]

        # ゲームループ
        while engine.phase() not in [ofc.GamePhase.SHOWDOWN, ofc.GamePhase.COMPLETE]:
            for player_idx in range(3):
                ps = engine.player(player_idx)

                if ps.in_fantasy_land:
                    # FL中はソルバーで処理
                    fl_cards = ps.get_hand()
                    solution = ofc.solve_fantasy_land(fl_cards, True)

                    action = ofc.FLAction()
                    for j, card in enumerate(solution.bot):
                        action.set_placement(j, card, ofc.BOTTOM)
                    for j, card in enumerate(solution.mid):
                        action.set_placement(j + 5, card, ofc.MIDDLE)
                    for j, card in enumerate(solution.top):
                        action.set_placement(j + 10, card, ofc.TOP)
                    action.discards = solution.discards

                    engine.apply_fl_action(player_idx, action)
                    continue

                phase = engine.phase()
                if phase == ofc.GamePhase.INITIAL_DEAL:
                    if ps.board.total_placed() == 0:
                        start_time = time.time()
                        if player_idx == 0:
                            action_id = mcts_agent.select_action(engine, player_idx)
                        else:
                            action_id = nn_agents[player_idx - 1].select_action(engine, player_idx)
                        time_totals[player_idx] += (time.time() - start_time) * 1000
                        action_counts[player_idx] += 1

                        _apply_action_3p(engine, player_idx, action_id)

                elif phase == ofc.GamePhase.TURN:
                    if ps.hand_count > 0:
                        start_time = time.time()
                        if player_idx == 0:
                            action_id = mcts_agent.select_action(engine, player_idx)
                        else:
                            action_id = nn_agents[player_idx - 1].select_action(engine, player_idx)
                        time_totals[player_idx] += (time.time() - start_time) * 1000
                        action_counts[player_idx] += 1

                        _apply_action_3p(engine, player_idx, action_id)

        # ショーダウン
        if engine.phase() == ofc.GamePhase.SHOWDOWN:
            engine.calculate_scores()

        result = engine.result()

        results['hands_played'] += 1
        for i in range(3):
            results['scores'][i] += result.get_score(i)
            if result.entered_fl(i):
                results['fl_entries'][i] += 1
            if result.stayed_fl(i):
                results['fl_stays'][i] += 1
            if result.is_fouled(i):
                results['fouls'][i] += 1

            if action_counts[i] > 0:
                avg_time = time_totals[i] / action_counts[i]
                results['avg_times_ms'][i] = (
                    results['avg_times_ms'][i] * 0.9 + avg_time * 0.1
                )

        if verbose and (hand_idx + 1) % 10 == 0:
            print(f"Hand {hand_idx + 1:3}/{num_hands} | "
                  f"Scores: [{results['scores'][0]:+.1f}, "
                  f"{results['scores'][1]:+.1f}, {results['scores'][2]:+.1f}]")

    # 最終結果表示
    print(f"\n{'=' * 60}")
    print("3-Player Match Summary (Phase 7 Task 3)")
    print(f"{'=' * 60}")
    print(f"Hands Played: {results['hands_played']}")
    print(f"\n--- Final Scores ---")
    print(f"  MCTS Agent (P0): {results['scores'][0]:+.1f} "
          f"(avg: {results['scores'][0] / results['hands_played']:.2f}/hand)")
    print(f"  NN Agent (P1):   {results['scores'][1]:+.1f} "
          f"(avg: {results['scores'][1] / results['hands_played']:.2f}/hand)")
    print(f"  NN Agent (P2):   {results['scores'][2]:+.1f} "
          f"(avg: {results['scores'][2] / results['hands_played']:.2f}/hand)")

    # EV差の計算
    mcts_ev = results['scores'][0] / results['hands_played']
    nn_avg_ev = (results['scores'][1] + results['scores'][2]) / (2 * results['hands_played'])
    ev_advantage = mcts_ev - nn_avg_ev

    print(f"\n--- EV Analysis ---")
    print(f"  MCTS EV: {mcts_ev:.2f}/hand")
    print(f"  NN Avg EV: {nn_avg_ev:.2f}/hand")
    print(f"  MCTS Advantage: {ev_advantage:+.2f}/hand")

    print(f"\n--- Fantasy Land Stats ---")
    print(f"  FL Entries: {results['fl_entries']}")
    print(f"  FL Stays:   {results['fl_stays']}")
    mcts_fl_rate = results['fl_entries'][0] / results['hands_played'] * 100
    nn_fl_rate = (results['fl_entries'][1] + results['fl_entries'][2]) / (2 * results['hands_played']) * 100
    print(f"  MCTS FL Rate: {mcts_fl_rate:.1f}%")
    print(f"  NN Avg FL Rate: {nn_fl_rate:.1f}%")

    print(f"\n--- Fouls ---")
    print(f"  {results['fouls']}")

    print(f"\n--- Time Performance ---")
    print(f"  Avg Times (ms): {[f'{t:.1f}' for t in results['avg_times_ms']]}")
    print(f"  MCTS Time Check: {'PASS' if results['avg_times_ms'][0] < 1000 else 'FAIL'} (< 1000ms)")

    # 検証結果
    print(f"\n{'=' * 60}")
    print("Phase 7 Verification Results")
    print(f"{'=' * 60}")

    checks = []

    # 1. FL突入率の検証
    if mcts_fl_rate > nn_fl_rate:
        checks.append(("FL Strategy", "PASS", f"MCTS FL rate ({mcts_fl_rate:.1f}%) > NN ({nn_fl_rate:.1f}%)"))
    else:
        checks.append(("FL Strategy", "FAIL", f"MCTS FL rate ({mcts_fl_rate:.1f}%) <= NN ({nn_fl_rate:.1f}%)"))

    # 2. 思考時間の検証
    if results['avg_times_ms'][0] < 1000:
        checks.append(("Time Limit", "PASS", f"Avg {results['avg_times_ms'][0]:.1f}ms < 1000ms"))
    else:
        checks.append(("Time Limit", "FAIL", f"Avg {results['avg_times_ms'][0]:.1f}ms >= 1000ms"))

    # 3. EV向上の検証
    if ev_advantage > 0:
        checks.append(("EV Improvement", "PASS", f"+{ev_advantage:.2f}/hand"))
    else:
        checks.append(("EV Improvement", "FAIL", f"{ev_advantage:.2f}/hand"))

    for check_name, status, detail in checks:
        print(f"  [{status}] {check_name}: {detail}")

    print(f"{'=' * 60}")

    return results


def _apply_action_3p(engine: Any, player_idx: int, action_id: int):
    """3人戦用アクション適用"""
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
        engine.apply_initial_action(player_idx, action)

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

        engine.apply_turn_action(player_idx, action)


def main():
    parser = argparse.ArgumentParser(
        description="OFC MCTS Agent Evaluation (Phase 7)"
    )
    parser.add_argument(
        "--hands", type=int, default=100,
        help="Number of hands to play"
    )
    parser.add_argument(
        "--simulations", type=int, default=200,
        help="MCTS simulations per action"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to MaskablePPO model"
    )
    parser.add_argument(
        "--fl-weight", type=float, default=0.6,
        help="FL value weight in MCTS (0-1)"
    )
    parser.add_argument(
        "--max-time", type=int, default=1000,
        help="Max time per action (ms)"
    )
    parser.add_argument(
        "--seed", type=int, default=12345,
        help="Random seed"
    )
    parser.add_argument(
        "--players", type=int, default=2, choices=[2, 3],
        help="Number of players (2 or 3)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # MCTS設定
    config = MCTSConfig(
        num_simulations=args.simulations,
        fl_weight=args.fl_weight,
        max_time_ms=args.max_time,
    )

    # エージェント作成
    print("=" * 60)
    print("OFC MCTS Evaluation - Phase 7")
    print("=" * 60)
    print(f"Simulations: {args.simulations}")
    print(f"FL Weight: {args.fl_weight}")
    print(f"Max Time: {args.max_time}ms")
    print(f"Model: {args.model or 'None (random)'}")
    print()

    print("Initializing agents...")
    mcts_agent = MCTSFLAgent(model_path=args.model, config=config)
    nn_agent = NNAgent(model_path=args.model)

    if args.players == 2:
        # 2人対戦
        evaluator = MCTSEvaluator(mcts_agent, nn_agent, verbose=args.verbose)
        results = evaluator.run_match(args.hands, args.seed)
        print(results.summary())

    else:
        # 3人対戦 (Phase 7 Task 3)
        nn_agents = [
            NNAgent(model_path=args.model),
            NNAgent(model_path=args.model)
        ]
        results = run_3player_evaluation(
            mcts_agent, nn_agents,
            num_hands=args.hands,
            seed=args.seed,
            verbose=args.verbose
        )

    print("\nMCTS Agent Stats:", mcts_agent.get_stats())


if __name__ == "__main__":
    main()
