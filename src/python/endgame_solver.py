"""
OFC Pineapple AI - End-game Solver
残り枚数が少ない終盤局面で全探索を行い、理論上の正解を導出する

用途:
- 教師あり学習のGround Truth生成
- 推論時の終盤精度向上
"""

import os
import sys
import numpy as np
from typing import List, Tuple, Dict, Optional
from itertools import permutations

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import ofc_engine as ofc
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False
    print("[EndgameSolver] Warning: ofc_engine not available")


class EndgameSolver:
    """
    終盤全探索ソルバー
    
    残りカードが少ない状況（≤5枚）で全ての配置パターンを試し、
    最適な手を見つける。
    """
    
    def __init__(self, max_remaining: int = 5):
        """
        Args:
            max_remaining: ソルブ対象とする残りカード枚数の上限
        """
        self.max_remaining = max_remaining
        self.stats = {
            'total_solves': 0,
            'cache_hits': 0,
            'positions_evaluated': 0,
        }
        self._cache: Dict[bytes, Tuple[int, float]] = {}
    
    def can_solve(self, engine: 'ofc.GameEngine', player: int = 0) -> bool:
        """
        この局面がソルブ可能か（残りカードが十分少ないか）
        """
        if not HAS_ENGINE:
            return False
        
        remaining = engine.remaining_cards_in_board(player)
        return remaining <= self.max_remaining
    
    def solve(
        self,
        engine: 'ofc.GameEngine',
        player: int = 0,
        use_cache: bool = True
    ) -> Tuple[int, float]:
        """
        最適な配置を全探索で求める
        
        Args:
            engine: ゲームエンジン
            player: 対象プレイヤー
            use_cache: キャッシュを使用するか
            
        Returns:
            (best_action_id, expected_score)
        """
        if not HAS_ENGINE:
            return 0, 0.0
        
        # キャッシュチェック
        if use_cache:
            state = engine.serialize()
            if state in self._cache:
                self.stats['cache_hits'] += 1
                return self._cache[state]
        
        self.stats['total_solves'] += 1
        
        # 現在の手札を取得
        player_state = engine.player(player)
        hand = [player_state.get_hand()[i] for i in range(player_state.hand_count)]
        
        if len(hand) == 0:
            return 0, 0.0
        
        # ボードの空きスロットを確認
        board = player_state.board
        available_rows = []
        if board.can_place(ofc.Row.TOP):
            available_rows.append(ofc.Row.TOP)
        if board.can_place(ofc.Row.MIDDLE):
            available_rows.append(ofc.Row.MIDDLE)
        if board.can_place(ofc.Row.BOTTOM):
            available_rows.append(ofc.Row.BOTTOM)
        
        if not available_rows:
            return 0, 0.0
        
        best_action = None
        best_score = float('-inf')
        
        # 全ての配置パターンを試す（初回ターンかどうかで分岐）
        phase = engine.phase()
        
        if phase == ofc.GamePhase.INITIAL_DEAL:
            # 初回5枚配置
            best_action, best_score = self._solve_initial(engine, player, hand, available_rows)
        else:
            # 通常ターン: 2枚配置 + 1枚捨て
            best_action, best_score = self._solve_turn(engine, player, hand, available_rows)
        
        self.stats['positions_evaluated'] += 1
        
        # キャッシュに保存
        if use_cache:
            self._cache[state] = (best_action, best_score)
        
        return best_action, best_score
    
    def _solve_initial(
        self,
        engine: 'ofc.GameEngine',
        player: int,
        hand: List['ofc.Card'],
        available_rows: List['ofc.Row']
    ) -> Tuple[int, float]:
        """初回5枚配置の全探索"""
        best_action = 0
        best_score = float('-inf')
        
        # 5枚の配置パターンを全列挙（5! * 3^5 は多すぎるので簡易化）
        # 各カードがどの行に行くかの組み合わせ
        rows = [ofc.Row.TOP, ofc.Row.MIDDLE, ofc.Row.BOTTOM]
        row_limits = [3, 5, 5]  # 各行の上限
        
        for row_combo in self._generate_row_combinations(len(hand), row_limits):
            # この配置が有効かチェック
            row_counts = [0, 0, 0]
            for r in row_combo:
                row_counts[r] += 1
            if row_counts[0] > 3 or row_counts[1] > 5 or row_counts[2] > 5:
                continue
            
            # 配置を適用してスコアを評価
            cloned = engine.clone()
            action = ofc.InitialAction()
            valid = True
            
            for i, (card, row) in enumerate(zip(hand, row_combo)):
                action.set_placement(i, card, rows[row])
            
            if cloned.apply_initial_action(player, action):
                # スコアを評価（終了まで進めて計算）
                score = self._evaluate_final_score(cloned, player)
                if score > best_score:
                    best_score = score
                    best_action = self._encode_initial_action(row_combo)
        
        return best_action, best_score
    
    def _solve_turn(
        self,
        engine: 'ofc.GameEngine',
        player: int,
        hand: List['ofc.Card'],
        available_rows: List['ofc.Row']
    ) -> Tuple[int, float]:
        """通常ターン（2枚配置 + 1枚捨て）の全探索"""
        best_action = 0
        best_score = float('-inf')
        
        if len(hand) < 3:
            return 0, 0.0
        
        # 3枚から2枚選んで配置、1枚捨てる
        rows = [ofc.Row.TOP, ofc.Row.MIDDLE, ofc.Row.BOTTOM]
        
        for discard_idx in range(3):
            place_indices = [i for i in range(3) if i != discard_idx]
            place_cards = [hand[i] for i in place_indices]
            discard_card = hand[discard_idx]
            
            # 2枚の配置パターン（各カードがどの行に行くか）
            for r1 in available_rows:
                for r2 in available_rows:
                    cloned = engine.clone()
                    action = ofc.TurnAction()
                    action.set_placement(0, place_cards[0], r1)
                    action.set_placement(1, place_cards[1], r2)
                    action.discard = discard_card
                    
                    if cloned.apply_turn_action(player, action):
                        score = self._evaluate_final_score(cloned, player)
                        if score > best_score:
                            best_score = score
                            # アクションをエンコード: card_idx * 3 + row
                            best_action = self._encode_turn_action(
                                place_indices[0], r1, 
                                place_indices[1], r2,
                                discard_idx
                            )
        
        return best_action, best_score
    
    def _evaluate_final_score(self, engine: 'ofc.GameEngine', player: int) -> float:
        """
        盤面の最終スコアを評価（実ゲームスコアリング準拠）

        C++ GameEngine の calculate_heads_up_score() と同じロジック:
        1. ファウル判定 → ファウル時は -6*相手数 - 相手royalty
        2. 3行比較（Top/Middle/Bottom）で各+1/-1
        3. スクープボーナス（3行全勝で+3）
        4. ロイヤリティ交換（自分royalty - 相手royalty）
        5. FL突入のEV加算

        未完成ボード（終盤ソルバーの途中評価）の場合:
        - 完成行同士は実際のスコアリングで比較
        - 未完成行はファウルリスクを考慮した推定値
        """
        player_state = engine.player(player)
        board = player_state.board
        num_players = engine.num_players()

        # === 1. ファウル判定（実際のゲームスコアリング公式） ===
        if board.is_foul():
            # P1ファウル時: 各相手に -(6 + 相手royalty) を支払う
            foul_penalty = 0.0
            for i in range(num_players):
                if i == player:
                    continue
                opp_board = engine.player(i).board
                if opp_board.is_foul():
                    # 両者ファウル: 相殺（0点）
                    pass
                else:
                    opp_royalty = opp_board.calculate_royalties() if opp_board.is_complete() else 0
                    foul_penalty -= (6 + opp_royalty)
            return foul_penalty

        # === 2. ボード完成度チェック ===
        top_count = board.count(ofc.Row.TOP)
        mid_count = board.count(ofc.Row.MIDDLE)
        bot_count = board.count(ofc.Row.BOTTOM)
        is_complete = board.is_complete()

        # 未完成でファウルリスクが高い場合のペナルティ
        if not is_complete:
            top_val = board.evaluate_top() if top_count >= 2 else None
            mid_val = board.evaluate_mid() if mid_count >= 3 else None
            bot_val = board.evaluate_bot() if bot_count >= 3 else None

            # 部分評価でファウル確定/危険を検出
            if bot_val and mid_val and bot_val < mid_val:
                if bot_count >= 5 and mid_count >= 5:
                    return self._foul_penalty(engine, player)  # 確定ファウル
            if mid_val and top_val and mid_val < top_val:
                if mid_count >= 5 and top_count >= 3:
                    return self._foul_penalty(engine, player)  # 確定ファウル

        # === 3. 実ゲームスコアリング: ヘッズアップ計算 ===
        own_royalty = board.calculate_royalties() if is_complete else 0
        total_score = 0.0

        for i in range(num_players):
            if i == player:
                continue

            opp_board = engine.player(i).board
            opp_complete = opp_board.is_complete()

            # 相手がファウルの場合
            if opp_complete and opp_board.is_foul():
                total_score += 6 + own_royalty
                continue

            # 通常対決: 3行比較
            line_wins = 0
            line_losses = 0

            if is_complete and opp_complete:
                # Top比較
                t1 = board.evaluate_top()
                t2 = opp_board.evaluate_top()
                if t1 > t2:
                    line_wins += 1
                elif t2 > t1:
                    line_losses += 1

                # Middle比較
                m1 = board.evaluate_mid()
                m2 = opp_board.evaluate_mid()
                if m1 > m2:
                    line_wins += 1
                elif m2 > m1:
                    line_losses += 1

                # Bottom比較
                b1 = board.evaluate_bot()
                b2 = opp_board.evaluate_bot()
                if b1 > b2:
                    line_wins += 1
                elif b2 > b1:
                    line_losses += 1

                # ライン点数
                line_diff = line_wins - line_losses
                total_score += line_diff

                # スクープボーナス
                if line_wins == 3:
                    total_score += 3
                elif line_losses == 3:
                    total_score -= 3

                # ロイヤリティ交換
                opp_royalty = opp_board.calculate_royalties()
                total_score += own_royalty - opp_royalty

            elif is_complete:
                # 自分完成、相手未完成: 推定スコア（自分のroyalty基準）
                total_score += own_royalty * 0.5

            else:
                # 両方未完成: ファウルリスクに基づく推定
                foul_risk = self._estimate_partial_foul_risk(board,
                                                             top_count, mid_count, bot_count)
                total_score -= foul_risk * 10.0

        # === 4. FL突入のEV加算 ===
        if is_complete and board.qualifies_for_fl():
            fl_cards = board.fl_card_count()
            # C++ mcts.hpp の FL_EXPECTED_SCORE 定数と同じ値
            fl_ev = {14: 18.0, 15: 22.0, 16: 26.0, 17: 32.0}
            total_score += fl_ev.get(fl_cards, 18.0)
        elif not is_complete and top_count >= 2:
            # 未完成だがFL可能性がある場合は部分的なボーナス
            top_val = board.evaluate_top()
            if top_val and top_val.rank == ofc.HandRank.THREE_OF_A_KIND:
                total_score += 25.0  # Trips: 高確率でFL
            elif top_val and top_val.rank == ofc.HandRank.ONE_PAIR:
                # QQ以上のペアのみFL対象
                if board.qualifies_for_fl():
                    fl_cards = board.fl_card_count()
                    fl_ev = {14: 18.0, 15: 22.0, 16: 26.0, 17: 32.0}
                    total_score += fl_ev.get(fl_cards, 18.0) * 0.8  # 確率調整

        return total_score

    def _foul_penalty(self, engine: 'ofc.GameEngine', player: int) -> float:
        """ファウル時の実際のペナルティを計算"""
        num_players = engine.num_players()
        penalty = 0.0
        for i in range(num_players):
            if i == player:
                continue
            opp_board = engine.player(i).board
            if opp_board.is_foul():
                pass  # 両者ファウル: 相殺
            else:
                opp_royalty = opp_board.calculate_royalties() if opp_board.is_complete() else 0
                penalty -= (6 + opp_royalty)
        return penalty

    def _estimate_partial_foul_risk(self, board, top_count: int,
                                     mid_count: int, bot_count: int) -> float:
        """未完成ボードのファウルリスクを推定 (0.0-1.0)"""
        risk = 0.0

        if top_count >= 2 and mid_count >= 3:
            top_val = board.evaluate_top()
            mid_val = board.evaluate_mid()
            if mid_val < top_val:
                risk += 0.6  # Middle < Top: 高リスク
            elif mid_val == top_val:
                risk += 0.3

        if mid_count >= 3 and bot_count >= 3:
            mid_val = board.evaluate_mid()
            bot_val = board.evaluate_bot()
            if bot_val < mid_val:
                risk += 0.5  # Bottom < Middle: 高リスク
            elif bot_val == mid_val:
                risk += 0.2

        return min(1.0, risk)
    
    def _generate_row_combinations(self, n: int, limits: List[int]):
        """n枚のカードを3行に分配する全組み合わせを生成"""
        if n == 0:
            yield []
            return
        
        for first_row in range(3):
            for rest in self._generate_row_combinations(n - 1, limits):
                yield [first_row] + rest
    
    def _encode_initial_action(self, row_combo: List[int]) -> int:
        """初回アクションをエンコード

        Environment expects: action_id = sum(row_combo[i] * 3^i for i in range(5))
        where row_combo[i] ∈ {0=TOP, 1=MIDDLE, 2=BOTTOM}
        """
        if not row_combo:
            return 0

        action_id = 0
        for i, row in enumerate(row_combo):
            action_id += row * (3 ** i)
        return action_id
    
    def _encode_turn_action(
        self,
        idx1: int, row1: 'ofc.Row',
        idx2: int, row2: 'ofc.Row',
        discard_idx: int
    ) -> int:
        """通常ターンアクションをエンコード

        Environment expects: action_id = discard_idx * 9 + first_row * 3 + second_row
        where row ∈ {0=TOP, 1=MIDDLE, 2=BOTTOM}, discard_idx ∈ {0, 1, 2}

        Note: The card indices (idx1, idx2) determine which cards go where,
        but the action encoding is based on the discard choice and row placements.
        """
        row1_val = row1.value if hasattr(row1, 'value') else int(row1)
        row2_val = row2.value if hasattr(row2, 'value') else int(row2)
        return discard_idx * 9 + row1_val * 3 + row2_val


def test_solver():
    """ソルバーのテスト"""
    if not HAS_ENGINE:
        print("ofc_engine not available, skipping test")
        return
    
    print("=" * 50)
    print("End-game Solver Test")
    print("=" * 50)
    
    solver = EndgameSolver(max_remaining=5)
    
    # ゲームを開始
    engine = ofc.GameEngine(2)
    engine.start_new_game(42)
    
    print(f"Game phase: {engine.phase()}")
    print(f"Remaining cards for player 0: {engine.remaining_cards_in_board(0)}")
    
    # ソルブ可能かチェック
    can_solve = solver.can_solve(engine, 0)
    print(f"Can solve: {can_solve}")
    
    if engine.remaining_cards_in_board(0) <= 5:
        best_action, best_score = solver.solve(engine, 0)
        print(f"Best action: {best_action}")
        print(f"Expected score: {best_score}")
    else:
        print("Too many remaining cards to solve (need <= 5)")
    
    print(f"Stats: {solver.stats}")
    print("Test complete!")


if __name__ == "__main__":
    test_solver()
