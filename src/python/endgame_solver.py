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
        盤面の最終スコアを評価
        
        ゲームが終了していない場合は、ヒューリスティックで評価
        """
        phase = engine.phase()
        
        if phase == ofc.GamePhase.COMPLETE:
            result = engine.result()
            return float(result.get_score(player))
        
        # 終了していない場合: ロイヤリティとファウルで評価
        player_state = engine.player(player)
        board = player_state.board
        
        if board.is_foul():
            return -30.0  # ファウルペナルティ
        
        royalty = board.calculate_royalties()
        return float(royalty)
    
    def _generate_row_combinations(self, n: int, limits: List[int]):
        """n枚のカードを3行に分配する全組み合わせを生成"""
        if n == 0:
            yield []
            return
        
        for first_row in range(3):
            for rest in self._generate_row_combinations(n - 1, limits):
                yield [first_row] + rest
    
    def _encode_initial_action(self, row_combo: List[int]) -> int:
        """初回アクションをエンコード"""
        # 最初のカードの配置をアクションIDとして返す
        if row_combo:
            return row_combo[0]
        return 0
    
    def _encode_turn_action(
        self,
        idx1: int, row1: 'ofc.Row',
        idx2: int, row2: 'ofc.Row',
        discard_idx: int
    ) -> int:
        """通常ターンアクションをエンコード"""
        # 簡易エンコーディング
        row1_val = row1.value if hasattr(row1, 'value') else int(row1)
        row2_val = row2.value if hasattr(row2, 'value') else int(row2)
        return idx1 * 9 + row1_val * 3 + row2_val


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
