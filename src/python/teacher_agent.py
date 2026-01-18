"""
OFC Pineapple AI - Teacher Heuristic Agent
人間のトッププレイヤーの戦略ロジックを実装した教師エージェント

JOPTルール準拠:
- OFC Pineapple (3-5-5)
- Fantasyland: TopにQQ以上でファウルなし
- 1-6-1方式スコアリング

戦略:
- FL至上主義: FantasyLand突入を最優先
- Bottom/Middle構成: フラッシュ優先、ファウル回避
- ライブカード計算: アウツ確率に基づく判断

使用例:
    from teacher_agent import TeacherHeuristicAgent

    teacher = TeacherHeuristicAgent(strategy='balanced')
    action = teacher.select_action(engine, player_idx)
"""

import os
import sys
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass
from collections import defaultdict
from enum import IntEnum

# パス設定
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import ofc_engine as ofc
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False
    print("[TeacherAgent] Warning: ofc_engine not available")


# カード定数
class Rank(IntEnum):
    TWO = 0
    THREE = 1
    FOUR = 2
    FIVE = 3
    SIX = 4
    SEVEN = 5
    EIGHT = 6
    NINE = 7
    TEN = 8
    JACK = 9
    QUEEN = 10
    KING = 11
    ACE = 12
    JOKER = 13  # Joker (2枚)

class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


# ロイヤリティテーブル (JOPTルール)
ROYALTY_BOTTOM = {
    'straight': 2,
    'flush': 4,
    'full_house': 6,
    'four_of_a_kind': 10,
    'straight_flush': 15,
    'royal_flush': 25,
}

ROYALTY_MIDDLE = {
    'three_of_a_kind': 2,
    'straight': 4,
    'flush': 8,
    'full_house': 12,
    'four_of_a_kind': 20,
    'straight_flush': 30,
    'royal_flush': 50,
}

ROYALTY_TOP = {
    '66': 1, '77': 2, '88': 3, '99': 4, 'TT': 5,
    'JJ': 6, 'QQ': 7, 'KK': 8, 'AA': 9,
    'trips_222': 10, 'trips_333': 11, 'trips_444': 12,
    'trips_555': 13, 'trips_666': 14, 'trips_777': 15,
    'trips_888': 16, 'trips_999': 17, 'trips_TTT': 18,
    'trips_JJJ': 19, 'trips_QQQ': 20, 'trips_KKK': 21, 'trips_AAA': 22,
}


@dataclass
class CardInfo:
    """カード情報"""
    index: int
    rank: int
    suit: int
    is_joker: bool = False

    @classmethod
    def from_engine_card(cls, card) -> 'CardInfo':
        """C++ Cardオブジェクトから変換"""
        idx = card.index
        if idx >= 52:  # Joker
            return cls(index=idx, rank=Rank.JOKER, suit=-1, is_joker=True)
        return cls(
            index=idx,
            rank=idx % 13,
            suit=idx // 13,
            is_joker=False
        )


@dataclass
class HandAnalysis:
    """ハンド分析結果"""
    pairs: List[int]  # ペアのランク
    trips: List[int]  # トリップスのランク
    quads: List[int]  # クワッズのランク
    suit_counts: Dict[int, int]  # スート別枚数
    rank_counts: Dict[int, int]  # ランク別枚数
    flush_suit: Optional[int]  # フラッシュ候補のスート
    flush_count: int  # フラッシュ候補の枚数
    straight_potential: float  # ストレート可能性
    high_cards: List[int]  # ハイカード (Q, K, A)
    joker_count: int  # ジョーカー枚数


@dataclass
class BoardState:
    """ボード状態"""
    top_cards: List[CardInfo]
    mid_cards: List[CardInfo]
    bot_cards: List[CardInfo]
    top_count: int
    mid_count: int
    bot_count: int


class TeacherHeuristicAgent:
    """
    ヒューリスティック教師エージェント

    人間のトッププレイヤーの戦略を実装:
    1. FL至上主義: QQ+をTopに置いてFantasyLand突入を狙う
    2. Bottom/Middle構成: フラッシュ優先、安定したハンド構成
    3. ライブカード計算: 残りカード確率に基づく判断
    """

    def __init__(
        self,
        strategy: str = 'balanced',
        fl_aggression: float = 0.7,
        foul_penalty: float = 50.0,
        verbose: bool = False
    ):
        """
        Args:
            strategy: 'aggressive' | 'balanced' | 'conservative'
            fl_aggression: FL狙いの積極性 (0-1)
            foul_penalty: ファウル時のペナルティスコア
            verbose: デバッグ出力
        """
        self.strategy = strategy
        self.fl_aggression = fl_aggression
        self.foul_penalty = foul_penalty
        self.verbose = verbose

        # 戦略別パラメータ
        self.strategy_params = {
            'aggressive': {
                'fl_threshold': 0.3,  # FL狙いの最低確率閾値
                'flush_min_cards': 3,
                'split_pair_threshold': Rank.TEN,  # ペアをスプリットする最低ランク
                'bottom_flush_priority': 1.5,  # フラッシュドロー優先度
            },
            'balanced': {
                'fl_threshold': 0.4,
                'flush_min_cards': 3,
                'split_pair_threshold': Rank.JACK,
                'bottom_flush_priority': 1.3,
            },
            'conservative': {
                'fl_threshold': 0.6,
                'flush_min_cards': 4,
                'split_pair_threshold': Rank.QUEEN,
                'bottom_flush_priority': 1.2,
            },
        }

        # 初手定石パターン
        self.opening_patterns = {
            'flush_draw_3': 15.0,      # 同スート3枚 → Bottomにフラッシュドロー
            'flush_draw_4': 25.0,      # 同スート4枚 → 確定的にBottom
            'high_pair_split': 10.0,   # QQ+のペア → スプリット検討
            'trips': 20.0,             # トリップス → Middleに
            'two_pair': 12.0,          # ツーペア → Bottomにフルハウス狙い
            'straight_draw': 8.0,      # ストレートドロー
        }

        # 統計
        self.stats = {
            'total_actions': 0,
            'fl_attempts': 0,
            'fouls': 0,
            'avg_score': 0,
        }

    def select_action(
        self,
        engine: Any,
        player_idx: int,
        deterministic: bool = True
    ) -> int:
        """
        最適なアクションを選択

        Args:
            engine: C++ GameEngine
            player_idx: プレイヤーインデックス
            deterministic: 決定的選択 (Falseで確率的)

        Returns:
            アクションID (0-242 or 0-26)
        """
        self.stats['total_actions'] += 1

        ps = engine.player(player_idx)
        hand = [CardInfo.from_engine_card(c) for c in ps.get_hand()]
        board = self._get_board_state(ps.board)
        phase = engine.phase()
        street = engine.current_turn()

        # 使用済みカード取得
        used_cards = self._get_used_cards(engine, player_idx)

        # 有効アクション取得
        valid_actions = self._get_valid_actions(engine, player_idx)

        if len(valid_actions) == 0:
            return 0
        if len(valid_actions) == 1:
            return valid_actions[0]

        # 各アクションを評価
        action_scores = {}
        for action in valid_actions:
            score = self._evaluate_action(
                action, hand, board, street, used_cards, engine, player_idx
            )
            action_scores[action] = score

        if self.verbose:
            top_actions = sorted(action_scores.items(), key=lambda x: -x[1])[:5]
            print(f"[Teacher] Street {street}, Top actions: {top_actions}")

        if deterministic:
            return max(action_scores.keys(), key=lambda a: action_scores[a])
        else:
            # Softmax確率選択
            scores = np.array([action_scores[a] for a in valid_actions])
            scores = scores - np.max(scores)  # オーバーフロー防止
            probs = np.exp(scores) / np.sum(np.exp(scores))
            return np.random.choice(valid_actions, p=probs)

    def _evaluate_action(
        self,
        action: int,
        hand: List[CardInfo],
        board: BoardState,
        street: int,
        used_cards: Set[int],
        engine: Any,
        player_idx: int
    ) -> float:
        """
        アクションを評価してスコアを返す

        評価基準:
        1. ファウル回避 (最優先)
        2. FL突入ポテンシャル
        3. ロイヤリティ期待値
        4. ハンド強度バランス
        """
        score = 0.0
        params = self.strategy_params[self.strategy]

        if street == 1:
            # 初期配置 - 定石パターンを適用
            placements = self._decode_initial_action(action)
            sim_board = self._simulate_initial_placement(hand, placements, board)
        else:
            # ターン配置
            row1, row2, discard_idx = self._decode_turn_action(action)
            sim_board = self._simulate_turn_placement(hand, row1, row2, discard_idx, board)

        # 1. ファウルチェック (致命的ペナルティ) - 最優先
        foul_risk = self._estimate_foul_risk_strict(sim_board, street, used_cards)
        if foul_risk >= 1.0:
            return -self.foul_penalty * 2
        if foul_risk > 0.6:
            return -self.foul_penalty
        if foul_risk > 0.4:
            return -self.foul_penalty * 0.5
        score -= foul_risk * 80.0  # ファウルペナルティ大幅強化

        # 2. ハンドバランス (Bottom >= Middle >= Top) - 最重要
        balance_score = self._evaluate_balance_strict(sim_board, street)
        if balance_score < -5:
            return -40.0  # バランス崩壊はファウル直結
        score += balance_score * 5.0  # バランス重視さらに強化

        # 3. 初手定石 or ディスカード評価 (ファウルチェック後に適用)
        if street == 1:
            opening_bonus = self._evaluate_opening_pattern(hand, placements, used_cards)
            # ファウルリスクが非常に低い場合のみボーナス適用
            if foul_risk < 0.15:
                score += opening_bonus * 0.5
        else:
            # ディスカード評価は常に適用（ファウル防止に寄与）
            discard_penalty = self._evaluate_discard(hand, discard_idx, sim_board, used_cards)
            score += discard_penalty * 0.3

        # 4. Bottom評価 (フラッシュ/フルハウス狙い)
        bottom_score = self._evaluate_bottom(sim_board, used_cards)
        score += bottom_score * 1.5

        # 4. Middle評価 (Bottomとのバランス)
        middle_score = self._evaluate_middle(sim_board, used_cards)
        score += middle_score * 1.2

        # 5. Top評価 (FL狙い or 安定) - 控えめに
        top_score = self._evaluate_top_conservative(sim_board, used_cards, street)
        score += top_score * 0.8

        # 6. FL突入ポテンシャル (後半ストリートのみ積極的に)
        if street >= 3 and not self._is_in_fantasy_land(engine, player_idx):
            fl_potential = self._estimate_fl_potential(sim_board, street, used_cards)
            if fl_potential > params['fl_threshold']:
                score += fl_potential * 20.0 * self.fl_aggression

        # 7. ライブカード評価
        live_card_score = self._evaluate_live_cards(sim_board, used_cards, street)
        score += live_card_score * 0.5

        return score

    def _estimate_foul_risk(
        self,
        board: BoardState,
        street: int,
        used_cards: Set[int]
    ) -> float:
        """ファウルリスクを推定 (0-1)"""
        # Top > Middle または Middle > Bottom でファウル

        # 現時点での強度を大まかに評価
        top_strength = self._estimate_row_strength(board.top_cards, 3, used_cards)
        mid_strength = self._estimate_row_strength(board.mid_cards, 5, used_cards)
        bot_strength = self._estimate_row_strength(board.bot_cards, 5, used_cards)

        # 残りストリートが少ないほどリスク判定を厳しく
        remaining_streets = 5 - street
        margin = 0.1 * remaining_streets

        risk = 0.0
        if top_strength > mid_strength + margin:
            risk += 0.5
        if mid_strength > bot_strength + margin:
            risk += 0.5

        return min(1.0, risk)

    def _estimate_foul_risk_strict(
        self,
        board: BoardState,
        street: int,
        used_cards: Set[int]
    ) -> float:
        """厳密なファウルリスク推定 (0-1) - シンプル版"""
        top_strength = self._calculate_hand_rank(board.top_cards, 3)
        mid_strength = self._calculate_hand_rank(board.mid_cards, 5)
        bot_strength = self._calculate_hand_rank(board.bot_cards, 5)

        remaining_streets = 5 - street
        risk = 0.0

        # 最終ストリート
        if remaining_streets == 0:
            if top_strength > mid_strength:
                return 1.0
            if mid_strength > bot_strength:
                return 1.0
            return 0.0

        # 安全マージン (残りストリートが多いほど余裕)
        margin = 10 * remaining_streets

        # Top vs Mid - Topが強すぎないか
        if top_strength > mid_strength + margin:
            risk += 0.8
        elif top_strength > mid_strength + margin // 2:
            risk += 0.5
        elif top_strength > mid_strength:
            risk += 0.3

        # Mid vs Bot - Midが強すぎないか
        if mid_strength > bot_strength + margin:
            risk += 0.8
        elif mid_strength > bot_strength + margin // 2:
            risk += 0.5
        elif mid_strength > bot_strength:
            risk += 0.3

        # 序盤で TopにペアがあるとMid/Botが追いつけない危険
        if street <= 2 and board.top_cards:
            top_analysis = self._analyze_cards(board.top_cards)
            if len(top_analysis.pairs) > 0:
                # ペアランクが高いほど危険
                pair_rank = max(top_analysis.pairs)
                if pair_rank >= Rank.TEN:
                    risk += 0.4
                elif pair_rank >= Rank.SIX:
                    risk += 0.2

        return min(1.0, risk)

    def _estimate_max_potential(
        self,
        cards: List[CardInfo],
        max_cards: int,
        used_cards: Set[int]
    ) -> int:
        """列の最大到達可能強度を推定"""
        if not cards:
            return 70  # 空なら何でも可能

        current_count = len(cards)
        remaining_slots = max_cards - current_count
        if remaining_slots <= 0:
            return self._calculate_hand_rank(cards, max_cards)

        analysis = self._analyze_cards(cards)
        potential = self._calculate_hand_rank(cards, max_cards)

        # フラッシュポテンシャル
        if analysis.flush_count >= 2 and analysis.flush_suit is not None:
            remaining_flush = self._count_remaining_suit(analysis.flush_suit, used_cards)
            needed = 5 - analysis.flush_count
            if remaining_flush >= needed and remaining_slots >= needed:
                potential = max(potential, 50)

        # ペア→トリップス/フルハウスポテンシャル
        if len(analysis.pairs) >= 1:
            potential = max(potential, 30)  # トリップス可能
            if remaining_slots >= 2:
                potential = max(potential, 60)  # フルハウス可能

        # ハイカードからのペア可能性
        if remaining_slots >= 1:
            potential = max(potential, 15)  # 最低でもペア可能

        return potential

    def _calculate_hand_rank(self, cards: List[CardInfo], max_cards: int) -> int:
        """
        ハンドランクを計算 (高いほど強い)
        0: ハイカード, 10: ペア, 20: ツーペア, 30: トリップス,
        40: ストレート, 50: フラッシュ, 60: フルハウス, 70: クワッズ, 80: ストフラ
        """
        if not cards:
            return 0

        analysis = self._analyze_cards(cards)
        base_rank = 0

        # 役の強度
        if len(analysis.quads) > 0:
            base_rank = 70 + analysis.quads[0]
        elif len(analysis.trips) > 0 and len(analysis.pairs) > 0:
            # フルハウス候補
            base_rank = 60 + analysis.trips[0]
        elif analysis.flush_count >= 5:
            base_rank = 50 + max(c.rank for c in cards if not c.is_joker)
        elif len(analysis.trips) > 0:
            base_rank = 30 + analysis.trips[0]
        elif len(analysis.pairs) >= 2:
            base_rank = 20 + max(analysis.pairs)
        elif len(analysis.pairs) == 1:
            base_rank = 10 + analysis.pairs[0]
        else:
            # ハイカード
            non_joker = [c.rank for c in cards if not c.is_joker]
            base_rank = max(non_joker) if non_joker else 0

        return base_rank

    def _estimate_fl_potential(
        self,
        board: BoardState,
        street: int,
        used_cards: Set[int]
    ) -> float:
        """FL突入ポテンシャルを推定 (0-1)"""
        # TopにQQ+が置けるか

        # 現在のTop分析
        top_analysis = self._analyze_cards(board.top_cards)

        # すでにQQ+がある
        for rank in top_analysis.pairs:
            if rank >= Rank.QUEEN:
                return 0.9

        # Topにハイカードがあり、ペアになる可能性
        high_card_count = len([c for c in board.top_cards if c.rank >= Rank.QUEEN])

        if high_card_count >= 1:
            # 残りカードから同ランクを引く確率
            for card in board.top_cards:
                if card.rank >= Rank.QUEEN:
                    remaining = self._count_remaining(card.rank, used_cards)
                    remaining_cards = 54 - len(used_cards)
                    remaining_streets = 5 - street

                    # 大まかな確率計算
                    prob = 1 - ((remaining_cards - remaining) / remaining_cards) ** (remaining_streets * 2)
                    return min(0.8, prob)

        # Topが空なら、今後の可能性
        if board.top_count == 0:
            return 0.3

        return 0.1

    def _evaluate_bottom(
        self,
        board: BoardState,
        used_cards: Set[int]
    ) -> float:
        """Bottom評価 (フラッシュ/フルハウス優先)"""
        score = 0.0
        analysis = self._analyze_cards(board.bot_cards)

        # フラッシュドロー - 最も重要
        if analysis.flush_count >= 3:
            remaining_flush = self._count_remaining_suit(analysis.flush_suit, used_cards)
            score += 10.0 + (analysis.flush_count - 3) * 6.0
            if remaining_flush >= (5 - analysis.flush_count):
                score += 8.0
            # フラッシュ完成
            if analysis.flush_count >= 5:
                score += 15.0

        # フルハウスドロー
        if len(analysis.trips) >= 1:
            score += 12.0
            # フルハウス可能性
            if len(analysis.pairs) >= 1:
                score += 8.0  # フルハウス確定に近い
        elif len(analysis.pairs) >= 2:
            score += 8.0
        elif len(analysis.pairs) == 1:
            score += 4.0
            # ペアのランクでボーナス
            score += analysis.pairs[0] * 0.3

        # ストレートドロー
        if analysis.straight_potential > 0.5:
            score += 5.0

        # ハイカードでも最低限のスコア
        if board.bot_cards:
            high = max((c.rank for c in board.bot_cards if not c.is_joker), default=0)
            score += high * 0.2

        return score

    def _evaluate_middle(
        self,
        board: BoardState,
        used_cards: Set[int]
    ) -> float:
        """Middle評価 (Bottomとのバランス)"""
        score = 0.0
        analysis = self._analyze_cards(board.mid_cards)

        # Middle用のボーナス (Bottomよりやや控えめなハンドでOK)
        if analysis.flush_count >= 3:
            score += 6.0

        if len(analysis.trips) >= 1:
            score += 8.0
        elif len(analysis.pairs) >= 1:
            score += 4.0

        return score

    def _evaluate_top(
        self,
        board: BoardState,
        used_cards: Set[int]
    ) -> float:
        """Top評価 (FL狙い or 安定)"""
        score = 0.0
        analysis = self._analyze_cards(board.top_cards)

        # FL条件: QQ+
        for rank in analysis.pairs:
            if rank >= Rank.QUEEN:
                score += 15.0 + (rank - Rank.QUEEN) * 2.0
            elif rank >= Rank.SIX:
                # ロイヤリティあり
                score += 3.0 + (rank - Rank.SIX) * 0.5

        # トリップス (大ボーナス + FL継続条件)
        if len(analysis.trips) >= 1:
            score += 20.0

        # ハイカード単体
        for card in board.top_cards:
            if card.rank >= Rank.QUEEN and card.rank not in analysis.pairs:
                score += 2.0  # ペアになる可能性

        return score

    def _evaluate_balance(self, board: BoardState) -> float:
        """ハンドバランス評価 (Bottom >= Middle >= Top)"""
        # 各列の強度を概算
        top_strength = len(board.top_cards) * 0.5
        mid_strength = len(board.mid_cards) * 0.5
        bot_strength = len(board.bot_cards) * 0.5

        # ペア等でボーナス
        for cards, strength_ref in [
            (board.top_cards, 'top'),
            (board.mid_cards, 'mid'),
            (board.bot_cards, 'bot')
        ]:
            analysis = self._analyze_cards(cards)
            bonus = len(analysis.pairs) * 2 + len(analysis.trips) * 5
            if strength_ref == 'top':
                top_strength += bonus
            elif strength_ref == 'mid':
                mid_strength += bonus
            else:
                bot_strength += bonus

        score = 0.0
        if bot_strength >= mid_strength:
            score += 5.0
        if mid_strength >= top_strength:
            score += 5.0

        return score

    def _evaluate_balance_strict(self, board: BoardState, street: int) -> float:
        """厳密なハンドバランス評価"""
        top_rank = self._calculate_hand_rank(board.top_cards, 3)
        mid_rank = self._calculate_hand_rank(board.mid_cards, 5)
        bot_rank = self._calculate_hand_rank(board.bot_cards, 5)

        score = 0.0
        remaining = 5 - street

        # Bottom >= Middle (必須条件)
        if bot_rank >= mid_rank:
            score += 15.0
            if bot_rank >= mid_rank + 15:
                score += 10.0  # 大きなマージンは安全
        else:
            # 逆転している - 厳しいペナルティ
            gap = mid_rank - bot_rank
            penalty_mult = 2.0 + (5 - remaining)  # 後半ほど厳しく
            score -= gap * penalty_mult

        # Middle >= Top (必須条件)
        if mid_rank >= top_rank:
            score += 15.0
            if mid_rank >= top_rank + 15:
                score += 10.0
        else:
            gap = top_rank - mid_rank
            penalty_mult = 2.5 + (5 - remaining)  # Top > Mid はより厳しく
            score -= gap * penalty_mult

        # 序盤でTopを空に保つボーナス
        if street <= 2 and len(board.top_cards) == 0:
            score += 5.0

        return score

    def _evaluate_top_conservative(
        self,
        board: BoardState,
        used_cards: Set[int],
        street: int
    ) -> float:
        """Top評価 (控えめ - ファウル回避優先)"""
        score = 0.0
        analysis = self._analyze_cards(board.top_cards)

        # 序盤はTopを弱く保つ
        if street <= 2:
            # ペアがあってもまだボーナスは少なめ
            if len(analysis.pairs) > 0:
                score += 3.0
            # ハイカード単体はほぼボーナスなし
            return score

        # 中盤以降
        for rank in analysis.pairs:
            if rank >= Rank.QUEEN:
                # FL条件達成
                score += 12.0 + (rank - Rank.QUEEN) * 2.0
            elif rank >= Rank.SIX:
                score += 2.0 + (rank - Rank.SIX) * 0.3

        # トリップス
        if len(analysis.trips) >= 1:
            score += 15.0

        return score

    def _evaluate_live_cards(
        self,
        board: BoardState,
        used_cards: Set[int],
        street: int
    ) -> float:
        """ライブカード評価 (アウツ確率)"""
        score = 0.0
        remaining_cards = 54 - len(used_cards)
        remaining_streets = 5 - street

        # 各ドローのアウツを評価
        for cards in [board.bot_cards, board.mid_cards]:
            analysis = self._analyze_cards(cards)

            # フラッシュドロー
            if analysis.flush_suit is not None:
                outs = self._count_remaining_suit(analysis.flush_suit, used_cards)
                if outs >= (5 - analysis.flush_count):
                    score += outs * 0.5

            # ペアアップ
            for card in cards:
                if not card.is_joker:
                    remaining = self._count_remaining(card.rank, used_cards)
                    score += remaining * 0.2

        return score

    def _evaluate_opening_pattern(
        self,
        hand: List[CardInfo],
        placements: List[int],
        used_cards: Set[int]
    ) -> float:
        """
        初手5枚の定石パターン評価

        定石:
        1. 同スート3枚以上 → Bottomにフラッシュドロー
        2. QQ+のハイペア → スプリットして後のFL狙い
        3. トリップス → Middle/Bottomに確保
        4. ツーペア → Bottomでフルハウス狙い
        5. コネクタ → ストレートドロー考慮
        """
        score = 0.0
        analysis = self._analyze_cards(hand)
        params = self.strategy_params[self.strategy]

        # 配置先別のカード
        top_placed = [hand[i] for i, p in enumerate(placements) if p == 0 and i < len(hand)]
        mid_placed = [hand[i] for i, p in enumerate(placements) if p == 1 and i < len(hand)]
        bot_placed = [hand[i] for i, p in enumerate(placements) if p == 2 and i < len(hand)]

        # パターン1: フラッシュドロー (同スート3枚以上)
        if analysis.flush_count >= 3:
            flush_suit = analysis.flush_suit
            flush_cards = [c for c in hand if c.suit == flush_suit and not c.is_joker]

            # フラッシュカードがBottomに置かれているか
            bot_flush = len([c for c in bot_placed if c.suit == flush_suit and not c.is_joker])
            if bot_flush >= 3:
                score += self.opening_patterns['flush_draw_3'] * params['bottom_flush_priority']
                if analysis.flush_count >= 4:
                    score += self.opening_patterns['flush_draw_4']
            elif bot_flush >= 2:
                score += self.opening_patterns['flush_draw_3'] * 0.5

        # パターン2: ハイペア (QQ+) のスプリット
        high_pairs = [r for r in analysis.pairs if r >= Rank.QUEEN]
        if high_pairs:
            for pair_rank in high_pairs:
                pair_cards = [c for c in hand if c.rank == pair_rank]
                # ペアがスプリットされているか (TopとBottom/Middle)
                top_has = any(c in top_placed for c in pair_cards)
                bot_mid_has = any(c in bot_placed or c in mid_placed for c in pair_cards)

                if top_has and bot_mid_has:
                    # スプリット: 後でTopにペアを作る余地
                    score += self.opening_patterns['high_pair_split']
                elif len([c for c in top_placed if c.rank == pair_rank]) == 2:
                    # 初手からTopにQQ+ → FL確定狙い (リスキーだが高リターン)
                    if pair_rank >= Rank.QUEEN:
                        score += 8.0

        # パターン3: トリップス
        if analysis.trips:
            for trip_rank in analysis.trips:
                trip_cards = [c for c in hand if c.rank == trip_rank]
                # Middleに置かれていれば高評価
                mid_trips = len([c for c in mid_placed if c.rank == trip_rank])
                if mid_trips >= 3:
                    score += self.opening_patterns['trips']
                # Bottomでも良い
                bot_trips = len([c for c in bot_placed if c.rank == trip_rank])
                if bot_trips >= 3:
                    score += self.opening_patterns['trips'] * 0.8

        # パターン4: ツーペア
        if len(analysis.pairs) >= 2:
            # Bottomに両ペアがあればフルハウス狙い
            bot_pairs = 0
            for pair_rank in analysis.pairs:
                if len([c for c in bot_placed if c.rank == pair_rank]) >= 2:
                    bot_pairs += 1
            if bot_pairs >= 2:
                score += self.opening_patterns['two_pair']

        # パターン5: Bottomに強いカードを集中させる
        if bot_placed:
            bot_high = max((c.rank for c in bot_placed if not c.is_joker), default=0)
            score += bot_high * 0.3

        # アンチパターン: Topに強いカードを置きすぎ
        if top_placed:
            top_analysis = self._analyze_cards(top_placed)
            if len(top_analysis.pairs) > 0 and top_analysis.pairs[0] < Rank.SIX:
                # 弱いペアをTopに → 将来的にファウルリスク
                score -= 5.0

        return score

    def _evaluate_discard(
        self,
        hand: List[CardInfo],
        discard_idx: int,
        sim_board: BoardState,
        used_cards: Set[int]
    ) -> float:
        """
        ディスカード選択の評価

        良いディスカード:
        - ドローに不要なカード
        - 孤立したカード (ペア候補なし)
        - 相手のアウツをブロックするカード (副次的)

        悪いディスカード:
        - フラッシュドローの一部
        - ペア候補のカード
        - ストレートドローの一部
        """
        if discard_idx >= len(hand):
            return 0.0

        discarded = hand[discard_idx]
        remaining = [c for i, c in enumerate(hand) if i != discard_idx]
        score = 0.0

        # ボード全体の分析
        all_board_cards = sim_board.top_cards + sim_board.mid_cards + sim_board.bot_cards
        board_analysis = self._analyze_cards(all_board_cards)

        # 1. フラッシュドローを捨てていないか
        if board_analysis.flush_suit is not None:
            if discarded.suit == board_analysis.flush_suit and not discarded.is_joker:
                # フラッシュドローの一部を捨てた → ペナルティ
                if board_analysis.flush_count >= 3:
                    score -= 8.0
                elif board_analysis.flush_count >= 2:
                    score -= 4.0

        # 2. ペア候補を捨てていないか
        remaining_ranks = [c.rank for c in remaining if not c.is_joker]
        board_ranks = [c.rank for c in all_board_cards if not c.is_joker]

        if not discarded.is_joker:
            # ボードに同ランクがある → ペア候補
            if discarded.rank in board_ranks:
                score -= 5.0
            # 残りの手札に同ランクがある → ペア候補
            if remaining_ranks.count(discarded.rank) > 0:
                score -= 3.0

        # 3. 孤立カード (良いディスカード)
        if not discarded.is_joker:
            is_isolated = True
            # 同ランクがボードや手札にない
            if discarded.rank in board_ranks or discarded.rank in remaining_ranks:
                is_isolated = False
            # 同スートが3枚以上ない
            suit_count = len([c for c in all_board_cards + remaining if c.suit == discarded.suit])
            if suit_count >= 2:
                is_isolated = False

            if is_isolated:
                score += 3.0

        # 4. ローカード優先 (高いカードは残す)
        if not discarded.is_joker:
            if discarded.rank <= Rank.SIX:
                score += 2.0  # ローカードを捨てるのは良い
            elif discarded.rank >= Rank.JACK:
                score -= 2.0  # ハイカードを捨てるのは避けたい

        # 5. Jokerを捨てるのは最悪
        if discarded.is_joker:
            score -= 15.0

        return score

    def _analyze_cards(self, cards: List[CardInfo]) -> HandAnalysis:
        """カードリストを分析"""
        rank_counts = defaultdict(int)
        suit_counts = defaultdict(int)
        joker_count = 0

        for card in cards:
            if card.is_joker:
                joker_count += 1
            else:
                rank_counts[card.rank] += 1
                suit_counts[card.suit] += 1

        pairs = [r for r, c in rank_counts.items() if c >= 2]
        trips = [r for r, c in rank_counts.items() if c >= 3]
        quads = [r for r, c in rank_counts.items() if c >= 4]

        # フラッシュ候補
        flush_suit = None
        flush_count = 0
        for suit, count in suit_counts.items():
            if count > flush_count:
                flush_suit = suit
                flush_count = count

        # ハイカード
        high_cards = [c.rank for c in cards if c.rank >= Rank.QUEEN and not c.is_joker]

        # ストレートポテンシャル (簡易版)
        ranks = sorted(set(c.rank for c in cards if not c.is_joker))
        straight_potential = 0.0
        if len(ranks) >= 3:
            for i in range(len(ranks) - 2):
                if ranks[i+2] - ranks[i] <= 4:
                    straight_potential = 0.5
                    break

        return HandAnalysis(
            pairs=pairs,
            trips=trips,
            quads=quads,
            suit_counts=dict(suit_counts),
            rank_counts=dict(rank_counts),
            flush_suit=flush_suit,
            flush_count=flush_count + joker_count,  # ジョーカーはワイルド
            straight_potential=straight_potential,
            high_cards=high_cards,
            joker_count=joker_count,
        )

    def _estimate_row_strength(
        self,
        cards: List[CardInfo],
        max_cards: int,
        used_cards: Set[int]
    ) -> float:
        """列の強度を推定 (0-100)"""
        if not cards:
            return 0.0

        analysis = self._analyze_cards(cards)
        strength = 0.0

        # 役の強度
        if len(analysis.quads) > 0:
            strength = 80.0
        elif len(analysis.trips) > 0:
            strength = 50.0
        elif len(analysis.pairs) >= 2:
            strength = 30.0
        elif len(analysis.pairs) == 1:
            strength = 20.0 + analysis.pairs[0]
        else:
            # ハイカード
            if cards:
                strength = max(c.rank for c in cards if not c.is_joker) if any(not c.is_joker for c in cards) else 0

        # フラッシュ/ストレートポテンシャル
        if analysis.flush_count >= 4:
            strength += 20.0
        if analysis.straight_potential > 0.5:
            strength += 10.0

        return strength

    def _count_remaining(self, rank: int, used_cards: Set[int]) -> int:
        """特定ランクの残り枚数"""
        count = 0
        for suit in range(4):
            card_idx = suit * 13 + rank
            if card_idx not in used_cards:
                count += 1
        return count

    def _count_remaining_suit(self, suit: int, used_cards: Set[int]) -> int:
        """特定スートの残り枚数"""
        count = 0
        for rank in range(13):
            card_idx = suit * 13 + rank
            if card_idx not in used_cards:
                count += 1
        return count

    def _get_board_state(self, board) -> BoardState:
        """C++ Boardから状態を取得"""
        top_cards = []
        mid_cards = []
        bot_cards = []

        top_mask = board.top_mask()
        mid_mask = board.mid_mask()
        bot_mask = board.bot_mask()

        for i in range(54):
            if (top_mask >> i) & 1:
                top_cards.append(CardInfo(
                    index=i,
                    rank=i % 13 if i < 52 else Rank.JOKER,
                    suit=i // 13 if i < 52 else -1,
                    is_joker=(i >= 52)
                ))
            if (mid_mask >> i) & 1:
                mid_cards.append(CardInfo(
                    index=i,
                    rank=i % 13 if i < 52 else Rank.JOKER,
                    suit=i // 13 if i < 52 else -1,
                    is_joker=(i >= 52)
                ))
            if (bot_mask >> i) & 1:
                bot_cards.append(CardInfo(
                    index=i,
                    rank=i % 13 if i < 52 else Rank.JOKER,
                    suit=i // 13 if i < 52 else -1,
                    is_joker=(i >= 52)
                ))

        return BoardState(
            top_cards=top_cards,
            mid_cards=mid_cards,
            bot_cards=bot_cards,
            top_count=board.count(ofc.TOP),
            mid_count=board.count(ofc.MIDDLE),
            bot_count=board.count(ofc.BOTTOM),
        )

    def _get_used_cards(self, engine, player_idx: int) -> Set[int]:
        """使用済みカードを取得"""
        used = set()
        num_players = engine.num_players()

        for i in range(num_players):
            ps = engine.player(i)
            mask = ps.board.all_mask()
            for j in range(54):
                if (mask >> j) & 1:
                    used.add(j)
            # 自分の手札
            if i == player_idx:
                for card in ps.get_hand():
                    used.add(card.index)

        return used

    def _get_valid_actions(self, engine, player_idx: int) -> List[int]:
        """有効なアクションを取得"""
        ps = engine.player(player_idx)
        hand = ps.get_hand()
        board = ps.board
        phase = engine.phase()

        valid_actions = []

        if phase == ofc.GamePhase.INITIAL_DEAL:
            for action in range(243):
                if self._is_valid_initial_action(board, hand, action):
                    valid_actions.append(action)
        elif phase == ofc.GamePhase.TURN:
            for action in range(27):
                if self._is_valid_turn_action(board, hand, action):
                    valid_actions.append(action)

        return valid_actions if valid_actions else [0]

    def _is_valid_initial_action(self, board, hand, action: int) -> bool:
        """初期配置アクションの有効性チェック"""
        top_count = board.count(ofc.TOP)
        mid_count = board.count(ofc.MIDDLE)
        bot_count = board.count(ofc.BOTTOM)

        temp = action
        for i in range(5):
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

    def _is_valid_turn_action(self, board, hand, action: int) -> bool:
        """ターンアクションの有効性チェック"""
        if len(hand) < 3:
            return False

        row1 = action % 3
        row2 = (action // 3) % 3

        top_slots = 3 - board.count(ofc.TOP)
        mid_slots = 5 - board.count(ofc.MIDDLE)
        bot_slots = 5 - board.count(ofc.BOTTOM)

        row_counts = [0, 0, 0]
        row_counts[row1] += 1
        row_counts[row2] += 1

        if row_counts[0] > top_slots:
            return False
        if row_counts[1] > mid_slots:
            return False
        if row_counts[2] > bot_slots:
            return False

        return True

    def _decode_initial_action(self, action: int) -> List[int]:
        """初期配置アクションをデコード"""
        placements = []
        temp = action
        for _ in range(5):
            placements.append(temp % 3)
            temp //= 3
        return placements

    def _decode_turn_action(self, action: int) -> Tuple[int, int, int]:
        """ターンアクションをデコード"""
        row1 = action % 3
        row2 = (action // 3) % 3
        discard_idx = (action // 9) % 3
        return row1, row2, discard_idx

    def _simulate_initial_placement(
        self,
        hand: List[CardInfo],
        placements: List[int],
        board: BoardState
    ) -> BoardState:
        """初期配置をシミュレート"""
        new_top = list(board.top_cards)
        new_mid = list(board.mid_cards)
        new_bot = list(board.bot_cards)

        for i, row in enumerate(placements):
            if i < len(hand):
                card = hand[i]
                if row == 0:
                    new_top.append(card)
                elif row == 1:
                    new_mid.append(card)
                else:
                    new_bot.append(card)

        return BoardState(
            top_cards=new_top,
            mid_cards=new_mid,
            bot_cards=new_bot,
            top_count=len(new_top),
            mid_count=len(new_mid),
            bot_count=len(new_bot),
        )

    def _simulate_turn_placement(
        self,
        hand: List[CardInfo],
        row1: int,
        row2: int,
        discard_idx: int,
        board: BoardState
    ) -> BoardState:
        """ターン配置をシミュレート"""
        new_top = list(board.top_cards)
        new_mid = list(board.mid_cards)
        new_bot = list(board.bot_cards)

        play_indices = [i for i in range(min(3, len(hand))) if i != discard_idx][:2]
        rows = [row1, row2]

        for i, play_idx in enumerate(play_indices):
            if play_idx < len(hand):
                card = hand[play_idx]
                row = rows[i]
                if row == 0:
                    new_top.append(card)
                elif row == 1:
                    new_mid.append(card)
                else:
                    new_bot.append(card)

        return BoardState(
            top_cards=new_top,
            mid_cards=new_mid,
            bot_cards=new_bot,
            top_count=len(new_top),
            mid_count=len(new_mid),
            bot_count=len(new_bot),
        )

    def _is_in_fantasy_land(self, engine, player_idx: int) -> bool:
        """FLに入っているかチェック"""
        ps = engine.player(player_idx)
        return ps.in_fantasy_land

    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        return self.stats.copy()


class RandomAgent:
    """ランダムエージェント (ベースライン比較用)"""

    def __init__(self):
        self.stats = {'total_actions': 0}

    def select_action(
        self,
        engine: Any,
        player_idx: int,
        deterministic: bool = False
    ) -> int:
        """ランダムにアクションを選択"""
        self.stats['total_actions'] += 1

        ps = engine.player(player_idx)
        hand = ps.get_hand()
        board = ps.board
        phase = engine.phase()

        valid_actions = []

        if phase == ofc.GamePhase.INITIAL_DEAL:
            for action in range(243):
                if self._is_valid_initial(board, action):
                    valid_actions.append(action)
        elif phase == ofc.GamePhase.TURN:
            for action in range(27):
                if self._is_valid_turn(board, len(hand), action):
                    valid_actions.append(action)

        if not valid_actions:
            return 0

        return np.random.choice(valid_actions)

    def _is_valid_initial(self, board, action: int) -> bool:
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

    def _is_valid_turn(self, board, hand_size: int, action: int) -> bool:
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


def test_teacher_vs_random(num_games: int = 100, verbose: bool = False):
    """Teacher vs Random の対戦テスト"""
    print("=" * 60)
    print("Teacher Heuristic Agent vs Random Agent - Benchmark")
    print("=" * 60)

    if not HAS_ENGINE:
        print("ofc_engine not available. Skipping test.")
        return

    teacher = TeacherHeuristicAgent(strategy='balanced', verbose=verbose)
    random_agent = RandomAgent()

    teacher_scores = []
    random_scores = []
    teacher_fouls = 0
    random_fouls = 0

    for game_idx in range(num_games):
        engine = ofc.GameEngine(2)
        engine.start_new_game(game_idx)

        agents = [teacher, random_agent]

        # ゲームプレイ
        while engine.phase() not in [ofc.GamePhase.SHOWDOWN, ofc.GamePhase.COMPLETE]:
            for player_idx in range(2):
                if engine.phase() in [ofc.GamePhase.SHOWDOWN, ofc.GamePhase.COMPLETE]:
                    break

                action = agents[player_idx].select_action(engine, player_idx)

                ps = engine.player(player_idx)
                hand = ps.get_hand()
                phase = engine.phase()

                if phase == ofc.GamePhase.INITIAL_DEAL:
                    initial_action = ofc.InitialAction()
                    placements = []
                    temp = action
                    for i in range(5):
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

        # 結果取得
        result = engine.result()
        t_score = result.get_score(0)
        r_score = result.get_score(1)

        teacher_scores.append(t_score)
        random_scores.append(r_score)

        if result.is_fouled(0):
            teacher_fouls += 1
        if result.is_fouled(1):
            random_fouls += 1

        if verbose and (game_idx + 1) % 10 == 0:
            print(f"Game {game_idx + 1}: Teacher={t_score:.1f}, Random={r_score:.1f}")

    print(f"\n{'='*60}")
    print(f"Results after {num_games} games:")
    print(f"{'='*60}")
    print(f"Teacher Avg Score: {np.mean(teacher_scores):.2f} (+/- {np.std(teacher_scores):.2f})")
    print(f"Random Avg Score:  {np.mean(random_scores):.2f} (+/- {np.std(random_scores):.2f})")
    print(f"Teacher Foul Rate: {teacher_fouls / num_games * 100:.1f}%")
    print(f"Random Foul Rate:  {random_fouls / num_games * 100:.1f}%")
    print(f"Score Difference:  {np.mean(teacher_scores) - np.mean(random_scores):.2f} pts/game")
    print(f"{'='*60}")

    # 目標: +5.0点/ハンド以上
    diff = np.mean(teacher_scores) - np.mean(random_scores)
    if diff >= 5.0:
        print("SUCCESS: Teacher beats Random by +5.0 pts/game or more!")
    else:
        print(f"Target: +5.0 pts/game. Current: {diff:.2f} pts/game")

    return {
        'teacher_avg': np.mean(teacher_scores),
        'random_avg': np.mean(random_scores),
        'teacher_foul_rate': teacher_fouls / num_games,
        'random_foul_rate': random_fouls / num_games,
        'score_diff': diff,
    }


if __name__ == "__main__":
    test_teacher_vs_random(num_games=100, verbose=True)
