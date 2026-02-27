"""
OFC Pineapple AI - Rule-Based Agent
Phase 0: ベンチマーク評価 + Self-Play混合用

2種類の戦略:
  1. SafeAgent: フォール回避最優先、FLは狙わない
  2. AggressiveAgent: FL積極追求、フォールリスク許容

使い方:
    agent = SafeAgent()
    action = agent.select_action(env, player_idx)
"""

import random
from typing import List, Optional, Tuple

try:
    import ofc_engine as ofc
except ImportError:
    raise ImportError("ofc_engine not found. Run 'python setup.py build_ext --inplace' first.")


# カードランク定数（C++エンジンと同じ: ACE=0, 2=1, ..., K=12）
ACE = 0
JACK = 10
QUEEN = 11
KING = 12

# 比較用ランク（ACE=14として扱う）
def cmp_rank(card_index: int) -> int:
    """カードindex(0-53)から比較用ランク(2-14)を返す。Jokerは15。"""
    if card_index >= 52:
        return 15  # Joker
    rank = card_index // 4  # ACE=0, 2=1, ..., K=12
    if rank == ACE:
        return 14  # ACEは最強
    return rank + 1  # 2=2, 3=3, ..., K=13


class BaseRuleAgent:
    """ルールベースエージェントの基底クラス"""

    def select_action(self, env, agent_name: str) -> int:
        """環境から有効アクションを取得し、最良のアクションを選択"""
        valid_actions = env.get_valid_actions(agent_name)
        if len(valid_actions) <= 1:
            return valid_actions[0]

        player_idx = env.agent_name_mapping[agent_name]
        ps = env.engine.player(player_idx)
        hand = ps.get_hand()
        board = ps.board

        if env.current_street == 1:
            return self._select_initial_action(hand, board, valid_actions)
        else:
            return self._select_turn_action(hand, board, valid_actions)

    def _select_initial_action(self, hand, board, valid_actions: List[int]) -> int:
        raise NotImplementedError

    def _select_turn_action(self, hand, board, valid_actions: List[int]) -> int:
        raise NotImplementedError

    def _decode_initial_action(self, action: int) -> List[int]:
        """初回アクション(0-242)を5枚の配置先リストにデコード"""
        placements = []
        temp = action
        for _ in range(5):
            placements.append(temp % 3)  # 0=TOP, 1=MID, 2=BOT
            temp //= 3
        return placements

    def _decode_turn_action(self, action: int) -> Tuple[int, int, int]:
        """ターンアクション(0-26)をデコード → (row1, row2, discard_idx)"""
        row1 = action % 3
        row2 = (action // 3) % 3
        discard_idx = (action // 9) % 3
        return row1, row2, discard_idx

    def _card_rank(self, card) -> int:
        """カードの比較用ランクを返す"""
        return cmp_rank(card.index)

    def _is_fl_card(self, card) -> bool:
        """FL資格カード(Q/K/A/Joker)か"""
        if card.index >= 52:
            return True  # Joker
        rank = card.index // 4
        return rank in (ACE, QUEEN, KING)

    def _count_row(self, board, row: int) -> int:
        """各列の現在のカード枚数"""
        rows = [ofc.TOP, ofc.MIDDLE, ofc.BOTTOM]
        return board.count(rows[row])

    def _row_capacity(self, board, row: int) -> int:
        """各列の残り空き"""
        max_sizes = [3, 5, 5]
        rows = [ofc.TOP, ofc.MIDDLE, ofc.BOTTOM]
        return max_sizes[row] - board.count(rows[row])


class SafeAgent(BaseRuleAgent):
    """
    安全型ルールベースエージェント

    戦略:
    - フォール回避が最優先
    - 強いカードを Bottom > Middle > Top の順に配置
    - FL は狙わない（Q/K/A を Top に意図的に置かない）
    - タイブレーク時はランダム
    """

    def _select_initial_action(self, hand, board, valid_actions: List[int]) -> int:
        best_score = -999
        best_actions = []

        for action in valid_actions:
            placements = self._decode_initial_action(action)
            score = self._score_initial_placement(hand, placements)
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return random.choice(best_actions)

    def _score_initial_placement(self, hand, placements: List[int]) -> float:
        """初回配置のスコアリング（高いほど良い）"""
        score = 0.0

        # 各列に配置されるカードのランクを集める
        rows = {0: [], 1: [], 2: []}  # TOP, MID, BOT
        for i, row in enumerate(placements):
            rows[row].append(self._card_rank(hand[i]))

        # 基本方針: 高いカードは Bot/Mid、低いカードは Top
        for rank in rows[0]:  # TOP
            # Topに高ランクカード(Q/K/A)を置くとフォールリスク → ペナルティ
            if rank >= 12:  # Q以上
                score -= 3.0
            # Topに低いカードを置く → 良い
            score -= rank * 0.5  # 低いランクほどスコアが高い

        for rank in rows[2]:  # BOT
            # Botに高ランクカードを置く → 良い
            score += rank * 0.3

        for rank in rows[1]:  # MID
            score += rank * 0.1

        # ペアボーナス: 同じランクが同じ列にある
        for row_cards in rows.values():
            if len(row_cards) >= 2:
                from collections import Counter
                counts = Counter(row_cards)
                for r, c in counts.items():
                    if c >= 2:
                        score += 2.0  # ペアボーナス
                    if c >= 3:
                        score += 5.0  # トリプスボーナス

        # Bot にペアがあると良い（フォール回避の基盤）
        if len(rows[2]) >= 2:
            from collections import Counter
            bot_counts = Counter(rows[2])
            for r, c in bot_counts.items():
                if c >= 2:
                    score += 3.0

        return score

    def _select_turn_action(self, hand, board, valid_actions: List[int]) -> int:
        best_score = -999
        best_actions = []

        for action in valid_actions:
            row1, row2, discard_idx = self._decode_turn_action(action)
            score = self._score_turn_placement(hand, board, row1, row2, discard_idx)
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return random.choice(best_actions)

    def _score_turn_placement(self, hand, board, row1: int, row2: int, discard_idx: int) -> float:
        """ターン配置のスコアリング"""
        score = 0.0

        play_indices = [i for i in range(3) if i != discard_idx]
        cards = [(hand[play_indices[0]], row1), (hand[play_indices[1]], row2)]
        discarded = hand[discard_idx]

        for card, row in cards:
            rank = self._card_rank(card)

            if row == 0:  # TOP
                # SafeAgent: Topには低ランクのみ
                if rank >= 12:  # Q以上
                    score -= 5.0
                score -= rank * 0.3
            elif row == 2:  # BOT
                score += rank * 0.4
            else:  # MID
                score += rank * 0.2

        # 捨て札: 低いカードを捨てるのが基本
        discard_rank = self._card_rank(discarded)
        # 中間ランクを捨てるのが最も無難
        if 5 <= discard_rank <= 9:
            score += 0.5

        return score


class AggressiveAgent(BaseRuleAgent):
    """
    攻撃型ルールベースエージェント

    戦略:
    - FL (Fantasy Land) を積極的に狙う
    - Q/K/A/Joker を Top に置く
    - Bot/Mid を強くしてフォール回避も試みる
    - フォールリスクが高くても FL が見えるなら攻める
    """

    def _select_initial_action(self, hand, board, valid_actions: List[int]) -> int:
        best_score = -999
        best_actions = []

        for action in valid_actions:
            placements = self._decode_initial_action(action)
            score = self._score_initial_placement(hand, placements)
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return random.choice(best_actions)

    def _score_initial_placement(self, hand, placements: List[int]) -> float:
        score = 0.0
        rows = {0: [], 1: [], 2: []}
        for i, row in enumerate(placements):
            rows[row].append((self._card_rank(hand[i]), hand[i]))

        # FL戦略: TopにQ/K/A/Jokerを置くとボーナス
        for rank, card in rows[0]:
            if self._is_fl_card(card):
                score += 5.0
                # ペアでさらにボーナス
                fl_count = sum(1 for r, c in rows[0] if self._is_fl_card(c))
                if fl_count >= 2:
                    score += 10.0  # FL資格ペア!
            else:
                # FL非対象カードをTopに置くのはやや悪い
                score -= 1.0

        # Botを強くする（フォール回避 + ロイヤリティ）
        for rank, card in rows[2]:
            score += rank * 0.3
            # 高ランクペアをBotに
            from collections import Counter
            bot_ranks = [r for r, c in rows[2]]
            counts = Counter(bot_ranks)
            for r, c in counts.items():
                if c >= 2:
                    score += 3.0
                if c >= 3:
                    score += 8.0

        # Midに中間ランク
        for rank, card in rows[1]:
            score += rank * 0.15

        return score

    def _select_turn_action(self, hand, board, valid_actions: List[int]) -> int:
        best_score = -999
        best_actions = []

        for action in valid_actions:
            row1, row2, discard_idx = self._decode_turn_action(action)
            score = self._score_turn_placement(hand, board, row1, row2, discard_idx)
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return random.choice(best_actions)

    def _score_turn_placement(self, hand, board, row1: int, row2: int, discard_idx: int) -> float:
        score = 0.0

        play_indices = [i for i in range(3) if i != discard_idx]
        cards = [(hand[play_indices[0]], row1), (hand[play_indices[1]], row2)]

        for card, row in cards:
            rank = self._card_rank(card)

            if row == 0:  # TOP
                if self._is_fl_card(card):
                    score += 8.0  # FL追求!
                    # Topに既にFL資格カードがあれば特大ボーナス
                    if board.count(ofc.TOP) >= 1 and board.qualifies_for_fl():
                        score += 15.0
                else:
                    score -= rank * 0.3
            elif row == 2:  # BOT
                score += rank * 0.4
            else:  # MID
                score += rank * 0.2

        # 捨て札: FL無関係かつ低ランクを捨てる
        discarded = hand[discard_idx]
        discard_rank = self._card_rank(discarded)
        if not self._is_fl_card(discarded):
            score += 1.0  # FL無関係カードを捨てるのは良い
        if discard_rank <= 7:
            score += 0.5

        return score


class RandomAgent(BaseRuleAgent):
    """ランダムエージェント（ベースラインの下限）"""

    def _select_initial_action(self, hand, board, valid_actions: List[int]) -> int:
        return random.choice(valid_actions)

    def _select_turn_action(self, hand, board, valid_actions: List[int]) -> int:
        return random.choice(valid_actions)


def get_agent(agent_type: str) -> BaseRuleAgent:
    """エージェントタイプ名からインスタンスを取得"""
    agents = {
        'random': RandomAgent,
        'safe': SafeAgent,
        'aggressive': AggressiveAgent,
    }
    if agent_type not in agents:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(agents.keys())}")
    return agents[agent_type]()


if __name__ == "__main__":
    # 簡易テスト: 各エージェントの1ゲーム
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))
    from ofc_3max_env import OFC3MaxEnv

    env = OFC3MaxEnv(enable_fl_turns=True, continuous_games=False, fl_solver_mode='greedy')
    agents_list = [SafeAgent(), AggressiveAgent(), RandomAgent()]
    agent_names = ['Safe', 'Aggressive', 'Random']

    for trial in range(3):
        env.reset()
        while not all(env.terminations.values()):
            agent_name = env.agent_selection
            if env.terminations.get(agent_name, False):
                env.step(None)
                continue

            pidx = env.agent_name_mapping[agent_name]
            ps = env.engine.player(pidx)
            if ps.board.total_placed() >= 13 or ps.in_fantasy_land:
                env.step(None)
                continue

            agent = agents_list[pidx]
            action = agent.select_action(env, agent_name)
            env.step(action)

        result = env.engine.result()
        scores = [result.get_score(i) for i in range(3)]
        fouls = [result.is_fouled(i) for i in range(3)]
        print(f"Game {trial+1}: ", end="")
        for i, name in enumerate(agent_names):
            f_mark = " [FOUL]" if fouls[i] else ""
            print(f"{name}={scores[i]:+.0f}{f_mark}  ", end="")
        print()
