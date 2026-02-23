#!/usr/bin/env python3
"""
C++ is_foul vs 精密フォールチェックの比較
C++のkicker情報が不足しているケースを検出する
"""

import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src/python'))

import numpy as np
import ofc_engine as ofc

RANKS = 'A23456789TJQK'
SUITS = 'shdc'
NUM_CARDS = 54

def card_str_to_index(card_str):
    if card_str == 'JK1': return 52
    if card_str == 'JK2': return 53
    rank_idx = RANKS.index(card_str[0])
    suit_idx = SUITS.index(card_str[1])
    return suit_idx * 13 + rank_idx

def index_to_card_str(idx):
    if idx == 52: return 'JK1'
    if idx == 53: return 'JK2'
    suit_idx = idx // 13
    rank_idx = idx % 13
    return RANKS[rank_idx] + SUITS[suit_idx]

def make_full_deck():
    deck = []
    for s in SUITS:
        for r in RANKS:
            deck.append(r + s)
    deck.extend(['JK1', 'JK2'])
    return deck


# ============================================
# 精密フォールチェック（TypeScript相当）
# ============================================

RANK_VALUES = {'A': 14, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
               '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13}

def parse_card(card_str):
    """カード文字列を (rank_value, suit) に変換"""
    if card_str in ('JK1', 'JK2'):
        return (0, 'joker')
    return (RANK_VALUES[card_str[0]], card_str[1])

def evaluate_5card_precise(card_strs):
    """5枚ハンドの精密評価 (Joker対応) → (rank_category, [highCards])"""
    cards = [parse_card(c) for c in card_strs]
    joker_count = sum(1 for r, s in cards if s == 'joker')
    normal = [(r, s) for r, s in cards if s != 'joker']

    if joker_count == 0:
        return _eval_5_no_joker(normal)

    # Jokerがある場合: 全候補を試して最強を返す
    used = set((r, s) for r, s in normal)
    all_cards = [(RANK_VALUES[r], s) for s in 'shdc' for r in RANKS]
    candidates = [c for c in all_cards if c not in used]

    best = (0, [0])
    if joker_count == 1:
        for c1 in candidates:
            hand = normal + [c1]
            result = _eval_5_no_joker(hand)
            if _is_better(result, best):
                best = result
    else:
        for i, c1 in enumerate(candidates):
            for c2 in candidates[i+1:]:
                hand = normal + [c1, c2]
                result = _eval_5_no_joker(hand)
                if _is_better(result, best):
                    best = result
    return best

def _is_better(a, b):
    if a[0] != b[0]: return a[0] > b[0]
    for x, y in zip(a[1], b[1]):
        if x != y: return x > y
    return False

def _eval_5_no_joker(cards):
    """5枚ハンド評価（Jokerなし） → (rank_category, [highCards_sorted_desc])"""
    ranks = sorted([r for r, s in cards], reverse=True)
    suits = [s for r, s in cards]
    is_flush = len(set(suits)) == 1

    # Check straight
    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = 0
    if len(unique_ranks) >= 5:
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i] - unique_ranks[i+4] == 4:
                is_straight = True
                straight_high = unique_ranks[i]
                break
        # Wheel: A-2-3-4-5
        if not is_straight and 14 in unique_ranks and set([2,3,4,5]).issubset(set(unique_ranks)):
            is_straight = True
            straight_high = 5

    from collections import Counter
    rank_counts = Counter(ranks)
    count_groups = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

    if is_straight and is_flush:
        return (8, [straight_high])  # Straight Flush

    if count_groups[0][1] == 4:
        quad_rank = count_groups[0][0]
        kicker = [r for r in ranks if r != quad_rank][0]
        return (7, [quad_rank, kicker])  # Four of a Kind

    if count_groups[0][1] == 3 and count_groups[1][1] == 2:
        return (6, [count_groups[0][0], count_groups[1][0]])  # Full House

    if is_flush:
        return (5, ranks)  # Flush - FULL kicker info

    if is_straight:
        return (4, [straight_high])  # Straight

    if count_groups[0][1] == 3:
        trip_rank = count_groups[0][0]
        kickers = sorted([r for r in ranks if r != trip_rank], reverse=True)
        return (3, [trip_rank] + kickers)  # Three of a Kind

    if count_groups[0][1] == 2 and count_groups[1][1] == 2:
        pair1 = count_groups[0][0]
        pair2 = count_groups[1][0]
        kicker = [r for r, c in rank_counts.items() if c == 1][0]
        high_pair = max(pair1, pair2)
        low_pair = min(pair1, pair2)
        return (2, [high_pair, low_pair, kicker])  # Two Pair - FULL info

    if count_groups[0][1] == 2:
        pair_rank = count_groups[0][0]
        kickers = sorted([r for r in ranks if r != pair_rank], reverse=True)
        return (1, [pair_rank] + kickers)  # One Pair

    return (0, ranks)  # High Card

def evaluate_3card_precise(card_strs):
    """3枚ハンドの精密評価"""
    cards = [parse_card(c) for c in card_strs]
    joker_count = sum(1 for r, s in cards if s == 'joker')
    normal = [(r, s) for r, s in cards if s != 'joker']

    if joker_count == 0:
        return _eval_3_no_joker(normal)

    used = set((r, s) for r, s in normal)
    all_cards = [(RANK_VALUES[r], s) for s in 'shdc' for r in RANKS]
    candidates = [c for c in all_cards if c not in used]

    best = (0, [0])
    if joker_count == 1:
        for c1 in candidates:
            hand = normal + [c1]
            result = _eval_3_no_joker(hand)
            if _is_better(result, best):
                best = result
    else:
        for i, c1 in enumerate(candidates):
            for c2 in candidates[i+1:]:
                hand = normal + [c1, c2]
                result = _eval_3_no_joker(hand)
                if _is_better(result, best):
                    best = result
    return best

def _eval_3_no_joker(cards):
    """3枚ハンド評価（Jokerなし）"""
    ranks = sorted([r for r, s in cards], reverse=True)
    from collections import Counter
    rank_counts = Counter(ranks)
    count_groups = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

    if count_groups[0][1] == 3:
        return (3, [count_groups[0][0]])  # Three of a Kind

    if count_groups[0][1] == 2:
        pair_rank = count_groups[0][0]
        kicker = [r for r in ranks if r != pair_rank][0]
        return (1, [pair_rank, kicker])  # One Pair

    return (0, ranks)  # High Card

def check_foul_precise(board_strs):
    """精密フォールチェック（TypeScript相当）"""
    top_strs, mid_strs, bot_strs = board_strs

    if len(top_strs) != 3 or len(mid_strs) != 5 or len(bot_strs) != 5:
        return True

    top_eval = evaluate_3card_precise(top_strs)
    mid_eval = evaluate_5card_precise(mid_strs)
    bot_eval = evaluate_5card_precise(bot_strs)

    # bot >= mid
    if _is_better(mid_eval, bot_eval):
        return True  # mid > bot → foul

    # mid >= top (cross-format: 5-card rank vs 3-card rank)
    if mid_eval[0] < top_eval[0]:
        return True
    if mid_eval[0] > top_eval[0]:
        return False
    # Same rank: compare highCards
    for x, y in zip(mid_eval[1], top_eval[1]):
        if x > y: return False
        if x < y: return True

    return False  # Equal is OK


def check_foul_cpp(board_indices):
    """C++ is_foul"""
    b = ofc.Board()
    rows = [ofc.TOP, ofc.MIDDLE, ofc.BOTTOM]
    for row_idx, row_enum in enumerate(rows):
        for card_idx in board_indices[row_idx]:
            b.place_card(row_enum, ofc.Card(card_idx))
    return b.is_foul()


def main():
    np.random.seed(42)
    num_trials = 5000

    agree_not_foul = 0
    agree_foul = 0
    cpp_no_ts_yes = 0  # C++ says OK but precise says foul
    cpp_yes_ts_no = 0  # C++ says foul but precise says OK

    mismatch_examples = []

    for trial in range(num_trials):
        # Generate random complete board
        deck = make_full_deck()
        np.random.shuffle(deck)

        # Split: 3 top, 5 mid, 5 bot
        top_strs = deck[:3]
        mid_strs = deck[3:8]
        bot_strs = deck[8:13]

        top_indices = [card_str_to_index(c) for c in top_strs]
        mid_indices = [card_str_to_index(c) for c in mid_strs]
        bot_indices = [card_str_to_index(c) for c in bot_strs]

        cpp_foul = check_foul_cpp([top_indices, mid_indices, bot_indices])
        precise_foul = check_foul_precise([top_strs, mid_strs, bot_strs])

        if cpp_foul and precise_foul:
            agree_foul += 1
        elif not cpp_foul and not precise_foul:
            agree_not_foul += 1
        elif not cpp_foul and precise_foul:
            cpp_no_ts_yes += 1
            if len(mismatch_examples) < 10:
                # Get hand info
                top_eval = evaluate_3card_precise(top_strs)
                mid_eval = evaluate_5card_precise(mid_strs)
                bot_eval = evaluate_5card_precise(bot_strs)

                b = ofc.Board()
                for ci in top_indices: b.place_card(ofc.TOP, ofc.Card(ci))
                for ci in mid_indices: b.place_card(ofc.MIDDLE, ofc.Card(ci))
                for ci in bot_indices: b.place_card(ofc.BOTTOM, ofc.Card(ci))
                cpp_top = b.evaluate_top()
                cpp_mid = b.evaluate_mid()
                cpp_bot = b.evaluate_bot()

                mismatch_examples.append({
                    'top': top_strs, 'mid': mid_strs, 'bot': bot_strs,
                    'precise_top': top_eval, 'precise_mid': mid_eval, 'precise_bot': bot_eval,
                    'cpp_top': f'rank={cpp_top.rank} k={cpp_top.kickers}',
                    'cpp_mid': f'rank={cpp_mid.rank} k={cpp_mid.kickers}',
                    'cpp_bot': f'rank={cpp_bot.rank} k={cpp_bot.kickers}',
                })
        else:
            cpp_yes_ts_no += 1

    total = num_trials
    print(f"=== Foul Check Comparison: C++ vs Precise (TypeScript-equivalent) ===")
    print(f"Total boards: {total}")
    print(f"")
    print(f"Both agree NOT foul:     {agree_not_foul} ({agree_not_foul/total*100:.1f}%)")
    print(f"Both agree FOUL:         {agree_foul} ({agree_foul/total*100:.1f}%)")
    print(f"C++ OK, Precise FOUL:    {cpp_no_ts_yes} ({cpp_no_ts_yes/total*100:.1f}%)  ← MISSED FOULS")
    print(f"C++ FOUL, Precise OK:    {cpp_yes_ts_no} ({cpp_yes_ts_no/total*100:.1f}%)  ← FALSE FOULS")
    print(f"")
    print(f"C++ foul rate:           {(agree_foul + cpp_yes_ts_no)/total*100:.1f}%")
    print(f"Precise foul rate:       {(agree_foul + cpp_no_ts_yes)/total*100:.1f}%")
    print(f"Discrepancy:             {cpp_no_ts_yes/total*100:.1f}% of boards are foul but C++ misses them")

    if mismatch_examples:
        rank_names = ['High Card', 'One Pair', 'Two Pair', 'Three of a Kind', 'Straight', 'Flush', 'Full House', 'Four of a Kind', 'Straight Flush']
        print(f"\n=== Mismatch Examples (C++ says OK, Precise says FOUL) ===")
        for i, ex in enumerate(mismatch_examples):
            print(f"\n--- Example {i+1} ---")
            print(f"  Top: {ex['top']}  → Precise: rank={ex['precise_top'][0]}({rank_names[ex['precise_top'][0]]}) hc={ex['precise_top'][1]}")
            print(f"  Mid: {ex['mid']}  → Precise: rank={ex['precise_mid'][0]}({rank_names[ex['precise_mid'][0]]}) hc={ex['precise_mid'][1]}")
            print(f"  Bot: {ex['bot']}  → Precise: rank={ex['precise_bot'][0]}({rank_names[ex['precise_bot'][0]]}) hc={ex['precise_bot'][1]}")
            print(f"  C++ Top: {ex['cpp_top']}, Mid: {ex['cpp_mid']}, Bot: {ex['cpp_bot']}")

if __name__ == '__main__':
    main()
