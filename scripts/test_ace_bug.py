#!/usr/bin/env python3
"""
C++ evaluator の ACE バグを検証
ACE=0 なので kicker比較で最弱扱いされている
"""

import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src/python'))

import ofc_engine as ofc

# Card index: suit * 13 + rank
# suit: s=0, h=1, d=2, c=3
# rank: A=0, 2=1, 3=2, ..., K=12

def make_card(rank_char, suit_char):
    ranks = 'A23456789TJQK'
    suits = 'shdc'
    return ofc.Card(suits.index(suit_char) * 13 + ranks.index(rank_char))

def test_pair_comparison():
    """AA vs KK: AA should be stronger"""
    print("=== Pair Comparison ===")

    # AA board
    b1 = ofc.Board()
    b1.place_card(ofc.TOP, make_card('A', 's'))
    b1.place_card(ofc.TOP, make_card('A', 'h'))
    b1.place_card(ofc.TOP, make_card('2', 's'))
    top_aa = b1.evaluate_top()
    print(f"AA2 top: rank={top_aa.rank} kickers={top_aa.kickers}")

    # KK board
    b2 = ofc.Board()
    b2.place_card(ofc.TOP, make_card('K', 's'))
    b2.place_card(ofc.TOP, make_card('K', 'h'))
    b2.place_card(ofc.TOP, make_card('2', 's'))
    top_kk = b2.evaluate_top()
    print(f"KK2 top: rank={top_kk.rank} kickers={top_kk.kickers}")

    # QQ board
    b3 = ofc.Board()
    b3.place_card(ofc.TOP, make_card('Q', 's'))
    b3.place_card(ofc.TOP, make_card('Q', 'h'))
    b3.place_card(ofc.TOP, make_card('2', 's'))
    top_qq = b3.evaluate_top()
    print(f"QQ2 top: rank={top_qq.rank} kickers={top_qq.kickers}")

    print(f"\nIn C++: AA kickers={top_aa.kickers}, KK kickers={top_kk.kickers}, QQ kickers={top_qq.kickers}")
    print(f"C++ thinks: AA {'>' if top_aa.kickers > top_kk.kickers else '<' if top_aa.kickers < top_kk.kickers else '=='} KK")
    print(f"Correct:    AA > KK > QQ")
    if top_aa.kickers < top_kk.kickers:
        print("BUG CONFIRMED: C++ treats AA as weaker than KK!")

def test_high_card_comparison():
    """A-high vs K-high: A-high should be stronger"""
    print("\n=== High Card Comparison ===")

    # A-high: A 8 5 3 2
    b1 = ofc.Board()
    for c in [('A','s'), ('8','h'), ('5','d'), ('3','c'), ('2','s')]:
        b1.place_card(ofc.BOTTOM, make_card(*c))
    bot_a = b1.evaluate_bot()
    print(f"A-high bottom: rank={bot_a.rank} kickers={bot_a.kickers}")

    # K-high: K Q J 9 7
    b2 = ofc.Board()
    for c in [('K','s'), ('Q','h'), ('J','d'), ('9','c'), ('7','s')]:
        b2.place_card(ofc.BOTTOM, make_card(*c))
    bot_k = b2.evaluate_bot()
    print(f"K-high bottom: rank={bot_k.rank} kickers={bot_k.kickers}")

    print(f"\nC++ thinks: A-high {'>' if bot_a.kickers > bot_k.kickers else '<' if bot_a.kickers < bot_k.kickers else '=='} K-high")
    print(f"Correct:    A-high > K-high")
    if bot_a.kickers < bot_k.kickers:
        print("BUG CONFIRMED: C++ treats A-high as weaker than K-high!")

def test_foul_with_ace():
    """
    Board: Top=22A, Mid=KQJ98 (High card K), Bot=AKQ76 (High card A)
    Correct: Bot(A-high) > Mid(K-high) → NOT foul for bot>=mid
    C++ bug: Bot has kickers=0(A), Mid has kickers=12(K) → Bot < Mid → FOUL
    """
    print("\n=== Foul Test with Ace ===")

    b = ofc.Board()
    # Top: pair of 2s + 3
    b.place_card(ofc.TOP, make_card('2', 's'))
    b.place_card(ofc.TOP, make_card('2', 'h'))
    b.place_card(ofc.TOP, make_card('3', 's'))

    # Mid: K-high
    b.place_card(ofc.MIDDLE, make_card('K', 's'))
    b.place_card(ofc.MIDDLE, make_card('Q', 'h'))
    b.place_card(ofc.MIDDLE, make_card('J', 'd'))
    b.place_card(ofc.MIDDLE, make_card('9', 'c'))
    b.place_card(ofc.MIDDLE, make_card('8', 's'))

    # Bot: A-high (stronger than K-high)
    b.place_card(ofc.BOTTOM, make_card('A', 'h'))
    b.place_card(ofc.BOTTOM, make_card('K', 'h'))
    b.place_card(ofc.BOTTOM, make_card('Q', 's'))
    b.place_card(ofc.BOTTOM, make_card('7', 'h'))
    b.place_card(ofc.BOTTOM, make_card('6', 'h'))

    top = b.evaluate_top()
    mid = b.evaluate_mid()
    bot = b.evaluate_bot()
    print(f"Top (22+3):  rank={top.rank} kickers={top.kickers}")
    print(f"Mid (KQJT8): rank={mid.rank} kickers={mid.kickers}")
    print(f"Bot (AKQ76): rank={bot.rank} kickers={bot.kickers}")
    print(f"C++ is_foul: {b.is_foul()}")
    print(f"Correct:     False (Bot A-high > Mid K-high, Mid HC > Top Pair, so Bot>=Mid is fine)")

    if b.is_foul():
        print("BUG CONFIRMED: C++ thinks A-high < K-high, causing false foul!")

def test_foul_pair_comparison():
    """
    Board: Top=222, Mid=AA+xyz, Bot=KK+xyz
    Correct: AA > KK → Mid > Bot → FOUL
    C++ bug: AA kickers=0, KK kickers=12 → KK > AA → Bot > Mid → NOT FOUL
    """
    print("\n=== Foul Test with Pair Aces ===")

    b = ofc.Board()
    # Top: trips of 2
    b.place_card(ofc.TOP, make_card('2', 's'))
    b.place_card(ofc.TOP, make_card('2', 'h'))
    b.place_card(ofc.TOP, make_card('2', 'd'))

    # Mid: AA + filler
    b.place_card(ofc.MIDDLE, make_card('A', 's'))
    b.place_card(ofc.MIDDLE, make_card('A', 'h'))
    b.place_card(ofc.MIDDLE, make_card('5', 's'))
    b.place_card(ofc.MIDDLE, make_card('4', 's'))
    b.place_card(ofc.MIDDLE, make_card('3', 's'))

    # Bot: KK + filler (should be weaker than AA)
    b.place_card(ofc.BOTTOM, make_card('K', 's'))
    b.place_card(ofc.BOTTOM, make_card('K', 'h'))
    b.place_card(ofc.BOTTOM, make_card('7', 'h'))
    b.place_card(ofc.BOTTOM, make_card('6', 'h'))
    b.place_card(ofc.BOTTOM, make_card('9', 'h'))

    top = b.evaluate_top()
    mid = b.evaluate_mid()
    bot = b.evaluate_bot()
    print(f"Top (222):   rank={top.rank} kickers={top.kickers}")
    print(f"Mid (AA543): rank={mid.rank} kickers={mid.kickers}")
    print(f"Bot (KK976): rank={bot.rank} kickers={bot.kickers}")
    print(f"C++ is_foul: {b.is_foul()}")
    print(f"Correct:     True (AA on Mid > KK on Bot → Bot < Mid → FOUL)")

    if not b.is_foul():
        print("BUG CONFIRMED: C++ thinks KK > AA, missing foul!")


if __name__ == '__main__':
    test_pair_comparison()
    test_high_card_comparison()
    test_foul_with_ace()
    test_foul_pair_comparison()
