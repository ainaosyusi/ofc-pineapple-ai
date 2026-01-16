import ofc_engine as ofc
import time

def test_solver():
    print("=== Fantasy Land Solver Test ===")
    
    # 14枚のカードを用意（Joker 2枚含む）
    # Royal Flush を狙える手
    cards = [
        ofc.Card(ofc.Suit.SPADE, ofc.Rank.ACE),
        ofc.Card(ofc.Suit.SPADE, ofc.Rank.KING),
        ofc.Card(ofc.Suit.SPADE, ofc.Rank.QUEEN),
        ofc.Card(ofc.Suit.SPADE, ofc.Rank.JACK),
        ofc.Card(ofc.Suit.SPADE, ofc.Rank.TEN),
        # Joker 2枚
        ofc.Card(52),
        ofc.Card(53),
        # その他
        ofc.Card(ofc.Suit.HEART, ofc.Rank.TWO),
        ofc.Card(ofc.Suit.HEART, ofc.Rank.THREE),
        ofc.Card(ofc.Suit.HEART, ofc.Rank.FOUR),
        ofc.Card(ofc.Suit.HEART, ofc.Rank.FIVE),
        ofc.Card(ofc.Suit.HEART, ofc.Rank.SIX),
        ofc.Card(ofc.Suit.DIAMOND, ofc.Rank.SEVEN),
        ofc.Card(ofc.Suit.DIAMOND, ofc.Rank.EIGHT),
    ]
    
    print(f"Testing with {len(cards)} cards (includes 2 Jokers)")
    
    start_time = time.time()
    solution = ofc.solve_fantasy_land(cards, True)
    elapsed = time.time() - start_time
    
    print(f"Solver (14 cards) finished in {elapsed:.4f}s")
    print(f"Optimal Score: {solution.score}")

    # 17枚テスト
    cards17 = cards + [
        ofc.Card(ofc.Suit.DIAMOND, ofc.Rank.TWO),
        ofc.Card(ofc.Suit.DIAMOND, ofc.Rank.THREE),
        ofc.Card(ofc.Suit.DIAMOND, ofc.Rank.FOUR),
    ]
    print(f"\nTesting with {len(cards17)} cards")
    start_time = time.time()
    solution17 = ofc.solve_fantasy_land(cards17, True)
    elapsed17 = time.time() - start_time
    print(f"Solver (17 cards) finished in {elapsed17:.4f}s")
    print(f"Optimal Score: {solution17.score}")
    print(f"Stayed FL: {solution.stayed}")
    
    print("\n[Board Layout]")
    print(f"Top: {[str(c) for c in solution.top]}")
    print(f"Mid: {[str(c) for c in solution.mid]}")
    print(f"Bot: {[str(c) for c in solution.bot]}")
    print(f"Discards: {[str(c) for c in solution.discards]}")

if __name__ == "__main__":
    test_solver()
