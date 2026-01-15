import ofc_engine as ofc
import numpy as np

def test_joker_eval():
    print("=== Joker Evaluation Test (Phase 4-0) ===\n")
    all_passed = True
    
    # ビット位置マッピング: suit * 13 + rank
    # スペード=0, ハート=1, ダイヤ=2, クラブ=3
    # A=0, 2=1, 3=2, 4=3, 5=4, 6=5, 7=6, 8=7, 9=8, T=9, J=10, Q=11, K=12
    
    def card_idx(suit, rank):
        """スートとランクからカードインデックスを取得"""
        return suit * 13 + rank
    
    JOKER1, JOKER2 = 52, 53
    SPADE, HEART, DIAMOND, CLUB = 0, 1, 2, 3
    A, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, J, Q, K = range(13)
    
    # ====================================================================
    # テスト 1: THREE_OF_A_KIND (As + Joker x 2)
    # ====================================================================
    mask = (1 << card_idx(SPADE, A)) | (1 << JOKER1) | (1 << JOKER2)
    val = ofc.evaluate_3card(mask)
    if val.rank == ofc.HandRank.THREE_OF_A_KIND and val.kickers == A:
        print(f"✅ Test 1: [As, JK, JK] -> THREE_OF_A_KIND (A)")
    else:
        print(f"❌ Test 1: Expected THREE_OF_A_KIND(A), got {val.rank}({val.kickers})")
        all_passed = False
    
    # ====================================================================
    # テスト 2: STRAIGHT_FLUSH (2s-5s + Joker = Wheel SF)
    # ====================================================================
    mask5 = (1 << card_idx(SPADE, TWO)) | (1 << card_idx(SPADE, THREE)) | \
            (1 << card_idx(SPADE, FOUR)) | (1 << card_idx(SPADE, FIVE)) | (1 << JOKER1)
    val5 = ofc.evaluate_5card(mask5)
    if val5.rank == ofc.HandRank.STRAIGHT_FLUSH:
        print(f"✅ Test 2: [2s, 3s, 4s, 5s, JK] -> STRAIGHT_FLUSH")
    else:
        print(f"❌ Test 2: Expected STRAIGHT_FLUSH, got {val5.rank}")
        all_passed = False
    
    # ====================================================================
    # テスト 3: フルハウス vs 3カード（意地悪テスト）
    # Hand: [2s, 2h, 3d, 4c, JK] -> 期待値: THREE_OF_A_KIND (4)
    # ※ 4 + JK = 44 より、4 + JK + JK_as_4 = 444 が最強
    # ただし JK は 1 枚なので、最大は 44x (2ペアより弱い)
    # 再考: 2が2枚、3が1枚、4が1枚、JK1枚
    # -> JKをどれかに足して最強を狙う: 4 + JK = 44, 3 + JK = 33, 2 + JK = 222 (Trips!)
    # ====================================================================
    mask3 = (1 << card_idx(SPADE, TWO)) | (1 << card_idx(HEART, TWO)) | \
            (1 << card_idx(DIAMOND, THREE)) | (1 << card_idx(CLUB, FOUR)) | (1 << JOKER1)
    val3 = ofc.evaluate_5card(mask3)
    if val3.rank == ofc.HandRank.THREE_OF_A_KIND:
        print(f"✅ Test 3: [2s, 2h, 3d, 4c, JK] -> THREE_OF_A_KIND")
    else:
        print(f"❌ Test 3: Expected THREE_OF_A_KIND, got {val3.rank}")
        all_passed = False
    
    # ====================================================================
    # テスト 4: 4カードの優先（意地悪テスト）
    # Hand: [8s, 8h, 8d, 9c, JK] -> 期待値: FOUR_OF_A_KIND (8)
    # ====================================================================
    mask4 = (1 << card_idx(SPADE, EIGHT)) | (1 << card_idx(HEART, EIGHT)) | \
            (1 << card_idx(DIAMOND, EIGHT)) | (1 << card_idx(CLUB, NINE)) | (1 << JOKER1)
    val4 = ofc.evaluate_5card(mask4)
    if val4.rank == ofc.HandRank.FOUR_OF_A_KIND:
        print(f"✅ Test 4: [8s, 8h, 8d, 9c, JK] -> FOUR_OF_A_KIND")
    else:
        print(f"❌ Test 4: Expected FOUR_OF_A_KIND, got {val4.rank}")
        all_passed = False
    
    # ====================================================================
    # テスト 5: ロイヤル完成（意地悪テスト）
    # Hand: [As, Ks, Qs, Ts, JK] -> 期待値: ROYAL_FLUSH
    # ※ JK が Js になる
    # ====================================================================
    mask_royal = (1 << card_idx(SPADE, A)) | (1 << card_idx(SPADE, K)) | \
                 (1 << card_idx(SPADE, Q)) | (1 << card_idx(SPADE, TEN)) | (1 << JOKER1)
    val_royal = ofc.evaluate_5card(mask_royal)
    if val_royal.rank == ofc.HandRank.ROYAL_FLUSH:
        print(f"✅ Test 5: [As, Ks, Qs, Ts, JK] -> ROYAL_FLUSH")
    else:
        print(f"❌ Test 5: Expected ROYAL_FLUSH, got {val_royal.rank}")
        all_passed = False
    
    # ====================================================================
    # テスト 6: Mid > Top の比較（Foul判定の要）
    # Top: [Ks, JK, JK] -> KKK (Trips)
    # Mid: [Qs, Qh, Qd, Qc, 2s] -> QQQQ (Quads)
    # 判定: Mid > Top なので Valid
    # ====================================================================
    top_mask = (1 << card_idx(SPADE, K)) | (1 << JOKER1) | (1 << JOKER2)
    mid_mask = (1 << card_idx(SPADE, Q)) | (1 << card_idx(HEART, Q)) | \
               (1 << card_idx(DIAMOND, Q)) | (1 << card_idx(CLUB, Q)) | (1 << card_idx(SPADE, TWO))
    
    top_val = ofc.evaluate_3card(top_mask)
    mid_val = ofc.evaluate_5card(mid_mask)
    
    # Mid (Quads) > Top (Trips) であることを確認
    if mid_val > top_val:
        print(f"✅ Test 6: Mid (QUADS) > Top (TRIPS) -> Valid (Not a Foul)")
    else:
        print(f"❌ Test 6: Mid should be > Top, but comparison failed")
        all_passed = False
    
    print()
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed. Please review.")
    
    return all_passed

if __name__ == "__main__":
    test_joker_eval()
