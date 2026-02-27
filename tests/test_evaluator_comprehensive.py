#!/usr/bin/env python3
"""
OFC Pineapple AI - C++ エンジン網羅テスト
Phase 0-1: ACE修正の検証 + 全役・境界条件・Joker・FL・スコア・フォール

優先度順:
  1. ACE比較テスト（V1崩壊の直接原因）
  2. 全役の大小関係テスト
  3. 役境界テスト（ストレート境界）
  4. ジョーカー代用テスト
  5. FL Entry/Stay 条件テスト
  6. スコア計算テスト
  7. フォール判定テスト

実行方法:
    cd /Users/naoai/試作品一覧/OFC\ NN
    python tests/test_evaluator_comprehensive.py
"""

import sys
import os

# プロジェクトルートをパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import ofc_engine as ofc

# ============================================================
# ヘルパー関数
# ============================================================

SPADE, HEART, DIAMOND, CLUB = 0, 1, 2, 3
A, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, J, Q, K = range(13)
JOKER1, JOKER2 = 52, 53

RANK_NAMES = {A: 'A', TWO: '2', THREE: '3', FOUR: '4', FIVE: '5',
              SIX: '6', SEVEN: '7', EIGHT: '8', NINE: '9', TEN: 'T',
              J: 'J', Q: 'Q', K: 'K'}
SUIT_NAMES = {SPADE: 's', HEART: 'h', DIAMOND: 'd', CLUB: 'c'}


def ci(suit, rank):
    """カードインデックス (0-51)"""
    return suit * 13 + rank


def mask_of(*indices):
    """カードインデックスのリストからビットマスクを作成"""
    m = 0
    for idx in indices:
        m |= (1 << idx)
    return m


def hand_5(*cards):
    """5枚のカード指定 → evaluate_5card"""
    return ofc.evaluate_5card(mask_of(*cards))


def hand_3(*cards):
    """3枚のカード指定 → evaluate_3card"""
    return ofc.evaluate_3card(mask_of(*cards))


def card_name(idx):
    """カードインデックスを文字列に変換"""
    if idx == JOKER1:
        return 'JK1'
    if idx == JOKER2:
        return 'JK2'
    suit = idx // 13
    rank = idx % 13
    return f"{RANK_NAMES[rank]}{SUIT_NAMES[suit]}"


class TestRunner:
    """テスト結果の集計"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, condition, test_name, detail=""):
        if condition:
            self.passed += 1
        else:
            self.failed += 1
            self.errors.append(f"{test_name}: {detail}")
            print(f"  FAIL: {test_name} - {detail}")

    def section(self, title):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"  RESULTS: {self.passed}/{total} passed, {self.failed} failed")
        print(f"{'='*60}")
        if self.errors:
            print("\n  Failed tests:")
            for err in self.errors:
                print(f"    - {err}")
        return self.failed == 0


T = TestRunner()


# ============================================================
# 1. ACE 比較テスト（最優先）
# ============================================================

def test_ace_comparison():
    T.section("1. ACE 比較テスト（V1崩壊の直接原因）")

    # --- 1-1. ペア比較: AA > KK > QQ > ... > 22 ---
    print("\n  [1-1] ペア比較 (5枚): AA > KK > QQ > ... > 22")

    # AA vs KK（同キッカー）
    aa = hand_5(ci(SPADE, A), ci(HEART, A), ci(DIAMOND, K), ci(CLUB, Q), ci(SPADE, J))
    kk = hand_5(ci(SPADE, K), ci(HEART, K), ci(DIAMOND, A), ci(CLUB, Q), ci(SPADE, J))
    T.check(aa > kk, "AA > KK (5card)",
            f"AA.rank={aa.rank}, kickers={aa.kickers:#x}; KK.rank={kk.rank}, kickers={kk.kickers:#x}")

    # AA vs 22
    aa2 = hand_5(ci(SPADE, A), ci(HEART, A), ci(DIAMOND, K), ci(CLUB, Q), ci(SPADE, J))
    twos = hand_5(ci(SPADE, TWO), ci(HEART, TWO), ci(DIAMOND, K), ci(CLUB, Q), ci(SPADE, J))
    T.check(aa2 > twos, "AA > 22 (5card)",
            f"AA.rank={aa2.rank}, kickers={aa2.kickers:#x}; 22.rank={twos.rank}, kickers={twos.kickers:#x}")

    # 全ペアの順序: AA > KK > QQ > JJ > TT > 99 > ... > 22
    ranks_desc = [A, K, Q, J, TEN, NINE, EIGHT, SEVEN, SIX, FIVE, FOUR, THREE, TWO]
    prev_val = None
    prev_name = None
    all_pair_order = True
    for r in ranks_desc:
        # ペアr + 3枚のキッカー（rと異なるランクから選ぶ）
        all_ranks = [A, K, Q, J, TEN, NINE, EIGHT, SEVEN, SIX, FIVE, FOUR, THREE, TWO]
        kickers = [x for x in all_ranks if x != r][:3]
        val = hand_5(ci(SPADE, r), ci(HEART, r),
                     ci(DIAMOND, kickers[0]), ci(CLUB, kickers[1]), ci(SPADE, kickers[2]))
        T.check(val.rank == ofc.HandRank.ONE_PAIR,
                f"{RANK_NAMES[r]}{RANK_NAMES[r]} is ONE_PAIR",
                f"got {val.rank}")
        if prev_val is not None:
            ok = prev_val > val
            T.check(ok, f"{prev_name} > {RANK_NAMES[r]}{RANK_NAMES[r]}",
                    f"prev.kickers={prev_val.kickers:#x}, curr.kickers={val.kickers:#x}")
            if not ok:
                all_pair_order = False
        prev_val = val
        prev_name = f"{RANK_NAMES[r]}{RANK_NAMES[r]}"

    # --- 1-2. Top 3枚でのペア比較 ---
    print("\n  [1-2] ペア比較 (3枚 Top): AA > KK > QQ > ... > 22")
    aa_top = hand_3(ci(SPADE, A), ci(HEART, A), ci(DIAMOND, K))
    kk_top = hand_3(ci(SPADE, K), ci(HEART, K), ci(DIAMOND, A))
    qq_top = hand_3(ci(SPADE, Q), ci(HEART, Q), ci(DIAMOND, A))
    twos_top = hand_3(ci(SPADE, TWO), ci(HEART, TWO), ci(DIAMOND, A))

    T.check(aa_top > kk_top, "AA > KK (3card Top)",
            f"AA={aa_top.rank},{aa_top.kickers:#x}; KK={kk_top.rank},{kk_top.kickers:#x}")
    T.check(kk_top > qq_top, "KK > QQ (3card Top)")
    T.check(aa_top > twos_top, "AA > 22 (3card Top)")

    # --- 1-3. ハイカードでのACE ---
    print("\n  [1-3] ハイカード: A ハイ > K ハイ")
    a_high = hand_5(ci(SPADE, A), ci(HEART, K), ci(DIAMOND, Q), ci(CLUB, TEN), ci(SPADE, EIGHT))
    k_high = hand_5(ci(SPADE, K), ci(HEART, Q), ci(DIAMOND, J), ci(CLUB, TEN), ci(SPADE, EIGHT))
    T.check(a_high > k_high, "A-high > K-high",
            f"A.rank={a_high.rank},{a_high.kickers:#x}; K.rank={k_high.rank},{k_high.kickers:#x}")

    # --- 1-4. キッカーでの ACE ---
    print("\n  [1-4] キッカーの ACE: KK-A > KK-Q")
    kk_a_kicker = hand_5(ci(SPADE, K), ci(HEART, K), ci(DIAMOND, A), ci(CLUB, THREE), ci(SPADE, TWO))
    kk_q_kicker = hand_5(ci(SPADE, K), ci(HEART, K), ci(DIAMOND, Q), ci(CLUB, THREE), ci(SPADE, TWO))
    T.check(kk_a_kicker > kk_q_kicker, "KK with A kicker > KK with Q kicker")

    # --- 1-5. トリプス比較 ---
    print("\n  [1-5] トリプス比較: AAA > KKK > 222")
    aaa = hand_5(ci(SPADE, A), ci(HEART, A), ci(DIAMOND, A), ci(CLUB, K), ci(SPADE, Q))
    kkk = hand_5(ci(SPADE, K), ci(HEART, K), ci(DIAMOND, K), ci(CLUB, A), ci(SPADE, Q))
    twos3 = hand_5(ci(SPADE, TWO), ci(HEART, TWO), ci(DIAMOND, TWO), ci(CLUB, A), ci(SPADE, K))
    T.check(aaa > kkk, "AAA > KKK")
    T.check(kkk > twos3, "KKK > 222")
    T.check(aaa > twos3, "AAA > 222")

    # --- 1-6. トリプス (3枚 Top) ---
    print("\n  [1-6] トリプス比較 (3枚 Top): AAA > KKK > 222")
    aaa_top = hand_3(ci(SPADE, A), ci(HEART, A), ci(DIAMOND, A))
    kkk_top = hand_3(ci(SPADE, K), ci(HEART, K), ci(DIAMOND, K))
    twos3_top = hand_3(ci(SPADE, TWO), ci(HEART, TWO), ci(DIAMOND, TWO))
    T.check(aaa_top > kkk_top, "AAA > KKK (3card)")
    T.check(kkk_top > twos3_top, "KKK > 222 (3card)")


# ============================================================
# 2. 全役の大小関係テスト
# ============================================================

def test_hand_rank_ordering():
    T.section("2. 全役の大小関係テスト")

    # 各役の代表的なハンド
    high_card = hand_5(ci(SPADE, A), ci(HEART, K), ci(DIAMOND, Q), ci(CLUB, TEN), ci(SPADE, EIGHT))
    one_pair = hand_5(ci(SPADE, A), ci(HEART, A), ci(DIAMOND, K), ci(CLUB, Q), ci(SPADE, J))
    two_pair = hand_5(ci(SPADE, A), ci(HEART, A), ci(DIAMOND, K), ci(CLUB, K), ci(SPADE, Q))
    trips = hand_5(ci(SPADE, A), ci(HEART, A), ci(DIAMOND, A), ci(CLUB, K), ci(SPADE, Q))
    straight = hand_5(ci(SPADE, TEN), ci(HEART, J), ci(DIAMOND, Q), ci(CLUB, K), ci(SPADE, A))
    flush = hand_5(ci(SPADE, A), ci(SPADE, K), ci(SPADE, Q), ci(SPADE, TEN), ci(SPADE, EIGHT))
    full_house = hand_5(ci(SPADE, A), ci(HEART, A), ci(DIAMOND, A), ci(CLUB, K), ci(SPADE, K))
    quads = hand_5(ci(SPADE, A), ci(HEART, A), ci(DIAMOND, A), ci(CLUB, A), ci(SPADE, K))
    straight_flush = hand_5(ci(SPADE, NINE), ci(SPADE, TEN), ci(SPADE, J), ci(SPADE, Q), ci(SPADE, K))
    royal_flush = hand_5(ci(SPADE, TEN), ci(SPADE, J), ci(SPADE, Q), ci(SPADE, K), ci(SPADE, A))

    # 役ランクの確認
    hands = [
        (high_card, ofc.HandRank.HIGH_CARD, "ハイカード"),
        (one_pair, ofc.HandRank.ONE_PAIR, "ワンペア"),
        (two_pair, ofc.HandRank.TWO_PAIR, "ツーペア"),
        (trips, ofc.HandRank.THREE_OF_A_KIND, "トリプス"),
        (straight, ofc.HandRank.STRAIGHT, "ストレート"),
        (flush, ofc.HandRank.FLUSH, "フラッシュ"),
        (full_house, ofc.HandRank.FULL_HOUSE, "フルハウス"),
        (quads, ofc.HandRank.FOUR_OF_A_KIND, "フォーカード"),
        (straight_flush, ofc.HandRank.STRAIGHT_FLUSH, "ストレートフラッシュ"),
        (royal_flush, ofc.HandRank.ROYAL_FLUSH, "ロイヤルフラッシュ"),
    ]

    print("\n  [2-1] 各役が正しいランクに判定されるか")
    for val, expected_rank, name in hands:
        T.check(val.rank == expected_rank, f"{name} の役判定",
                f"expected {expected_rank}, got {val.rank}")

    print("\n  [2-2] 役の大小: RF > SF > Quads > FH > Flush > Str > Trips > 2P > 1P > HC")
    for i in range(len(hands) - 1):
        val_lower, _, name_lower = hands[i]
        val_upper, _, name_upper = hands[i + 1]
        T.check(val_upper > val_lower, f"{name_upper} > {name_lower}",
                f"upper={val_upper.rank},{val_upper.kickers:#x}; lower={val_lower.rank},{val_lower.kickers:#x}")

    # --- 同じ役内での比較 ---
    print("\n  [2-3] 同役内比較: ツーペア AA-KK > AA-QQ > KK-QQ")
    aakk = hand_5(ci(SPADE, A), ci(HEART, A), ci(DIAMOND, K), ci(CLUB, K), ci(SPADE, Q))
    aaqq = hand_5(ci(SPADE, A), ci(HEART, A), ci(DIAMOND, Q), ci(CLUB, Q), ci(SPADE, K))
    kkqq = hand_5(ci(SPADE, K), ci(HEART, K), ci(DIAMOND, Q), ci(CLUB, Q), ci(SPADE, A))
    T.check(aakk > aaqq, "AA-KK > AA-QQ (2pair)")
    T.check(aaqq > kkqq, "AA-QQ > KK-QQ (2pair)")

    print("\n  [2-4] フルハウス: AAA-KK > KKK-AA > QQQ-AA")
    fh_aak = hand_5(ci(SPADE, A), ci(HEART, A), ci(DIAMOND, A), ci(CLUB, K), ci(SPADE, K))
    fh_kka = hand_5(ci(SPADE, K), ci(HEART, K), ci(DIAMOND, K), ci(CLUB, A), ci(SPADE, A))
    fh_qqa = hand_5(ci(SPADE, Q), ci(HEART, Q), ci(DIAMOND, Q), ci(CLUB, A), ci(SPADE, A))
    T.check(fh_aak > fh_kka, "AAA-KK > KKK-AA (full house)")
    T.check(fh_kka > fh_qqa, "KKK-AA > QQQ-AA (full house)")

    print("\n  [2-5] フラッシュ: A-K-Q-J-9 > A-K-Q-J-8")
    flush1 = hand_5(ci(SPADE, A), ci(SPADE, K), ci(SPADE, Q), ci(SPADE, J), ci(SPADE, NINE))
    flush2 = hand_5(ci(HEART, A), ci(HEART, K), ci(HEART, Q), ci(HEART, J), ci(HEART, EIGHT))
    T.check(flush1 > flush2, "Flush A-K-Q-J-9 > A-K-Q-J-8")


# ============================================================
# 3. 役境界テスト（ストレート境界）
# ============================================================

def test_straight_boundaries():
    T.section("3. 役境界テスト（ストレート境界）")

    # --- 3-1. ホイール (A-2-3-4-5): 最弱ストレート ---
    print("\n  [3-1] ホイール A-2-3-4-5")
    wheel = hand_5(ci(SPADE, A), ci(HEART, TWO), ci(DIAMOND, THREE), ci(CLUB, FOUR), ci(SPADE, FIVE))
    T.check(wheel.rank == ofc.HandRank.STRAIGHT, "A-2-3-4-5 はストレート",
            f"got {wheel.rank}")

    # --- 3-2. ブロードウェイ (T-J-Q-K-A): 最強ストレート ---
    print("\n  [3-2] ブロードウェイ T-J-Q-K-A")
    broadway = hand_5(ci(SPADE, TEN), ci(HEART, J), ci(DIAMOND, Q), ci(CLUB, K), ci(HEART, A))
    T.check(broadway.rank == ofc.HandRank.STRAIGHT, "T-J-Q-K-A はストレート",
            f"got {broadway.rank}")

    # --- 3-3. ホイール < 2-3-4-5-6 < ... < ブロードウェイ ---
    print("\n  [3-3] ストレートの大小順序")
    str_2_6 = hand_5(ci(SPADE, TWO), ci(HEART, THREE), ci(DIAMOND, FOUR), ci(CLUB, FIVE), ci(SPADE, SIX))
    str_9_k = hand_5(ci(SPADE, NINE), ci(HEART, TEN), ci(DIAMOND, J), ci(CLUB, Q), ci(SPADE, K))

    T.check(broadway > str_9_k, "Broadway > 9-K straight")
    T.check(str_9_k > str_2_6, "9-K > 2-6 straight")
    T.check(str_2_6 > wheel, "2-6 > Wheel (A-5)")
    T.check(broadway > wheel, "Broadway > Wheel")

    # --- 3-4. ラップアラウンド不可: Q-K-A-2-3 はストレートではない ---
    print("\n  [3-4] ラップアラウンド不可: Q-K-A-2-3 はストレートではない")
    wrap = hand_5(ci(SPADE, Q), ci(HEART, K), ci(DIAMOND, A), ci(CLUB, TWO), ci(SPADE, THREE))
    T.check(wrap.rank != ofc.HandRank.STRAIGHT, "Q-K-A-2-3 はストレートではない",
            f"got {wrap.rank}")

    # --- 3-5. K-A-2-3-4 もストレートではない ---
    print("\n  [3-5] K-A-2-3-4 もストレートではない")
    wrap2 = hand_5(ci(SPADE, K), ci(HEART, A), ci(DIAMOND, TWO), ci(CLUB, THREE), ci(SPADE, FOUR))
    T.check(wrap2.rank != ofc.HandRank.STRAIGHT, "K-A-2-3-4 はストレートではない",
            f"got {wrap2.rank}")

    # --- 3-6. ストレートフラッシュのホイール ---
    print("\n  [3-6] ホイール・ストレートフラッシュ (As-2s-3s-4s-5s)")
    wheel_sf = hand_5(ci(SPADE, A), ci(SPADE, TWO), ci(SPADE, THREE), ci(SPADE, FOUR), ci(SPADE, FIVE))
    T.check(wheel_sf.rank == ofc.HandRank.STRAIGHT_FLUSH,
            "A-2-3-4-5 同スート = ストレートフラッシュ",
            f"got {wheel_sf.rank}")

    # --- 3-7. ロイヤルフラッシュ > ストレートフラッシュ > ストレート ---
    print("\n  [3-7] ロイヤルフラッシュ > ストレートフラッシュ > ストレート")
    royal = hand_5(ci(SPADE, TEN), ci(SPADE, J), ci(SPADE, Q), ci(SPADE, K), ci(SPADE, A))
    sf_9_k = hand_5(ci(SPADE, NINE), ci(SPADE, TEN), ci(SPADE, J), ci(SPADE, Q), ci(SPADE, K))
    T.check(royal > sf_9_k, "Royal > SF (9-K)")
    T.check(sf_9_k > broadway, "SF (9-K) > Straight (Broadway)")


# ============================================================
# 4. ジョーカー代用テスト
# ============================================================

def test_joker_substitution():
    T.section("4. ジョーカー代用テスト")

    # --- 4-1. ジョーカー1枚でペア形成 ---
    print("\n  [4-1] Joker + A = AA (ペア)")
    pair_jk = hand_3(ci(SPADE, A), JOKER1, ci(DIAMOND, K))
    pair_real = hand_3(ci(SPADE, A), ci(HEART, A), ci(DIAMOND, K))
    T.check(pair_jk.rank == ofc.HandRank.ONE_PAIR, "A+JK = ペア (3card)",
            f"got {pair_jk.rank}")
    # Joker ペアは実ペアと等価（同ランク）
    # >= が未実装のため > or == で代替
    T.check(pair_jk > pair_real or pair_jk == pair_real,
            "A+JK のペア >= 実AA (3card)",
            f"jk={pair_jk.rank},{pair_jk.kickers:#x}; real={pair_real.rank},{pair_real.kickers:#x}")

    # --- 4-2. ジョーカー1枚でトリプス (5枚) ---
    print("\n  [4-2] AA + JK = AAA (トリプス 5card)")
    trips_jk = hand_5(ci(SPADE, A), ci(HEART, A), JOKER1, ci(DIAMOND, K), ci(CLUB, Q))
    T.check(trips_jk.rank == ofc.HandRank.THREE_OF_A_KIND, "AA+JK = トリプス",
            f"got {trips_jk.rank}")

    # --- 4-3. ジョーカー2枚でトリプス (3枚 Top) ---
    print("\n  [4-3] A + JK + JK = AAA (トリプス 3card)")
    trips_2jk = hand_3(ci(SPADE, A), JOKER1, JOKER2)
    T.check(trips_2jk.rank == ofc.HandRank.THREE_OF_A_KIND, "A+JK+JK = トリプス (3card)",
            f"got {trips_2jk.rank}")

    # --- 4-4. ジョーカーでストレート完成 ---
    print("\n  [4-4] 9-T-J-Q + JK = ストレート (JK=K or 8)")
    str_jk = hand_5(ci(SPADE, NINE), ci(HEART, TEN), ci(DIAMOND, J), ci(CLUB, Q), JOKER1)
    T.check(str_jk.rank == ofc.HandRank.STRAIGHT, "9-T-J-Q+JK = ストレート",
            f"got {str_jk.rank}")

    # --- 4-5. ジョーカーでフラッシュ完成 ---
    print("\n  [4-5] 4枚同スート + JK = フラッシュ")
    flush_jk = hand_5(ci(SPADE, A), ci(SPADE, K), ci(SPADE, Q), ci(SPADE, TEN), JOKER1)
    T.check(flush_jk.rank >= ofc.HandRank.FLUSH, "4枚スペード+JK = フラッシュ以上",
            f"got {flush_jk.rank}")

    # --- 4-6. ジョーカーでフォーカード ---
    print("\n  [4-6] AAA + JK = AAAA (フォーカード)")
    quads_jk = hand_5(ci(SPADE, A), ci(HEART, A), ci(DIAMOND, A), JOKER1, ci(CLUB, K))
    T.check(quads_jk.rank == ofc.HandRank.FOUR_OF_A_KIND, "AAA+JK = フォーカード",
            f"got {quads_jk.rank}")

    # --- 4-7. ジョーカー2枚でフォーカード ---
    print("\n  [4-7] AA + JK + JK = AAAA (フォーカード)")
    quads_2jk = hand_5(ci(SPADE, A), ci(HEART, A), JOKER1, JOKER2, ci(CLUB, K))
    T.check(quads_2jk.rank == ofc.HandRank.FOUR_OF_A_KIND, "AA+JK+JK = フォーカード",
            f"got {quads_2jk.rank}")

    # --- 4-8. ジョーカーでロイヤルフラッシュ ---
    print("\n  [4-8] Ts-Js-Qs-Ks + JK = ロイヤルフラッシュ")
    royal_jk = hand_5(ci(SPADE, TEN), ci(SPADE, J), ci(SPADE, Q), ci(SPADE, K), JOKER1)
    T.check(royal_jk.rank == ofc.HandRank.ROYAL_FLUSH, "T-J-Q-K+JK (spade) = ロイヤルフラッシュ",
            f"got {royal_jk.rank}")

    # --- 4-9. ジョーカーは最も有利な役に使われる ---
    print("\n  [4-9] KKK + Q + JK: フォーカード(KKKK) > フルハウス(KKK-QQ)")
    best_jk = hand_5(ci(SPADE, K), ci(HEART, K), ci(DIAMOND, K), ci(CLUB, Q), JOKER1)
    T.check(best_jk.rank == ofc.HandRank.FOUR_OF_A_KIND,
            "JK は KKKK(Quads) に使われる（FH にはならない）",
            f"got {best_jk.rank}")


# ============================================================
# 5. FL Entry/Stay 条件テスト
# ============================================================

def test_fl_conditions():
    T.section("5. FL Entry/Stay 条件テスト")

    def make_board(top_cards, mid_cards, bot_cards):
        """ボードを構築してFL判定をテスト"""
        board = ofc.Board()
        for c in top_cards:
            board.place_card(ofc.Row.TOP, ofc.Card(c))
        for c in mid_cards:
            board.place_card(ofc.Row.MIDDLE, ofc.Card(c))
        for c in bot_cards:
            board.place_card(ofc.Row.BOTTOM, ofc.Card(c))
        return board

    # --- 5-1. FL Entry: QQ+ on Top ---
    print("\n  [5-1] FL Entry 条件: Top が QQ以上")

    # QQ on Top → FL Entry
    board_qq = make_board(
        [ci(SPADE, Q), ci(HEART, Q), ci(DIAMOND, TWO)],         # Top: QQ
        [ci(SPADE, K), ci(HEART, K), ci(DIAMOND, K), ci(CLUB, THREE), ci(CLUB, FOUR)],  # Mid: KKK
        [ci(SPADE, A), ci(HEART, A), ci(DIAMOND, A), ci(CLUB, A), ci(CLUB, FIVE)]       # Bot: AAAA
    )
    T.check(not board_qq.is_foul(), "QQ top, KKK mid, AAAA bot: フォールではない")
    T.check(board_qq.qualifies_for_fl(), "QQ on Top → FL Entry",
            f"qualifies={board_qq.qualifies_for_fl()}")

    # KK on Top → FL Entry
    board_kk = make_board(
        [ci(SPADE, K), ci(HEART, K), ci(DIAMOND, TWO)],
        [ci(SPADE, A), ci(HEART, A), ci(DIAMOND, A), ci(CLUB, THREE), ci(CLUB, FOUR)],
        [ci(SPADE, A), ci(HEART, TEN), ci(DIAMOND, J), ci(CLUB, Q), ci(CLUB, K)]  # 使えるカード
    )
    # Note: Bot needs to be >= Mid. Let me fix this
    board_kk2 = make_board(
        [ci(SPADE, K), ci(HEART, K), ci(DIAMOND, TWO)],
        [ci(CLUB, THREE), ci(CLUB, FOUR), ci(CLUB, FIVE), ci(CLUB, SIX), ci(CLUB, SEVEN)],  # Mid: Flush
        [ci(SPADE, A), ci(HEART, A), ci(DIAMOND, A), ci(HEART, J), ci(HEART, TEN)]          # Bot: Trips A
    )
    # フォールの可能性があるので先にチェック
    if not board_kk2.is_foul():
        T.check(board_kk2.qualifies_for_fl(), "KK on Top → FL Entry")
    else:
        # フォールになる場合は別の組み合わせで
        board_kk3 = make_board(
            [ci(SPADE, K), ci(HEART, K), ci(DIAMOND, TWO)],
            [ci(SPADE, THREE), ci(HEART, THREE), ci(DIAMOND, THREE), ci(CLUB, FOUR), ci(CLUB, FIVE)],  # Mid: Trips 3
            [ci(SPADE, A), ci(HEART, A), ci(DIAMOND, A), ci(CLUB, A), ci(HEART, TEN)]                  # Bot: Quads A
        )
        T.check(not board_kk3.is_foul(), "KK/Trips3/QuadsA: フォールではない")
        T.check(board_kk3.qualifies_for_fl(), "KK on Top → FL Entry")

    # AA on Top → FL Entry
    board_aa = make_board(
        [ci(SPADE, A), ci(HEART, A), ci(DIAMOND, TWO)],
        [ci(SPADE, THREE), ci(HEART, THREE), ci(DIAMOND, THREE), ci(CLUB, FOUR), ci(CLUB, FIVE)],
        [ci(SPADE, K), ci(HEART, K), ci(DIAMOND, K), ci(CLUB, K), ci(HEART, TEN)]
    )
    T.check(not board_aa.is_foul(), "AA/Trips3/QuadsK: フォールではない")
    T.check(board_aa.qualifies_for_fl(), "AA on Top → FL Entry")

    # Trips on Top → FL Entry
    board_trips_top = make_board(
        [ci(SPADE, TWO), ci(HEART, TWO), ci(DIAMOND, TWO)],
        [ci(SPADE, THREE), ci(HEART, THREE), ci(DIAMOND, THREE), ci(CLUB, FOUR), ci(CLUB, FIVE)],
        [ci(SPADE, K), ci(HEART, K), ci(DIAMOND, K), ci(CLUB, K), ci(HEART, TEN)]
    )
    T.check(not board_trips_top.is_foul(), "222/Trips3/QuadsK: フォールではない")
    T.check(board_trips_top.qualifies_for_fl(), "Trips on Top → FL Entry")

    # JJ on Top → NO FL Entry
    print("\n  [5-2] FL 非Entry 条件: JJ 以下は FL に入れない")
    board_jj = make_board(
        [ci(SPADE, J), ci(HEART, J), ci(DIAMOND, TWO)],
        [ci(SPADE, Q), ci(HEART, Q), ci(DIAMOND, Q), ci(CLUB, THREE), ci(CLUB, FOUR)],
        [ci(SPADE, A), ci(HEART, A), ci(DIAMOND, A), ci(CLUB, A), ci(CLUB, FIVE)]
    )
    T.check(not board_jj.is_foul(), "JJ/TripsQ/QuadsA: フォールではない")
    T.check(not board_jj.qualifies_for_fl(), "JJ on Top → NO FL Entry",
            f"qualifies={board_jj.qualifies_for_fl()}")

    # --- 5-3. FL 枚数: QQ=14, KK=15, AA=16, Trips=17 ---
    print("\n  [5-3] FL 配布枚数 (Ultimate Rules)")
    top_qq = ofc.evaluate_3card(mask_of(ci(SPADE, Q), ci(HEART, Q), ci(DIAMOND, TWO)))
    top_kk = ofc.evaluate_3card(mask_of(ci(SPADE, K), ci(HEART, K), ci(DIAMOND, TWO)))
    top_aa = ofc.evaluate_3card(mask_of(ci(SPADE, A), ci(HEART, A), ci(DIAMOND, TWO)))
    top_trips = ofc.evaluate_3card(mask_of(ci(SPADE, TWO), ci(HEART, TWO), ci(DIAMOND, TWO)))

    T.check(ofc.fantasy_land_cards(top_qq) == 14, "QQ → 14枚",
            f"got {ofc.fantasy_land_cards(top_qq)}")
    T.check(ofc.fantasy_land_cards(top_kk) == 15, "KK → 15枚",
            f"got {ofc.fantasy_land_cards(top_kk)}")
    T.check(ofc.fantasy_land_cards(top_aa) == 16, "AA → 16枚",
            f"got {ofc.fantasy_land_cards(top_aa)}")
    T.check(ofc.fantasy_land_cards(top_trips) == 17, "Trips → 17枚",
            f"got {ofc.fantasy_land_cards(top_trips)}")

    # --- 5-4. FL Stay 条件 ---
    print("\n  [5-4] FL Stay 条件: Trips on Top or Quads on Bot")

    # Trips on Top → Stay (Mid >= Trips 必須)
    board_stay1 = make_board(
        [ci(SPADE, SEVEN), ci(HEART, SEVEN), ci(DIAMOND, SEVEN)],
        [ci(SPADE, EIGHT), ci(HEART, EIGHT), ci(DIAMOND, EIGHT), ci(CLUB, TWO), ci(CLUB, THREE)],
        [ci(SPADE, A), ci(HEART, A), ci(DIAMOND, A), ci(CLUB, A), ci(CLUB, FIVE)]
    )
    T.check(not board_stay1.is_foul(), "777/888x/AAAAx: フォールではない")
    T.check(board_stay1.can_stay_fl(), "Trips on Top → FL Stay")

    # Quads on Bot → Stay
    board_stay2 = make_board(
        [ci(SPADE, TWO), ci(HEART, THREE), ci(DIAMOND, FOUR)],  # Top: ハイカード
        [ci(SPADE, FIVE), ci(HEART, FIVE), ci(DIAMOND, FIVE), ci(CLUB, SIX), ci(CLUB, SEVEN)],  # Mid: Trips 5
        [ci(SPADE, K), ci(HEART, K), ci(DIAMOND, K), ci(CLUB, K), ci(HEART, TEN)]  # Bot: Quads K
    )
    T.check(not board_stay2.is_foul(), "HC/Trips5/QuadsK: フォールではない")
    T.check(board_stay2.can_stay_fl(), "Quads on Bot → FL Stay")

    # Pair on Top, Trips on Bot → NO Stay
    board_nostay = make_board(
        [ci(SPADE, A), ci(HEART, A), ci(DIAMOND, TWO)],         # Top: AA
        [ci(SPADE, K), ci(HEART, K), ci(DIAMOND, K), ci(CLUB, THREE), ci(CLUB, FOUR)],  # Mid: KKK
        [ci(SPADE, Q), ci(HEART, Q), ci(DIAMOND, Q), ci(CLUB, Q), ci(HEART, TEN)]       # Bot: QQQQ
    )
    # AA on Top + QQQQ on Bot: qualifies for FL Entry, but NOT Stay (need Trips on Top or Quads on Bot)
    # Wait, QQQQ IS Quads on Bot, so this SHOULD stay
    T.check(board_nostay.can_stay_fl(), "AA top + QQQQ bot → FL Stay (Quads on Bot)")

    # 本当にStayできないケース: AA on Top, Full House on Bot
    board_nostay2 = make_board(
        [ci(SPADE, A), ci(HEART, A), ci(DIAMOND, TWO)],
        [ci(SPADE, K), ci(HEART, K), ci(DIAMOND, K), ci(CLUB, THREE), ci(CLUB, FOUR)],
        [ci(SPADE, Q), ci(HEART, Q), ci(DIAMOND, Q), ci(CLUB, J), ci(HEART, J)]  # Bot: FH
    )
    T.check(not board_nostay2.is_foul(), "AA/KKK/QQQ-JJ: フォールではない")
    T.check(not board_nostay2.can_stay_fl(), "AA top + FH bot → NO FL Stay",
            f"can_stay={board_nostay2.can_stay_fl()}")

    # --- 5-5. フォール時は FL Entry できない ---
    print("\n  [5-5] フォール時は FL Entry できない")
    board_foul_fl = make_board(
        [ci(SPADE, A), ci(HEART, A), ci(DIAMOND, A)],           # Top: AAA (Trips)
        [ci(SPADE, K), ci(HEART, K), ci(DIAMOND, TWO), ci(CLUB, THREE), ci(CLUB, FOUR)],  # Mid: KK (Pair)
        [ci(SPADE, Q), ci(HEART, Q), ci(DIAMOND, Q), ci(CLUB, Q), ci(HEART, TEN)]         # Bot: QQQQ
    )
    T.check(board_foul_fl.is_foul(), "AAA top > KK mid → フォール")
    T.check(not board_foul_fl.qualifies_for_fl(), "フォール時は FL Entry できない")


# ============================================================
# 6. スコア計算テスト
# ============================================================

def test_score_and_royalties():
    T.section("6. スコア計算・ロイヤリティテスト")

    def make_board(top_cards, mid_cards, bot_cards):
        board = ofc.Board()
        for c in top_cards:
            board.place_card(ofc.Row.TOP, ofc.Card(c))
        for c in mid_cards:
            board.place_card(ofc.Row.MIDDLE, ofc.Card(c))
        for c in bot_cards:
            board.place_card(ofc.Row.BOTTOM, ofc.Card(c))
        return board

    # --- 6-1. Top ロイヤリティ: 66=1, 77=2, ..., AA=9, 222=10, ..., AAA=22 ---
    print("\n  [6-1] Top ロイヤリティ (ペア)")

    # 66 → 1pt
    board_66 = make_board(
        [ci(SPADE, SIX), ci(HEART, SIX), ci(DIAMOND, TWO)],
        [ci(SPADE, THREE), ci(HEART, THREE), ci(DIAMOND, THREE), ci(CLUB, FOUR), ci(CLUB, FIVE)],
        [ci(SPADE, K), ci(HEART, K), ci(DIAMOND, K), ci(CLUB, K), ci(HEART, TEN)]
    )
    T.check(not board_66.is_foul(), "66/Trips3/QuadsK: フォールではない")
    # 55 以下はロイヤリティなし
    board_55 = make_board(
        [ci(SPADE, FIVE), ci(HEART, FIVE), ci(DIAMOND, TWO)],
        [ci(SPADE, THREE), ci(HEART, THREE), ci(DIAMOND, THREE), ci(CLUB, FOUR), ci(CLUB, SIX)],
        [ci(SPADE, K), ci(HEART, K), ci(DIAMOND, K), ci(CLUB, K), ci(HEART, TEN)]
    )

    royalty_66 = board_66.calculate_royalties()
    royalty_55 = board_55.calculate_royalties()
    T.check(royalty_66 > royalty_55, "66 on Top のロイヤリティ > 55 on Top",
            f"66={royalty_66}, 55={royalty_55}")

    # AA → 9pt (Top)
    board_aa_top = make_board(
        [ci(SPADE, A), ci(HEART, A), ci(DIAMOND, TWO)],
        [ci(SPADE, THREE), ci(HEART, THREE), ci(DIAMOND, THREE), ci(CLUB, FOUR), ci(CLUB, FIVE)],
        [ci(SPADE, K), ci(HEART, K), ci(DIAMOND, K), ci(CLUB, K), ci(HEART, TEN)]
    )
    T.check(not board_aa_top.is_foul(), "AA/Trips3/QuadsK: フォールではない")
    royalty_aa = board_aa_top.calculate_royalties()
    T.check(royalty_aa > royalty_66, "AA on Top のロイヤリティ > 66 on Top",
            f"AA={royalty_aa}, 66={royalty_66}")

    # --- 6-2. Mid/Bot ロイヤリティ ---
    print("\n  [6-2] フォール時はロイヤリティ 0")
    board_foul = make_board(
        [ci(SPADE, A), ci(HEART, A), ci(DIAMOND, A)],  # Trips
        [ci(SPADE, K), ci(HEART, K), ci(DIAMOND, TWO), ci(CLUB, THREE), ci(CLUB, FOUR)],  # Pair
        [ci(SPADE, Q), ci(HEART, Q), ci(DIAMOND, Q), ci(CLUB, J), ci(HEART, J)]  # FH
    )
    T.check(board_foul.is_foul(), "Trips top > Pair mid → フォール")
    T.check(board_foul.calculate_royalties() == 0, "フォール時ロイヤリティ = 0",
            f"got {board_foul.calculate_royalties()}")

    # --- 6-3. ボードが完成していない場合 ---
    print("\n  [6-3] 未完成ボード")
    board_incomplete = ofc.Board()
    board_incomplete.place_card(ofc.Row.TOP, ofc.Card(ci(SPADE, A)))
    T.check(not board_incomplete.is_complete(), "1枚配置 → 未完成")
    T.check(not board_incomplete.is_foul(), "未完成ボードはフォールではない")


# ============================================================
# 7. フォール判定テスト
# ============================================================

def test_foul_detection():
    T.section("7. フォール判定テスト")

    def make_board(top_cards, mid_cards, bot_cards):
        board = ofc.Board()
        for c in top_cards:
            board.place_card(ofc.Row.TOP, ofc.Card(c))
        for c in mid_cards:
            board.place_card(ofc.Row.MIDDLE, ofc.Card(c))
        for c in bot_cards:
            board.place_card(ofc.Row.BOTTOM, ofc.Card(c))
        return board

    # --- 7-1. 正常: Bot >= Mid >= Top ---
    print("\n  [7-1] 正常ボード: Bot >= Mid >= Top")

    # HC < Pair < Trips
    board_valid1 = make_board(
        [ci(SPADE, TWO), ci(HEART, THREE), ci(DIAMOND, FOUR)],     # HC
        [ci(SPADE, K), ci(HEART, K), ci(DIAMOND, TWO), ci(CLUB, THREE), ci(CLUB, FOUR)],  # Pair
        [ci(SPADE, A), ci(HEART, A), ci(DIAMOND, A), ci(CLUB, FIVE), ci(HEART, SIX)]      # Trips
    )
    T.check(not board_valid1.is_foul(), "HC top < Pair mid < Trips bot → Valid")

    # Pair < Pair (強) < Two Pair
    board_valid2 = make_board(
        [ci(SPADE, TWO), ci(HEART, TWO), ci(DIAMOND, THREE)],     # Pair 22
        [ci(SPADE, K), ci(HEART, K), ci(DIAMOND, Q), ci(CLUB, Q), ci(CLUB, FOUR)],  # Two Pair KK-QQ
        [ci(SPADE, A), ci(HEART, A), ci(DIAMOND, A), ci(CLUB, A), ci(HEART, SIX)]   # Quads AAAA
    )
    T.check(not board_valid2.is_foul(), "22 < KK-QQ < AAAA → Valid")

    # --- 7-2. フォール: Top > Mid ---
    print("\n  [7-2] フォール: Top > Mid")
    board_foul1 = make_board(
        [ci(SPADE, A), ci(HEART, A), ci(DIAMOND, TWO)],           # AA (Pair)
        [ci(SPADE, K), ci(HEART, TWO), ci(DIAMOND, THREE), ci(CLUB, FOUR), ci(CLUB, FIVE)],  # K-high (HC)
        [ci(SPADE, Q), ci(HEART, Q), ci(DIAMOND, Q), ci(CLUB, Q), ci(HEART, TEN)]            # QQQQ
    )
    T.check(board_foul1.is_foul(), "AA top > K-high mid → フォール")

    # --- 7-3. フォール: Mid > Bot ---
    print("\n  [7-3] フォール: Mid > Bot")
    board_foul2 = make_board(
        [ci(SPADE, TWO), ci(HEART, THREE), ci(DIAMOND, FOUR)],     # HC
        [ci(SPADE, A), ci(HEART, A), ci(DIAMOND, A), ci(CLUB, K), ci(HEART, K)],  # FH AAA-KK
        [ci(SPADE, Q), ci(HEART, Q), ci(DIAMOND, Q), ci(CLUB, J), ci(HEART, TEN)]  # Trips QQQ
    )
    T.check(board_foul2.is_foul(), "FH mid > Trips bot → フォール")

    # --- 7-4. 境界: 同じ役の比較でフォール ---
    print("\n  [7-4] 境界: 同じ役でもランクが逆ならフォール")
    # Top: KK, Mid: QQ → Top > Mid → Foul
    board_foul3 = make_board(
        [ci(SPADE, K), ci(HEART, K), ci(DIAMOND, TWO)],
        [ci(SPADE, Q), ci(HEART, Q), ci(DIAMOND, THREE), ci(CLUB, FOUR), ci(CLUB, FIVE)],
        [ci(SPADE, A), ci(HEART, A), ci(DIAMOND, A), ci(CLUB, A), ci(HEART, TEN)]
    )
    T.check(board_foul3.is_foul(), "KK top > QQ mid (同Pair) → フォール")

    # --- 7-5. 同じ役・同じランク → フォールではない ---
    print("\n  [7-5] 同じ役・同じランク → フォールではない")
    board_equal = make_board(
        [ci(SPADE, K), ci(HEART, K), ci(DIAMOND, TWO)],           # KK
        [ci(DIAMOND, K), ci(CLUB, K), ci(DIAMOND, THREE), ci(CLUB, FOUR), ci(CLUB, FIVE)],  # KK (Mid)
        [ci(SPADE, A), ci(HEART, A), ci(DIAMOND, A), ci(CLUB, A), ci(HEART, TEN)]           # AAAA
    )
    # Mid(KK) >= Top(KK) should be true
    T.check(not board_equal.is_foul(), "KK top == KK mid → Valid (等しい場合はフォールではない)",
            f"foul={board_equal.is_foul()}")


# ============================================================
# 8. 3人対戦のスコア計算
# ============================================================

def test_three_player_scoring():
    T.section("8. 3人対戦のスコア計算")

    print("\n  [8-1] GameEngine を使った3人対戦スコア")

    engine = ofc.GameEngine(3)
    engine.start_new_game(12345)

    # エンジンが正常に初期化されるか
    T.check(engine.num_players() == 3, "3人対戦エンジンの初期化")

    # フェーズ確認
    phase = engine.phase()
    T.check(phase is not None, "ゲームフェーズが取得できる",
            f"phase={phase}")

    # プレイヤー状態
    for i in range(3):
        ps = engine.player(i)
        T.check(ps is not None, f"Player {i} の状態が取得できる")

    print("\n  [8-2] ゼロサム確認: 3人のスコア合計は 0 に近い")
    # 完全なゲームを手動で進めるのは複雑なので、
    # 環境を使ったテストは evaluate_benchmark.py に委任
    print("  (環境を使ったスコア計算テストは evaluate_benchmark.py で実施)")


# ============================================================
# メイン実行
# ============================================================

if __name__ == "__main__":
    print("OFC Pineapple AI - C++ エンジン網羅テスト")
    print("=" * 60)

    test_ace_comparison()
    test_hand_rank_ordering()
    test_straight_boundaries()
    test_joker_substitution()
    test_fl_conditions()
    test_score_and_royalties()
    test_foul_detection()
    test_three_player_scoring()

    success = T.summary()
    sys.exit(0 if success else 1)
