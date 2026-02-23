/**
 * OFC Pineapple AI - Hand Evaluator
 *
 * OFC用の役判定エンジン。
 * Top(3枚)とMid/Bottom(5枚)の役を高速に判定し、ロイヤリティを計算。
 *
 * ランク比較値 (cmp_rank):
 *   ACE=14(最強), KING=13, QUEEN=12, ..., TWO=2
 *   内部表現(ACE=0)とは異なり、正しいポーカーの強さ順を反映。
 */

#ifndef OFC_EVALUATOR_HPP
#define OFC_EVALUATOR_HPP

#include "card.hpp"
#include <algorithm>
#include <array>

namespace ofc {

// ============================================
// 役の種類（HandRank）
// ============================================

enum HandRank : uint8_t {
  HIGH_CARD = 0,
  ONE_PAIR = 1,
  TWO_PAIR = 2,
  THREE_OF_A_KIND = 3,
  STRAIGHT = 4,
  FLUSH = 5,
  FULL_HOUSE = 6,
  FOUR_OF_A_KIND = 7,
  STRAIGHT_FLUSH = 8,
  ROYAL_FLUSH = 9
};

// ============================================
// HandValue - 役の値を表現
// ============================================

struct HandValue {
  HandRank rank;
  uint32_t kickers; // 比較用キッカー情報 (cmp_rank値を使用)

  HandValue() : rank(HIGH_CARD), kickers(0) {}
  HandValue(HandRank r, uint32_t k = 0) : rank(r), kickers(k) {}

  // 比較演算子（役の強さ比較用）
  bool operator<(const HandValue &other) const {
    if (rank != other.rank)
      return rank < other.rank;
    return kickers < other.kickers;
  }
  bool operator>(const HandValue &other) const { return other < *this; }
  bool operator<=(const HandValue &other) const { return !(other < *this); }
  bool operator>=(const HandValue &other) const { return !(*this < other); }
  bool operator==(const HandValue &other) const {
    return rank == other.rank && kickers == other.kickers;
  }
};

// ============================================
// ランク比較ヘルパー
// ============================================

namespace detail {

// 内部ランク(ACE=0, TWO=1, ..., KING=12) → 比較値(ACE=14, TWO=2, ..., KING=13)
inline constexpr int to_cmp_rank(int raw_rank) {
  return raw_rank == 0 ? 14 : raw_rank + 1;
}

// 比較値 → 内部ランク
inline constexpr Rank from_cmp_rank(int cmp_rank) {
  return static_cast<Rank>(cmp_rank == 14 ? 0 : cmp_rank - 1);
}

// 降順反復用: ACE(0)が最強なので最初、次にKING(12), QUEEN(11), ..., TWO(1)
inline constexpr std::array<int, NUM_RANKS> make_rank_desc_order() {
  return {0, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
}
static constexpr auto RANK_DESC = make_rank_desc_order();

// kicker値をパック: 最大5つの比較値を1つのuint32_tに (各4bit, MSB=最重要)
inline uint32_t pack5(int a, int b = 0, int c = 0, int d = 0, int e = 0) {
  return (uint32_t(a) << 16) | (uint32_t(b) << 12) | (uint32_t(c) << 8) |
         (uint32_t(d) << 4) | uint32_t(e);
}

} // namespace detail

// ============================================
// ロイヤリティ計算（JOPT準拠）
// ============================================

// Top (3枚) のロイヤリティ
// pair_rank: 内部ランク値 (ACE=0, TWO=1, ..., KING=12)
inline int calculate_top_royalty(HandRank rank, Rank pair_rank) {
  if (rank == THREE_OF_A_KIND) {
    // トリップス: 10点(222) ~ 22点(AAA)
    if (pair_rank == ACE)
      return 22;
    return 10 + static_cast<int>(pair_rank) - 1; // TWO=1→10, THREE=2→11, ..., KING=12→21
  }
  if (rank == ONE_PAIR) {
    // ペア 66以上: 1点(66) ~ 9点(AA)
    if (pair_rank == ACE)
      return 9;
    if (pair_rank >= SIX) {
      return static_cast<int>(pair_rank) - static_cast<int>(SIX) + 1;
    }
  }
  return 0;
}

// Middle (5枚) のロイヤリティ
inline int calculate_middle_royalty(HandRank rank) {
  switch (rank) {
  case THREE_OF_A_KIND:
    return 2;
  case STRAIGHT:
    return 4;
  case FLUSH:
    return 8;
  case FULL_HOUSE:
    return 12;
  case FOUR_OF_A_KIND:
    return 20;
  case STRAIGHT_FLUSH:
    return 30;
  case ROYAL_FLUSH:
    return 50;
  default:
    return 0;
  }
}

// Bottom (5枚) のロイヤリティ
inline int calculate_bottom_royalty(HandRank rank) {
  switch (rank) {
  case STRAIGHT:
    return 2;
  case FLUSH:
    return 4;
  case FULL_HOUSE:
    return 6;
  case FOUR_OF_A_KIND:
    return 10;
  case STRAIGHT_FLUSH:
    return 15;
  case ROYAL_FLUSH:
    return 25;
  default:
    return 0;
  }
}

// ============================================
// 役判定ヘルパー関数
// ============================================

namespace detail {

// ランクごとのカード枚数をカウント
inline std::array<int, NUM_RANKS> count_ranks(CardMask hand) {
  std::array<int, NUM_RANKS> counts{};
  for (int r = 0; r < NUM_RANKS; ++r) {
    counts[r] = count_cards(hand & RANK_MASKS[r]);
  }
  return counts;
}

// フラッシュ判定（5枚以上同スート）
inline bool is_flush(CardMask hand) {
  for (int s = 0; s < NUM_SUITS; ++s) {
    if (count_cards(hand & SUIT_MASKS[s]) >= 5) {
      return true;
    }
  }
  return false;
}

// フラッシュのスートを取得
inline Suit get_flush_suit(CardMask hand) {
  for (int s = 0; s < NUM_SUITS; ++s) {
    if (count_cards(hand & SUIT_MASKS[s]) >= 5) {
      return static_cast<Suit>(s);
    }
  }
  return SPADE; // デフォルト
}

// ストレート判定（5連続）
// 戻り値: ストレートの最高ランク（なければ-1）
// 注意: ACE=0, TWO=1, ..., KING=12
inline int check_straight(const std::array<int, NUM_RANKS> &counts) {
  // A-high straight (T-J-Q-K-A) のチェック
  if (counts[ACE] > 0 && counts[KING] > 0 && counts[QUEEN] > 0 &&
      counts[JACK] > 0 && counts[TEN] > 0) {
    return 14; // A-high (Aceを14として返す)
  }

  // 通常のストレート (2から始まる連続)
  // K(12)から2(1)まで逆順にチェック
  for (int high = KING; high >= FIVE; --high) {
    bool is_straight = true;
    for (int i = 0; i < 5; ++i) {
      if (counts[high - i] == 0) {
        is_straight = false;
        break;
      }
    }
    if (is_straight) {
      return high; // 最高ランク (内部値)
    }
  }

  // A-2-3-4-5 (wheel) のチェック
  if (counts[ACE] > 0 && counts[TWO] > 0 && counts[THREE] > 0 &&
      counts[FOUR] > 0 && counts[FIVE] > 0) {
    return FIVE; // 5-high straight
  }

  return -1; // ストレートなし
}

// ストレートフラッシュ判定
inline int check_straight_flush(CardMask hand) {
  for (int s = 0; s < NUM_SUITS; ++s) {
    CardMask suit_hand = hand & SUIT_MASKS[s];
    if (count_cards(suit_hand) >= 5) {
      // このスートのカードでストレートチェック
      auto counts = count_ranks(suit_hand);
      int straight_high = check_straight(counts);
      if (straight_high >= 0) {
        return straight_high;
      }
    }
  }
  return -1;
}

// カードの比較ランク値を降順にソートして返す (最大n個)
inline void get_sorted_cmp_ranks(const std::array<int, NUM_RANKS> &counts,
                                  int *out, int max_n, int exclude_rank = -1,
                                  int exclude_rank2 = -1) {
  int idx = 0;
  for (int ii = 0; ii < NUM_RANKS && idx < max_n; ++ii) {
    int r = RANK_DESC[ii];
    if (r == exclude_rank || r == exclude_rank2)
      continue;
    for (int c = 0; c < counts[r] && idx < max_n; ++c) {
      out[idx++] = to_cmp_rank(r);
    }
  }
  while (idx < max_n)
    out[idx++] = 0;
}

} // namespace detail

// ============================================
// 5枚役判定（Mid / Bottom用）
// ============================================

inline HandValue evaluate_5card(CardMask hand) {
  using namespace detail;

  int num_jokers = count_cards(hand & JOKER_MASK);
  CardMask normal_cards = hand & ~JOKER_MASK;
  auto counts = count_ranks(normal_cards);

  // 1. ストレートフラッシュ / ロイヤル
  for (int s = 0; s < NUM_SUITS; ++s) {
    CardMask suit_hand = normal_cards & SUIT_MASKS[s];
    auto suit_counts = count_ranks(suit_hand);

    // A-high (Royal Flush)
    int present = (suit_counts[ACE] > 0) + (suit_counts[KING] > 0) +
                  (suit_counts[QUEEN] > 0) + (suit_counts[JACK] > 0) +
                  (suit_counts[TEN] > 0);
    if (present + num_jokers >= 5)
      return HandValue(ROYAL_FLUSH, 14);

    // Normal SF
    for (int high = KING; high >= FIVE; --high) {
      int p = 0;
      for (int i = 0; i < 5; ++i)
        if (suit_counts[high - i] > 0)
          p++;
      if (p + num_jokers >= 5)
        return HandValue(STRAIGHT_FLUSH, to_cmp_rank(high));
    }
    // Wheel SF
    int p_wheel = (suit_counts[ACE] > 0) + (suit_counts[TWO] > 0) +
                  (suit_counts[THREE] > 0) + (suit_counts[FOUR] > 0) +
                  (suit_counts[FIVE] > 0);
    if (p_wheel + num_jokers >= 5)
      return HandValue(STRAIGHT_FLUSH, to_cmp_rank(FIVE));
  }

  // 2. フォーカード (ACEが最強なのでRANK_DESC順に探索)
  for (int ii = 0; ii < NUM_RANKS; ++ii) {
    int r = RANK_DESC[ii];
    if (counts[r] + num_jokers >= 4) {
      int kickers[1] = {0};
      get_sorted_cmp_ranks(counts, kickers, 1, r);
      return HandValue(FOUR_OF_A_KIND, pack5(to_cmp_rank(r), kickers[0]));
    }
  }

  // 3. フルハウス (トリップス + ペア, ACEが最強順)
  for (int ii = 0; ii < NUM_RANKS; ++ii) {
    int r1 = RANK_DESC[ii]; // trips rank
    for (int jj = 0; jj < NUM_RANKS; ++jj) {
      int r2 = RANK_DESC[jj]; // pair rank
      if (r1 == r2)
        continue;
      int needed = std::max(0, 3 - counts[r1]) + std::max(0, 2 - counts[r2]);
      if (needed <= num_jokers)
        return HandValue(FULL_HOUSE,
                         pack5(to_cmp_rank(r1), to_cmp_rank(r2)));
    }
  }

  // 4. フラッシュ (キッカー情報を保存)
  for (int s = 0; s < NUM_SUITS; ++s) {
    int suit_count = count_cards(normal_cards & SUIT_MASKS[s]);
    if (suit_count + num_jokers >= 5) {
      auto suit_counts = count_ranks(normal_cards & SUIT_MASKS[s]);
      int kickers[5] = {0};
      int ki = 0;
      // Jokerは最強の未使用ランクとして扱う
      for (int j = 0; j < num_jokers && ki < 5; ++j) {
        kickers[ki++] = 15; // any joker > ACE(14)
      }
      for (int jj = 0; jj < NUM_RANKS && ki < 5; ++jj) {
        int r = RANK_DESC[jj];
        if (suit_counts[r] > 0) {
          kickers[ki++] = to_cmp_rank(r);
        }
      }
      return HandValue(FLUSH, pack5(kickers[0], kickers[1], kickers[2],
                                    kickers[3], kickers[4]));
    }
  }

  // 5. ストレート
  // A-high
  int p_ace = (counts[ACE] > 0) + (counts[KING] > 0) + (counts[QUEEN] > 0) +
              (counts[JACK] > 0) + (counts[TEN] > 0);
  if (p_ace + num_jokers >= 5)
    return HandValue(STRAIGHT, 14);
  for (int high = KING; high >= FIVE; --high) {
    int p = 0;
    for (int i = 0; i < 5; ++i)
      if (counts[high - i] > 0)
        p++;
    if (p + num_jokers >= 5)
      return HandValue(STRAIGHT, to_cmp_rank(high));
  }
  int p_wheel = (counts[ACE] > 0) + (counts[TWO] > 0) + (counts[THREE] > 0) +
                (counts[FOUR] > 0) + (counts[FIVE] > 0);
  if (p_wheel + num_jokers >= 5)
    return HandValue(STRAIGHT, to_cmp_rank(FIVE));

  // 6. スリーオブアカインド (ACEが最強順)
  for (int ii = 0; ii < NUM_RANKS; ++ii) {
    int r = RANK_DESC[ii];
    if (counts[r] + num_jokers >= 3) {
      int kickers[2] = {0};
      get_sorted_cmp_ranks(counts, kickers, 2, r);
      return HandValue(THREE_OF_A_KIND,
                       pack5(to_cmp_rank(r), kickers[0], kickers[1]));
    }
  }

  // 7. ツーペア (ACEが最強順)
  for (int ii = 0; ii < NUM_RANKS; ++ii) {
    int r1 = RANK_DESC[ii];
    if (counts[r1] < 2)
      continue;
    for (int jj = ii + 1; jj < NUM_RANKS; ++jj) {
      int r2 = RANK_DESC[jj];
      int needed = std::max(0, 2 - counts[r1]) + std::max(0, 2 - counts[r2]);
      if (needed <= num_jokers) {
        int kickers[1] = {0};
        get_sorted_cmp_ranks(counts, kickers, 1, r1, r2);
        return HandValue(TWO_PAIR, pack5(to_cmp_rank(r1), to_cmp_rank(r2),
                                         kickers[0]));
      }
    }
  }

  // 8. ワンペア (ACEが最強順)
  for (int ii = 0; ii < NUM_RANKS; ++ii) {
    int r = RANK_DESC[ii];
    if (counts[r] + num_jokers >= 2) {
      int kickers[3] = {0};
      get_sorted_cmp_ranks(counts, kickers, 3, r);
      return HandValue(ONE_PAIR,
                       pack5(to_cmp_rank(r), kickers[0], kickers[1], kickers[2]));
    }
  }

  // 9. ハイカード
  {
    int kickers[5] = {0};
    get_sorted_cmp_ranks(counts, kickers, 5);
    return HandValue(HIGH_CARD,
                     pack5(kickers[0], kickers[1], kickers[2], kickers[3],
                           kickers[4]));
  }
}

// ============================================
// 3枚役判定（Top用）
// ============================================

inline HandValue evaluate_3card(CardMask hand) {
  using namespace detail;

  int num_jokers = count_cards(hand & JOKER_MASK);
  CardMask normal_cards = hand & ~JOKER_MASK;
  auto counts = count_ranks(normal_cards);

  // トリップス (ACEが最強順)
  for (int ii = 0; ii < NUM_RANKS; ++ii) {
    int r = RANK_DESC[ii];
    if (counts[r] + num_jokers >= 3)
      return HandValue(THREE_OF_A_KIND, to_cmp_rank(r));
  }

  // ペア (ACEが最強順)
  for (int ii = 0; ii < NUM_RANKS; ++ii) {
    int r = RANK_DESC[ii];
    if (counts[r] + num_jokers >= 2) {
      int kicker[1] = {0};
      get_sorted_cmp_ranks(counts, kicker, 1, r);
      return HandValue(ONE_PAIR, pack5(to_cmp_rank(r), kicker[0]));
    }
  }

  // ハイカード
  {
    int kickers[3] = {0};
    get_sorted_cmp_ranks(counts, kickers, 3);
    return HandValue(HIGH_CARD, pack5(kickers[0], kickers[1], kickers[2]));
  }
}

// ============================================
// ファウル判定
// ============================================

// OFCルール: Bottom >= Middle >= Top でなければファウル
inline bool is_foul(const HandValue &top, const HandValue &mid,
                    const HandValue &bot) {
  return !(bot >= mid && mid >= top);
}

// ============================================
// ファンタジーランド判定
// ============================================

// kickers から内部ランクを取得するヘルパー
// 3枚ハンドのONE_PAIRは pack5(cmp_rank, kicker) なので上位4bitがペアランク
inline Rank get_top_pair_rank(const HandValue &top) {
  int cmp = (top.kickers >> 16) & 0xF;
  return detail::from_cmp_rank(cmp);
}

// 3枚ハンドのTHREE_OF_A_KINDは cmp_rank そのまま
inline Rank get_top_trips_rank(const HandValue &top) {
  return detail::from_cmp_rank(top.kickers);
}

// FL突入条件: TopがQQ以上
inline bool qualifies_for_fantasy_land(const HandValue &top) {
  if (top.rank == THREE_OF_A_KIND)
    return true;
  if (top.rank == ONE_PAIR) {
    Rank pair_rank = get_top_pair_rank(top);
    if (pair_rank == ACE || pair_rank == KING || pair_rank == QUEEN)
      return true;
  }
  return false;
}

// プログレッシブFL: 配布枚数を決定
// QQ=14枚, KK=15枚, AA=16枚, Trips=17枚
inline int fantasy_land_cards(const HandValue &top) {
  if (top.rank == THREE_OF_A_KIND)
    return 17;
  if (top.rank == ONE_PAIR) {
    Rank pair_rank = get_top_pair_rank(top);
    if (pair_rank == ACE)
      return 16; // AA
    if (pair_rank == KING)
      return 15; // KK
    if (pair_rank == QUEEN)
      return 14; // QQ
  }
  return 0; // FL対象外
}

// FL継続条件
inline bool can_stay_in_fantasy_land(const HandValue &top,
                                     const HandValue &bot) {
  // Top: トリップス(222以上)
  if (top.rank == THREE_OF_A_KIND)
    return true;
  // Bottom: フォーカード以上
  if (bot.rank >= FOUR_OF_A_KIND)
    return true;
  return false;
}

} // namespace ofc

#endif // OFC_EVALUATOR_HPP
