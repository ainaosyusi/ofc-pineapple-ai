/**
 * OFC Pineapple AI - Hand Evaluator
 *
 * OFC用の役判定エンジン。
 * Top(3枚)とMid/Bottom(5枚)の役を高速に判定し、ロイヤリティを計算。
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
  uint32_t kickers; // 比較用キッカー情報

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
// ロイヤリティ計算（JOPT準拠）
// ============================================

// Top (3枚) のロイヤリティ
inline int calculate_top_royalty(HandRank rank, Rank pair_rank) {
  if (rank == THREE_OF_A_KIND) {
    // トリップス: 10点(222) ~ 22点(AAA)
    // ACE=0なので特殊処理: AAA = 10 + 12 = 22点
    if (pair_rank == ACE)
      return 22;
    return 10 + static_cast<int>(pair_rank);
  }
  if (rank == ONE_PAIR) {
    // ペア 66以上: 1点(66) ~ 9点(AA)
    // ACE=0なので特殊処理: AA = 9点
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
      return high; // 最高ランク
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
        return HandValue(STRAIGHT_FLUSH, high);
    }
    // Wheel SF
    int p_wheel = (suit_counts[ACE] > 0) + (suit_counts[TWO] > 0) +
                  (suit_counts[THREE] > 0) + (suit_counts[FOUR] > 0) +
                  (suit_counts[FIVE] > 0);
    if (p_wheel + num_jokers >= 5)
      return HandValue(STRAIGHT_FLUSH, FIVE);
  }

  // 2. フォーカード
  for (int r = NUM_RANKS - 1; r >= 0; --r) {
    if (counts[r] + num_jokers >= 4)
      return HandValue(FOUR_OF_A_KIND, r);
  }

  // 3. フルハウス
  // トリップス + ペア
  for (int r1 = NUM_RANKS - 1; r1 >= 0; --r1) {
    for (int r2 = NUM_RANKS - 1; r2 >= 0; --r2) {
      if (r1 == r2)
        continue;
      // 必要枚数 = (3 - counts[r1]) + (2 - counts[r2])
      int needed = std::max(0, 3 - counts[r1]) + std::max(0, 2 - counts[r2]);
      if (needed <= num_jokers)
        return HandValue(FULL_HOUSE, r1 * 16 + r2);
    }
  }

  // 4. フラッシュ
  for (int s = 0; s < NUM_SUITS; ++s) {
    if (count_cards(normal_cards & SUIT_MASKS[s]) + num_jokers >= 5)
      return HandValue(FLUSH, 0);
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
      return HandValue(STRAIGHT, high);
  }
  int p_wheel = (counts[ACE] > 0) + (counts[TWO] > 0) + (counts[THREE] > 0) +
                (counts[FOUR] > 0) + (counts[FIVE] > 0);
  if (p_wheel + num_jokers >= 5)
    return HandValue(STRAIGHT, FIVE);

  // 6. スリーオブアカインド
  for (int r = NUM_RANKS - 1; r >= 0; --r) {
    if (counts[r] + num_jokers >= 3)
      return HandValue(THREE_OF_A_KIND, r);
  }

  // 7. ツーペア
  // 3枚の非ジョーカーカードのうち、ペア + シングルの場合: (X,X,Y,J,J) は
  // フルハウスになるはずなので、ここに来るのは num_jokers が少ない場合
  for (int r1 = NUM_RANKS - 1; r1 >= 0; --r1) {
    for (int r2 = r1 - 1; r2 >= 0; --r2) {
      int needed = std::max(0, 2 - counts[r1]) + std::max(0, 2 - counts[r2]);
      if (needed <= num_jokers)
        return HandValue(TWO_PAIR, r1);
    }
  }

  // 8. ワンペア
  for (int r = NUM_RANKS - 1; r >= 0; --r) {
    if (counts[r] + num_jokers >= 2)
      return HandValue(ONE_PAIR, r);
  }

  // 9. ハイカード
  for (int r = NUM_RANKS - 1; r >= 0; --r) {
    if (counts[r] > 0)
      return HandValue(HIGH_CARD, r);
  }

  return HandValue(HIGH_CARD, 0);
}

// ============================================
// 3枚役判定（Top用）
// ============================================

inline HandValue evaluate_3card(CardMask hand) {
  using namespace detail;

  int num_jokers = count_cards(hand & JOKER_MASK);
  CardMask normal_cards = hand & ~JOKER_MASK;
  auto counts = count_ranks(normal_cards);

  // トリップス
  for (int r = NUM_RANKS - 1; r >= 0; --r) {
    if (counts[r] + num_jokers >= 3)
      return HandValue(THREE_OF_A_KIND, r);
  }

  // ペア
  for (int r = NUM_RANKS - 1; r >= 0; --r) {
    if (counts[r] + num_jokers >= 2)
      return HandValue(ONE_PAIR, r);
  }

  // ハイカード
  for (int r = NUM_RANKS - 1; r >= 0; --r) {
    if (counts[r] > 0)
      return HandValue(HIGH_CARD, r);
  }

  return HandValue(HIGH_CARD, 0);
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

// FL突入条件: TopがQQ以上
inline bool qualifies_for_fantasy_land(const HandValue &top) {
  if (top.rank == THREE_OF_A_KIND)
    return true;
  if (top.rank == ONE_PAIR) {
    // ACE=0なので特殊処理: AA, KK, QQ がFL条件
    Rank pair_rank = static_cast<Rank>(top.kickers);
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
    Rank pair_rank = static_cast<Rank>(top.kickers);
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
