/**
 * OFC Pineapple AI - Card Module
 *
 * カードをBitboardで表現するための基本構造体と関数。
 * 52枚のカードを64ビット整数で効率的に管理。
 */

#ifndef OFC_CARD_HPP
#define OFC_CARD_HPP

#include <array>
#include <cstdint>
#include <string>

namespace ofc {

// ============================================
// 定数定義
// ============================================

constexpr int NUM_SUITS = 4;
constexpr int NUM_RANKS = 13;
constexpr int NUM_CARDS = 54; // 52 + Joker 2枚

// ジョーカーのインデックス定義
constexpr uint8_t JOKER_INDEX_A = 52;
constexpr uint8_t JOKER_INDEX_B = 53;

// ============================================
// スートとランクの列挙型
// ============================================

enum Suit : uint8_t {
  SPADE = 0,     // ♠
  HEART = 1,     // ♥
  DIAMOND = 2,   // ♦
  CLUB = 3,      // ♣
  JOKER_SUIT = 4 // ジョーカー用ダミースート
};

enum Rank : uint8_t {
  ACE = 0,
  TWO = 1,
  THREE = 2,
  FOUR = 3,
  FIVE = 4,
  SIX = 5,
  SEVEN = 6,
  EIGHT = 7,
  NINE = 8,
  TEN = 9,
  JACK = 10,
  QUEEN = 11,
  KING = 12,
  JOKER_RANK = 13 // ジョーカー用ダミーランク
};

// ============================================
// CardMask - ビットボードによるカード表現
// ============================================

using CardMask = uint64_t;

// ビット位置: suit * 13 + rank
// 例: A♠ = 0, 2♠ = 1, ..., K♣ = 51

constexpr CardMask EMPTY_MASK = 0ULL;
constexpr CardMask FULL_DECK = (1ULL << 54) - 1; // 下位54ビットすべて1
constexpr CardMask JOKER_MASK =
    (1ULL << JOKER_INDEX_A) | (1ULL << JOKER_INDEX_B);

// 特定のカード1枚をマスクに変換
constexpr CardMask card_to_mask(Suit suit, Rank rank) {
  return 1ULL << (static_cast<int>(suit) * NUM_RANKS + static_cast<int>(rank));
}

// インデックス(0-51)からマスクに変換
constexpr CardMask index_to_mask(int index) { return 1ULL << index; }

// マスクからカード枚数をカウント (popcount)
inline int count_cards(CardMask mask) { return __builtin_popcountll(mask); }

// ============================================
// Card構造体 - 単一カードの表現
// ============================================

struct Card {
  uint8_t index; // 0-51

  constexpr Card() : index(0) {}
  constexpr Card(uint8_t idx) : index(idx) {}
  constexpr Card(Suit suit, Rank rank)
      : index(static_cast<uint8_t>(suit) * NUM_RANKS +
              static_cast<uint8_t>(rank)) {}

  // スートとランクの取得
  constexpr Suit suit() const {
    if (index >= 52)
      return JOKER_SUIT;
    return static_cast<Suit>(index / NUM_RANKS);
  }
  constexpr Rank rank() const {
    if (index >= 52)
      return JOKER_RANK;
    return static_cast<Rank>(index % NUM_RANKS);
  }

  // ビットマスクへの変換
  constexpr CardMask to_mask() const { return 1ULL << index; }

  // 有効なカードか判定
  constexpr bool is_valid() const { return index < NUM_CARDS; }

  // 比較演算子
  constexpr bool operator==(const Card &other) const {
    return index == other.index;
  }
  constexpr bool operator!=(const Card &other) const {
    return index != other.index;
  }
  constexpr bool operator<(const Card &other) const {
    return index < other.index;
  }
};

// ============================================
// 文字列変換ユーティリティ
// ============================================

inline const char *suit_to_char(Suit suit) {
  static const char *suits[] = {"s", "h", "d", "c"};
  return suits[suit];
}

inline const char *suit_to_symbol(Suit suit) {
  static const char *symbols[] = {"♠", "♥", "♦", "♣"};
  return symbols[suit];
}

inline char rank_to_char(Rank rank) {
  static const char ranks[] = "A23456789TJQK";
  return ranks[rank];
}

inline std::string card_to_string(const Card &card) {
  if (card.index >= 52)
    return "JK";
  std::string s;
  s += rank_to_char(card.rank());
  s += suit_to_char(card.suit());
  return s;
}

inline std::string card_to_symbol(const Card &card) {
  if (card.index >= 52)
    return "🃏";
  std::string s;
  s += rank_to_char(card.rank());
  s += suit_to_symbol(card.suit());
  return s;
}

// ============================================
// ランク別マスク（ペア判定用）
// ============================================

// 特定ランクのカード4枚のマスクを取得
constexpr CardMask rank_mask(Rank rank) {
  CardMask m = 0;
  for (int s = 0; s < NUM_SUITS; ++s) {
    m |= 1ULL << (s * NUM_RANKS + static_cast<int>(rank));
  }
  return m;
}

// ランク別マスクのテーブル生成用
namespace detail {
inline CardMask create_rank_mask(Rank rank) {
  CardMask m = 0;
  for (int s = 0; s < NUM_SUITS; ++s) {
    m |= 1ULL << (s * NUM_RANKS + static_cast<int>(rank));
  }
  return m;
}
} // namespace detail

// ランク別マスクのテーブル
const std::array<CardMask, NUM_RANKS> RANK_MASKS = {
    detail::create_rank_mask(ACE),   detail::create_rank_mask(TWO),
    detail::create_rank_mask(THREE), detail::create_rank_mask(FOUR),
    detail::create_rank_mask(FIVE),  detail::create_rank_mask(SIX),
    detail::create_rank_mask(SEVEN), detail::create_rank_mask(EIGHT),
    detail::create_rank_mask(NINE),  detail::create_rank_mask(TEN),
    detail::create_rank_mask(JACK),  detail::create_rank_mask(QUEEN),
    detail::create_rank_mask(KING)};

// ============================================
// スート別マスク（フラッシュ判定用）
// ============================================

constexpr CardMask SUIT_MASK = (1ULL << NUM_RANKS) - 1; // 0x1FFF (下位13ビット)

constexpr CardMask suit_mask(Suit suit) {
  return SUIT_MASK << (static_cast<int>(suit) * NUM_RANKS);
}

constexpr std::array<CardMask, NUM_SUITS> SUIT_MASKS = {
    suit_mask(SPADE), suit_mask(HEART), suit_mask(DIAMOND), suit_mask(CLUB)};

} // namespace ofc

#endif // OFC_CARD_HPP
