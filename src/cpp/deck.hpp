/**
 * OFC Pineapple AI - Deck Module
 *
 * デッキ管理とシャッフル機能。
 * 高速な乱数生成でゲームシミュレーションを効率化。
 */

#ifndef OFC_DECK_HPP
#define OFC_DECK_HPP

#include "card.hpp"
#include <algorithm>
#include <array>
#include <random>

namespace ofc {

// ============================================
// Xoroshiro128+ 高速乱数生成器
// ============================================

class FastRNG {
public:
  using result_type = uint64_t;

  FastRNG(uint64_t seed = 12345) {
    state_[0] = splitmix64(seed);
    state_[1] = splitmix64(state_[0]);
  }

  uint64_t operator()() {
    const uint64_t s0 = state_[0];
    uint64_t s1 = state_[1];
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    state_[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    state_[1] = rotl(s1, 37);

    return result;
  }

  // 0からn-1までの一様乱数
  uint32_t next_int(uint32_t n) { return static_cast<uint32_t>((*this)() % n); }

  static constexpr uint64_t min() { return 0; }
  static constexpr uint64_t max() { return UINT64_MAX; }

private:
  uint64_t state_[2];

  static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

  static uint64_t splitmix64(uint64_t &x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
  }
};

// ============================================
// Deck クラス
// ============================================

class Deck {
public:
  Deck() { reset(); }

  // デッキをリセット（全52枚を順番に）
  void reset() {
    for (int i = 0; i < NUM_CARDS; ++i) {
      cards_[i] = Card(static_cast<uint8_t>(i));
    }
    top_ = 0;
    used_mask_ = EMPTY_MASK;
  }

  // Fisher-Yatesシャッフル
  template <typename RNG> void shuffle(RNG &rng) {
    for (int i = NUM_CARDS - 1; i > 0; --i) {
      int j = rng.next_int(i + 1);
      std::swap(cards_[i], cards_[j]);
    }
    top_ = 0;
  }

  // std::default_random_engine用のオーバーロード
  void shuffle(std::mt19937 &rng) {
    std::shuffle(cards_.begin(), cards_.end(), rng);
    top_ = 0;
  }

  // カードを1枚引く
  Card draw() {
    if (top_ >= NUM_CARDS) {
      return Card(255); // 無効なカード
    }
    Card c = cards_[top_++];
    used_mask_ |= c.to_mask();
    return c;
  }

  // 複数枚引く
  template <size_t N> std::array<Card, N> draw_n() {
    std::array<Card, N> result;
    for (size_t i = 0; i < N; ++i) {
      result[i] = draw();
    }
    return result;
  }

  // 残りのカード枚数
  int remaining() const { return NUM_CARDS - top_; }

  // 使用済みカードのマスク
  CardMask used_mask() const { return used_mask_; }

  // 残っているカードのマスク
  CardMask remaining_mask() const { return FULL_DECK & ~used_mask_; }

  // 特定のカードがデッキに残っているか
  bool contains(const Card &card) const {
    return (used_mask_ & card.to_mask()) == 0 &&
           card.index >= top_; // 簡易チェック
  }

  // デバッグ用: デッキの状態を文字列で取得
  std::string to_string() const {
    std::string s = "Deck[";
    for (int i = top_; i < NUM_CARDS && i < top_ + 10; ++i) {
      if (i > top_)
        s += " ";
      s += card_to_string(cards_[i]);
    }
    if (remaining() > 10)
      s += " ...";
    s += "] (" + std::to_string(remaining()) + " cards)";
    return s;
  }

  // ============================================
  // シリアライズ（MCTS用状態保存）
  // ============================================

  // 状態をバイト列にシリアライズ
  std::vector<uint8_t> serialize() const {
    std::vector<uint8_t> data;
    data.reserve(60);

    // top_インデックス
    data.push_back(static_cast<uint8_t>(top_));

    // 残りカード（順番が重要）
    for (int i = top_; i < NUM_CARDS; ++i) {
      data.push_back(cards_[i].index);
    }

    return data;
  }

  // バイト列から状態を復元
  bool deserialize(const std::vector<uint8_t> &data) {
    if (data.empty())
      return false;

    top_ = data[0];
    used_mask_ = EMPTY_MASK;

    // 使用済みカードのマスクを再計算
    for (int i = 0; i < top_; ++i) {
      // シリアライズ時に保存されていないカードは使用済み
      if (i < static_cast<int>(data.size()) - 1) {
        used_mask_ |= Card(data[i + 1]).to_mask();
      }
    }

    // 残りカードを復元
    for (size_t i = 1; i < data.size() && (top_ + i - 1) < NUM_CARDS; ++i) {
      cards_[top_ + i - 1] = Card(data[i]);
    }

    // 使用済みカードのマスクを計算
    used_mask_ = EMPTY_MASK;
    for (int i = 0; i < top_; ++i) {
      used_mask_ |= cards_[i].to_mask();
    }

    return true;
  }

private:
  std::array<Card, NUM_CARDS> cards_;
  int top_;            // 次に引くカードのインデックス
  CardMask used_mask_; // 既に使用されたカードのビットマスク
};

} // namespace ofc

#endif // OFC_DECK_HPP
