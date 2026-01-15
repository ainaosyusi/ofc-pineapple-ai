/**
 * OFC Pineapple AI - Probability Calculator
 *
 * フラッシュ・ストレート等の完成確率を計算。
 * 観測空間の特徴量として使用し、AIの判断を強化する。
 */

#ifndef OFC_PROBABILITY_HPP
#define OFC_PROBABILITY_HPP

#include "board.hpp"
#include "card.hpp"
#include <cmath>

namespace ofc {

// ============================================
// 確率計算クラス
// ============================================

class ProbabilityCalculator {
public:
  /**
   * フラッシュ完成確率を計算
   *
   * @param current_cards 現在の段のカード（マスク）
   * @param current_count 現在のカード枚数
   * @param target_count 目標枚数（5枚）
   * @param remaining_deck 残りデッキ（マスク）
   * @param remaining_count 残りデッキ枚数
   */
  static double flush_probability(CardMask current_cards, int current_count,
                                  int target_count, CardMask remaining_deck,
                                  int remaining_count) {
    if (current_count >= target_count) {
      // 既に完成しているならフラッシュかチェック
      return is_flush_mask(current_cards) ? 1.0 : 0.0;
    }

    int needed = target_count - current_count;
    if (remaining_count < needed)
      return 0.0;

    // 各スートごとに計算
    double best_prob = 0.0;

    for (int suit = 0; suit < NUM_SUITS; ++suit) {
      int suit_in_hand = count_suit(current_cards, static_cast<Suit>(suit));
      int suit_remaining = count_suit(remaining_deck, static_cast<Suit>(suit));

      // 現在のスート枚数 + 残りから引ける枚数 >= 5 か？
      int total_possible = suit_in_hand + suit_remaining;
      if (total_possible < 5)
        continue;

      int need_more = 5 - suit_in_hand;
      if (need_more <= 0) {
        best_prob = 1.0;
        break;
      }

      if (need_more <= needed && suit_remaining >= need_more) {
        // 必要枚数を引ける確率（組み合わせ）
        double prob = combination_probability(
            suit_remaining, need_more, remaining_count - suit_remaining,
            needed - need_more, remaining_count, needed);
        if (prob > best_prob)
          best_prob = prob;
      }
    }

    return best_prob;
  }

  /**
   * ストレート完成確率を計算（簡易版）
   */
  static double straight_probability(CardMask current_cards, int current_count,
                                     CardMask remaining_deck,
                                     int remaining_count) {
    if (remaining_count == 0)
      return 0.0;

    // ランクごとの存在チェック
    std::array<bool, NUM_RANKS> has_rank = {};
    std::array<int, NUM_RANKS> available = {};

    for (int r = 0; r < NUM_RANKS; ++r) {
      for (int s = 0; s < NUM_SUITS; ++s) {
        uint8_t idx = static_cast<uint8_t>(r * 4 + s);
        CardMask mask = 1ULL << idx;
        if (current_cards & mask)
          has_rank[r] = true;
        if (remaining_deck & mask)
          available[r]++;
      }
    }

    // 5連続のパターンをチェック
    double best_prob = 0.0;

    // A-5ストレート（ホイール）
    int wheel_ranks[5] = {0, 1, 2, 3, 4}; // A-5
    double wheel_prob = calculate_straight_prob_for_ranks(
        wheel_ranks, has_rank, available, remaining_count);
    if (wheel_prob > best_prob)
      best_prob = wheel_prob;

    // 通常のストレート
    for (int start = 1; start <= 9; ++start) {
      int ranks[5];
      for (int i = 0; i < 5; ++i) {
        ranks[i] = start + i; // 2-6, 3-7, ... T-A
      }
      double prob = calculate_straight_prob_for_ranks(
          ranks, has_rank, available, remaining_count);
      if (prob > best_prob)
        best_prob = prob;
    }

    return best_prob;
  }

  /**
   * 行ごとの完成状況を数値化
   */
  static double row_completion_score(const Board &board, Row row) {
    int placed = board.count(row);
    int capacity = row_capacity(row);
    return static_cast<double>(placed) / capacity;
  }

private:
  static bool is_flush_mask(CardMask cards) {
    for (int s = 0; s < NUM_SUITS; ++s) {
      int count = count_suit(cards, static_cast<Suit>(s));
      if (count >= 5)
        return true;
    }
    return false;
  }

  static int count_suit(CardMask cards, Suit suit) {
    int count = 0;
    for (int r = 0; r < NUM_RANKS; ++r) {
      uint8_t idx = static_cast<uint8_t>(r * 4 + static_cast<int>(suit));
      if (cards & (1ULL << idx))
        count++;
    }
    return count;
  }

  /**
   * 組み合わせ確率計算
   * 必要なカードを引ける確率
   */
  static double combination_probability(int good_cards, int need_good,
                                        int other_cards, int need_other,
                                        int total_cards, int total_draw) {
    if (need_good < 0 || need_other < 0)
      return 0.0;
    if (good_cards < need_good || other_cards < need_other)
      return 0.0;
    if (total_cards < total_draw)
      return 0.0;

    // 超幾何分布の確率
    // C(good, need_good) * C(other, need_other) / C(total, total_draw)

    double numerator = log_combination(good_cards, need_good) +
                       log_combination(other_cards, need_other);
    double denominator = log_combination(total_cards, total_draw);

    return std::exp(numerator - denominator);
  }

  static double log_combination(int n, int k) {
    if (k < 0 || k > n)
      return -1e9;
    if (k == 0 || k == n)
      return 0;

    // log(C(n,k)) = log(n!) - log(k!) - log((n-k)!)
    double result = 0;
    for (int i = 0; i < k; ++i) {
      result += std::log(n - i) - std::log(i + 1);
    }
    return result;
  }

  static double calculate_straight_prob_for_ranks(
      int ranks[5], const std::array<bool, NUM_RANKS> &has_rank,
      const std::array<int, NUM_RANKS> &available, int remaining_count) {
    int missing = 0;
    double prob = 1.0;

    for (int i = 0; i < 5; ++i) {
      int r = ranks[i];
      if (r < 0 || r >= NUM_RANKS)
        return 0.0;

      if (!has_rank[r]) {
        missing++;
        if (available[r] == 0)
          return 0.0;

        // 残りデッキから引ける確率（独立近似）
        prob *= static_cast<double>(available[r]) / remaining_count;
      }
    }

    return prob;
  }
};

} // namespace ofc

#endif // OFC_PROBABILITY_HPP
