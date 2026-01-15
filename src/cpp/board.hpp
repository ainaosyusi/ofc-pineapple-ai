/**
 * OFC Pineapple AI - Board Module
 *
 * プレイヤーのボード（Top/Middle/Bottom）を管理。
 * カードの配置とスロット管理。
 */

#ifndef OFC_BOARD_HPP
#define OFC_BOARD_HPP

#include "card.hpp"
#include "evaluator.hpp"
#include <array>
#include <sstream>
#include <string>

namespace ofc {

// ============================================
// 定数
// ============================================

constexpr int TOP_SIZE = 3;
constexpr int MID_SIZE = 5;
constexpr int BOT_SIZE = 5;
constexpr int TOTAL_SLOTS = TOP_SIZE + MID_SIZE + BOT_SIZE; // 13

// ============================================
// Row（行）列挙型
// ============================================

enum Row : uint8_t { TOP = 0, MIDDLE = 1, BOTTOM = 2 };

// 各行の最大カード枚数
constexpr int row_capacity(Row row) {
  switch (row) {
  case TOP:
    return TOP_SIZE;
  case MIDDLE:
    return MID_SIZE;
  case BOTTOM:
    return BOT_SIZE;
  default:
    return 0;
  }
}

// ============================================
// Board クラス
// ============================================

class Board {
public:
  Board() { clear(); }

  // ボードをクリア
  void clear() {
    top_cards_.fill(Card(255));
    mid_cards_.fill(Card(255));
    bot_cards_.fill(Card(255));
    top_count_ = 0;
    mid_count_ = 0;
    bot_count_ = 0;
    all_cards_mask_ = EMPTY_MASK;
  }

  // カードを配置
  bool place_card(Row row, const Card &card) {
    if (!can_place(row))
      return false;
    if (all_cards_mask_ & card.to_mask())
      return false; // 既に配置済み

    switch (row) {
    case TOP:
      top_cards_[top_count_++] = card;
      break;
    case MIDDLE:
      mid_cards_[mid_count_++] = card;
      break;
    case BOTTOM:
      bot_cards_[bot_count_++] = card;
      break;
    }
    all_cards_mask_ |= card.to_mask();
    return true;
  }

  // 配置可能か
  bool can_place(Row row) const {
    switch (row) {
    case TOP:
      return top_count_ < TOP_SIZE;
    case MIDDLE:
      return mid_count_ < MID_SIZE;
    case BOTTOM:
      return bot_count_ < BOT_SIZE;
    default:
      return false;
    }
  }

  // 各行の現在のカード枚数
  int count(Row row) const {
    switch (row) {
    case TOP:
      return top_count_;
    case MIDDLE:
      return mid_count_;
    case BOTTOM:
      return bot_count_;
    default:
      return 0;
    }
  }

  // 各行の空きスロット数
  int remaining_slots(Row row) const { return row_capacity(row) - count(row); }

  // ボードが完成しているか
  bool is_complete() const {
    return top_count_ == TOP_SIZE && mid_count_ == MID_SIZE &&
           bot_count_ == BOT_SIZE;
  }

  // 配置済みカード枚数の合計
  int total_placed() const { return top_count_ + mid_count_ + bot_count_; }

  // 各行のカードマスクを取得
  CardMask top_mask() const {
    CardMask m = 0;
    for (int i = 0; i < top_count_; ++i)
      m |= top_cards_[i].to_mask();
    return m;
  }

  CardMask mid_mask() const {
    CardMask m = 0;
    for (int i = 0; i < mid_count_; ++i)
      m |= mid_cards_[i].to_mask();
    return m;
  }

  CardMask bot_mask() const {
    CardMask m = 0;
    for (int i = 0; i < bot_count_; ++i)
      m |= bot_cards_[i].to_mask();
    return m;
  }

  // 全カードのマスク
  CardMask all_mask() const { return all_cards_mask_; }

  // 役判定
  HandValue evaluate_top() const {
    if (top_count_ < TOP_SIZE)
      return HandValue();
    return evaluate_3card(top_mask());
  }

  HandValue evaluate_mid() const {
    if (mid_count_ < MID_SIZE)
      return HandValue();
    return evaluate_5card(mid_mask());
  }

  HandValue evaluate_bot() const {
    if (bot_count_ < BOT_SIZE)
      return HandValue();
    return evaluate_5card(bot_mask());
  }

  // ファウル判定
  bool is_foul() const {
    if (!is_complete())
      return false;
    return ofc::is_foul(evaluate_top(), evaluate_mid(), evaluate_bot());
  }

  // ロイヤリティ計算
  int calculate_royalties() const {
    if (!is_complete() || is_foul())
      return 0;

    int total = 0;

    // Top
    HandValue top_val = evaluate_top();
    if (top_val.rank == THREE_OF_A_KIND) {
      total += 10 + top_val.kickers; // 222=10, ..., AAA=22
    } else if (top_val.rank == ONE_PAIR) {
      Rank pair_rank = static_cast<Rank>(top_val.kickers);
      if (pair_rank >= SIX) {
        total += static_cast<int>(pair_rank) - static_cast<int>(SIX) + 1;
      }
    }

    // Middle
    total += calculate_middle_royalty(evaluate_mid().rank);

    // Bottom
    total += calculate_bottom_royalty(evaluate_bot().rank);

    return total;
  }

  // ファンタジーランド突入判定
  bool qualifies_for_fl() const {
    if (!is_complete() || is_foul())
      return false;
    return qualifies_for_fantasy_land(evaluate_top());
  }

  // FL配布枚数
  int fl_card_count() const {
    if (!qualifies_for_fl())
      return 0;
    return fantasy_land_cards(evaluate_top());
  }

  // FL継続判定
  bool can_stay_fl() const {
    if (!is_complete() || is_foul())
      return false;
    return can_stay_in_fantasy_land(evaluate_top(), evaluate_bot());
  }

  // デバッグ用文字列出力
  std::string to_string() const {
    std::ostringstream ss;

    ss << "Top:    [";
    for (int i = 0; i < TOP_SIZE; ++i) {
      if (i > 0)
        ss << " ";
      if (i < top_count_)
        ss << card_to_string(top_cards_[i]);
      else
        ss << "__";
    }
    ss << "]\n";

    ss << "Middle: [";
    for (int i = 0; i < MID_SIZE; ++i) {
      if (i > 0)
        ss << " ";
      if (i < mid_count_)
        ss << card_to_string(mid_cards_[i]);
      else
        ss << "__";
    }
    ss << "]\n";

    ss << "Bottom: [";
    for (int i = 0; i < BOT_SIZE; ++i) {
      if (i > 0)
        ss << " ";
      if (i < bot_count_)
        ss << card_to_string(bot_cards_[i]);
      else
        ss << "__";
    }
    ss << "]";

    return ss.str();
  }

  // カード配列へのアクセス
  const std::array<Card, TOP_SIZE> &top_cards() const { return top_cards_; }
  const std::array<Card, MID_SIZE> &mid_cards() const { return mid_cards_; }
  const std::array<Card, BOT_SIZE> &bot_cards() const { return bot_cards_; }

  // ============================================
  // シリアライズ用：マスクから状態を復元
  // ============================================

  void restore_from_masks(CardMask top_mask, CardMask mid_mask,
                          CardMask bot_mask) {
    clear();

    // マスクからカードを復元
    for (uint8_t i = 0; i < NUM_CARDS && top_count_ < TOP_SIZE; ++i) {
      Card c(i);
      if (top_mask & c.to_mask()) {
        top_cards_[top_count_++] = c;
        all_cards_mask_ |= c.to_mask();
      }
    }

    for (uint8_t i = 0; i < NUM_CARDS && mid_count_ < MID_SIZE; ++i) {
      Card c(i);
      if (mid_mask & c.to_mask()) {
        mid_cards_[mid_count_++] = c;
        all_cards_mask_ |= c.to_mask();
      }
    }

    for (uint8_t i = 0; i < NUM_CARDS && bot_count_ < BOT_SIZE; ++i) {
      Card c(i);
      if (bot_mask & c.to_mask()) {
        bot_cards_[bot_count_++] = c;
        all_cards_mask_ |= c.to_mask();
      }
    }
  }

private:
  std::array<Card, TOP_SIZE> top_cards_;
  std::array<Card, MID_SIZE> mid_cards_;
  std::array<Card, BOT_SIZE> bot_cards_;

  int top_count_;
  int mid_count_;
  int bot_count_;

  CardMask all_cards_mask_;
};

} // namespace ofc

#endif // OFC_BOARD_HPP
