/**
 * OFC Pineapple AI - Fantasy Land Solver (Extreme Performance)
 */

#ifndef OFC_SOLVER_HPP
#define OFC_SOLVER_HPP

#include "board.hpp"
#include "card.hpp"
#include "evaluator.hpp"
#include <algorithm>
#include <vector>

namespace ofc {

struct FantasySolution {
  std::vector<Card> top;
  std::vector<Card> mid;
  std::vector<Card> bot;
  std::vector<Card> discards;
  float score = -100.0f;
  bool stayed = false;
};

class FantasySolver {
public:
  static FantasySolution solve(const std::vector<Card> &cards,
                               bool already_in_fl = true) {
    FantasySolver solver(cards, already_in_fl);
    return solver.run();
  }

private:
  std::vector<Card> cards_;
  CardMask card_masks_[17];
  int n_;
  bool already_in_fl_;
  const float STAY_BONUS = 15.0f;

  float best_score_ = -100.0f;
  int best_assignment_[17];
  bool best_stayed_ = false;

  FantasySolver(const std::vector<Card> &cards, bool in_fl)
      : cards_(cards), already_in_fl_(in_fl) {
    n_ = static_cast<int>(cards_.size());
    for (int i = 0; i < n_; ++i)
      card_masks_[i] = cards_[i].to_mask();
  }

  FantasySolution run() {
    // 高得点を見つけやすくするために降順ソート
    std::sort(cards_.begin(), cards_.end(),
              [](const Card &a, const Card &b) { return a.index > b.index; });
    for (int i = 0; i < n_; ++i)
      card_masks_[i] = cards_[i].to_mask();

    int assignment[17];
    search(0, 0, 0, 0, 0, 0, 0, HandValue(), HandValue(), assignment);

    FantasySolution sol;
    sol.score = best_score_;
    sol.stayed = best_stayed_;
    CardMask all_mask = 0;
    for (int i = 0; i < n_; ++i) {
      all_mask |= card_masks_[i];
      if (best_assignment_[i] == 0)
        sol.bot.push_back(cards_[i]);
      else if (best_assignment_[i] == 1)
        sol.mid.push_back(cards_[i]);
      else if (best_assignment_[i] == 2)
        sol.top.push_back(cards_[i]);
      else
        sol.discards.push_back(cards_[i]);
    }
    return sol;
  }

  // 171M states max. Each choice is 4-way. Pruning is key.
  void search(int idx, int b_cnt, int m_cnt, int t_cnt, CardMask b_m,
              CardMask m_m, CardMask t_m, HandValue b_v, HandValue m_v,
              int assignment[]) {
    if (idx == n_) {
      HandValue t_v = evaluate_3card(t_m);
      if (b_v < m_v || m_v < t_v)
        return;

      int r = calculate_bottom_royalty(b_v.rank) +
              calculate_middle_royalty(m_v.rank) +
              calculate_top_royalty(t_v.rank, static_cast<Rank>(t_v.kickers));

      float s = static_cast<float>(r);
      bool stayed = false;
      if (already_in_fl_) {
        stayed = can_stay_in_fantasy_land(t_v, b_v);
        if (stayed)
          s += STAY_BONUS;
      }

      if (s > best_score_) {
        best_score_ = s;
        best_stayed_ = stayed;
        for (int i = 0; i < n_; ++i)
          best_assignment_[i] = assignment[i];
      }
      return;
    }

    int rem = n_ - idx;

    // 必須枚数チェック
    if (b_cnt + rem < 5 || m_cnt + rem < 5 || t_cnt + rem < 3)
      return;

    // スコア枝刈り（簡易版）
    // TODO: もし現在のスコア + 最大ロイヤリティ < best_score なら return

    // 試行順序
    // Bottom
    if (b_cnt < 5) {
      assignment[idx] = 0;
      HandValue next_bv = b_v;
      if (b_cnt == 4)
        next_bv = evaluate_5card(b_m | card_masks_[idx]);
      search(idx + 1, b_cnt + 1, m_cnt, t_cnt, b_m | card_masks_[idx], m_m, t_m,
             next_bv, m_v, assignment);
    }
    // Middle
    if (m_cnt < 5) {
      assignment[idx] = 1;
      HandValue next_mv = m_v;
      if (m_cnt == 4) {
        next_mv = evaluate_5card(m_m | card_masks_[idx]);
        if (b_cnt == 5 && b_v < next_mv)
          goto try_top;
      }
      search(idx + 1, b_cnt, m_cnt + 1, t_cnt, b_m, m_m | card_masks_[idx], t_m,
             b_v, next_mv, assignment);
    }
  try_top:
    // Top
    if (t_cnt < 3) {
      assignment[idx] = 2;
      if (t_cnt == 2) {
        HandValue tv = evaluate_3card(t_m | card_masks_[idx]);
        if (m_cnt == 5 && m_v < tv)
          goto try_discard;
      }
      search(idx + 1, b_cnt, m_cnt, t_cnt + 1, b_m, m_m, t_m | card_masks_[idx],
             b_v, m_v, assignment);
    }
  try_discard:
    // Discard
    int d_cnt = idx - b_cnt - m_cnt - t_cnt;
    if (d_cnt < (n_ - 13)) {
      assignment[idx] = 3;
      search(idx + 1, b_cnt, m_cnt, t_cnt, b_m, m_m, t_m, b_v, m_v, assignment);
    }
  }
};

} // namespace ofc

#endif // OFC_SOLVER_HPP
