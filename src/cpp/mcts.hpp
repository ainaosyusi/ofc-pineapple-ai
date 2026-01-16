/**
 * OFC Pineapple AI - MCTS Support Module
 *
 * MCTS (Monte Carlo Tree Search) のためのヘルパー関数群。
 * FL突入確率の計算とFantasySolverとの連携を提供。
 */

#ifndef OFC_MCTS_HPP
#define OFC_MCTS_HPP

#include "board.hpp"
#include "card.hpp"
#include "evaluator.hpp"
#include "solver.hpp"
#include "probability.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace ofc {

// ============================================
// FL Expected Value 計算用定数
// ============================================

// FL配布枚数に応じた期待スコア（経験的推定値）
// QQ=14枚, KK=15枚, AA=16枚, Trips=17枚
constexpr float FL_EXPECTED_SCORE_14 = 18.0f;  // QQ: 平均18pt (FL内ロイヤリティ + 対戦勝利)
constexpr float FL_EXPECTED_SCORE_15 = 22.0f;  // KK: 平均22pt
constexpr float FL_EXPECTED_SCORE_16 = 26.0f;  // AA: 平均26pt
constexpr float FL_EXPECTED_SCORE_17 = 32.0f;  // Trips: 平均32pt

// FLステイボーナス (次のハンドでもFLを継続できる価値)
constexpr float FL_STAY_VALUE = 15.0f;

// ============================================
// FL突入確率計算
// ============================================

struct FLProbabilityResult {
    float prob_qq;      // QQ以上でFL突入できる確率
    float prob_kk;      // KK以上でFL突入できる確率
    float prob_aa;      // AA以上でFL突入できる確率
    float prob_trips;   // トリップスでFL突入できる確率
    float total_prob;   // いずれかでFL突入できる確率
    float expected_ev;  // FL突入の期待値 (確率 × 期待スコア)
};

/**
 * TopにQQ+を配置してFL突入できる確率を計算
 *
 * @param board 現在のボード状態
 * @param remaining_deck 残りデッキのカードマスク
 * @param cards_to_draw 残りターン数 × 引くカード数
 * @return FL突入確率と期待値
 */
inline FLProbabilityResult calculate_fl_probability(
    const Board& board,
    CardMask remaining_deck,
    int remaining_turns
) {
    FLProbabilityResult result = {0, 0, 0, 0, 0, 0};

    CardMask top_mask = board.top_mask();
    int top_count = board.count(TOP);
    int top_slots = TOP_SIZE - top_count;

    if (top_slots == 0) {
        // Topが完成済み → FL判定は確定
        HandValue top_val = board.evaluate_top();
        if (top_val.rank == THREE_OF_A_KIND) {
            result.prob_trips = 1.0f;
            result.total_prob = 1.0f;
            result.expected_ev = FL_EXPECTED_SCORE_17;
        } else if (top_val.rank == ONE_PAIR) {
            Rank pair_rank = static_cast<Rank>(top_val.kickers);
            if (pair_rank == ACE) {
                result.prob_aa = 1.0f;
                result.total_prob = 1.0f;
                result.expected_ev = FL_EXPECTED_SCORE_16;
            } else if (pair_rank == KING) {
                result.prob_kk = 1.0f;
                result.total_prob = 1.0f;
                result.expected_ev = FL_EXPECTED_SCORE_15;
            } else if (pair_rank == QUEEN) {
                result.prob_qq = 1.0f;
                result.total_prob = 1.0f;
                result.expected_ev = FL_EXPECTED_SCORE_14;
            }
        }
        return result;
    }

    // 残りカード枚数
    int remaining_count = count_cards(remaining_deck);
    if (remaining_count == 0) return result;

    // Topに既存のカードのランクを確認
    auto counts = detail::count_ranks(top_mask);

    // 簡易的な確率計算（モンテカルロより高速なヒューリスティック）
    //
    // 残りスロット数と残りターン数に基づいて確率を推定
    // cards_to_consider = min(remaining_turns * 3, remaining_count)

    int cards_available = std::min(remaining_turns * 3, remaining_count);

    // 各ランクの残り枚数を計算
    int queens_remaining = 0, kings_remaining = 0, aces_remaining = 0;
    for (int s = 0; s < NUM_SUITS; ++s) {
        if (remaining_deck & card_to_mask(static_cast<Suit>(s), QUEEN)) queens_remaining++;
        if (remaining_deck & card_to_mask(static_cast<Suit>(s), KING)) kings_remaining++;
        if (remaining_deck & card_to_mask(static_cast<Suit>(s), ACE)) aces_remaining++;
    }

    // 既存のペア/トリップスを考慮
    int existing_queens = counts[QUEEN];
    int existing_kings = counts[KING];
    int existing_aces = counts[ACE];

    // ハイパージオメトリック分布を使った近似計算
    // P(QQ+) = P(少なくとも2枚のQを引く | 既にn枚のQがある)

    auto calc_pair_prob = [&](int existing, int remaining_in_deck, int slots) -> float {
        if (existing >= 2) return 1.0f;
        if (existing == 1 && remaining_in_deck >= 1) {
            // 1枚あり、もう1枚引く確率
            return 1.0f - static_cast<float>(
                std::max(0, remaining_count - remaining_in_deck)) /
                static_cast<float>(std::max(1, remaining_count));
        }
        if (existing == 0 && remaining_in_deck >= 2) {
            // 0枚から2枚引く確率（近似）
            float p = static_cast<float>(remaining_in_deck) / remaining_count;
            return p * p * slots * 0.5f;  // 簡略化
        }
        return 0.0f;
    };

    // トリップス確率（より高度な役なので低い確率）
    auto calc_trips_prob = [&](int existing, int remaining_in_deck) -> float {
        if (existing >= 3) return 1.0f;
        if (existing == 2 && remaining_in_deck >= 1) {
            return 0.3f;  // 経験的近似
        }
        if (existing == 1 && remaining_in_deck >= 2) {
            return 0.05f;
        }
        return 0.01f;  // ほぼ不可能
    };

    // 各FL条件の確率計算
    result.prob_trips = calc_trips_prob(
        std::max({existing_queens, existing_kings, existing_aces}),
        std::max({queens_remaining, kings_remaining, aces_remaining})
    );

    result.prob_aa = calc_pair_prob(existing_aces, aces_remaining, top_slots);
    result.prob_kk = calc_pair_prob(existing_kings, kings_remaining, top_slots);
    result.prob_qq = calc_pair_prob(existing_queens, queens_remaining, top_slots);

    // 総合確率（包除原理の近似）
    result.total_prob = std::min(1.0f,
        result.prob_trips +
        (1.0f - result.prob_trips) * result.prob_aa +
        (1.0f - result.prob_trips - result.prob_aa) * result.prob_kk +
        (1.0f - result.prob_trips - result.prob_aa - result.prob_kk) * result.prob_qq
    );

    // 期待値計算
    result.expected_ev =
        result.prob_trips * FL_EXPECTED_SCORE_17 +
        result.prob_aa * FL_EXPECTED_SCORE_16 +
        result.prob_kk * FL_EXPECTED_SCORE_15 +
        result.prob_qq * FL_EXPECTED_SCORE_14;

    return result;
}

// ============================================
// MCTS評価関数
// ============================================

struct MCTSEvaluation {
    float base_score;      // 現在のボードスコア (ロイヤリティ等)
    float fl_value;        // FL価値 (突入確率 × 期待スコア)
    float foul_penalty;    // ファウルリスクペナルティ
    float total_value;     // 統合評価値
};

/**
 * MCTSノードの評価
 *
 * @param board プレイヤーのボード
 * @param remaining_deck 残りデッキ
 * @param remaining_turns 残りターン数
 * @param fl_weight FL価値の重み（0-1）
 * @return 評価結果
 */
inline MCTSEvaluation evaluate_mcts_node(
    const Board& board,
    CardMask remaining_deck,
    int remaining_turns,
    float fl_weight = 0.5f
) {
    MCTSEvaluation eval = {0, 0, 0, 0};

    // 1. 基本スコア（完成した段のロイヤリティ）
    if (board.count(BOTTOM) == BOT_SIZE) {
        eval.base_score += calculate_bottom_royalty(board.evaluate_bot().rank);
    }
    if (board.count(MIDDLE) == MID_SIZE) {
        eval.base_score += calculate_middle_royalty(board.evaluate_mid().rank);
    }
    if (board.count(TOP) == TOP_SIZE) {
        HandValue top_val = board.evaluate_top();
        eval.base_score += calculate_top_royalty(
            top_val.rank,
            static_cast<Rank>(top_val.kickers)
        );
    }

    // 2. FL価値計算
    FLProbabilityResult fl_prob = calculate_fl_probability(
        board, remaining_deck, remaining_turns
    );
    eval.fl_value = fl_prob.expected_ev;

    // 3. ファウルリスク（簡易判定）
    // Bot < Mid または Mid < Top になる可能性をチェック
    if (board.count(BOTTOM) == BOT_SIZE && board.count(MIDDLE) == MID_SIZE) {
        HandValue bot_val = board.evaluate_bot();
        HandValue mid_val = board.evaluate_mid();
        if (bot_val < mid_val) {
            eval.foul_penalty = -50.0f;  // 確定ファウル
        }
    }
    // TODO: より詳細なファウルリスク計算

    // 4. 統合評価値
    eval.total_value =
        eval.base_score +
        fl_weight * eval.fl_value +
        eval.foul_penalty;

    return eval;
}

// ============================================
// Fantasy Land ソルバー連携
// ============================================

/**
 * FL突入時の期待スコアを FantasySolver で計算
 *
 * @param fl_cards FL配布カード
 * @param already_in_fl すでにFL中かどうか
 * @return 期待スコア (ロイヤリティ + ステイボーナス)
 */
inline float calculate_fl_expected_score(
    const std::vector<Card>& fl_cards,
    bool already_in_fl = true
) {
    FantasySolution solution = FantasySolver::solve(fl_cards, already_in_fl);
    return solution.score;
}

// ============================================
// モンテカルロ・ロールアウト用ヘルパー
// ============================================

/**
 * 指定されたシードでデッキをシャッフルし、残りカードをリストで返す
 */
inline std::vector<Card> get_remaining_cards_shuffled(
    CardMask used_cards,
    uint64_t seed
) {
    std::vector<Card> remaining;
    remaining.reserve(52);

    for (uint8_t i = 0; i < NUM_CARDS; ++i) {
        Card c(i);
        if (!(used_cards & c.to_mask())) {
            remaining.push_back(c);
        }
    }

    // シャッフル
    std::mt19937_64 rng(seed);
    std::shuffle(remaining.begin(), remaining.end(), rng);

    return remaining;
}

/**
 * ランダムロールアウト: ボードをランダムに完成させてスコアを計算
 *
 * @param board 現在のボード (コピーで渡す)
 * @param remaining_cards 使用可能な残りカード
 * @param seed 乱数シード
 * @return 最終スコア (ファウル時は-6)
 */
inline float random_rollout(
    Board board,
    const std::vector<Card>& remaining_cards,
    uint64_t seed
) {
    std::mt19937_64 rng(seed);

    // 残りのスロットを埋める
    int card_idx = 0;

    while (!board.is_complete() && card_idx < static_cast<int>(remaining_cards.size())) {
        Card c = remaining_cards[card_idx++];

        // 配置可能な行を選択（ランダムまたはヒューリスティック）
        std::vector<Row> available;
        if (board.can_place(BOTTOM)) available.push_back(BOTTOM);
        if (board.can_place(MIDDLE)) available.push_back(MIDDLE);
        if (board.can_place(TOP)) available.push_back(TOP);

        if (available.empty()) break;

        // 簡易ヒューリスティック: 高いカードはTopへ、低いカードはBotへ
        Row chosen;
        Rank r = c.rank();
        if (r >= QUEEN && board.can_place(TOP)) {
            chosen = TOP;
        } else if (r >= EIGHT && board.can_place(MIDDLE)) {
            chosen = MIDDLE;
        } else if (board.can_place(BOTTOM)) {
            chosen = BOTTOM;
        } else {
            chosen = available[rng() % available.size()];
        }

        board.place_card(chosen, c);
    }

    if (!board.is_complete()) {
        return -10.0f;  // 完成できなかった
    }

    if (board.is_foul()) {
        return -6.0f;  // ファウル
    }

    float score = static_cast<float>(board.calculate_royalties());

    // FL突入ボーナス
    if (board.qualifies_for_fl()) {
        int fl_cards = board.fl_card_count();
        if (fl_cards == 17) score += FL_EXPECTED_SCORE_17;
        else if (fl_cards == 16) score += FL_EXPECTED_SCORE_16;
        else if (fl_cards == 15) score += FL_EXPECTED_SCORE_15;
        else score += FL_EXPECTED_SCORE_14;
    }

    return score;
}

/**
 * 複数回のロールアウトを実行して平均スコアを返す
 *
 * @param board 現在のボード
 * @param used_cards 使用済みカードマスク
 * @param num_rollouts ロールアウト回数
 * @param base_seed 基本乱数シード
 * @return 平均期待スコア
 */
inline float monte_carlo_evaluation(
    const Board& board,
    CardMask used_cards,
    int num_rollouts = 100,
    uint64_t base_seed = 12345
) {
    float total_score = 0.0f;

    for (int i = 0; i < num_rollouts; ++i) {
        uint64_t seed = base_seed + i;
        auto remaining = get_remaining_cards_shuffled(used_cards, seed);
        float score = random_rollout(board, remaining, seed + 1000000);
        total_score += score;
    }

    return total_score / static_cast<float>(num_rollouts);
}

// ============================================
// 配置評価（アクション選択支援）
// ============================================

struct PlacementEvaluation {
    int action_id;          // アクションID
    float immediate_value;  // 即時価値（ロイヤリティ増分）
    float fl_potential;     // FL突入ポテンシャル
    float foul_risk;        // ファウルリスク
    float total_score;      // 統合スコア
};

/**
 * カード配置候補を評価
 *
 * @param board 現在のボード
 * @param card 配置するカード
 * @param row 配置先の行
 * @param remaining_deck 残りデッキ
 * @param remaining_turns 残りターン数
 * @return 評価結果
 */
inline PlacementEvaluation evaluate_placement(
    const Board& board,
    const Card& card,
    Row row,
    CardMask remaining_deck,
    int remaining_turns
) {
    PlacementEvaluation eval;
    eval.action_id = card.index * 3 + static_cast<int>(row);

    // 配置後のボードを作成
    Board new_board = board;
    if (!new_board.place_card(row, card)) {
        eval.total_score = -1000.0f;  // 無効な配置
        return eval;
    }

    // 即時価値（完成した行のロイヤリティ増分）
    eval.immediate_value = 0.0f;
    if (row == BOTTOM && new_board.count(BOTTOM) == BOT_SIZE) {
        eval.immediate_value = calculate_bottom_royalty(new_board.evaluate_bot().rank);
    } else if (row == MIDDLE && new_board.count(MIDDLE) == MID_SIZE) {
        eval.immediate_value = calculate_middle_royalty(new_board.evaluate_mid().rank);
    } else if (row == TOP && new_board.count(TOP) == TOP_SIZE) {
        HandValue top_val = new_board.evaluate_top();
        eval.immediate_value = calculate_top_royalty(
            top_val.rank,
            static_cast<Rank>(top_val.kickers)
        );
        // FL突入は大きなボーナス
        if (qualifies_for_fantasy_land(top_val)) {
            eval.immediate_value += FL_EXPECTED_SCORE_14;
        }
    }

    // FL突入ポテンシャル
    FLProbabilityResult fl_prob = calculate_fl_probability(
        new_board, remaining_deck, remaining_turns
    );
    eval.fl_potential = fl_prob.expected_ev;

    // ファウルリスク
    eval.foul_risk = 0.0f;
    // 簡易判定: BottomとMiddleが完成している場合
    if (new_board.count(BOTTOM) == BOT_SIZE && new_board.count(MIDDLE) == MID_SIZE) {
        HandValue bot_val = new_board.evaluate_bot();
        HandValue mid_val = new_board.evaluate_mid();
        if (bot_val < mid_val) {
            eval.foul_risk = -100.0f;  // 確定ファウル
        }
    }
    // MiddleとTopが完成している場合
    if (new_board.count(MIDDLE) == MID_SIZE && new_board.count(TOP) == TOP_SIZE) {
        HandValue mid_val = new_board.evaluate_mid();
        HandValue top_val = new_board.evaluate_top();
        if (mid_val < top_val) {
            eval.foul_risk = -100.0f;  // 確定ファウル
        }
    }

    // 統合スコア
    eval.total_score = eval.immediate_value + 0.5f * eval.fl_potential + eval.foul_risk;

    return eval;
}

} // namespace ofc

#endif // OFC_MCTS_HPP
