/**
 * OFC Pineapple AI - Game Engine
 *
 * ゲーム進行を管理するメインエンジン。
 * プレイヤー管理、配布、点数計算を統括。
 */

#ifndef OFC_GAME_HPP
#define OFC_GAME_HPP

#include "board.hpp"
#include "card.hpp"
#include "deck.hpp"
#include "evaluator.hpp"
#include <array>
#include <vector>

namespace ofc {

// ============================================
// 定数
// ============================================

constexpr int MAX_PLAYERS = 3;
constexpr int INITIAL_CARDS = 5;  // 初回配布枚数
constexpr int TURN_CARDS = 3;     // 各ターンの配布枚数
constexpr int NUM_TURNS = 5;      // 総ターン数（初回+4回）
constexpr int FL_BASE_CARDS = 14; // FL基本配布枚数

// ============================================
// ゲームフェーズ
// ============================================

enum GamePhase : uint8_t {
  PHASE_INIT = 0,     // 初期化
  PHASE_INITIAL_DEAL, // 初回5枚配布
  PHASE_TURN,         // 通常ターン（3枚配布）
  PHASE_SHOWDOWN,     // ショーダウン
  PHASE_COMPLETE      // ゲーム終了
};

// ============================================
// プレイヤー状態
// ============================================

struct PlayerState {
  Board board;
  std::array<Card, 5> hand; // 現在の手札（最大5枚）
  int hand_count = 0;
  bool in_fantasy_land = false;
  int fl_cards_to_receive = 0; // FL時の配布枚数

  // アクション用
  std::vector<Card> fl_hand; // FL時の手札（14-17枚）

  void clear() {
    board.clear();
    hand_count = 0;
    in_fantasy_land = false;
    fl_cards_to_receive = 0;
    fl_hand.clear();
  }

  // 手札から指定カードを削除
  bool remove_from_hand(const Card &card) {
    for (int i = 0; i < hand_count; ++i) {
      if (hand[i] == card) {
        // 最後のカードと入れ替えて削除
        hand[i] = hand[hand_count - 1];
        hand_count--;
        return true;
      }
    }
    return false;
  }
};

// ============================================
// アクション定義
// ============================================

struct PlaceAction {
  Card card;
  Row row;
};

struct TurnAction {
  PlaceAction placements[2]; // 2枚配置
  Card discard;              // 1枚捨て
};

struct InitialAction {
  PlaceAction placements[5]; // 5枚すべて配置
};

struct FLAction {
  PlaceAction placements[13]; // 13枚配置
  std::vector<Card> discards; // 残りを捨て
};

// ============================================
// ゲーム結果
// ============================================

struct GameResult {
  std::array<int, MAX_PLAYERS> scores;
  std::array<int, MAX_PLAYERS> royalties;
  std::array<bool, MAX_PLAYERS> fouled;
  std::array<bool, MAX_PLAYERS> entered_fl;
  std::array<bool, MAX_PLAYERS> stayed_fl;

  void clear() {
    scores.fill(0);
    royalties.fill(0);
    fouled.fill(false);
    entered_fl.fill(false);
    stayed_fl.fill(false);
  }
};

// ============================================
// GameEngine クラス
// ============================================

class GameEngine {
public:
  GameEngine(int num_players = 2) : num_players_(num_players) { reset(); }

  // ゲームをリセット
  void reset() {
    deck_.reset();
    for (int i = 0; i < num_players_; ++i) {
      players_[i].clear();
    }
    phase_ = PHASE_INIT;
    current_turn_ = 0;
    current_player_ = 0;
    result_.clear();
  }

  // シャッフルして新規ゲーム開始
  template <typename RNG> void start_new_game(RNG &rng) {
    reset();
    deck_.shuffle(rng);
    phase_ = PHASE_INITIAL_DEAL;

    // 初回5枚配布
    for (int p = 0; p < num_players_; ++p) {
      if (!players_[p].in_fantasy_land) {
        deal_initial_cards(p);
      }
    }
    current_turn_ = 1;
  }

  // FL状態を引き継いで開始（boolのみ - 後方互換用）
  template <typename RNG>
  void start_with_fl(RNG &rng, const std::array<bool, MAX_PLAYERS> &fl_status) {
    // デフォルトは14枚
    std::array<int, MAX_PLAYERS> fl_cards = {0};
    for (int p = 0; p < MAX_PLAYERS; ++p) {
      fl_cards[p] = fl_status[p] ? FL_BASE_CARDS : 0;
    }
    start_with_fl_cards(rng, fl_cards);
  }

  // FL状態を引き継いで開始（Ultimate Rules: カード枚数指定）
  // fl_cards[p] = 0: 通常プレイヤー
  // fl_cards[p] = 14-17: FLプレイヤー（枚数に応じた配布）
  template <typename RNG>
  void start_with_fl_cards(RNG &rng, const std::array<int, MAX_PLAYERS> &fl_cards) {
    reset();
    deck_.shuffle(rng);

    for (int p = 0; p < num_players_; ++p) {
      if (fl_cards[p] > 0) {
        players_[p].in_fantasy_land = true;
        players_[p].fl_cards_to_receive = fl_cards[p];
      } else {
        players_[p].in_fantasy_land = false;
        players_[p].fl_cards_to_receive = 0;
      }
    }

    phase_ = PHASE_INITIAL_DEAL;

    // 配布
    for (int p = 0; p < num_players_; ++p) {
      if (players_[p].in_fantasy_land) {
        deal_fl_cards(p);
      } else {
        deal_initial_cards(p);
      }
    }
    current_turn_ = 1;
  }

  // 初回アクション適用（5枚配置）
  bool apply_initial_action(int player, const InitialAction &action) {
    if (phase_ != PHASE_INITIAL_DEAL)
      return false;

    PlayerState &ps = players_[player];
    for (int i = 0; i < 5; ++i) {
      if (!ps.board.place_card(action.placements[i].row,
                               action.placements[i].card)) {
        return false;
      }
    }
    ps.hand_count = 0;

    advance_game_state();
    return true;
  }

  // 通常ターンアクション適用（2枚配置、1枚捨て）
  bool apply_turn_action(int player, const TurnAction &action) {
    if (phase_ != PHASE_TURN)
      return false;

    PlayerState &ps = players_[player];

    // 2枚配置
    for (int i = 0; i < 2; ++i) {
      if (!ps.board.place_card(action.placements[i].row,
                               action.placements[i].card)) {
        return false;
      }
    }

    // 1枚は捨て札（何もしない - デッキから引かれている）
    ps.hand_count = 0;

    advance_game_state();
    return true;
  }

  // FLアクション適用（13枚配置、残り捨て）
  bool apply_fl_action(int player, const FLAction &action) {
    if (!players_[player].in_fantasy_land)
      return false;

    PlayerState &ps = players_[player];
    for (int i = 0; i < 13; ++i) {
      if (!ps.board.place_card(action.placements[i].row,
                               action.placements[i].card)) {
        return false;
      }
    }
    ps.fl_hand.clear();

    advance_game_state();
    return true;
  }

  // ショーダウン：点数計算
  void calculate_scores() {
    if (phase_ != PHASE_SHOWDOWN)
      return;

    result_.clear();

    // 各プレイヤーのロイヤリティとファウル判定
    for (int p = 0; p < num_players_; ++p) {
      result_.fouled[p] = players_[p].board.is_foul();
      if (!result_.fouled[p]) {
        result_.royalties[p] = players_[p].board.calculate_royalties();
        result_.entered_fl[p] = players_[p].board.qualifies_for_fl();
        result_.stayed_fl[p] =
            players_[p].in_fantasy_land && players_[p].board.can_stay_fl();
      }
    }

    // ヘッズアップの点数計算
    if (num_players_ == 2) {
      calculate_heads_up_score(0, 1);
    } else if (num_players_ == 3) {
      // 3人戦：各ペアで計算
      calculate_heads_up_score(0, 1);
      calculate_heads_up_score(1, 2);
      calculate_heads_up_score(0, 2);
    }

    phase_ = PHASE_COMPLETE;
  }

  // アクセサ
  GamePhase phase() const { return phase_; }
  int current_turn() const { return current_turn_; }
  int current_player() const { return current_player_; }
  int num_players() const { return num_players_; }
  const PlayerState &player(int idx) const { return players_[idx]; }
  PlayerState &player_mut(int idx) { return players_[idx]; }
  const GameResult &result() const { return result_; }
  const Deck &deck() const { return deck_; }

  // ============================================
  // シリアライズ（MCTS用状態保存）
  // ============================================

  // 状態をバイト列にシリアライズ
  std::vector<uint8_t> serialize() const {
    std::vector<uint8_t> data;
    data.reserve(512); // 予想サイズ

    // ゲーム状態
    data.push_back(static_cast<uint8_t>(phase_));
    data.push_back(static_cast<uint8_t>(current_turn_));
    data.push_back(static_cast<uint8_t>(current_player_));
    data.push_back(static_cast<uint8_t>(num_players_));

    // デッキ状態
    auto deck_data = deck_.serialize();
    uint16_t deck_size = static_cast<uint16_t>(deck_data.size());
    data.push_back(deck_size & 0xFF);
    data.push_back((deck_size >> 8) & 0xFF);
    data.insert(data.end(), deck_data.begin(), deck_data.end());

    // プレイヤー状態
    for (int p = 0; p < num_players_; ++p) {
      const PlayerState &ps = players_[p];

      // ボード（各段のカードマスク）
      auto top_mask = ps.board.top_mask();
      auto mid_mask = ps.board.mid_mask();
      auto bot_mask = ps.board.bot_mask();

      // 8バイト x 3段 = 24バイト
      for (int i = 0; i < 8; ++i) {
        data.push_back((top_mask >> (i * 8)) & 0xFF);
      }
      for (int i = 0; i < 8; ++i) {
        data.push_back((mid_mask >> (i * 8)) & 0xFF);
      }
      for (int i = 0; i < 8; ++i) {
        data.push_back((bot_mask >> (i * 8)) & 0xFF);
      }

      // 手札
      data.push_back(static_cast<uint8_t>(ps.hand_count));
      for (int i = 0; i < ps.hand_count; ++i) {
        data.push_back(ps.hand[i].index);
      }

      // FL状態
      data.push_back(ps.in_fantasy_land ? 1 : 0);
      data.push_back(static_cast<uint8_t>(ps.fl_cards_to_receive));
    }

    return data;
  }

  // バイト列から状態を復元
  bool deserialize(const std::vector<uint8_t> &data) {
    if (data.size() < 10)
      return false;

    size_t idx = 0;

    // ゲーム状態
    phase_ = static_cast<GamePhase>(data[idx++]);
    current_turn_ = data[idx++];
    current_player_ = data[idx++];
    num_players_ = data[idx++];

    // デッキ状態
    uint16_t deck_size = data[idx] | (data[idx + 1] << 8);
    idx += 2;
    if (idx + deck_size > data.size())
      return false;
    std::vector<uint8_t> deck_data(data.begin() + idx,
                                   data.begin() + idx + deck_size);
    deck_.deserialize(deck_data);
    idx += deck_size;

    // プレイヤー状態
    for (int p = 0; p < num_players_; ++p) {
      if (idx + 27 > data.size())
        return false; // 最小サイズチェック

      PlayerState &ps = players_[p];
      ps.clear();

      // ボード（マスクから復元）
      uint64_t top_mask = 0, mid_mask = 0, bot_mask = 0;
      for (int i = 0; i < 8; ++i) {
        top_mask |= static_cast<uint64_t>(data[idx++]) << (i * 8);
      }
      for (int i = 0; i < 8; ++i) {
        mid_mask |= static_cast<uint64_t>(data[idx++]) << (i * 8);
      }
      for (int i = 0; i < 8; ++i) {
        bot_mask |= static_cast<uint64_t>(data[idx++]) << (i * 8);
      }
      ps.board.restore_from_masks(top_mask, mid_mask, bot_mask);

      // 手札
      ps.hand_count = data[idx++];
      for (int i = 0; i < ps.hand_count; ++i) {
        ps.hand[i] = Card(data[idx++]);
      }

      // FL状態
      ps.in_fantasy_land = (data[idx++] != 0);
      ps.fl_cards_to_receive = data[idx++];
    }

    return true;
  }

  // 状態のクローン（MCTS用）
  GameEngine clone() const {
    GameEngine cloned(num_players_);
    auto data = serialize();
    cloned.deserialize(data);
    return cloned;
  }

  // 残りカード枚数（終盤判定用）
  int remaining_cards_in_board(int player) const {
    return 13 - players_[player].board.total_placed();
  }

private:
  int num_players_;
  Deck deck_;
  std::array<PlayerState, MAX_PLAYERS> players_;
  GamePhase phase_;
  int current_turn_;
  int current_player_;
  GameResult result_;

  // 初回5枚配布
  void deal_initial_cards(int player) {
    PlayerState &ps = players_[player];
    for (int i = 0; i < INITIAL_CARDS; ++i) {
      ps.hand[i] = deck_.draw();
    }
    ps.hand_count = INITIAL_CARDS;
  }

  // 通常ターン3枚配布
  void deal_turn_cards(int player) {
    PlayerState &ps = players_[player];
    for (int i = 0; i < TURN_CARDS; ++i) {
      ps.hand[i] = deck_.draw();
    }
    ps.hand_count = TURN_CARDS;
  }

  // FL配布
  void deal_fl_cards(int player) {
    PlayerState &ps = players_[player];
    int cards = ps.fl_cards_to_receive;
    ps.fl_hand.resize(cards);
    for (int i = 0; i < cards; ++i) {
      ps.fl_hand[i] = deck_.draw();
    }
  }

  // ゲーム状態を進める
  void advance_game_state() {
    // 全プレイヤーのボードが完成済みなら即SHOWDOWN
    // (全員FL時にcurrent_turn_が不足する問題への対策)
    {
      bool all_boards_complete = true;
      for (int p = 0; p < num_players_; ++p) {
        if (players_[p].board.total_placed() < 13) {
          all_boards_complete = false;
          break;
        }
      }
      if (all_boards_complete) {
        phase_ = PHASE_SHOWDOWN;
        calculate_scores();
        return;
      }
    }

    // 全プレイヤーがこのフェーズのアクションを完了したかチェック
    bool all_ready = true;

    if (phase_ == PHASE_INITIAL_DEAL) {
      // 初回フェーズ: 全員が5枚配置済みかチェック
      for (int p = 0; p < num_players_; ++p) {
        if (players_[p].in_fantasy_land)
          continue;
        if (players_[p].board.total_placed() < 5) {
          all_ready = false;
          break;
        }
      }
    } else if (phase_ == PHASE_TURN) {
      // ターンフェーズ: 全員がボード完成または手札なしかチェック
      for (int p = 0; p < num_players_; ++p) {
        if (players_[p].in_fantasy_land)
          continue;
        // このターンで2枚追加したか（hand_count == 0になっている）
        if (players_[p].hand_count > 0) {
          all_ready = false;
          break;
        }
      }
    }

    if (!all_ready)
      return;

    // 全員アクション完了 → 次のターンへ
    current_turn_++;

    if (current_turn_ > NUM_TURNS) {
      phase_ = PHASE_SHOWDOWN;
      calculate_scores();
    } else {
      phase_ = PHASE_TURN;
      // 次のターンのカード配布
      for (int p = 0; p < num_players_; ++p) {
        if (!players_[p].in_fantasy_land) {
          deal_turn_cards(p);
        }
      }
    }
  }

  // ヘッズアップ点数計算
  void calculate_heads_up_score(int p1, int p2) {
    // ファウルチェック
    if (result_.fouled[p1] && result_.fouled[p2]) {
      // 両者ファウル：相殺
      return;
    }
    if (result_.fouled[p1]) {
      // P1ファウル：P2に6点+ロイヤリティ
      result_.scores[p2] += 6 + result_.royalties[p2];
      result_.scores[p1] -= 6 + result_.royalties[p2];
      return;
    }
    if (result_.fouled[p2]) {
      // P2ファウル：P1に6点+ロイヤリティ
      result_.scores[p1] += 6 + result_.royalties[p1];
      result_.scores[p2] -= 6 + result_.royalties[p1];
      return;
    }

    // 通常対決
    const Board &b1 = players_[p1].board;
    const Board &b2 = players_[p2].board;

    int line_wins_p1 = 0;
    int line_wins_p2 = 0;

    // 各段の勝敗
    // Top
    HandValue t1 = b1.evaluate_top();
    HandValue t2 = b2.evaluate_top();
    if (t1 > t2)
      line_wins_p1++;
    else if (t2 > t1)
      line_wins_p2++;

    // Middle
    HandValue m1 = b1.evaluate_mid();
    HandValue m2 = b2.evaluate_mid();
    if (m1 > m2)
      line_wins_p1++;
    else if (m2 > m1)
      line_wins_p2++;

    // Bottom
    HandValue bot1 = b1.evaluate_bot();
    HandValue bot2 = b2.evaluate_bot();
    if (bot1 > bot2)
      line_wins_p1++;
    else if (bot2 > bot1)
      line_wins_p2++;

    // ライン点数
    int line_diff = line_wins_p1 - line_wins_p2;
    result_.scores[p1] += line_diff;
    result_.scores[p2] -= line_diff;

    // スクープボーナス
    if (line_wins_p1 == 3) {
      result_.scores[p1] += 3;
      result_.scores[p2] -= 3;
    } else if (line_wins_p2 == 3) {
      result_.scores[p2] += 3;
      result_.scores[p1] -= 3;
    }

    // ロイヤリティ交換
    result_.scores[p1] += result_.royalties[p1] - result_.royalties[p2];
    result_.scores[p2] += result_.royalties[p2] - result_.royalties[p1];
  }
};

} // namespace ofc

#endif // OFC_GAME_HPP
