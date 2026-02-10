/**
 * OFC Pineapple AI - Python Bindings
 *
 * pybind11を使用してC++エンジンをPythonから呼び出し可能にする。
 */

#include "../board.hpp"
#include "../card.hpp"
#include "../deck.hpp"
#include "../evaluator.hpp"
#include "../game.hpp"
#include "../mcts.hpp"
#include "../probability.hpp"
#include "../solver.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(ofc_engine, m) {
  m.doc() = "OFC Pineapple AI - High-performance game engine";

  // ============================================
  // Enums
  // ============================================

  py::enum_<ofc::Suit>(m, "Suit")
      .value("SPADE", ofc::SPADE)
      .value("HEART", ofc::HEART)
      .value("DIAMOND", ofc::DIAMOND)
      .value("CLUB", ofc::CLUB)
      .export_values();

  py::enum_<ofc::Rank>(m, "Rank")
      .value("ACE", ofc::ACE)
      .value("TWO", ofc::TWO)
      .value("THREE", ofc::THREE)
      .value("FOUR", ofc::FOUR)
      .value("FIVE", ofc::FIVE)
      .value("SIX", ofc::SIX)
      .value("SEVEN", ofc::SEVEN)
      .value("EIGHT", ofc::EIGHT)
      .value("NINE", ofc::NINE)
      .value("TEN", ofc::TEN)
      .value("JACK", ofc::JACK)
      .value("QUEEN", ofc::QUEEN)
      .value("KING", ofc::KING)
      .export_values();

  py::enum_<ofc::Row>(m, "Row")
      .value("TOP", ofc::TOP)
      .value("MIDDLE", ofc::MIDDLE)
      .value("BOTTOM", ofc::BOTTOM)
      .export_values();

  py::enum_<ofc::HandRank>(m, "HandRank", py::arithmetic())
      .value("HIGH_CARD", ofc::HIGH_CARD)
      .value("ONE_PAIR", ofc::ONE_PAIR)
      .value("TWO_PAIR", ofc::TWO_PAIR)
      .value("THREE_OF_A_KIND", ofc::THREE_OF_A_KIND)
      .value("STRAIGHT", ofc::STRAIGHT)
      .value("FLUSH", ofc::FLUSH)
      .value("FULL_HOUSE", ofc::FULL_HOUSE)
      .value("FOUR_OF_A_KIND", ofc::FOUR_OF_A_KIND)
      .value("STRAIGHT_FLUSH", ofc::STRAIGHT_FLUSH)
      .value("ROYAL_FLUSH", ofc::ROYAL_FLUSH)
      .export_values();

  py::enum_<ofc::GamePhase>(m, "GamePhase")
      .value("INIT", ofc::PHASE_INIT)
      .value("INITIAL_DEAL", ofc::PHASE_INITIAL_DEAL)
      .value("TURN", ofc::PHASE_TURN)
      .value("SHOWDOWN", ofc::PHASE_SHOWDOWN)
      .value("COMPLETE", ofc::PHASE_COMPLETE)
      .export_values();

  // ============================================
  // Card
  // ============================================

  py::class_<ofc::Card>(m, "Card")
      .def(py::init<>())
      .def(py::init<uint8_t>())
      .def(py::init<ofc::Suit, ofc::Rank>())
      .def_readwrite("index", &ofc::Card::index)
      .def("suit", &ofc::Card::suit)
      .def("rank", &ofc::Card::rank)
      .def("is_valid", &ofc::Card::is_valid)
      .def("__repr__",
           [](const ofc::Card &c) {
             return "Card(" + ofc::card_to_string(c) + ")";
           })
      .def("__str__", [](const ofc::Card &c) { return ofc::card_to_string(c); })
      .def("__eq__", &ofc::Card::operator==);

  // ============================================
  // HandValue
  // ============================================

  py::class_<ofc::HandValue>(m, "HandValue")
      .def(py::init<>())
      .def(py::init<ofc::HandRank, uint32_t>())
      .def_readwrite("rank", &ofc::HandValue::rank)
      .def_readwrite("kickers", &ofc::HandValue::kickers)
      .def("__lt__", &ofc::HandValue::operator<)
      .def("__gt__", &ofc::HandValue::operator>)
      .def("__eq__", &ofc::HandValue::operator==);

  // ============================================
  // Board
  // ============================================

  py::class_<ofc::Board>(m, "Board")
      .def(py::init<>())
      .def("clear", &ofc::Board::clear)
      .def("place_card", &ofc::Board::place_card)
      .def("can_place", &ofc::Board::can_place)
      .def("count", &ofc::Board::count)
      .def("remaining_slots", &ofc::Board::remaining_slots)
      .def("is_complete", &ofc::Board::is_complete)
      .def("total_placed", &ofc::Board::total_placed)
      .def("evaluate_top", &ofc::Board::evaluate_top)
      .def("evaluate_mid", &ofc::Board::evaluate_mid)
      .def("evaluate_bot", &ofc::Board::evaluate_bot)
      .def("is_foul", &ofc::Board::is_foul)
      .def("calculate_royalties", &ofc::Board::calculate_royalties)
      .def("qualifies_for_fl", &ofc::Board::qualifies_for_fl)
      .def("fl_card_count", &ofc::Board::fl_card_count)
      .def("can_stay_fl", &ofc::Board::can_stay_fl)
      .def("top_mask", &ofc::Board::top_mask)
      .def("mid_mask", &ofc::Board::mid_mask)
      .def("bot_mask", &ofc::Board::bot_mask)
      .def("all_mask", &ofc::Board::all_mask)
      .def("to_string", &ofc::Board::to_string)
      .def("__repr__", &ofc::Board::to_string);

  // ============================================
  // Deck
  // ============================================

  py::class_<ofc::Deck>(m, "Deck")
      .def(py::init<>())
      .def("reset", &ofc::Deck::reset)
      .def("draw", &ofc::Deck::draw)
      .def("remaining", &ofc::Deck::remaining)
      .def("to_string", &ofc::Deck::to_string)
      .def("shuffle_with_seed", [](ofc::Deck &deck, uint64_t seed) {
        ofc::FastRNG rng(seed);
        deck.shuffle(rng);
      });

  // ============================================
  // PlayerState
  // ============================================

  py::class_<ofc::PlayerState>(m, "PlayerState")
      .def(py::init<>())
      .def_readwrite("board", &ofc::PlayerState::board)
      .def_readonly("hand_count", &ofc::PlayerState::hand_count)
      .def_readonly("in_fantasy_land", &ofc::PlayerState::in_fantasy_land)
      .def_readonly("fl_cards_to_receive", &ofc::PlayerState::fl_cards_to_receive)
      .def("get_hand", [](const ofc::PlayerState &ps) {
        if (!ps.fl_hand.empty())
          return ps.fl_hand;
        std::vector<ofc::Card> hand;
        for (int i = 0; i < ps.hand_count; ++i) {
          hand.push_back(ps.hand[i]);
        }
        return hand;
      });

  // ============================================
  // GameResult
  // ============================================

  py::class_<ofc::GameResult>(m, "GameResult")
      .def(py::init<>())
      .def("get_score",
           [](const ofc::GameResult &r, int p) { return r.scores[p]; })
      .def("get_royalty",
           [](const ofc::GameResult &r, int p) { return r.royalties[p]; })
      .def("is_fouled",
           [](const ofc::GameResult &r, int p) { return r.fouled[p]; })
      .def("entered_fl",
           [](const ofc::GameResult &r, int p) { return r.entered_fl[p]; })
      .def("stayed_fl",
           [](const ofc::GameResult &r, int p) { return r.stayed_fl[p]; });

  // ============================================
  // PlaceAction, TurnAction, InitialAction
  // ============================================

  py::class_<ofc::PlaceAction>(m, "PlaceAction")
      .def(py::init<>())
      .def_readwrite("card", &ofc::PlaceAction::card)
      .def_readwrite("row", &ofc::PlaceAction::row);

  py::class_<ofc::InitialAction>(m, "InitialAction")
      .def(py::init<>())
      .def("set_placement", [](ofc::InitialAction &a, int i, ofc::Card c,
                               ofc::Row r) { a.placements[i] = {c, r}; });

  py::class_<ofc::TurnAction>(m, "TurnAction")
      .def(py::init<>())
      .def_readwrite("discard", &ofc::TurnAction::discard)
      .def("set_placement", [](ofc::TurnAction &a, int i, ofc::Card c,
                               ofc::Row r) { a.placements[i] = {c, r}; });

  py::class_<ofc::FLAction>(m, "FLAction")
      .def(py::init<>())
      .def_readwrite("discards", &ofc::FLAction::discards)
      .def("set_placement", [](ofc::FLAction &a, int i, ofc::Card c,
                               ofc::Row r) { a.placements[i] = {c, r}; });

  // ============================================
  // GameEngine
  // ============================================

  py::class_<ofc::GameEngine>(m, "GameEngine")
      .def(py::init<int>(), py::arg("num_players") = 2)
      .def("reset", &ofc::GameEngine::reset)
      .def("start_new_game",
           [](ofc::GameEngine &e, uint64_t seed) {
             ofc::FastRNG rng(seed);
             e.start_new_game(rng);
           })
      .def("start_with_fl",
           [](ofc::GameEngine &e, uint64_t seed, std::vector<bool> fl_status) {
             ofc::FastRNG rng(seed);
             std::array<bool, ofc::MAX_PLAYERS> fl_array = {false};
             for (size_t i = 0;
                  i < std::min(fl_status.size(), (size_t)ofc::MAX_PLAYERS);
                  ++i) {
               fl_array[i] = fl_status[i];
             }
             e.start_with_fl(rng, fl_array);
           })
      .def("start_with_fl_cards",
           [](ofc::GameEngine &e, uint64_t seed, std::vector<int> fl_cards) {
             // Ultimate Rules: FL枚数を指定して開始
             // fl_cards[p] = 0: 通常, 14-17: FL枚数
             ofc::FastRNG rng(seed);
             std::array<int, ofc::MAX_PLAYERS> fl_array = {0};
             for (size_t i = 0;
                  i < std::min(fl_cards.size(), (size_t)ofc::MAX_PLAYERS);
                  ++i) {
               fl_array[i] = fl_cards[i];
             }
             e.start_with_fl_cards(rng, fl_array);
           },
           "Start game with FL card counts (Ultimate Rules: QQ=14, KK=15, AA=16, Trips=17)")
      .def("apply_initial_action", &ofc::GameEngine::apply_initial_action)
      .def("apply_turn_action", &ofc::GameEngine::apply_turn_action)
      .def("apply_fl_action", &ofc::GameEngine::apply_fl_action)
      .def("phase", &ofc::GameEngine::phase)
      .def("current_turn", &ofc::GameEngine::current_turn)
      .def("current_player", &ofc::GameEngine::current_player)
      .def("num_players", &ofc::GameEngine::num_players)
      .def("player", &ofc::GameEngine::player,
           py::return_value_policy::reference)
      .def("result", &ofc::GameEngine::result,
           py::return_value_policy::reference)
      // MCTS用シリアライズ
      .def("serialize",
           [](const ofc::GameEngine &e) {
             auto data = e.serialize();
             return py::bytes(reinterpret_cast<const char *>(data.data()),
                              data.size());
           })
      .def("deserialize",
           [](ofc::GameEngine &e, py::bytes data) {
             std::string s = data;
             std::vector<uint8_t> vec(s.begin(), s.end());
             return e.deserialize(vec);
           })
      .def("clone", &ofc::GameEngine::clone)
      .def("remaining_cards_in_board",
           &ofc::GameEngine::remaining_cards_in_board);

  // ============================================
  // Utility Functions
  // ============================================

  m.def("evaluate_5card", &ofc::evaluate_5card, "Evaluate a 5-card hand");
  m.def("evaluate_3card", &ofc::evaluate_3card,
        "Evaluate a 3-card hand (for Top)");
  m.def("card_to_mask", &ofc::card_to_mask,
        "Convert suit and rank to CardMask");
  m.def("qualifies_for_fantasy_land", &ofc::qualifies_for_fantasy_land);
  m.def("fantasy_land_cards", &ofc::fantasy_land_cards);

  // ============================================
  // Probability Calculator
  // ============================================

  m.def("flush_probability", &ofc::ProbabilityCalculator::flush_probability,
        "Calculate flush completion probability", py::arg("current_cards"),
        py::arg("current_count"), py::arg("target_count"),
        py::arg("remaining_deck"), py::arg("remaining_count"));

  m.def("straight_probability",
        &ofc::ProbabilityCalculator::straight_probability,
        "Calculate straight completion probability", py::arg("current_cards"),
        py::arg("current_count"), py::arg("remaining_deck"),
        py::arg("remaining_count"));

  // ============================================
  // Fantasy Solver
  // ============================================

  py::class_<ofc::FantasySolution>(m, "FantasySolution")
      .def_readwrite("top", &ofc::FantasySolution::top)
      .def_readwrite("mid", &ofc::FantasySolution::mid)
      .def_readwrite("bot", &ofc::FantasySolution::bot)
      .def_readwrite("discards", &ofc::FantasySolution::discards)
      .def_readwrite("score", &ofc::FantasySolution::score)
      .def_readwrite("stayed", &ofc::FantasySolution::stayed);

  m.def("solve_fantasy_land", &ofc::FantasySolver::solve,
        "Solve Fantasy Land optimal placement", py::arg("cards"),
        py::arg("already_in_fl") = true);

  // ============================================
  // MCTS Support Functions
  // ============================================

  py::class_<ofc::FLProbabilityResult>(m, "FLProbabilityResult")
      .def_readonly("prob_qq", &ofc::FLProbabilityResult::prob_qq)
      .def_readonly("prob_kk", &ofc::FLProbabilityResult::prob_kk)
      .def_readonly("prob_aa", &ofc::FLProbabilityResult::prob_aa)
      .def_readonly("prob_trips", &ofc::FLProbabilityResult::prob_trips)
      .def_readonly("total_prob", &ofc::FLProbabilityResult::total_prob)
      .def_readonly("expected_ev", &ofc::FLProbabilityResult::expected_ev);

  py::class_<ofc::MCTSEvaluation>(m, "MCTSEvaluation")
      .def_readonly("base_score", &ofc::MCTSEvaluation::base_score)
      .def_readonly("fl_value", &ofc::MCTSEvaluation::fl_value)
      .def_readonly("foul_penalty", &ofc::MCTSEvaluation::foul_penalty)
      .def_readonly("total_value", &ofc::MCTSEvaluation::total_value);

  py::class_<ofc::PlacementEvaluation>(m, "PlacementEvaluation")
      .def_readonly("action_id", &ofc::PlacementEvaluation::action_id)
      .def_readonly("immediate_value", &ofc::PlacementEvaluation::immediate_value)
      .def_readonly("fl_potential", &ofc::PlacementEvaluation::fl_potential)
      .def_readonly("foul_risk", &ofc::PlacementEvaluation::foul_risk)
      .def_readonly("total_score", &ofc::PlacementEvaluation::total_score);

  m.def(
      "calculate_fl_probability",
      [](const ofc::Board &board, ofc::CardMask remaining_deck,
         int remaining_turns) {
        return ofc::calculate_fl_probability(board, remaining_deck,
                                             remaining_turns);
      },
      "Calculate Fantasy Land entry probability", py::arg("board"),
      py::arg("remaining_deck"), py::arg("remaining_turns"));

  m.def(
      "evaluate_mcts_node",
      [](const ofc::Board &board, ofc::CardMask remaining_deck,
         int remaining_turns, float fl_weight) {
        return ofc::evaluate_mcts_node(board, remaining_deck, remaining_turns,
                                       fl_weight);
      },
      "Evaluate MCTS node with FL consideration", py::arg("board"),
      py::arg("remaining_deck"), py::arg("remaining_turns"),
      py::arg("fl_weight") = 0.5f);

  m.def(
      "evaluate_placement",
      [](const ofc::Board &board, const ofc::Card &card, ofc::Row row,
         ofc::CardMask remaining_deck, int remaining_turns) {
        return ofc::evaluate_placement(board, card, row, remaining_deck,
                                       remaining_turns);
      },
      "Evaluate a card placement action", py::arg("board"), py::arg("card"),
      py::arg("row"), py::arg("remaining_deck"), py::arg("remaining_turns"));

  m.def(
      "monte_carlo_evaluation",
      [](const ofc::Board &board, ofc::CardMask used_cards, int num_rollouts,
         uint64_t seed) {
        return ofc::monte_carlo_evaluation(board, used_cards, num_rollouts,
                                           seed);
      },
      "Run Monte Carlo rollouts to evaluate board position", py::arg("board"),
      py::arg("used_cards"), py::arg("num_rollouts") = 100,
      py::arg("seed") = 12345);

  m.def(
      "calculate_fl_expected_score",
      [](const std::vector<ofc::Card> &fl_cards, bool already_in_fl) {
        return ofc::calculate_fl_expected_score(fl_cards, already_in_fl);
      },
      "Calculate expected score for Fantasy Land hand", py::arg("fl_cards"),
      py::arg("already_in_fl") = true);

  // MCTS関連定数
  m.attr("FL_EXPECTED_SCORE_14") = ofc::FL_EXPECTED_SCORE_14;
  m.attr("FL_EXPECTED_SCORE_15") = ofc::FL_EXPECTED_SCORE_15;
  m.attr("FL_EXPECTED_SCORE_16") = ofc::FL_EXPECTED_SCORE_16;
  m.attr("FL_EXPECTED_SCORE_17") = ofc::FL_EXPECTED_SCORE_17;
  m.attr("FL_STAY_VALUE") = ofc::FL_STAY_VALUE;
}
