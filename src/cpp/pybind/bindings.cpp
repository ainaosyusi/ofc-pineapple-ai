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
#include "../probability.hpp"
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

  py::enum_<ofc::HandRank>(m, "HandRank")
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
      .def("get_hand", [](const ofc::PlayerState &ps) {
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
      .def("apply_initial_action", &ofc::GameEngine::apply_initial_action)
      .def("apply_turn_action", &ofc::GameEngine::apply_turn_action)
      .def("phase", &ofc::GameEngine::phase)
      .def("current_turn", &ofc::GameEngine::current_turn)
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
}
