#!/usr/bin/env python3
"""
FL (Fantasy Land) テストスクリプト

AIがFL中にどのようにプレイするかを確認
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src/python'))

import numpy as np
from sb3_contrib import MaskablePPO
import ofc_engine as ofc
from ofc_3max_env import OFC3MaxEnv

RANKS = 'A23456789TJQK'
SUITS = ['♠', '♥', '♦', '♣']

def card_to_str(card):
    """カードを文字列に変換"""
    rank = int(card.rank())
    suit = int(card.suit())
    if rank >= 13 or suit >= 4:
        return "🃏"
    return f"{RANKS[rank]}{SUITS[suit]}"

def mask_to_cards(mask):
    """ビットマスクからカードリストを取得"""
    cards = []
    SUIT_ENUMS = [ofc.SPADE, ofc.HEART, ofc.DIAMOND, ofc.CLUB]
    RANK_ENUMS = [ofc.ACE, ofc.TWO, ofc.THREE, ofc.FOUR, ofc.FIVE, ofc.SIX,
                  ofc.SEVEN, ofc.EIGHT, ofc.NINE, ofc.TEN, ofc.JACK, ofc.QUEEN, ofc.KING]

    for i in range(54):
        if (mask >> i) & 1:
            if i < 52:
                suit_idx = i // 13
                rank_idx = i % 13
                card = ofc.Card(SUIT_ENUMS[suit_idx], RANK_ENUMS[rank_idx])
                cards.append(card_to_str(card))
            else:
                cards.append("🃏")
    return cards

def show_board(board, title="Board"):
    """ボードを表示"""
    print(f"\n{title}:")
    print(f"  Top:    {mask_to_cards(board.top_mask())}")
    print(f"  Mid:    {mask_to_cards(board.mid_mask())}")
    print(f"  Bottom: {mask_to_cards(board.bot_mask())}")

    if board.is_foul():
        print("  ⚠️ FOUL!")
    else:
        print(f"  Royalty: {board.calculate_royalties()}")
        if board.qualifies_for_fl():
            print("  ✅ FL Entry qualified!")
        if board.can_stay_fl():
            print("  🎉 FL STAY qualified!")

def test_fl_game(model_path, num_games=10):
    """FL ゲームをテスト"""
    print(f"Loading model: {model_path}")
    model = MaskablePPO.load(model_path)
    print("Model loaded.\n")

    # FL専用環境（continuous_games=Trueで連続ゲーム）
    env = OFC3MaxEnv(continuous_games=True, fl_solver_mode='greedy')

    fl_entry_count = 0
    fl_stay_count = 0
    fl_games_played = 0
    foul_count = 0

    print("="*60)
    print("Testing FL Entry and Stay")
    print("="*60)

    for game_num in range(num_games):
        env.reset()

        # FL強制開始（テスト用）
        if game_num < 5:
            # 最初の5ゲームはFL開始
            fl_types = ['qq', 'kk', 'aa', 'trips', 'kk']
            fl_type = fl_types[game_num]
            fl_cards = {'qq': 14, 'kk': 15, 'aa': 16, 'trips': 17}

            for agent in env.possible_agents:
                env.fl_status[agent] = True
                env.fl_cards_count[agent] = fl_cards[fl_type]

            env.reset()
            print(f"\n--- Game {game_num+1}: FL Start ({fl_type.upper()}, {fl_cards[fl_type]} cards) ---")
            fl_games_played += 1
        else:
            print(f"\n--- Game {game_num+1}: Normal Start ---")

        done = False
        step = 0

        while not done:
            current_agent = env.agent_selection

            if env.terminations[current_agent] or env.truncations[current_agent]:
                env.step(None)
                done = all(env.terminations.values()) or all(env.truncations.values())
                continue

            player_idx = int(current_agent.split('_')[1])
            ps = env.engine.player(player_idx)

            obs = env.observe(current_agent)
            mask = env.action_masks(current_agent)

            if not np.any(mask):
                env.step(None)
                done = all(env.terminations.values()) or all(env.truncations.values())
                continue

            try:
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                action = int(action)
            except ValueError:
                valid_actions = np.where(mask)[0]
                action = int(np.random.choice(valid_actions))

            # player_0 のFL状態を表示
            if current_agent == 'player_0' and ps.in_fantasy_land:
                hand_mask = ps.hand_mask()
                hand_cards = mask_to_cards(hand_mask)
                print(f"  FL Hand ({len(hand_cards)} cards): {hand_cards}")

            env.step(action)
            step += 1
            done = all(env.terminations.values()) or all(env.truncations.values())

        # ゲーム終了後の結果
        player = env.engine.player(0)
        board = player.board

        show_board(board, f"Player 0 Final Board (Game {game_num+1})")

        if board.is_foul():
            foul_count += 1
        else:
            if board.qualifies_for_fl():
                fl_entry_count += 1
            if board.can_stay_fl():
                fl_stay_count += 1

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total games: {num_games}")
    print(f"FL games (forced start): {fl_games_played}")
    print(f"Foul rate: {100*foul_count/num_games:.1f}%")
    print(f"FL Entry count: {fl_entry_count} ({100*fl_entry_count/num_games:.1f}%)")
    print(f"FL Stay count: {fl_stay_count} ({100*fl_stay_count/num_games:.1f}%)")

    if fl_games_played > 0:
        print(f"\nFL Stay rate (from FL starts): {100*fl_stay_count/fl_games_played:.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='models/phase9/p9_fl_mastery_250000000.zip')
    parser.add_argument('--games', type=int, default=20)
    args = parser.parse_args()

    test_fl_game(args.model, args.games)
