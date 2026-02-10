#!/usr/bin/env python3
"""
AIプレイヤーのハンド記録を表示
Phase 9 FL Mastery モデルを使用
"""

import sys
import os
# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src/python'))

import numpy as np
from sb3_contrib import MaskablePPO
import ofc_engine as ofc
from ofc_3max_env import OFC3MaxEnv

# カード表記変換 (ACE=0, TWO=1, ..., KING=12)
RANKS = 'A23456789TJQK'
SUITS = ['♠', '♥', '♦', '♣']

def card_to_str(card):
    """C++カードを文字列に変換"""
    rank = int(card.rank())
    suit = int(card.suit())
    # Jokerの場合
    if rank >= 13 or suit >= 4:
        return "🃏"
    return f"{RANKS[rank]}{SUITS[suit]}"

SUIT_ENUMS = [ofc.SPADE, ofc.HEART, ofc.DIAMOND, ofc.CLUB]
RANK_ENUMS = [ofc.TWO, ofc.THREE, ofc.FOUR, ofc.FIVE, ofc.SIX, ofc.SEVEN,
              ofc.EIGHT, ofc.NINE, ofc.TEN, ofc.JACK, ofc.QUEEN, ofc.KING, ofc.ACE]

def mask_to_cards(mask):
    """ビットマスクからカードリストを取得"""
    cards = []
    for i in range(52):
        if mask & (1 << i):
            suit_idx = i // 13
            rank_idx = i % 13
            cards.append(ofc.Card(SUIT_ENUMS[suit_idx], RANK_ENUMS[rank_idx]))
    return cards

def row_to_str(board, row):
    """列のカードを文字列に変換"""
    if row == 'top':
        mask = board.top_mask()
    elif row == 'middle':
        mask = board.mid_mask()
    else:
        mask = board.bot_mask()

    cards = mask_to_cards(mask)
    return ' '.join(card_to_str(c) for c in cards) if cards else '-'

def hand_to_str(cards):
    """手札を文字列に変換"""
    return ' '.join(card_to_str(c) for c in cards)

def rank_to_str(rank):
    """ハンドランクを文字列に変換"""
    if rank is None:
        return "N/A"
    names = {
        ofc.HIGH_CARD: "High",
        ofc.ONE_PAIR: "Pair",
        ofc.TWO_PAIR: "2Pair",
        ofc.THREE_OF_A_KIND: "Trips",
        ofc.STRAIGHT: "Str",
        ofc.FLUSH: "Flush",
        ofc.FULL_HOUSE: "FH",
        ofc.FOUR_OF_A_KIND: "Quads",
        ofc.STRAIGHT_FLUSH: "SF",
        ofc.ROYAL_FLUSH: "RF",
    }
    return names.get(rank.rank, str(rank.rank))

def main():
    # モデルロード
    model_path = "models/phase9/p9_fl_mastery_150000000.zip"
    print(f"Loading model: {model_path}")
    model = MaskablePPO.load(model_path)
    print("Model loaded.\n")

    # 環境作成
    env = OFC3MaxEnv(continuous_games=True, fl_solver_mode='greedy')

    num_hands = 100
    hand_records = []

    print("Generating hands...")
    for hand_num in range(num_hands):
        env.reset()
        done = False

        # このハンドの初期手札を記録
        initial_hands = {}
        for agent in env.possible_agents:
            player_idx = int(agent.split('_')[1])
            player = env.engine.player(player_idx)
            initial_hands[agent] = player.get_hand().copy()

        while not done:
            current_agent = env.agent_selection

            if env.terminations[current_agent] or env.truncations[current_agent]:
                env.step(None)
                done = all(env.terminations.values()) or all(env.truncations.values())
                continue

            # 観測とマスク取得
            obs = env.observe(current_agent)
            mask = env.action_masks(current_agent)

            # AI推論
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            action = int(action)

            # ステップ実行
            env.step(action)
            done = all(env.terminations.values()) or all(env.truncations.values())

        # 最終結果を記録
        hand_data = {
            'hand_num': hand_num + 1,
            'players': {}
        }

        for agent in env.possible_agents:
            player_idx = int(agent.split('_')[1])
            player = env.engine.player(player_idx)
            board = player.board

            foul = board.is_foul()
            fl_qualify = board.qualifies_for_fl() if not foul else False

            top_str = row_to_str(board, 'top')
            mid_str = row_to_str(board, 'middle')
            bot_str = row_to_str(board, 'bottom')

            if not foul and board.is_complete():
                top_rank = board.evaluate_top()
                mid_rank = board.evaluate_mid()
                bot_rank = board.evaluate_bot()
                ranks_str = f"{rank_to_str(top_rank)}/{rank_to_str(mid_rank)}/{rank_to_str(bot_rank)}"
            else:
                ranks_str = "FOUL" if foul else "incomplete"

            hand_data['players'][agent] = {
                'initial_hand': hand_to_str(initial_hands.get(agent, [])),
                'top': top_str,
                'middle': mid_str,
                'bottom': bot_str,
                'ranks': ranks_str,
                'foul': foul,
                'fl_qualify': fl_qualify,
            }

        hand_records.append(hand_data)

        # 進捗表示
        if (hand_num + 1) % 10 == 0:
            print(f"  {hand_num + 1}/{num_hands}")

    # 結果表示
    print("\n" + "=" * 100)
    print("AI HAND RECORDS - Phase 9 FL Mastery (150M steps)")
    print("=" * 100)

    total_fouls = 0
    total_fl = 0

    for record in hand_records:
        print(f"\n--- Hand #{record['hand_num']} ---")

        for agent in ['player_0', 'player_1', 'player_2']:
            p = record['players'][agent]
            prefix = "🤖" if agent == 'player_0' else "  "

            if p['foul']:
                status = "🚫FOUL"
            elif p['fl_qualify']:
                status = "⭐FL!"
            else:
                status = ""

            print(f"{prefix} {agent}: Top[{p['top']:15}] Mid[{p['middle']:20}] Bot[{p['bottom']:20}] → {p['ranks']:15} {status}")

            if agent == 'player_0':
                total_fouls += 1 if p['foul'] else 0
                total_fl += 1 if p['fl_qualify'] else 0

    # サマリー
    print("\n" + "=" * 100)
    print("SUMMARY (🤖 = Player 0 = AI)")
    print("=" * 100)
    print(f"Total Hands: {num_hands}")
    print(f"Fouls: {total_fouls} ({100*total_fouls/num_hands:.1f}%)")
    print(f"FL Qualify: {total_fl} ({100*total_fl/num_hands:.1f}%)")

if __name__ == "__main__":
    main()
