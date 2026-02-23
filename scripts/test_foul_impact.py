#!/usr/bin/env python3
"""
モデル生成ボードでのC++ vs 精密フォールチェック比較
実際のAIプレイでどれだけの差が出るかを計測
"""

import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src/python'))

import numpy as np
import torch
torch.distributions.Distribution.set_default_validate_args(False)
from sb3_contrib import MaskablePPO
import ofc_engine as ofc
from collections import Counter

NUM_CARDS = 54
RANKS = 'A23456789TJQK'
SUITS = 'shdc'
RANK_VALUES = {'A': 14, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
               '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13}

def card_str_to_index(card_str):
    if card_str == 'JK1': return 52
    if card_str == 'JK2': return 53
    return SUITS.index(card_str[1]) * 13 + RANKS.index(card_str[0])

def index_to_card_str(idx):
    if idx == 52: return 'JK1'
    if idx == 53: return 'JK2'
    return RANKS[idx % 13] + SUITS[idx // 13]

def make_full_deck():
    deck = []
    for s in SUITS:
        for r in RANKS:
            deck.append(r + s)
    deck.extend(['JK1', 'JK2'])
    return deck


# ===== 精密評価（TypeScript相当）=====

def parse_card(card_str):
    if card_str in ('JK1', 'JK2'):
        return (0, 'joker')
    return (RANK_VALUES[card_str[0]], card_str[1])

def _is_better(a, b):
    if a[0] != b[0]: return a[0] > b[0]
    for x, y in zip(a[1], b[1]):
        if x != y: return x > y
    return False

def _eval_5_no_joker(cards):
    ranks_sorted = sorted([r for r, s in cards], reverse=True)
    suits = [s for r, s in cards]
    is_flush = len(set(suits)) == 1
    unique_ranks = sorted(set(ranks_sorted), reverse=True)
    is_straight = False
    straight_high = 0
    if len(unique_ranks) >= 5:
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i] - unique_ranks[i+4] == 4:
                is_straight = True
                straight_high = unique_ranks[i]
                break
        if not is_straight and 14 in unique_ranks and {2,3,4,5}.issubset(set(unique_ranks)):
            is_straight = True
            straight_high = 5

    rank_counts = Counter(ranks_sorted)
    count_groups = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

    if is_straight and is_flush:
        return (8, [straight_high])
    if count_groups[0][1] == 4:
        quad_rank = count_groups[0][0]
        kicker = [r for r in ranks_sorted if r != quad_rank][0]
        return (7, [quad_rank, kicker])
    if count_groups[0][1] == 3 and count_groups[1][1] == 2:
        return (6, [count_groups[0][0], count_groups[1][0]])
    if is_flush:
        return (5, ranks_sorted)
    if is_straight:
        return (4, [straight_high])
    if count_groups[0][1] == 3:
        trip_rank = count_groups[0][0]
        kickers = sorted([r for r in ranks_sorted if r != trip_rank], reverse=True)
        return (3, [trip_rank] + kickers)
    if count_groups[0][1] == 2 and count_groups[1][1] == 2:
        pair1, pair2 = count_groups[0][0], count_groups[1][0]
        kicker = [r for r, c in rank_counts.items() if c == 1][0]
        return (2, [max(pair1,pair2), min(pair1,pair2), kicker])
    if count_groups[0][1] == 2:
        pair_rank = count_groups[0][0]
        kickers = sorted([r for r in ranks_sorted if r != pair_rank], reverse=True)
        return (1, [pair_rank] + kickers)
    return (0, ranks_sorted)

def _eval_3_no_joker(cards):
    ranks_sorted = sorted([r for r, s in cards], reverse=True)
    rank_counts = Counter(ranks_sorted)
    count_groups = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    if count_groups[0][1] == 3:
        return (3, [count_groups[0][0]])
    if count_groups[0][1] == 2:
        pair_rank = count_groups[0][0]
        kicker = [r for r in ranks_sorted if r != pair_rank][0]
        return (1, [pair_rank, kicker])
    return (0, ranks_sorted)

def eval_5_precise(card_strs):
    cards = [parse_card(c) for c in card_strs]
    joker_count = sum(1 for r, s in cards if s == 'joker')
    normal = [(r, s) for r, s in cards if s != 'joker']
    if joker_count == 0:
        return _eval_5_no_joker(normal)
    used = set((r, s) for r, s in normal)
    all_cards = [(RANK_VALUES[r], s) for s in 'shdc' for r in RANKS]
    candidates = [c for c in all_cards if c not in used]
    best = (0, [0])
    if joker_count == 1:
        for c1 in candidates:
            result = _eval_5_no_joker(normal + [c1])
            if _is_better(result, best): best = result
    else:
        for i, c1 in enumerate(candidates):
            for c2 in candidates[i+1:]:
                result = _eval_5_no_joker(normal + [c1, c2])
                if _is_better(result, best): best = result
    return best

def eval_3_precise(card_strs):
    cards = [parse_card(c) for c in card_strs]
    joker_count = sum(1 for r, s in cards if s == 'joker')
    normal = [(r, s) for r, s in cards if s != 'joker']
    if joker_count == 0:
        return _eval_3_no_joker(normal)
    used = set((r, s) for r, s in normal)
    all_cards = [(RANK_VALUES[r], s) for s in 'shdc' for r in RANKS]
    candidates = [c for c in all_cards if c not in used]
    best = (0, [0])
    if joker_count == 1:
        for c1 in candidates:
            result = _eval_3_no_joker(normal + [c1])
            if _is_better(result, best): best = result
    else:
        for i, c1 in enumerate(candidates):
            for c2 in candidates[i+1:]:
                result = _eval_3_no_joker(normal + [c1, c2])
                if _is_better(result, best): best = result
    return best

def check_foul_precise(top_strs, mid_strs, bot_strs):
    if len(top_strs) != 3 or len(mid_strs) != 5 or len(bot_strs) != 5:
        return True
    top_eval = eval_3_precise(top_strs)
    mid_eval = eval_5_precise(mid_strs)
    bot_eval = eval_5_precise(bot_strs)
    if _is_better(mid_eval, bot_eval):
        return True
    if mid_eval[0] < top_eval[0]:
        return True
    if mid_eval[0] > top_eval[0]:
        return False
    for x, y in zip(mid_eval[1], top_eval[1]):
        if x > y: return False
        if x < y: return True
    return False

def check_foul_cpp(top_idx, mid_idx, bot_idx):
    b = ofc.Board()
    for ci in top_idx: b.place_card(ofc.TOP, ofc.Card(ci))
    for ci in mid_idx: b.place_card(ofc.MIDDLE, ofc.Card(ci))
    for ci in bot_idx: b.place_card(ofc.BOTTOM, ofc.Card(ci))
    return b.is_foul()


# ===== Model simulation =====

def build_obs(hand_indices, my_board, opp_boards, round_num, position, my_discards, opp_fl):
    obs = np.zeros(881, dtype=np.float32)
    obs[0] = round_num
    obs[1] = len(my_board['top'])
    obs[2] = len(my_board['middle'])
    obs[3] = len(my_board['bottom'])
    for i in range(2):
        if i < len(opp_boards):
            obs[4+i*3] = len(opp_boards[i]['top'])
            obs[5+i*3] = len(opp_boards[i]['middle'])
            obs[6+i*3] = len(opp_boards[i]['bottom'])
    obs[10] = 0; obs[11] = 0
    obs[12] = 1 if (len(opp_fl) > 0 and opp_fl[0]) else 0
    obs[13] = 1 if (len(opp_fl) > 1 and opp_fl[1]) else 0

    for row_idx, row_name in enumerate(['top','middle','bottom']):
        for ci in my_board[row_name]:
            obs[14 + row_idx * NUM_CARDS + ci] = 1
    for d in my_discards:
        obs[176 + d] = 1
    for i, ci in enumerate(hand_indices[:5]):
        obs[230 + i * NUM_CARDS + ci] = 1
    if len(opp_boards) > 0 and not (len(opp_fl) > 0 and opp_fl[0]):
        for row_idx, row_name in enumerate(['top','middle','bottom']):
            for ci in opp_boards[0][row_name]:
                obs[500 + row_idx * NUM_CARDS + ci] = 1
    obs[662 + min(max(position,0),2)] = 1
    if len(opp_boards) > 1 and not (len(opp_fl) > 1 and opp_fl[1]):
        for row_idx, row_name in enumerate(['top','middle','bottom']):
            for ci in opp_boards[1][row_name]:
                obs[665 + row_idx * NUM_CARDS + ci] = 1
    seen = np.zeros(NUM_CARDS, dtype=np.int8)
    for rn in ['top','middle','bottom']:
        for ci in my_board[rn]: seen[ci] = 1
    for ci in hand_indices: seen[ci] = 1
    for d in my_discards: seen[d] = 1
    for opp in opp_boards:
        for rn in ['top','middle','bottom']:
            for ci in opp[rn]: seen[ci] = 1
    uc = NUM_CARDS - np.sum(seen)
    if uc > 0:
        p = 1.0/uc
        for i in range(NUM_CARDS):
            obs[827+i] = 0 if seen[i] else p
    return obs

def build_mask(hand_indices, my_board, phase):
    tc = 3 - len(my_board['top'])
    mc = 5 - len(my_board['middle'])
    bc = 5 - len(my_board['bottom'])
    mask = np.zeros(243, dtype=np.int8)
    if phase == 'initial':
        for a in range(243):
            t = a; rows = []
            for _ in range(5):
                rows.append(t%3); t//=3
            if rows.count(0)<=tc and rows.count(1)<=mc and rows.count(2)<=bc:
                mask[a] = 1
    else:
        for di in range(min(3,len(hand_indices))):
            for pa in range(9):
                r1=pa%3; r2=pa//3
                tn=(1 if r1==0 else 0)+(1 if r2==0 else 0)
                mn=(1 if r1==1 else 0)+(1 if r2==1 else 0)
                bn=(1 if r1==2 else 0)+(1 if r2==2 else 0)
                if tn<=tc and mn<=mc and bn<=bc:
                    mask[di*9+r2*3+r1] = 1
    if np.sum(mask)==0: mask[0]=1
    return mask


def main():
    model_path = 'models/phase10_gcp/p10_fl_stay_150000000.zip'
    num_games = 500
    num_players = 3

    print(f"Loading model: {model_path}")
    sb3_model = MaskablePPO.load(model_path)
    policy_net = sb3_model.policy.mlp_extractor.policy_net
    action_net = sb3_model.policy.action_net
    policy_net.eval()
    action_net.eval()

    np.random.seed(12345)

    cpp_fouls = 0
    precise_fouls = 0
    both_fouls = 0
    cpp_only_foul = 0
    precise_only_foul = 0
    total_hands = 0

    mismatch_details = []
    ace_related = 0

    for game_idx in range(num_games):
        deck_strs = make_full_deck()
        np.random.shuffle(deck_strs)
        deck_idx = [card_str_to_index(c) for c in deck_strs]
        dp = 0

        players = [{'board': {'top':[], 'middle':[], 'bottom':[]}, 'discards':[], 'position':i,
                     'board_strs': {'top':[], 'middle':[], 'bottom':[]}} for i in range(num_players)]

        # Round 1
        for i in range(num_players):
            hand = deck_idx[dp:dp+5]; hand_strs = deck_strs[dp:dp+5]; dp += 5
            ni = (i+1)%num_players; pi = (i-1)%num_players
            obs = build_obs(hand, players[i]['board'], [players[ni]['board'], players[pi]['board']], 1, i, [], [])
            mask = build_mask(hand, players[i]['board'], 'initial')
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                logits = action_net(policy_net(obs_t))
                action = torch.argmax(logits + (1-mask_t)*-1e8).item()
            t = action; rn = ['top','middle','bottom']
            for j in range(5):
                r = t%3; t//=3
                players[i]['board'][rn[r]].append(hand[j])
                players[i]['board_strs'][rn[r]].append(hand_strs[j])

        # Rounds 2-5
        for rd in range(2,6):
            for i in range(num_players):
                hand = deck_idx[dp:dp+3]; hand_strs = deck_strs[dp:dp+3]; dp += 3
                ni = (i+1)%num_players; pi = (i-1)%num_players
                obs = build_obs(hand, players[i]['board'], [players[ni]['board'], players[pi]['board']],
                                rd, i, players[i]['discards'], [])
                mask = build_mask(hand, players[i]['board'], 'pineapple')
                obs_t = torch.from_numpy(obs).unsqueeze(0)
                mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
                with torch.no_grad():
                    logits = action_net(policy_net(obs_t))
                    action = torch.argmax(logits + (1-mask_t)*-1e8).item()
                r1 = action%3; r2 = (action//3)%3; di = (action//9)%3
                pi_list = [j for j in range(3) if j != di][:2]
                players[i]['board'][rn[r1]].append(hand[pi_list[0]])
                players[i]['board_strs'][rn[r1]].append(hand_strs[pi_list[0]])
                players[i]['board'][rn[r2]].append(hand[pi_list[1]])
                players[i]['board_strs'][rn[r2]].append(hand_strs[pi_list[1]])
                players[i]['discards'].append(hand[di])

        # Evaluate
        for p in players:
            total_hands += 1
            b = p['board']
            bs = p['board_strs']

            cf = check_foul_cpp(b['top'], b['middle'], b['bottom'])
            pf = check_foul_precise(bs['top'], bs['middle'], bs['bottom'])

            if cf: cpp_fouls += 1
            if pf: precise_fouls += 1
            if cf and pf: both_fouls += 1
            elif cf and not pf: cpp_only_foul += 1
            elif not cf and pf:
                precise_only_foul += 1
                # Check if Ace-related
                has_ace = any('A' in c for c in bs['top'] + bs['middle'] + bs['bottom'])
                if has_ace:
                    ace_related += 1
                if len(mismatch_details) < 5:
                    mismatch_details.append(bs)

        if (game_idx+1) % 100 == 0:
            print(f"  Game {game_idx+1}/{num_games}: "
                  f"C++ foul {cpp_fouls/total_hands*100:.1f}%, "
                  f"Precise foul {precise_fouls/total_hands*100:.1f}%, "
                  f"C++ only {cpp_only_foul}, Precise only {precise_only_foul}")

    print(f"\n{'='*60}")
    print(f"MODEL-PRODUCED BOARD FOUL CHECK COMPARISON")
    print(f"{'='*60}")
    print(f"Total hands: {total_hands}")
    print(f"")
    print(f"C++ foul rate:      {cpp_fouls/total_hands*100:.1f}%")
    print(f"Precise foul rate:  {precise_fouls/total_hands*100:.1f}%")
    print(f"")
    print(f"Both agree foul:    {both_fouls} ({both_fouls/total_hands*100:.1f}%)")
    print(f"C++ only foul:      {cpp_only_foul} ({cpp_only_foul/total_hands*100:.1f}%) ← C++ false positive")
    print(f"Precise only foul:  {precise_only_foul} ({precise_only_foul/total_hands*100:.1f}%) ← C++ missed foul")
    print(f"Neither foul:       {total_hands-both_fouls-cpp_only_foul-precise_only_foul}")
    print(f"")
    print(f"Ace-related mismatches: {ace_related}/{precise_only_foul}")

    if mismatch_details:
        rank_names = ['High Card','Pair','Two Pair','Trips','Straight','Flush','Full House','Quads','SF']
        print(f"\n=== Mismatch Examples (C++ OK but Precise says FOUL) ===")
        for i, bs in enumerate(mismatch_details):
            t = eval_3_precise(bs['top'])
            m = eval_5_precise(bs['middle'])
            b = eval_5_precise(bs['bottom'])
            print(f"\n--- Example {i+1} ---")
            print(f"  Top: {bs['top']} → {rank_names[t[0]]} {t[1]}")
            print(f"  Mid: {bs['middle']} → {rank_names[m[0]]} {m[1]}")
            print(f"  Bot: {bs['bottom']} → {rank_names[b[0]]} {b[1]}")
            # Why is it foul?
            if _is_better(m, b):
                print(f"  FOUL: Mid > Bot ({rank_names[m[0]]} vs {rank_names[b[0]]})")
            elif m[0] < t[0]:
                print(f"  FOUL: Mid < Top (rank {m[0]} < {t[0]})")

if __name__ == '__main__':
    main()
