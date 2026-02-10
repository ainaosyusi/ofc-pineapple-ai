"""
Greedy Fantasy Land Solver for Training (v3)

Major improvements in v3:
1. Flush detection and prioritization
2. Straight detection and prioritization
3. Enhanced Trips on Top search with flush/straight mid/bot
4. Better royalty maximization

Speed: ~20-50ms per solve
"""

import random
from collections import defaultdict
from itertools import combinations


def greedy_solve_fl(cards, ofc, already_in_fl=True, n_samples=500):
    n = len(cards)
    if n < 13:
        return cards[:3], cards[3:8], cards[8:13], [], 0.0, False

    best_score = -1000.0
    best_perm = None
    best_stayed = False

    indices = list(range(n))

    # Shared state across all phases
    state = _State(best_score, best_perm, best_stayed)

    # === Phase 0: High-royalty structured search (flush/straight aware) ===
    _royalty_aware_search(cards, ofc, n, already_in_fl, state)

    # === Phase 1: FL-Stay targeted sampling ===
    if already_in_fl:
        _fl_stay_search(cards, ofc, n, indices, state)

    # === Phase 2: Royalty-aware structured sampling ===
    _structured_search(cards, ofc, n, already_in_fl, state, n_trials=200)

    # === Phase 3: Random sampling ===
    _random_search(cards, ofc, n, indices, already_in_fl, state, n_trials=n_samples)

    if state.best_perm is None:
        # Fallback
        sorted_idx = sorted(range(n), key=lambda i: cards[i].index, reverse=True)
        state.best_perm = sorted_idx
        state.best_stayed = False
        state.best_score = 0.0

    p = state.best_perm
    bot = [cards[p[i]] for i in range(5)]
    mid = [cards[p[i]] for i in range(5, 10)]
    top = [cards[p[i]] for i in range(10, 13)]
    discards = [cards[p[i]] for i in range(13, n)]

    return top, mid, bot, discards, state.best_score, state.best_stayed


class _State:
    __slots__ = ('best_score', 'best_perm', 'best_stayed')
    def __init__(self, score, perm, stayed):
        self.best_score = score
        self.best_perm = perm
        self.best_stayed = stayed


def _evaluate_assignment(cards, ofc, bot_idx, mid_idx, top_idx, disc_idx,
                         already_in_fl, state):
    """Evaluate a specific card assignment and update state if better."""
    board = ofc.Board()
    for i in bot_idx:
        board.place_card(ofc.BOTTOM, cards[i])
    for i in mid_idx:
        board.place_card(ofc.MIDDLE, cards[i])
    for i in top_idx:
        board.place_card(ofc.TOP, cards[i])

    if board.is_foul():
        return False

    score = float(board.calculate_royalties())
    stayed = False
    if already_in_fl and board.can_stay_fl():
        stayed = True
        score += 15.0  # Stay bonus

    if score > state.best_score:
        state.best_score = score
        # Reconstruct perm: [bot(5), mid(5), top(3), disc(rest)]
        state.best_perm = list(bot_idx) + list(mid_idx) + list(top_idx) + list(disc_idx)
        state.best_stayed = stayed
        return True
    return False


def _get_suit(card):
    """Get suit as int (0-3)"""
    return int(card.suit())


def _get_rank(card):
    """Get rank as int (0=A, 1=2, ..., 12=K)"""
    r = int(card.rank())
    return r if r < 13 else 0  # Joker treated as Ace for sorting


def _find_flushes(cards, indices):
    """Find all possible 5+ card flushes.
    Returns list of (suit, card_indices) for suits with 5+ cards.
    """
    suit_groups = defaultdict(list)
    for idx in indices:
        c = cards[idx]
        s = _get_suit(c)
        if s < 4:  # Not joker
            suit_groups[s].append(idx)

    flushes = []
    for suit, group in suit_groups.items():
        if len(group) >= 5:
            # Sort by rank descending for best flush
            group_sorted = sorted(group, key=lambda i: _get_rank(cards[i]), reverse=True)
            flushes.append((suit, group_sorted))

    return flushes


def _find_straights(cards, indices):
    """Find all possible 5-card straights.
    Returns list of card_indices forming straights.
    """
    # Group by rank
    rank_groups = defaultdict(list)
    for idx in indices:
        c = cards[idx]
        r = _get_rank(c)
        if r < 13:  # Not joker
            rank_groups[r].append(idx)

    straights = []

    # Check each starting rank (A can be high: A-K-Q-J-T or low: A-2-3-4-5)
    # Ranks: A=0, 2=1, 3=2, ..., T=9, J=10, Q=11, K=12

    # Standard straights
    for start in range(9):  # 0-8 (A-2-3-4-5 through 9-T-J-Q-K)
        ranks_needed = [(start + i) % 13 for i in range(5)]
        if all(len(rank_groups[r]) > 0 for r in ranks_needed):
            # Found a straight, pick one card from each rank
            straight_indices = [rank_groups[r][0] for r in ranks_needed]
            straights.append(straight_indices)

    # Wheel (A-2-3-4-5)
    wheel_ranks = [0, 1, 2, 3, 4]  # A, 2, 3, 4, 5
    if all(len(rank_groups[r]) > 0 for r in wheel_ranks):
        straight_indices = [rank_groups[r][0] for r in wheel_ranks]
        straights.append(straight_indices)

    # Broadway (T-J-Q-K-A)
    broadway_ranks = [9, 10, 11, 12, 0]  # T, J, Q, K, A
    if all(len(rank_groups[r]) > 0 for r in broadway_ranks):
        straight_indices = [rank_groups[r][0] for r in broadway_ranks]
        straights.append(straight_indices)

    return straights


def _royalty_aware_search(cards, ofc, n, already_in_fl, state):
    """Phase 0: Explicitly search for high-royalty hands (flushes, straights).

    This is the key improvement - actually detect and place flushes/straights.
    """
    indices = list(range(n))
    n_discard = n - 13

    # Group cards
    rank_groups = defaultdict(list)
    suit_groups = defaultdict(list)
    for idx, c in enumerate(cards):
        rank_groups[c.rank()].append(idx)
        s = _get_suit(c)
        if s < 4:
            suit_groups[s].append(idx)

    # Find trips candidates for Top (for FL Stay)
    trips_candidates = []
    for rank, group in rank_groups.items():
        if len(group) >= 3:
            for trips in combinations(group, 3):
                trips_candidates.append(list(trips))

    # Find flushes
    flushes = _find_flushes(cards, indices)

    # Find straights
    straights = _find_straights(cards, indices)

    # === Strategy A: Trips on Top + Flush Mid + Flush Bot ===
    for trips_idx in trips_candidates:
        remaining = [i for i in indices if i not in trips_idx]
        remaining_flushes = _find_flushes(cards, remaining)

        for flush_suit, flush_cards in remaining_flushes:
            if len(flush_cards) >= 5:
                # Use flush for bottom
                bot_idx = flush_cards[:5]
                remaining2 = [i for i in remaining if i not in bot_idx]

                # Check for another flush for mid
                remaining_flushes2 = _find_flushes(cards, remaining2)
                for flush_suit2, flush_cards2 in remaining_flushes2:
                    if len(flush_cards2) >= 5:
                        mid_idx = flush_cards2[:5]
                        disc_idx = [i for i in remaining2 if i not in mid_idx][:n_discard]
                        _evaluate_assignment(cards, ofc, bot_idx, mid_idx, trips_idx,
                                             disc_idx, already_in_fl, state)

                # Or use remaining for mid (any 5)
                if len(remaining2) >= 5:
                    for _ in range(10):
                        random.shuffle(remaining2)
                        mid_idx = remaining2[:5]
                        disc_idx = remaining2[5:5+n_discard]
                        _evaluate_assignment(cards, ofc, bot_idx, mid_idx, trips_idx,
                                             disc_idx, already_in_fl, state)

    # === Strategy B: Trips on Top + Straight Mid/Bot ===
    for trips_idx in trips_candidates:
        remaining = [i for i in indices if i not in trips_idx]
        remaining_straights = _find_straights(cards, remaining)
        remaining_flushes = _find_flushes(cards, remaining)

        for straight_idx in remaining_straights:
            # Use straight for bot or mid
            remaining2 = [i for i in remaining if i not in straight_idx]

            # Straight on bottom
            if len(remaining2) >= 5:
                for _ in range(10):
                    random.shuffle(remaining2)
                    mid_idx = remaining2[:5]
                    disc_idx = remaining2[5:5+n_discard]
                    _evaluate_assignment(cards, ofc, straight_idx, mid_idx, trips_idx,
                                         disc_idx, already_in_fl, state)

            # Straight on middle (need stronger bottom)
            if len(remaining2) >= 5:
                remaining2_sorted = sorted(remaining2, key=lambda i: cards[i].index, reverse=True)
                bot_idx = remaining2_sorted[:5]
                disc_idx = remaining2_sorted[5:5+n_discard]
                _evaluate_assignment(cards, ofc, bot_idx, straight_idx, trips_idx,
                                     disc_idx, already_in_fl, state)

            # Straight on middle + Flush on bottom (HIGH VALUE!)
            remaining2_flushes = _find_flushes(cards, remaining2)
            for flush_suit, flush_cards in remaining2_flushes:
                if len(flush_cards) >= 5:
                    bot_idx = flush_cards[:5]
                    disc_idx = [i for i in remaining2 if i not in bot_idx][:n_discard]
                    _evaluate_assignment(cards, ofc, bot_idx, straight_idx, trips_idx,
                                         disc_idx, already_in_fl, state)

        # === NEW: Flush on Bot + Straight on Mid (key combination) ===
        # Try ALL 5-card combinations from flush suit to find one compatible with straight
        for flush_suit, flush_cards in remaining_flushes:
            if len(flush_cards) >= 5:
                # Try all combinations of 5 cards from flush suit
                for flush_combo in combinations(flush_cards, 5):
                    bot_idx = list(flush_combo)
                    remaining3 = [i for i in remaining if i not in bot_idx]
                    remaining3_straights = _find_straights(cards, remaining3)

                    for straight_idx in remaining3_straights:
                        disc_idx = [i for i in remaining3 if i not in straight_idx][:n_discard]
                        _evaluate_assignment(cards, ofc, bot_idx, straight_idx, trips_idx,
                                             disc_idx, already_in_fl, state)

    # === Strategy C: Flush on Bottom + various Mid/Top ===
    for flush_suit, flush_cards in flushes:
        bot_idx = flush_cards[:5]
        remaining = [i for i in indices if i not in bot_idx]

        if len(remaining) >= 8:
            # Try trips on top if available
            for trips_idx in trips_candidates:
                if all(i in remaining for i in trips_idx):
                    remaining2 = [i for i in remaining if i not in trips_idx]
                    if len(remaining2) >= 5:
                        # Check for straight in remaining
                        remaining_straights = _find_straights(cards, remaining2)
                        for straight_idx in remaining_straights:
                            disc_idx = [i for i in remaining2 if i not in straight_idx][:n_discard]
                            _evaluate_assignment(cards, ofc, bot_idx, straight_idx, trips_idx,
                                                 disc_idx, already_in_fl, state)

                        # Random mid
                        for _ in range(5):
                            random.shuffle(remaining2)
                            mid_idx = remaining2[:5]
                            disc_idx = remaining2[5:5+n_discard]
                            _evaluate_assignment(cards, ofc, bot_idx, mid_idx, trips_idx,
                                                 disc_idx, already_in_fl, state)

            # High pairs on top
            high_ranks = [12, 11, 0]  # K, Q, A
            for hr in high_ranks:
                hr_enum = None
                for rank in rank_groups:
                    if hasattr(rank, 'value') and rank.value == hr:
                        hr_enum = rank
                        break
                    elif int(rank) == hr:
                        hr_enum = rank
                        break

                if hr_enum and len(rank_groups[hr_enum]) >= 2:
                    pair_candidates = list(combinations(rank_groups[hr_enum], 2))
                    for pair in pair_candidates:
                        pair_idx = list(pair)
                        if all(i in remaining for i in pair_idx):
                            remaining2 = [i for i in remaining if i not in pair_idx]
                            if len(remaining2) >= 6:
                                third = remaining2[0]
                                top_idx = pair_idx + [third]
                                mid_idx = remaining2[1:6]
                                disc_idx = remaining2[6:6+n_discard]
                                _evaluate_assignment(cards, ofc, bot_idx, mid_idx, top_idx,
                                                     disc_idx, already_in_fl, state)

    # === Strategy D: Straight on Bottom + various Mid/Top ===
    for straight_idx in straights:
        remaining = [i for i in indices if i not in straight_idx]

        if len(remaining) >= 8:
            # Try trips on top
            for trips_idx in trips_candidates:
                if all(i in remaining for i in trips_idx):
                    remaining2 = [i for i in remaining if i not in trips_idx]
                    if len(remaining2) >= 5:
                        # Check for flush in remaining for mid
                        remaining_flushes = _find_flushes(cards, remaining2)
                        for flush_suit, flush_cards in remaining_flushes:
                            if len(flush_cards) >= 5:
                                mid_idx = flush_cards[:5]
                                disc_idx = [i for i in remaining2 if i not in mid_idx][:n_discard]
                                _evaluate_assignment(cards, ofc, straight_idx, mid_idx, trips_idx,
                                                     disc_idx, already_in_fl, state)

                        # Random mid
                        for _ in range(5):
                            random.shuffle(remaining2)
                            mid_idx = remaining2[:5]
                            disc_idx = remaining2[5:5+n_discard]
                            _evaluate_assignment(cards, ofc, straight_idx, mid_idx, trips_idx,
                                                 disc_idx, already_in_fl, state)


def _fl_stay_search(cards, ofc, n, indices, state):
    """Phase 1: Specifically target FL-stay-eligible placements.

    FL Stay requires EITHER:
    1. Trips on Top (any rank), OR
    2. Quads on Bottom

    Strategy: First try Trips on Top, then Quads on Bottom.
    """
    # Group cards by rank
    rank_groups = defaultdict(list)
    for idx, c in enumerate(cards):
        rank_groups[c.rank()].append(idx)

    n_discard = n - 13

    # === Strategy 1: Trips on Top ===
    # Find any rank with 3+ cards
    trips_candidates = []
    for rank, group in rank_groups.items():
        if len(group) >= 3:
            # Try all combinations of 3 cards from this rank for Top
            for trips in combinations(group, 3):
                trips_candidates.append(list(trips))

    for trips_idx in trips_candidates:
        remaining = [i for i in range(n) if i not in trips_idx]

        # Try multiple B/M arrangements
        for trial in range(50):  # Increased from 30
            random.shuffle(remaining)

            disc_idx = remaining[:n_discard]
            play_cards = remaining[n_discard:n_discard + 10]

            if len(play_cards) < 10:
                continue

            # Try different B/M splits
            for split_trial in range(10):  # Increased from 5
                random.shuffle(play_cards)
                bot_idx = play_cards[:5]
                mid_idx = play_cards[5:10]

                _evaluate_assignment(cards, ofc, bot_idx, mid_idx, trips_idx,
                                     disc_idx, True, state)

        # Also try sorted arrangements (strong bottom to avoid foul)
        remaining_sorted = sorted(remaining, key=lambda i: cards[i].index, reverse=True)
        disc_idx = remaining_sorted[-n_discard:] if n_discard > 0 else []
        play_cards = remaining_sorted[:10]

        if len(play_cards) >= 10:
            bot_idx = play_cards[:5]
            mid_idx = play_cards[5:10]
            _evaluate_assignment(cards, ofc, bot_idx, mid_idx, trips_idx,
                                 disc_idx, True, state)

    # === Strategy 2: Quads on Bottom (with Joker) ===
    # Find joker
    joker_idx = None
    for idx, c in enumerate(cards):
        if _get_suit(c) >= 4 or _get_rank(c) >= 13:
            joker_idx = idx
            break

    # Quads with joker (3 of a kind + joker)
    if joker_idx is not None:
        for rank, group in rank_groups.items():
            if len(group) >= 3:
                for trips in combinations(group, 3):
                    quads_idx = list(trips) + [joker_idx]
                    remaining = [i for i in range(n) if i not in quads_idx]

                    for trial in range(30):
                        random.shuffle(remaining)
                        fifth = remaining[0]
                        bot_idx = quads_idx + [fifth]
                        rest = remaining[1:]

                        disc_idx = rest[:n_discard]
                        play_cards = rest[n_discard:n_discard + 8]

                        if len(play_cards) < 8:
                            continue

                        random.shuffle(play_cards)
                        mid_idx = play_cards[:5]
                        top_idx = play_cards[5:8]

                        _evaluate_assignment(cards, ofc, bot_idx, mid_idx, top_idx,
                                             disc_idx, True, state)

    # Natural quads (4 of same rank)
    quads_candidates = []
    for rank, group in rank_groups.items():
        if len(group) >= 4:
            for quads in combinations(group, 4):
                quads_candidates.append(list(quads))

    for quads_idx in quads_candidates:
        remaining = [i for i in range(n) if i not in quads_idx]

        for trial in range(30):
            random.shuffle(remaining)

            # Pick 1 card to complete bottom (5 cards)
            fifth_bot_idx = remaining[0]
            bot_idx = quads_idx + [fifth_bot_idx]

            rest = remaining[1:]
            disc_idx = rest[:n_discard]
            play_cards = rest[n_discard:n_discard + 8]  # 5 for mid, 3 for top

            if len(play_cards) < 8:
                continue

            for split_trial in range(5):
                random.shuffle(play_cards)
                mid_idx = play_cards[:5]
                top_idx = play_cards[5:8]

                _evaluate_assignment(cards, ofc, bot_idx, mid_idx, top_idx,
                                     disc_idx, True, state)

    # === Strategy 3: High pairs on Top (for FL Entry, bonus points) ===
    high_ranks = []
    for rank in rank_groups:
        r_val = rank.value if hasattr(rank, 'value') else int(rank)
        if r_val in [0, 11, 12]:  # A, Q, K
            high_ranks.append(rank)

    for r in high_ranks:
        group = rank_groups.get(r, [])
        if len(group) >= 2:
            for pair in combinations(group, 2):
                pair_idx = list(pair)
                remaining = [i for i in range(n) if i not in pair_idx]

                for trial in range(20):
                    random.shuffle(remaining)
                    third_idx = remaining[0]
                    top_idx = pair_idx + [third_idx]

                    rest = remaining[1:]
                    disc_idx = rest[:n_discard]
                    play_cards = rest[n_discard:n_discard + 10]

                    if len(play_cards) < 10:
                        continue

                    random.shuffle(play_cards)
                    bot_idx = play_cards[:5]
                    mid_idx = play_cards[5:10]

                    _evaluate_assignment(cards, ofc, bot_idx, mid_idx, top_idx,
                                         disc_idx, True, state)


def _structured_search(cards, ofc, n, already_in_fl, state, n_trials=200):
    """Phase 2: Structured sampling that groups pairs/sets together."""
    rank_groups = defaultdict(list)
    for idx, c in enumerate(cards):
        rank_groups[c.rank()].append(idx)

    n_discard = n - 13

    for _ in range(n_trials):
        # Sort by rank with some randomness
        shuffled_groups = list(rank_groups.items())
        random.shuffle(shuffled_groups)
        # Sort groups: larger groups and higher ranks first (for bottom)
        shuffled_groups.sort(key=lambda x: (len(x[1]), x[0].value if hasattr(x[0], 'value') else 0), reverse=True)

        # Flatten: keep pairs/sets together
        ordered = []
        for rank, group in shuffled_groups:
            random.shuffle(group)
            ordered.extend(group)

        # Assign: first 5 → bot (strongest groups), next 5 → mid, next 3 → top
        if len(ordered) < 13:
            continue

        disc_idx = ordered[13:13+n_discard] if n_discard > 0 else []
        bot_idx = ordered[:5]
        mid_idx = ordered[5:10]
        top_idx = ordered[10:13]

        _evaluate_assignment(cards, ofc, bot_idx, mid_idx, top_idx,
                             disc_idx, already_in_fl, state)


def _random_search(cards, ofc, n, indices, already_in_fl, state, n_trials=500):
    """Phase 3: Pure random sampling for broad coverage."""
    n_discard = n - 13

    for _ in range(n_trials):
        random.shuffle(indices)
        bot_idx = indices[:5]
        mid_idx = indices[5:10]
        top_idx = indices[10:13]
        disc_idx = indices[13:13+n_discard] if n_discard > 0 else []

        _evaluate_assignment(cards, ofc, bot_idx, mid_idx, top_idx,
                             disc_idx, already_in_fl, state)
