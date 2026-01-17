#!/usr/bin/env python3
"""
OFC Pineapple AI - Visual Demo
AIのプレイを視覚的に表示するデモスクリプト

使用方法:
    python src/python/visual_demo.py                    # 1ゲーム表示
    python src/python/visual_demo.py --games 5          # 5ゲーム表示
    python src/python/visual_demo.py --stats 100        # 100ゲームの統計
"""

import os
import sys
import argparse
import random
import numpy as np
from collections import defaultdict
import torch
torch.distributions.Distribution.set_default_validate_args(False)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

import ofc_engine as ofc
from sb3_contrib import MaskablePPO
from ofc_3max_env import OFC3MaxEnv


# カード表示用
SUIT_SYMBOLS = {'s': '\u2660', 'h': '\u2665', 'd': '\u2666', 'c': '\u2663'}
RANK_NAMES = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']


def card_to_string(card) -> str:
    """カードを見やすい文字列に変換"""
    idx = card.index
    if idx >= 52:  # Joker
        return "JK" if idx == 52 else "jk"
    rank = RANK_NAMES[idx % 13]
    suits = ['s', 'h', 'd', 'c']
    suit = SUIT_SYMBOLS[suits[idx // 13]]
    return f"{rank}{suit}"


def format_board(board, title: str = "") -> str:
    """ボードを見やすく整形"""
    lines = []
    if title:
        lines.append(f"  {title}")
        lines.append("  " + "-" * 20)

    # Top (3 cards)
    top_cards = []
    for i in range(54):
        if (board.top_mask() >> i) & 1:
            c = ofc.Card(i)
            top_cards.append(card_to_string(c))
    top_str = " ".join(top_cards) if top_cards else "[ ][ ][ ]"
    lines.append(f"  Top:    {top_str:20}")

    # Middle (5 cards)
    mid_cards = []
    for i in range(54):
        if (board.mid_mask() >> i) & 1:
            c = ofc.Card(i)
            mid_cards.append(card_to_string(c))
    mid_str = " ".join(mid_cards) if mid_cards else "[ ][ ][ ][ ][ ]"
    lines.append(f"  Middle: {mid_str:20}")

    # Bottom (5 cards)
    bot_cards = []
    for i in range(54):
        if (board.bot_mask() >> i) & 1:
            c = ofc.Card(i)
            bot_cards.append(card_to_string(c))
    bot_str = " ".join(bot_cards) if bot_cards else "[ ][ ][ ][ ][ ]"
    lines.append(f"  Bottom: {bot_str:20}")

    return "\n".join(lines)


def evaluate_hand_name(mask: int, row: str) -> str:
    """役名を取得"""
    if row == "top":
        # 3枚なのでペア/トリップスのみ
        cards = []
        for i in range(54):
            if (mask >> i) & 1:
                cards.append(i)
        if len(cards) < 3:
            return "未完成"
        ranks = [c % 13 for c in cards if c < 52]
        jokers = sum(1 for c in cards if c >= 52)

        from collections import Counter
        cnt = Counter(ranks)
        if cnt:
            max_cnt = max(cnt.values()) + jokers
        else:
            max_cnt = jokers

        if max_cnt >= 3:
            return "Trips"
        elif max_cnt >= 2:
            return "Pair"
        return "High"
    else:
        # 5枚役
        cards = []
        for i in range(54):
            if (mask >> i) & 1:
                cards.append(i)
        if len(cards) < 5:
            return "未完成"

        try:
            rank = ofc.evaluate_5cards(mask)
            names = ["High", "Pair", "2Pair", "Trips", "Straight",
                     "Flush", "FullHouse", "Quads", "SF", "Royal"]
            return names[min(rank >> 20, 9)]
        except Exception as e:
            # フォールバック: 簡易判定
            ranks = [c % 13 for c in cards if c < 52]
            suits = [c // 13 for c in cards if c < 52]
            jokers = sum(1 for c in cards if c >= 52)

            from collections import Counter
            rank_cnt = Counter(ranks)
            suit_cnt = Counter(suits)

            if rank_cnt:
                max_rank_cnt = max(rank_cnt.values()) + jokers
            else:
                max_rank_cnt = jokers

            if max_rank_cnt >= 4:
                return "Quads"
            elif max_rank_cnt >= 3:
                if len(rank_cnt) <= 2:
                    return "FullHouse"
                return "Trips"
            elif max_rank_cnt >= 2:
                pairs = sum(1 for c in rank_cnt.values() if c >= 2)
                if pairs >= 2 or (pairs == 1 and jokers >= 1):
                    return "2Pair"
                return "Pair"

            # フラッシュチェック
            if suit_cnt and max(suit_cnt.values()) + jokers >= 5:
                return "Flush"

            return "High"


class VisualDemo:
    """視覚的デモクラス"""

    def __init__(self, model_path: str = None):
        self.env = OFC3MaxEnv()
        self.model = None
        self.model_name = "Random"

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, path: str):
        """モデルをロード"""
        import zipfile
        import io
        import torch

        try:
            # ダミー環境でロード
            from sb3_contrib.common.wrappers import ActionMasker
            import gymnasium as gym

            class DummyEnv(gym.Env):
                def __init__(self, base_env):
                    self.base = base_env
                    self.observation_space = base_env.observation_space("player_0")
                    self.action_space = base_env.action_space("player_0")
                def reset(self, **kwargs): return self.base.observe("player_0"), {}
                def step(self, a): return self.base.observe("player_0"), 0, False, False, {}
                def action_masks(self): return self.base.action_masks("player_0")

            dummy = ActionMasker(DummyEnv(self.env), lambda e: e.action_masks())

            # まず新規モデル作成
            self.model = MaskablePPO(
                "MultiInputPolicy",
                dummy,
                verbose=0,
            )

            # 手動で重みをロード（NumPy互換性問題回避）
            with zipfile.ZipFile(path, 'r') as zip_ref:
                with zip_ref.open('policy.pth') as f:
                    state_dict = torch.load(io.BytesIO(f.read()), map_location='cpu')
                    self.model.policy.load_state_dict(state_dict, strict=False)

            self.model_name = os.path.basename(path).replace('.zip', '')
            print(f"[*] Loaded model: {self.model_name}")
        except Exception as e:
            print(f"[!] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.model_name = "Random"

    def get_action(self, agent: str) -> int:
        """エージェントの行動を取得"""
        valid = self.env.get_valid_actions(agent)
        if not valid:
            return 0

        if agent == "player_0" and self.model is not None:
            obs = self.env.observe(agent)
            mask = self.env.action_masks(agent)
            action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
            return int(action)
        return random.choice(valid)

    def play_visual_game(self, verbose: bool = True) -> dict:
        """1ゲームを視覚的に表示しながらプレイ"""
        self.env.reset(seed=random.randint(0, 100000))

        if verbose:
            print("\n" + "=" * 60)
            print(f"  OFC Pineapple AI Demo - {self.model_name}")
            print("=" * 60)
            print("  Player 0 = AI | Player 1, 2 = Random")
            print("=" * 60)

        step = 0
        while not all(self.env.terminations.values()):
            agent = self.env.agent_selection

            if self.env.terminations.get(agent, False):
                self.env.step(None)
                continue

            action = self.get_action(agent)
            self.env.step(action)
            step += 1

        # 結果表示
        result = self.env.engine.result()

        if verbose:
            print("\n" + "-" * 60)
            print("  FINAL BOARDS")
            print("-" * 60)

            for i in range(3):
                ps = self.env.engine.player(i)
                label = "AI" if i == 0 else f"Random {i}"
                fouled = result.is_fouled(i)
                royalty = result.get_royalty(i)
                score = result.get_score(i)

                status = "FOUL!" if fouled else f"Royalty: {royalty}"
                print(f"\n  [{label}] Score: {score:+.0f} | {status}")
                print(format_board(ps.board))

                # 役名表示
                if not fouled:
                    top_hand = evaluate_hand_name(ps.board.top_mask(), "top")
                    mid_hand = evaluate_hand_name(ps.board.mid_mask(), "mid")
                    bot_hand = evaluate_hand_name(ps.board.bot_mask(), "bot")
                    print(f"  Hands: {top_hand} / {mid_hand} / {bot_hand}")

            print("\n" + "=" * 60)
            ai_score = result.get_score(0)
            if ai_score > 0:
                print(f"  Result: AI WINS! (+{ai_score:.0f})")
            elif ai_score < 0:
                print(f"  Result: AI LOSES ({ai_score:.0f})")
            else:
                print(f"  Result: DRAW")
            print("=" * 60)

        return {
            'score': result.get_score(0),
            'royalty': result.get_royalty(0),
            'fouled': result.is_fouled(0),
            'entered_fl': result.entered_fl(0),
            'won': result.get_score(0) > 0
        }

    def run_stats(self, num_games: int = 100) -> dict:
        """複数ゲームの統計を取得"""
        print(f"\n[*] Running {num_games} games for statistics...")
        print(f"[*] Model: {self.model_name}")
        print("-" * 40)

        stats = defaultdict(list)

        for i in range(num_games):
            result = self.play_visual_game(verbose=False)
            for k, v in result.items():
                stats[k].append(v)

            if (i + 1) % 50 == 0:
                foul = np.mean(stats['fouled']) * 100
                roy = np.mean(stats['royalty'])
                win = np.mean(stats['won']) * 100
                print(f"  [{i+1}/{num_games}] Foul: {foul:.1f}% | Royalty: {roy:.2f} | Win: {win:.1f}%")

        summary = {
            'games': num_games,
            'foul_rate': np.mean(stats['fouled']) * 100,
            'mean_royalty': np.mean(stats['royalty']),
            'fl_rate': np.mean(stats['entered_fl']) * 100,
            'win_rate': np.mean(stats['won']) * 100,
            'avg_score': np.mean(stats['score']),
            'score_std': np.std(stats['score'])
        }

        print("\n" + "=" * 50)
        print("  STATISTICS SUMMARY")
        print("=" * 50)
        print(f"  Total Games:    {summary['games']}")
        print(f"  Foul Rate:      {summary['foul_rate']:.1f}%")
        print(f"  Mean Royalty:   {summary['mean_royalty']:.2f}")
        print(f"  FL Entry Rate:  {summary['fl_rate']:.1f}%")
        print(f"  Win Rate:       {summary['win_rate']:.1f}% (vs Random)")
        print(f"  Avg Score:      {summary['avg_score']:+.2f} +/- {summary['score_std']:.2f}")
        print("=" * 50)

        # 強さの評価
        print("\n  [Strength Assessment]")
        if summary['foul_rate'] < 20:
            print("  Foul Control: Excellent (Pro level)")
        elif summary['foul_rate'] < 30:
            print("  Foul Control: Good (Advanced)")
        elif summary['foul_rate'] < 40:
            print("  Foul Control: Fair (Intermediate)")
        else:
            print("  Foul Control: Needs improvement")

        if summary['mean_royalty'] > 7:
            print("  Hand Building: Excellent (Pro level)")
        elif summary['mean_royalty'] > 5:
            print("  Hand Building: Good (Advanced)")
        elif summary['mean_royalty'] > 3:
            print("  Hand Building: Fair (Intermediate)")
        else:
            print("  Hand Building: Basic")

        return summary


def find_latest_model() -> str:
    """最新のモデルを探す"""
    import glob

    model_dirs = [
        "models/p8_selfplay_*.zip",
        "models/p7_parallel_*.zip",
    ]

    for pattern in model_dirs:
        files = glob.glob(pattern)
        if files:
            latest = max(files, key=lambda f: int(f.split('_')[-1].replace('.zip', '')))
            return latest

    return None


def main():
    parser = argparse.ArgumentParser(description="OFC AI Visual Demo")
    parser.add_argument("--model", type=str, help="Model path to use")
    parser.add_argument("--games", type=int, default=1, help="Number of games to show visually")
    parser.add_argument("--stats", type=int, help="Run N games and show statistics only")
    parser.add_argument("--no-pause", action="store_true", help="Don't pause between games")

    args = parser.parse_args()

    # モデルを探す
    model_path = args.model
    if not model_path:
        model_path = find_latest_model()
        if model_path:
            print(f"[*] Auto-detected model: {model_path}")
        else:
            print("[*] No model found, using random policy")

    demo = VisualDemo(model_path)

    if args.stats:
        demo.run_stats(args.stats)
    else:
        for i in range(args.games):
            demo.play_visual_game(verbose=True)
            if i < args.games - 1 and not args.no_pause:
                try:
                    print("\n[Press Enter for next game...]")
                    input()
                except EOFError:
                    pass  # Non-interactive mode


if __name__ == "__main__":
    main()
