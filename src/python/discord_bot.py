"""
OFC Pineapple AI - Discord Bot
/play コマンドでAIと対戦

セットアップ:
    1. Discord Developer Portal でBotを作成
    2. BOT_TOKEN環境変数を設定
    3. python src/python/discord_bot.py

コマンド:
    /play     - 新しいゲームを開始
    /status   - 学習状況を表示
    /board    - 現在のボード状態を表示
    /place    - カードを配置
    /quit     - ゲームを終了
"""

import os
import sys
import asyncio
import random
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

try:
    import discord
    from discord import app_commands
    from discord.ext import commands
    HAS_DISCORD = True
except ImportError:
    HAS_DISCORD = False
    print("[Bot] discord.py not installed. Run: pip install discord.py")

try:
    import ofc_engine as ofc
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False
    print("[Bot] ofc_engine not available")

from sb3_contrib import MaskablePPO

# カードの絵文字
SUIT_EMOJI = {
    's': '<:spade:>',   # スペード (実際のサーバー絵文字IDに置換)
    'h': ':heart:',      # ハート
    'd': ':diamonds:',   # ダイヤ
    'c': '<:club:>',     # クラブ
}

RANK_DISPLAY = {
    0: 'A', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7',
    7: '8', 8: '9', 9: 'T', 10: 'J', 11: 'Q', 12: 'K', 13: '🃏'
}


@dataclass
class GameSession:
    """ユーザーのゲームセッション"""
    user_id: int
    channel_id: int
    engine: any = None
    model: any = None
    phase: str = "waiting"
    last_activity: datetime = field(default_factory=datetime.now)
    player_position: int = 0  # 0 or 1
    current_hand: List[int] = field(default_factory=list)


class OFCBot(commands.Bot):
    """OFC Pineapple Discord Bot"""

    def __init__(self, model_path: Optional[str] = None):
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(
            command_prefix="!",
            intents=intents,
            description="OFC Pineapple AI Bot"
        )

        self.model_path = model_path
        self.model = None
        self.sessions: Dict[int, GameSession] = {}  # user_id -> session

    async def setup_hook(self):
        """Bot起動時の初期化"""
        # モデルをロード
        if self.model_path and os.path.exists(self.model_path):
            print(f"[Bot] Loading model: {self.model_path}")
            try:
                # ダミー環境でモデルをロード
                self.model = MaskablePPO.load(self.model_path)
                print("[Bot] Model loaded successfully")
            except Exception as e:
                print(f"[Bot] Failed to load model: {e}")

        # スラッシュコマンドを登録
        await self.tree.sync()
        print("[Bot] Commands synced")

    async def on_ready(self):
        print(f"[Bot] Logged in as {self.user}")


def card_to_str(card_id: int) -> str:
    """カードIDを表示用文字列に変換"""
    if card_id >= 52:
        return "🃏 Joker"

    rank = card_id % 13
    suit = card_id // 13  # 0=s, 1=h, 2=d, 3=c

    suit_chars = ['♠', '♥', '♦', '♣']
    rank_chars = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']

    return f"{suit_chars[suit]}{rank_chars[rank]}"


def format_board(board) -> str:
    """ボードを表示用文字列に変換"""
    lines = []

    # Top row (3 cards)
    top_cards = []
    for i in range(3):
        card = board.get_card(ofc.TOP, i) if hasattr(board, 'get_card') else -1
        if card >= 0:
            top_cards.append(card_to_str(card))
        else:
            top_cards.append("[ ]")
    lines.append(f"Top:    {' '.join(top_cards)}")

    # Middle row (5 cards)
    mid_cards = []
    for i in range(5):
        card = board.get_card(ofc.MIDDLE, i) if hasattr(board, 'get_card') else -1
        if card >= 0:
            mid_cards.append(card_to_str(card))
        else:
            mid_cards.append("[ ]")
    lines.append(f"Middle: {' '.join(mid_cards)}")

    # Bottom row (5 cards)
    bot_cards = []
    for i in range(5):
        card = board.get_card(ofc.BOTTOM, i) if hasattr(board, 'get_card') else -1
        if card >= 0:
            bot_cards.append(card_to_str(card))
        else:
            bot_cards.append("[ ]")
    lines.append(f"Bottom: {' '.join(bot_cards)}")

    return "```\n" + "\n".join(lines) + "\n```"


def format_hand(hand: List[int]) -> str:
    """手札を表示用文字列に変換"""
    return " ".join([f"[{i+1}] {card_to_str(c)}" for i, c in enumerate(hand)])


def create_bot(model_path: Optional[str] = None) -> OFCBot:
    """Botインスタンスを作成"""
    bot = OFCBot(model_path=model_path)

    @bot.tree.command(name="play", description="OFC Pineapple AIと対戦を開始")
    async def play(interaction: discord.Interaction):
        user_id = interaction.user.id

        if user_id in bot.sessions:
            await interaction.response.send_message(
                "既にゲーム中です。`/quit` で終了するか、`/board` で現在の状態を確認してください。",
                ephemeral=True
            )
            return

        # 新しいゲームセッションを作成
        session = GameSession(
            user_id=user_id,
            channel_id=interaction.channel_id
        )

        # ゲームエンジンを初期化
        if HAS_ENGINE:
            session.engine = ofc.GameEngine(2)
            session.engine.start_new_game(random.randint(0, 1000000))
            session.phase = "initial"

            # 初期手札を取得
            ps = session.engine.player(0)
            session.current_hand = list(ps.get_hand())

        bot.sessions[user_id] = session

        embed = discord.Embed(
            title="🃏 OFC Pineapple - 新しいゲーム",
            description="ゲームを開始しました！",
            color=0x00ff00
        )

        if session.current_hand:
            embed.add_field(
                name="あなたの手札",
                value=format_hand(session.current_hand),
                inline=False
            )
            embed.add_field(
                name="配置方法",
                value="`/place 1 top` - 1番目のカードをTopに配置\n"
                      "`/place 2 mid` - 2番目のカードをMiddleに配置\n"
                      "`/place 3 bot` - 3番目のカードをBottomに配置",
                inline=False
            )

        await interaction.response.send_message(embed=embed)

    @bot.tree.command(name="board", description="現在のボード状態を表示")
    async def board(interaction: discord.Interaction):
        user_id = interaction.user.id

        if user_id not in bot.sessions:
            await interaction.response.send_message(
                "ゲームが開始されていません。`/play` で開始してください。",
                ephemeral=True
            )
            return

        session = bot.sessions[user_id]

        embed = discord.Embed(
            title="🃏 現在のボード",
            color=0x0099ff
        )

        if session.engine:
            # プレイヤーのボード
            ps = session.engine.player(0)
            embed.add_field(
                name="あなたのボード",
                value=format_board(ps.board),
                inline=False
            )

            # AIのボード
            ai_ps = session.engine.player(1)
            embed.add_field(
                name="AIのボード",
                value=format_board(ai_ps.board),
                inline=False
            )

            # 現在の手札
            if session.current_hand:
                embed.add_field(
                    name="あなたの手札",
                    value=format_hand(session.current_hand),
                    inline=False
                )

        await interaction.response.send_message(embed=embed)

    @bot.tree.command(name="quit", description="ゲームを終了")
    async def quit(interaction: discord.Interaction):
        user_id = interaction.user.id

        if user_id not in bot.sessions:
            await interaction.response.send_message(
                "ゲームが開始されていません。",
                ephemeral=True
            )
            return

        del bot.sessions[user_id]

        await interaction.response.send_message(
            "ゲームを終了しました。また遊んでくださいね！ 🎮",
            ephemeral=False
        )

    @bot.tree.command(name="status", description="学習状況を表示")
    async def status(interaction: discord.Interaction):
        embed = discord.Embed(
            title="📊 OFC Pineapple AI - 学習状況",
            color=0x0099ff
        )

        embed.add_field(
            name="Phase 7: Parallel Training",
            value="```\n"
                  "Progress: ~12.5%\n"
                  "FPS: 4,494-12,382\n"
                  "Foul Rate: ~34%\n"
                  "Instance: GCP n2-standard-4\n"
                  "```",
            inline=False
        )

        embed.add_field(
            name="Best Model (Phase 4)",
            value="```\n"
                  "Foul Rate: 25.1%\n"
                  "Royalty: 0.85\n"
                  "FL Rate: 1.1%\n"
                  "```",
            inline=True
        )

        if bot.model:
            embed.add_field(
                name="Loaded Model",
                value=f"`{bot.model_path}`",
                inline=False
            )

        embed.set_footer(text=f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        await interaction.response.send_message(embed=embed)

    @bot.tree.command(name="help", description="ヘルプを表示")
    async def help_cmd(interaction: discord.Interaction):
        embed = discord.Embed(
            title="🃏 OFC Pineapple AI - ヘルプ",
            description="Open-Face Chinese Poker Pineapple AI Bot",
            color=0x00ff00
        )

        embed.add_field(
            name="コマンド一覧",
            value="```\n"
                  "/play   - 新しいゲームを開始\n"
                  "/board  - 現在のボードを表示\n"
                  "/quit   - ゲームを終了\n"
                  "/status - 学習状況を表示\n"
                  "/help   - このヘルプを表示\n"
                  "```",
            inline=False
        )

        embed.add_field(
            name="ゲームルール",
            value="• 13枚のカードを3つの列に配置\n"
                  "• Top: 3枚, Middle: 5枚, Bottom: 5枚\n"
                  "• Bottom ≥ Middle ≥ Top の強さが必要\n"
                  "• 違反するとファウル（0点）",
            inline=False
        )

        await interaction.response.send_message(embed=embed)

    return bot


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="OFC Pineapple Discord Bot")
    parser.add_argument("--token", type=str, help="Discord Bot Token")
    parser.add_argument("--model", type=str, help="Path to model file")
    args = parser.parse_args()

    if not HAS_DISCORD:
        print("[Error] discord.py is not installed")
        print("Run: pip install discord.py")
        return

    token = args.token or os.getenv("DISCORD_BOT_TOKEN")

    if not token:
        print("[Error] Discord Bot Token not provided")
        print("Set DISCORD_BOT_TOKEN environment variable or use --token")
        return

    # モデルパスを探索
    model_path = args.model
    if not model_path:
        # デフォルトのモデルを探す
        candidates = [
            "models/phase4/ofc_phase4_joker_20260115_190744_10500000_steps.zip",
            "models/p7_parallel_2400000.zip",
            "models/p7_mcts_2200000.zip",
        ]
        for path in candidates:
            if os.path.exists(path):
                model_path = path
                break

    print("=" * 60)
    print("OFC Pineapple AI - Discord Bot")
    print("=" * 60)
    print(f"Model: {model_path or 'None'}")
    print()

    bot = create_bot(model_path=model_path)
    bot.run(token)


if __name__ == "__main__":
    main()
