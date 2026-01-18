"""
OFC Pineapple AI - Web Application
デプロイ可能なFastAPI Webアプリケーション

デプロイ先:
    - Render: render.yaml で設定
    - Railway: Procfile で設定
    - Heroku: Procfile で設定
    - Docker: Dockerfile で設定

ローカル実行:
    cd webapp && uvicorn app:app --reload --port 8000
"""

import os
import sys
import json
import random
import uuid
import numpy as np
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

# FastAPI
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# CORS for frontend
from fastapi.middleware.cors import CORSMiddleware

# ========== Configuration ==========

BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
MODELS_DIR = PROJECT_DIR / "models"

# Pythonパス追加
sys.path.insert(0, str(PROJECT_DIR / "src" / "python"))

# 環境変数
MODEL_PATH = os.getenv("MODEL_PATH", None)
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# ========== AI Integration ==========

HAS_AI = False
MaskablePPO = None
MCTSFLAgent = None

try:
    from sb3_contrib import MaskablePPO as _MaskablePPO
    MaskablePPO = _MaskablePPO
    try:
        from mcts_agent import MCTSFLAgent as _MCTSFLAgent
        MCTSFLAgent = _MCTSFLAgent
    except ImportError:
        pass
    HAS_AI = True
except ImportError:
    pass


class AIPlayer:
    """AIプレイヤー抽象クラス"""

    def __init__(self, model_path: Optional[str] = None, agent_type: str = "random"):
        self.agent_type = agent_type
        self.model = None
        self.agent = None

        if agent_type == "mcts" and MCTSFLAgent is not None and model_path:
            self.agent = MCTSFLAgent(model_path=model_path)
        elif agent_type == "ppo" and MaskablePPO is not None and model_path:
            try:
                self.model = MaskablePPO.load(model_path)
            except Exception as e:
                print(f"[AIPlayer] Failed to load PPO model: {e}")
                self.agent_type = "random"

    def select_action_initial(self, hand: List[int], board: List[List[int]]) -> List[int]:
        """
        初期配置アクション選択 (5枚 -> 各行へ)
        Returns: [row0, row1, row2, row3, row4] (0=Top, 1=Middle, 2=Bottom)
        """
        if self.agent_type == "random" or self.model is None:
            return self._random_initial(hand, board)

        # PPO/MCTSの場合は観測データが必要だが、
        # 簡易版ではランダムにフォールバック
        return self._random_initial(hand, board)

    def select_action_turn(self, hand: List[int], board: List[List[int]]) -> tuple:
        """
        ターンアクション選択 (3枚 -> 2枚配置, 1枚捨て)
        Returns: (placements, discard_idx)
            placements: [(card_idx, row), (card_idx, row)]
            discard_idx: 捨てるカードのインデックス
        """
        if self.agent_type == "random" or self.model is None:
            return self._random_turn(hand, board)

        return self._random_turn(hand, board)

    def _random_initial(self, hand: List[int], board: List[List[int]]) -> List[int]:
        """ランダム初期配置"""
        limits = [3, 5, 5]
        current = [len(board[0]), len(board[1]), len(board[2])]
        rows = []
        for _ in hand:
            available = [r for r in range(3) if current[r] < limits[r]]
            row = random.choice(available) if available else 0
            current[row] += 1
            rows.append(row)
        return rows

    def _random_turn(self, hand: List[int], board: List[List[int]]) -> tuple:
        """ランダムターンアクション"""
        limits = [3, 5, 5]
        current = [len(board[0]), len(board[1]), len(board[2])]

        # 捨てるカードをランダム選択
        discard_idx = random.randint(0, 2)

        # 残り2枚を配置
        placements = []
        place_indices = [i for i in range(3) if i != discard_idx]

        for card_idx in place_indices:
            available = [r for r in range(3) if current[r] < limits[r]]
            row = random.choice(available) if available else 0
            current[row] += 1
            placements.append((card_idx, row))

        return (placements, discard_idx)


# ========== Data Models ==========

@dataclass
class Card:
    """カード"""
    id: int
    rank: int  # 0-12 (A-K)
    suit: int  # 0-3 (s,h,d,c)

    @classmethod
    def from_id(cls, card_id: int) -> "Card":
        if card_id >= 52:
            return cls(id=card_id, rank=13, suit=4)  # Joker
        return cls(id=card_id, rank=card_id % 13, suit=card_id // 13)

    def to_str(self) -> str:
        suits = ['♠', '♥', '♦', '♣', '🃏']
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', '']
        if self.id >= 52:
            return '🃏'
        return f"{ranks[self.rank]}{suits[self.suit]}"


@dataclass
class GameState:
    """ゲーム状態"""
    game_id: str
    phase: str = "waiting"
    turn: int = 0
    num_players: int = 2
    ai_type: str = "random"
    player_board: List[List[int]] = field(default_factory=lambda: [[], [], []])
    ai_boards: List[List[List[int]]] = field(default_factory=lambda: [[[], [], []]])  # 複数AI対応
    player_hand: List[int] = field(default_factory=list)
    deck: List[int] = field(default_factory=list)
    discards: List[int] = field(default_factory=list)
    message: str = ""
    result: Optional[Dict] = None
    ai_thinking: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # ターンアクション用（一括処理）
    pending_placements: List[Dict] = field(default_factory=list)  # [{card_idx, row}]
    pending_discard: Optional[int] = None

    # 取り消し用履歴
    action_history: List[Dict] = field(default_factory=list)

    # AI学習用ゲーム履歴
    game_history: List[Dict] = field(default_factory=list)

    # 後方互換性のためのプロパティ
    @property
    def ai_board(self) -> List[List[int]]:
        return self.ai_boards[0] if self.ai_boards else [[], [], []]

    @ai_board.setter
    def ai_board(self, value: List[List[int]]):
        if not self.ai_boards:
            self.ai_boards = [value]
        else:
            self.ai_boards[0] = value

    def to_dict(self) -> dict:
        ai_boards_display = []
        for ai_board in self.ai_boards:
            ai_boards_display.append([
                [Card.from_id(c).to_str() for c in row]
                for row in ai_board
            ])

        # ペンディング配置のプレビュー用ボード
        preview_board = [list(row) for row in self.player_board]
        for p in self.pending_placements:
            preview_board[p["row"]].append(self.player_hand[p["card_idx"]])

        preview_board_display = [
            [Card.from_id(c).to_str() for c in row]
            for row in preview_board
        ]

        # 残り手札（ペンディング分を除く）
        pending_indices = {p["card_idx"] for p in self.pending_placements}
        if self.pending_discard is not None:
            pending_indices.add(self.pending_discard)
        remaining_hand = [c for i, c in enumerate(self.player_hand) if i not in pending_indices]

        return {
            "game_id": self.game_id,
            "phase": self.phase,
            "turn": self.turn,
            "num_players": self.num_players,
            "ai_type": self.ai_type,
            "player_board": self.player_board,
            "player_board_display": [
                [Card.from_id(c).to_str() for c in row]
                for row in self.player_board
            ],
            "preview_board": preview_board,
            "preview_board_display": preview_board_display,
            "ai_boards": self.ai_boards,
            "ai_boards_display": ai_boards_display,
            # 後方互換性
            "ai_board": self.ai_board,
            "ai_board_display": ai_boards_display[0] if ai_boards_display else [[], [], []],
            "player_hand": self.player_hand,
            "player_hand_display": [Card.from_id(c).to_str() for c in self.player_hand],
            "remaining_hand": remaining_hand,
            "remaining_hand_display": [Card.from_id(c).to_str() for c in remaining_hand],
            "pending_placements": self.pending_placements,
            "pending_discard": self.pending_discard,
            "can_undo": len(self.action_history) > 0,
            "message": self.message,
            "result": self.result,
            "ai_thinking": self.ai_thinking,
        }


class PlaceAction(BaseModel):
    """配置アクション"""
    game_id: str
    card_index: int
    row: str  # "top", "middle", "bottom"


class DiscardAction(BaseModel):
    """捨てるアクション"""
    game_id: str
    card_index: int


class NewGameRequest(BaseModel):
    """新規ゲームリクエスト"""
    ai_type: str = "random"  # random / ppo / mcts
    num_players: int = 2     # 2 or 3


class TurnActionRequest(BaseModel):
    """ターンアクション（2枚配置+1枚捨て）"""
    game_id: str
    placements: List[Dict[str, Any]]  # [{card_idx: int, row: str}]
    discard_idx: int


class PendingAction(BaseModel):
    """仮配置/仮捨てアクション"""
    game_id: str
    card_idx: int
    action_type: str  # "place" or "discard"
    row: Optional[str] = None  # place時のみ


class UndoRequest(BaseModel):
    """取り消しリクエスト"""
    game_id: str


# ========== Game Logic ==========

class GameManager:
    """ゲーム管理"""

    def __init__(self):
        self.games: Dict[str, GameState] = {}
        self.ai_players: Dict[str, List[AIPlayer]] = {}  # game_id -> AIPlayers
        self.max_games = 1000  # メモリ制限
        self._default_model_path = self._find_default_model()

    def _find_default_model(self) -> Optional[str]:
        """デフォルトモデルパスを探す"""
        if not MODELS_DIR.exists():
            return None

        # 優先順: p7_mcts > p7_parallel > phase5 > enhanced_ppo
        patterns = [
            "p7_mcts_*.zip",
            "p7_parallel_*.zip",
            "phase5/*.zip",
            "enhanced_ppo_final.zip",
        ]

        for pattern in patterns:
            matches = list(MODELS_DIR.glob(pattern))
            if matches:
                # 最新のものを選択
                return str(max(matches, key=lambda p: p.stat().st_mtime))

        # どれもなければ最初に見つかった.zip
        all_zips = list(MODELS_DIR.glob("**/*.zip"))
        return str(all_zips[0]) if all_zips else None

    def list_available_models(self) -> List[Dict[str, str]]:
        """利用可能なモデル一覧"""
        models = [
            {"id": "random", "name": "Random AI", "type": "random"}
        ]

        if not MODELS_DIR.exists():
            return models

        # PPO models
        for path in MODELS_DIR.glob("**/*.zip"):
            rel_path = path.relative_to(MODELS_DIR)
            name = path.stem
            model_type = "ppo"
            if "mcts" in name.lower():
                model_type = "mcts"

            models.append({
                "id": str(rel_path),
                "name": name,
                "type": model_type,
                "path": str(path)
            })

        return models

    def create_game(
        self,
        ai_type: str = "random",
        num_players: int = 2
    ) -> GameState:
        """新規ゲーム作成"""
        # 古いゲームを削除
        if len(self.games) >= self.max_games:
            oldest = min(self.games.values(), key=lambda g: g.created_at)
            self._cleanup_game(oldest.game_id)

        game_id = str(uuid.uuid4())[:8]
        num_players = max(2, min(3, num_players))  # 2-3人に制限
        num_ais = num_players - 1

        # デッキ作成（52枚、ジョーカーなし）
        deck = list(range(52))
        random.shuffle(deck)

        # 手札配布: プレイヤー5枚 + AI各5枚
        cards_needed = 5 * num_players
        player_hand = deck[:5]
        ai_hands = [deck[5 + i*5 : 5 + (i+1)*5] for i in range(num_ais)]
        remaining_deck = deck[cards_needed:]

        # AIプレイヤー初期化
        model_path = self._default_model_path if ai_type != "random" else None
        ai_players = [AIPlayer(model_path=model_path, agent_type=ai_type) for _ in range(num_ais)]
        self.ai_players[game_id] = ai_players

        # AI初期配置
        ai_boards = []
        for i, ai in enumerate(ai_players):
            rows = ai.select_action_initial(ai_hands[i], [[], [], []])
            board = self._apply_initial_placement(ai_hands[i], rows)
            ai_boards.append(board)

        state = GameState(
            game_id=game_id,
            num_players=num_players,
            ai_type=ai_type,
            deck=remaining_deck,
            player_hand=player_hand,
            ai_boards=ai_boards,
            phase="initial",
            message=f"5枚のカードを配置してください（Top: 3枚, Middle: 5枚, Bottom: 5枚）\n対戦: {num_players}人戦 / AI: {ai_type.upper()}"
        )

        self.games[game_id] = state
        return state

    def _cleanup_game(self, game_id: str):
        """ゲームをクリーンアップ"""
        if game_id in self.games:
            del self.games[game_id]
        if game_id in self.ai_players:
            del self.ai_players[game_id]

    def _apply_initial_placement(self, hand: List[int], rows: List[int]) -> List[List[int]]:
        """初期配置を適用"""
        board = [[], [], []]
        for card, row in zip(hand, rows):
            board[row].append(card)
        return board

    def _random_initial_placement(self, hand: List[int]) -> List[List[int]]:
        """ランダム初期配置"""
        board = [[], [], []]
        for card in hand:
            row = random.choice([0, 1, 2])
            # 行の上限チェック
            limits = [3, 5, 5]
            while len(board[row]) >= limits[row]:
                row = (row + 1) % 3
            board[row].append(card)
        return board

    def place_card(self, game_id: str, card_index: int, row: str) -> GameState:
        """カード配置"""
        if game_id not in self.games:
            raise ValueError("ゲームが見つかりません")

        state = self.games[game_id]

        if state.phase not in ["initial", "turn"]:
            raise ValueError(f"配置できません: {state.phase}")

        if card_index < 0 or card_index >= len(state.player_hand):
            raise ValueError("無効なカードインデックス")

        row_map = {"top": 0, "middle": 1, "bottom": 2}
        row_idx = row_map.get(row.lower())
        if row_idx is None:
            raise ValueError("無効な行")

        # 行の上限チェック
        limits = [3, 5, 5]
        if len(state.player_board[row_idx]) >= limits[row_idx]:
            raise ValueError(f"{row}は満杯です")

        # カード配置
        card = state.player_hand.pop(card_index)
        state.player_board[row_idx].append(card)

        # 配置完了チェック
        total = sum(len(r) for r in state.player_board)

        if total == 13:
            # ゲーム終了
            state.phase = "complete"
            state.result = self._calculate_result(state)
            state.message = state.result["message"]
        elif len(state.player_hand) == 0:
            # 次のターン
            state = self._next_turn(state)
        else:
            state.message = f"残り {len(state.player_hand)} 枚を配置してください"

        self.games[game_id] = state
        return state

    def discard_card(self, game_id: str, card_index: int) -> GameState:
        """カードを捨てる（ターン中）"""
        if game_id not in self.games:
            raise ValueError("ゲームが見つかりません")

        state = self.games[game_id]

        if state.phase != "turn":
            raise ValueError("捨てられません")

        if len(state.player_hand) != 1:
            raise ValueError("2枚配置してから捨ててください")

        card = state.player_hand.pop(card_index)
        state.discards.append(card)

        # 次のターンへ
        state = self._next_turn(state)
        self.games[game_id] = state
        return state

    def _next_turn(self, state: GameState) -> GameState:
        """次のターンへ"""
        state.turn += 1

        total = sum(len(r) for r in state.player_board)
        if total >= 13:
            state.phase = "complete"
            state.result = self._calculate_result(state)
            state.message = state.result["message"]
            return state

        # プレイヤー + AI分の3枚配布が必要
        cards_per_turn = 3 * state.num_players
        if len(state.deck) >= cards_per_turn:
            state.player_hand = state.deck[:3]
            state.deck = state.deck[3:]

            # 各AI配置
            ai_players = self.ai_players.get(state.game_id, [])
            for i, ai_board in enumerate(state.ai_boards):
                if len(state.deck) >= 3:
                    ai_hand = state.deck[:3]
                    state.deck = state.deck[3:]

                    # AIプレイヤーを使用
                    if i < len(ai_players):
                        ai = ai_players[i]
                        placements, discard_idx = ai.select_action_turn(ai_hand, ai_board)
                        for card_idx, row in placements:
                            ai_board[row].append(ai_hand[card_idx])
                    else:
                        # フォールバック: ランダム
                        for card in ai_hand[:2]:
                            row = random.choice([0, 1, 2])
                            limits = [3, 5, 5]
                            while len(ai_board[row]) >= limits[row]:
                                row = (row + 1) % 3
                            ai_board[row].append(card)

            state.phase = "turn"
            state.message = f"ターン {state.turn}: 2枚配置し、1枚捨ててください"
        else:
            state.phase = "complete"
            state.result = self._calculate_result(state)
            state.message = state.result["message"]

        return state

    def _calculate_result(self, state: GameState) -> Dict:
        """結果計算 (2人/3人対応)"""
        player_fouled = not self._is_valid_board(state.player_board)
        player_royalty = self._calculate_royalty(state.player_board) if not player_fouled else 0

        # 各AIの状態
        ai_results = []
        for i, ai_board in enumerate(state.ai_boards):
            fouled = not self._is_valid_board(ai_board)
            royalty = self._calculate_royalty(ai_board) if not fouled else 0
            ai_results.append({
                "index": i,
                "fouled": fouled,
                "royalty": royalty
            })

        # スコア計算 (簡易版: ロイヤリティ比較)
        total_player_score = 0
        ai_scores = [0] * len(ai_results)

        if player_fouled:
            # プレイヤーがファウル: 各AIに-6
            for i, ai in enumerate(ai_results):
                if not ai["fouled"]:
                    total_player_score -= 6
                    ai_scores[i] += 6
            message = "ファウル！役の順序が正しくありません (Bottom ≥ Middle ≥ Top)"
        else:
            # 各AIとの対戦
            wins = 0
            losses = 0
            for i, ai in enumerate(ai_results):
                if ai["fouled"]:
                    total_player_score += 6
                    ai_scores[i] -= 6
                    wins += 1
                elif player_royalty > ai["royalty"]:
                    pts = 1 + player_royalty - ai["royalty"]
                    total_player_score += pts
                    ai_scores[i] -= pts
                    wins += 1
                elif player_royalty < ai["royalty"]:
                    pts = 1 + ai["royalty"] - player_royalty
                    total_player_score -= pts
                    ai_scores[i] += pts
                    losses += 1

            if wins > losses:
                message = f"勝利！スコア: +{total_player_score} (ロイヤリティ: {player_royalty})"
            elif losses > wins:
                message = f"敗北... スコア: {total_player_score}"
            else:
                message = f"引き分け スコア: {total_player_score}"

        winner = "player" if total_player_score > 0 else ("ai" if total_player_score < 0 else "draw")

        return {
            "winner": winner,
            "player_score": total_player_score,
            "player_royalty": player_royalty,
            "player_fouled": player_fouled,
            "ai_scores": ai_scores,
            "ai_results": ai_results,
            "message": message
        }

    def _is_valid_board(self, board: List[List[int]]) -> bool:
        """ボード有効性チェック"""
        def hand_strength(cards: List[int]) -> tuple:
            if not cards:
                return (0, 0)
            ranks = sorted([c % 13 for c in cards], reverse=True)
            # ペア検出
            from collections import Counter
            counts = Counter(ranks)
            max_count = max(counts.values())
            return (max_count, ranks[0] if ranks else 0)

        top_str = hand_strength(board[0])
        mid_str = hand_strength(board[1])
        bot_str = hand_strength(board[2])

        return bot_str >= mid_str >= top_str

    def _calculate_royalty(self, board: List[List[int]]) -> int:
        """ロイヤリティ計算（簡易版）"""
        royalty = 0

        # Top: ペア66以上
        if len(board[0]) == 3:
            ranks = [c % 13 for c in board[0]]
            from collections import Counter
            counts = Counter(ranks)
            if max(counts.values()) >= 2:
                pair_rank = max(r for r, c in counts.items() if c >= 2)
                if pair_rank >= 5:  # 66以上
                    royalty += pair_rank - 3
            if max(counts.values()) >= 3:
                royalty += 10  # トリップス

        return royalty

    def get_game(self, game_id: str) -> Optional[GameState]:
        return self.games.get(game_id)

    # ========== 新しいターンアクション処理 ==========

    def add_pending_action(self, game_id: str, card_idx: int, action_type: str, row: Optional[str] = None) -> GameState:
        """仮配置/仮捨てを追加"""
        if game_id not in self.games:
            raise ValueError("ゲームが見つかりません")

        state = self.games[game_id]

        if state.phase not in ["initial", "turn"]:
            raise ValueError(f"アクションできません: {state.phase}")

        if card_idx < 0 or card_idx >= len(state.player_hand):
            raise ValueError("無効なカードインデックス")

        # 既に使用済みかチェック
        used_indices = {p["card_idx"] for p in state.pending_placements}
        if state.pending_discard is not None:
            used_indices.add(state.pending_discard)
        if card_idx in used_indices:
            raise ValueError("このカードは既に選択されています")

        row_map = {"top": 0, "middle": 1, "bottom": 2}

        if action_type == "place":
            if row is None:
                raise ValueError("配置先を指定してください")
            row_idx = row_map.get(row.lower())
            if row_idx is None:
                raise ValueError("無効な行")

            # 行の上限チェック（既存 + pending）
            limits = [3, 5, 5]
            current_count = len(state.player_board[row_idx])
            pending_count = sum(1 for p in state.pending_placements if p["row"] == row_idx)
            if current_count + pending_count >= limits[row_idx]:
                raise ValueError(f"{row}は満杯です")

            # ターン中の配置上限チェック
            if state.phase == "turn" and len(state.pending_placements) >= 2:
                raise ValueError("ターン中は2枚まで配置できます")

            state.pending_placements.append({"card_idx": card_idx, "row": row_idx})
            state.action_history.append({"type": "place", "card_idx": card_idx, "row": row_idx})

        elif action_type == "discard":
            if state.phase != "turn":
                raise ValueError("初期配置では捨てられません")
            if state.pending_discard is not None:
                raise ValueError("既に捨てるカードが選択されています")

            state.pending_discard = card_idx
            state.action_history.append({"type": "discard", "card_idx": card_idx})

        self._update_message(state)
        self.games[game_id] = state
        return state

    def undo_last_action(self, game_id: str) -> GameState:
        """最後のアクションを取り消し"""
        if game_id not in self.games:
            raise ValueError("ゲームが見つかりません")

        state = self.games[game_id]

        if not state.action_history:
            raise ValueError("取り消すアクションがありません")

        last_action = state.action_history.pop()

        if last_action["type"] == "place":
            # 仮配置を取り消し
            state.pending_placements = [
                p for p in state.pending_placements
                if not (p["card_idx"] == last_action["card_idx"] and p["row"] == last_action["row"])
            ]
        elif last_action["type"] == "discard":
            state.pending_discard = None

        self._update_message(state)
        self.games[game_id] = state
        return state

    def submit_turn(self, game_id: str) -> GameState:
        """ターンアクションを確定"""
        if game_id not in self.games:
            raise ValueError("ゲームが見つかりません")

        state = self.games[game_id]

        if state.phase == "initial":
            # 初期配置: 5枚全て配置されているか
            if len(state.pending_placements) != len(state.player_hand):
                raise ValueError(f"全てのカードを配置してください（残り {len(state.player_hand) - len(state.pending_placements)} 枚）")

            # 配置を確定
            self._record_game_history(state, "initial")
            for p in sorted(state.pending_placements, key=lambda x: -x["card_idx"]):
                card = state.player_hand[p["card_idx"]]
                state.player_board[p["row"]].append(card)

            state.player_hand = []
            state.pending_placements = []
            state.action_history = []

            # 次のターンへ
            state = self._next_turn(state)

        elif state.phase == "turn":
            # ターン中: 2枚配置 + 1枚捨て
            if len(state.pending_placements) != 2:
                raise ValueError(f"2枚配置してください（現在 {len(state.pending_placements)} 枚）")
            if state.pending_discard is None:
                raise ValueError("捨てるカードを選択してください")

            # 配置を確定
            self._record_game_history(state, "turn")

            # カードIDを取得（インデックスが変わる前に）
            place_cards = [(state.player_hand[p["card_idx"]], p["row"]) for p in state.pending_placements]
            discard_card = state.player_hand[state.pending_discard]

            # ボードに配置
            for card, row in place_cards:
                state.player_board[row].append(card)

            # 捨て札に追加
            state.discards.append(discard_card)

            state.player_hand = []
            state.pending_placements = []
            state.pending_discard = None
            state.action_history = []

            # ゲーム終了チェック
            total = sum(len(r) for r in state.player_board)
            if total >= 13:
                state.phase = "complete"
                state.result = self._calculate_result(state)
                state.message = state.result["message"]
                self._save_game_history(state)
            else:
                state = self._next_turn(state)

        self.games[game_id] = state
        return state

    def _update_message(self, state: GameState):
        """状態に応じたメッセージを更新"""
        if state.phase == "initial":
            placed = len(state.pending_placements)
            total = len(state.player_hand)
            state.message = f"初期配置: {placed}/{total} 枚選択済み"
            if placed == total:
                state.message += "\n「確定」ボタンで配置を確定"
        elif state.phase == "turn":
            placed = len(state.pending_placements)
            has_discard = state.pending_discard is not None
            state.message = f"ターン {state.turn}: 配置 {placed}/2枚"
            if has_discard:
                state.message += ", 捨て札 選択済み"
            else:
                state.message += ", 捨て札 未選択"
            if placed == 2 and has_discard:
                state.message += "\n「確定」ボタンでターンを確定"

    def _record_game_history(self, state: GameState, action_type: str):
        """ゲーム履歴を記録（AI学習用）"""
        record = {
            "turn": state.turn,
            "action_type": action_type,
            "player_hand": list(state.player_hand),
            "player_board_before": [list(row) for row in state.player_board],
            "ai_boards_before": [[list(row) for row in board] for board in state.ai_boards],
            "placements": list(state.pending_placements),
            "discard": state.pending_discard,
            "timestamp": datetime.now().isoformat()
        }
        state.game_history.append(record)

    def _save_game_history(self, state: GameState):
        """ゲーム履歴をファイルに保存"""
        history_dir = PROJECT_DIR / "game_history"
        history_dir.mkdir(exist_ok=True)

        filename = f"game_{state.game_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = history_dir / filename

        data = {
            "game_id": state.game_id,
            "num_players": state.num_players,
            "ai_type": state.ai_type,
            "result": state.result,
            "player_board_final": state.player_board,
            "ai_boards_final": state.ai_boards,
            "history": state.game_history,
            "created_at": state.created_at,
            "completed_at": datetime.now().isoformat()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[GameManager] Game history saved: {filepath}")


# ========== FastAPI App ==========

app = FastAPI(
    title="OFC Pineapple AI",
    description="Open-Face Chinese Poker Pineapple - Play against AI",
    version="1.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静的ファイル
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# テンプレート
templates = None
if TEMPLATES_DIR.exists():
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ゲームマネージャー
manager = GameManager()


# ========== HTML Template (Inline) ==========

GAME_HTML = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OFC Pineapple AI</title>
    <style>
        :root {
            --bg-primary: #0f0f23;
            --bg-secondary: #1a1a3e;
            --accent: #ffd700;
            --success: #00ff88;
            --danger: #ff4757;
            --warning: #ff9f43;
            --text: #ffffff;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            min-height: 100vh;
            color: var(--text);
            padding: 20px;
        }

        .container { max-width: 1100px; margin: 0 auto; }

        header { text-align: center; margin-bottom: 30px; }

        h1 {
            font-size: 2.5rem;
            background: linear-gradient(90deg, var(--accent), var(--success));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }

        .subtitle { color: #888; font-size: 1rem; }

        .settings-panel {
            background: rgba(255,255,255,0.08);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 25px;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            align-items: center;
            justify-content: center;
        }

        .setting-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .setting-group label { font-weight: 600; color: #aaa; }

        select {
            background: rgba(255,255,255,0.1);
            color: white;
            border: 1px solid rgba(255,255,255,0.3);
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
        }

        select:focus { outline: none; border-color: var(--accent); }

        .message-box {
            background: rgba(255,255,255,0.1);
            border-left: 4px solid var(--accent);
            padding: 15px 20px;
            margin-bottom: 25px;
            border-radius: 0 10px 10px 0;
            font-size: 1.1rem;
            white-space: pre-line;
        }

        .ai-thinking {
            display: none;
            background: rgba(255,215,0,0.2);
            border: 2px solid var(--accent);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            text-align: center;
            animation: pulse 1.5s infinite;
        }

        .ai-thinking.active { display: block; }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .board-section {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
        }

        .board-title {
            font-size: 1.2rem;
            margin-bottom: 15px;
            color: var(--accent);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .board-title .badge {
            background: rgba(255,255,255,0.2);
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            color: #ccc;
        }

        .ai-boards-container {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }

        .ai-boards-container .board-section {
            flex: 1;
            min-width: 300px;
            opacity: 0.85;
        }

        .row {
            display: flex;
            align-items: center;
            margin-bottom: 12px;
            gap: 15px;
        }

        .row-label {
            width: 80px;
            font-weight: 600;
            color: #aaa;
        }

        .cards {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .card {
            width: 55px;
            height: 75px;
            background: white;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            position: relative;
        }

        .card:hover:not(.empty):not(.used) {
            transform: translateY(-8px);
            box-shadow: 0 8px 25px rgba(255,215,0,0.3);
        }

        .card.empty {
            background: rgba(255,255,255,0.1);
            border: 2px dashed rgba(255,255,255,0.3);
            cursor: default;
        }

        .card.selected {
            border: 3px solid var(--accent);
            box-shadow: 0 0 20px var(--accent);
        }

        .card.pending {
            border: 3px solid var(--warning);
            box-shadow: 0 0 15px var(--warning);
            opacity: 0.7;
        }

        .card.used {
            opacity: 0.3;
            cursor: not-allowed;
        }

        .card.discard-pending {
            border: 3px solid var(--danger);
            box-shadow: 0 0 15px var(--danger);
        }

        .card.spade, .card.club { color: #1a1a2e; }
        .card.heart, .card.diamond { color: #e63946; }

        .card .pending-label {
            position: absolute;
            bottom: -8px;
            left: 50%;
            transform: translateX(-50%);
            background: var(--warning);
            color: black;
            font-size: 0.6rem;
            padding: 1px 5px;
            border-radius: 3px;
            white-space: nowrap;
        }

        .card .discard-label {
            position: absolute;
            bottom: -8px;
            left: 50%;
            transform: translateX(-50%);
            background: var(--danger);
            color: white;
            font-size: 0.6rem;
            padding: 1px 5px;
            border-radius: 3px;
        }

        .hand-section {
            background: linear-gradient(135deg, rgba(255,215,0,0.1), rgba(255,215,0,0.05));
            border: 2px solid var(--accent);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 25px;
        }

        .hand-title {
            color: var(--accent);
            margin-bottom: 15px;
            font-size: 1.2rem;
        }

        .actions {
            display: flex;
            gap: 12px;
            justify-content: center;
            flex-wrap: wrap;
            margin-bottom: 15px;
        }

        button {
            background: linear-gradient(135deg, #4361ee, #3a0ca3);
            color: white;
            border: none;
            padding: 14px 28px;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }

        button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(67,97,238,0.4);
        }

        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        button.new-game { background: linear-gradient(135deg, var(--success), #00cc6a); }
        button.discard { background: linear-gradient(135deg, var(--danger), #cc3344); }
        button.undo { background: linear-gradient(135deg, #666, #444); }
        button.submit { background: linear-gradient(135deg, var(--accent), #cc9900); color: black; }

        .result-panel {
            background: linear-gradient(135deg, #f72585, #7209b7);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 25px;
        }

        .result-panel h2 { margin-bottom: 10px; font-size: 1.8rem; }
        .result-panel .score { font-size: 2rem; margin-bottom: 10px; }
        .result-panel .details { font-size: 1rem; color: rgba(255,255,255,0.8); white-space: pre-line; }

        .status-bar {
            background: rgba(0,0,0,0.3);
            padding: 10px 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            font-size: 0.9rem;
            color: #aaa;
        }

        .hidden { display: none !important; }

        @media (max-width: 600px) {
            h1 { font-size: 1.8rem; }
            .card { width: 45px; height: 65px; font-size: 1.1rem; }
            button { padding: 12px 20px; font-size: 0.9rem; }
            .settings-panel { flex-direction: column; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>OFC Pineapple AI</h1>
            <p class="subtitle">Open-Face Chinese Poker - AIと対戦</p>
        </header>

        <div class="settings-panel">
            <div class="setting-group">
                <label for="num-players">プレイヤー数:</label>
                <select id="num-players">
                    <option value="2">2人戦 (1 vs AI)</option>
                    <option value="3">3人戦 (1 vs AI x2)</option>
                </select>
            </div>
            <div class="setting-group">
                <label for="ai-type">AIタイプ:</label>
                <select id="ai-type">
                    <option value="random">Random AI</option>
                    <option value="ppo">PPO (強化学習)</option>
                    <option value="mcts">MCTS (探索)</option>
                </select>
            </div>
            <button class="new-game" onclick="newGame()">新しいゲーム</button>
        </div>

        <div class="message-box" id="message">
            設定を選択して「新しいゲーム」をクリックしてください
        </div>

        <div class="ai-thinking" id="ai-thinking">AI思考中...</div>

        <div id="result-panel" class="result-panel hidden">
            <h2 id="result-title">結果</h2>
            <p class="score" id="result-score"></p>
            <p class="details" id="result-details"></p>
        </div>

        <div class="board-section">
            <h3 class="board-title">
                <span>あなたのボード</span>
                <span class="badge" id="player-royalty-badge"></span>
            </h3>
            <div class="row">
                <span class="row-label">Top (3)</span>
                <div class="cards" id="player-top"></div>
            </div>
            <div class="row">
                <span class="row-label">Middle (5)</span>
                <div class="cards" id="player-middle"></div>
            </div>
            <div class="row">
                <span class="row-label">Bottom (5)</span>
                <div class="cards" id="player-bottom"></div>
            </div>
        </div>

        <div class="ai-boards-container" id="ai-boards-container"></div>

        <div class="hand-section hidden" id="hand-section">
            <h3 class="hand-title">あなたの手札 (カードを選択してから配置先を選ぶ)</h3>
            <div class="cards" id="hand"></div>
        </div>

        <div class="status-bar hidden" id="status-bar">
            配置: <span id="place-count">0</span>/2, 捨て: <span id="discard-status">未選択</span>
        </div>

        <div class="actions">
            <button onclick="placeCard('top')" id="btn-top">Topに配置</button>
            <button onclick="placeCard('middle')" id="btn-middle">Middleに配置</button>
            <button onclick="placeCard('bottom')" id="btn-bottom">Bottomに配置</button>
            <button class="discard" onclick="discardCard()" id="btn-discard">捨てる</button>
            <button class="undo" onclick="undoAction()" id="btn-undo">取り消し</button>
            <button class="submit" onclick="submitTurn()" id="btn-submit">確定</button>
        </div>
    </div>

    <script>
        let gameId = null;
        let selectedCardIndex = null;
        let currentState = null;

        function getCardClass(cardStr) {
            if (cardStr.includes('\\u2660') || cardStr.includes('\\u2663') ||
                cardStr.includes('\u2660') || cardStr.includes('\u2663')) return 'spade';
            if (cardStr.includes('\\u2665') || cardStr.includes('\\u2666') ||
                cardStr.includes('\u2665') || cardStr.includes('\u2666')) return 'heart';
            return '';
        }

        function isCardUsed(idx, state) {
            if (!state) return false;
            const pendingIndices = new Set((state.pending_placements || []).map(p => p.card_idx));
            if (state.pending_discard !== null && state.pending_discard !== undefined) {
                pendingIndices.add(state.pending_discard);
            }
            return pendingIndices.has(idx);
        }

        function getCardPendingInfo(idx, state) {
            if (!state) return null;
            for (const p of (state.pending_placements || [])) {
                if (p.card_idx === idx) {
                    const rowNames = ['Top', 'Middle', 'Bottom'];
                    return { type: 'place', row: rowNames[p.row] };
                }
            }
            if (state.pending_discard === idx) {
                return { type: 'discard' };
            }
            return null;
        }

        function renderHandCards(state) {
            const container = document.getElementById('hand');
            if (!container) return;
            container.innerHTML = '';

            const cards = state.player_hand_display || [];
            cards.forEach((card, idx) => {
                const div = document.createElement('div');
                let className = 'card ' + getCardClass(card);

                const pendingInfo = getCardPendingInfo(idx, state);
                if (pendingInfo) {
                    if (pendingInfo.type === 'place') {
                        className += ' pending';
                    } else {
                        className += ' discard-pending';
                    }
                }

                if (idx === selectedCardIndex && !pendingInfo) {
                    className += ' selected';
                }

                div.className = className;
                div.textContent = card;

                if (pendingInfo) {
                    const label = document.createElement('span');
                    if (pendingInfo.type === 'place') {
                        label.className = 'pending-label';
                        label.textContent = pendingInfo.row;
                    } else {
                        label.className = 'discard-label';
                        label.textContent = '捨';
                    }
                    div.appendChild(label);
                } else {
                    div.onclick = () => selectCard(idx, div);
                }

                container.appendChild(div);
            });
        }

        function renderBoardCards(cards, containerId) {
            const container = document.getElementById(containerId);
            if (!container) return;
            container.innerHTML = '';
            cards.forEach((card) => {
                const div = document.createElement('div');
                div.className = 'card ' + getCardClass(card);
                div.textContent = card;
                container.appendChild(div);
            });
        }

        function addEmptySlots(containerId, count, current) {
            const container = document.getElementById(containerId);
            if (!container) return;
            for (let i = current; i < count; i++) {
                const div = document.createElement('div');
                div.className = 'card empty';
                container.appendChild(div);
            }
        }

        function selectCard(idx, element) {
            if (isCardUsed(idx, currentState)) return;
            document.querySelectorAll('#hand .card').forEach(c => c.classList.remove('selected'));
            element.classList.add('selected');
            selectedCardIndex = idx;
        }

        function updateButtons(state) {
            const phase = state?.phase;
            const pendingCount = (state?.pending_placements || []).length;
            const hasDiscard = state?.pending_discard !== null && state?.pending_discard !== undefined;
            const canUndo = state?.can_undo || false;

            const btnTop = document.getElementById('btn-top');
            const btnMid = document.getElementById('btn-middle');
            const btnBot = document.getElementById('btn-bottom');
            const btnDiscard = document.getElementById('btn-discard');
            const btnUndo = document.getElementById('btn-undo');
            const btnSubmit = document.getElementById('btn-submit');
            const statusBar = document.getElementById('status-bar');

            if (phase === 'initial') {
                btnTop.disabled = selectedCardIndex === null;
                btnMid.disabled = selectedCardIndex === null;
                btnBot.disabled = selectedCardIndex === null;
                btnDiscard.classList.add('hidden');
                btnUndo.disabled = !canUndo;
                btnSubmit.disabled = pendingCount !== (state?.player_hand?.length || 5);
                statusBar.classList.add('hidden');
            } else if (phase === 'turn') {
                const canPlace = selectedCardIndex !== null && pendingCount < 2;
                btnTop.disabled = !canPlace;
                btnMid.disabled = !canPlace;
                btnBot.disabled = !canPlace;
                btnDiscard.classList.remove('hidden');
                btnDiscard.disabled = selectedCardIndex === null || hasDiscard;
                btnUndo.disabled = !canUndo;
                btnSubmit.disabled = !(pendingCount === 2 && hasDiscard);

                statusBar.classList.remove('hidden');
                document.getElementById('place-count').textContent = pendingCount;
                document.getElementById('discard-status').textContent = hasDiscard ? '選択済' : '未選択';
            } else {
                btnTop.disabled = true;
                btnMid.disabled = true;
                btnBot.disabled = true;
                btnDiscard.disabled = true;
                btnUndo.disabled = true;
                btnSubmit.disabled = true;
                statusBar.classList.add('hidden');
            }
        }

        async function newGame() {
            const numPlayers = parseInt(document.getElementById('num-players').value);
            const aiType = document.getElementById('ai-type').value;

            try {
                const res = await fetch('/api/game/new', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ai_type: aiType, num_players: numPlayers })
                });
                const data = await res.json();
                gameId = data.game_id;
                selectedCardIndex = null;
                updateUI(data);
            } catch (e) {
                alert('ゲームの作成に失敗しました');
            }
        }

        async function placeCard(row) {
            if (!gameId || selectedCardIndex === null) {
                alert('カードを選択してください');
                return;
            }

            try {
                const res = await fetch('/api/game/pending', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        game_id: gameId,
                        card_idx: selectedCardIndex,
                        action_type: 'place',
                        row: row
                    })
                });

                if (!res.ok) {
                    const err = await res.json();
                    alert(err.detail);
                    return;
                }

                const data = await res.json();
                selectedCardIndex = null;
                updateUI(data);
            } catch (e) {
                alert('エラーが発生しました');
            }
        }

        async function discardCard() {
            if (!gameId || selectedCardIndex === null) {
                alert('捨てるカードを選択してください');
                return;
            }

            try {
                const res = await fetch('/api/game/pending', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        game_id: gameId,
                        card_idx: selectedCardIndex,
                        action_type: 'discard'
                    })
                });

                if (!res.ok) {
                    const err = await res.json();
                    alert(err.detail);
                    return;
                }

                const data = await res.json();
                selectedCardIndex = null;
                updateUI(data);
            } catch (e) {
                alert('エラーが発生しました');
            }
        }

        async function undoAction() {
            if (!gameId) return;

            try {
                const res = await fetch('/api/game/undo', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ game_id: gameId })
                });

                if (!res.ok) {
                    const err = await res.json();
                    alert(err.detail);
                    return;
                }

                const data = await res.json();
                selectedCardIndex = null;
                updateUI(data);
            } catch (e) {
                alert('エラーが発生しました');
            }
        }

        async function submitTurn() {
            if (!gameId) return;

            try {
                const res = await fetch('/api/game/submit', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ game_id: gameId })
                });

                if (!res.ok) {
                    const err = await res.json();
                    alert(err.detail);
                    return;
                }

                const data = await res.json();
                selectedCardIndex = null;
                updateUI(data);
            } catch (e) {
                alert('エラーが発生しました');
            }
        }

        function renderAIBoards(state) {
            const container = document.getElementById('ai-boards-container');
            container.innerHTML = '';

            const aiBoards = state.ai_boards_display || [state.ai_board_display];

            aiBoards.forEach((boardDisplay, idx) => {
                const section = document.createElement('div');
                section.className = 'board-section';

                const aiLabel = aiBoards.length > 1 ? `AI ${idx + 1}` : 'AI';
                const aiTypeLabel = state.ai_type ? state.ai_type.toUpperCase() : 'RANDOM';

                section.innerHTML = `
                    <h3 class="board-title">
                        <span>${aiLabel}のボード</span>
                        <span class="badge">${aiTypeLabel}</span>
                    </h3>
                    <div class="row">
                        <span class="row-label">Top (3)</span>
                        <div class="cards" id="ai-${idx}-top"></div>
                    </div>
                    <div class="row">
                        <span class="row-label">Middle (5)</span>
                        <div class="cards" id="ai-${idx}-middle"></div>
                    </div>
                    <div class="row">
                        <span class="row-label">Bottom (5)</span>
                        <div class="cards" id="ai-${idx}-bottom"></div>
                    </div>
                `;
                container.appendChild(section);

                renderBoardCards(boardDisplay[0], `ai-${idx}-top`);
                addEmptySlots(`ai-${idx}-top`, 3, boardDisplay[0].length);
                renderBoardCards(boardDisplay[1], `ai-${idx}-middle`);
                addEmptySlots(`ai-${idx}-middle`, 5, boardDisplay[1].length);
                renderBoardCards(boardDisplay[2], `ai-${idx}-bottom`);
                addEmptySlots(`ai-${idx}-bottom`, 5, boardDisplay[2].length);
            });
        }

        function updateUI(state) {
            currentState = state;
            document.getElementById('message').textContent = state.message;

            const aiThinking = document.getElementById('ai-thinking');
            if (state.ai_thinking) {
                aiThinking.classList.add('active');
            } else {
                aiThinking.classList.remove('active');
            }

            // Player board with preview
            const boardDisplay = state.preview_board_display || state.player_board_display;
            renderBoardCards(boardDisplay[0], 'player-top');
            addEmptySlots('player-top', 3, boardDisplay[0].length);
            renderBoardCards(boardDisplay[1], 'player-middle');
            addEmptySlots('player-middle', 5, boardDisplay[1].length);
            renderBoardCards(boardDisplay[2], 'player-bottom');
            addEmptySlots('player-bottom', 5, boardDisplay[2].length);

            renderAIBoards(state);

            const handSection = document.getElementById('hand-section');
            if (state.player_hand_display && state.player_hand_display.length > 0) {
                handSection.classList.remove('hidden');
                renderHandCards(state);
            } else {
                handSection.classList.add('hidden');
            }

            updateButtons(state);

            const resultPanel = document.getElementById('result-panel');
            if (state.result) {
                resultPanel.classList.remove('hidden');
                document.getElementById('result-title').textContent =
                    state.result.winner === 'player' ? '勝利!' :
                    state.result.winner === 'ai' ? '敗北...' : '引き分け';
                document.getElementById('result-score').textContent =
                    `スコア: ${state.result.player_score > 0 ? '+' : ''}${state.result.player_score}`;

                let details = '';
                if (state.result.player_royalty !== undefined) {
                    details += `あなたのロイヤリティ: ${state.result.player_royalty}`;
                }
                if (state.result.player_fouled) {
                    details += ' (ファウル)';
                }
                if (state.result.ai_results) {
                    state.result.ai_results.forEach((ai, idx) => {
                        const label = state.result.ai_results.length > 1 ? `AI${idx+1}` : 'AI';
                        details += `\\n${label}: ロイヤリティ ${ai.royalty}${ai.fouled ? ' (ファウル)' : ''}`;
                    });
                }
                document.getElementById('result-details').textContent = details;
                document.getElementById('player-royalty-badge').textContent =
                    state.result.player_fouled ? 'FOUL' : `Royalty: ${state.result.player_royalty}`;
            } else {
                resultPanel.classList.add('hidden');
                document.getElementById('player-royalty-badge').textContent = '';
            }
        }
    </script>
</body>
</html>
"""


# ========== Routes ==========

@app.get("/", response_class=HTMLResponse)
async def index():
    """メインページ"""
    return HTMLResponse(content=GAME_HTML)


@app.post("/api/game/new")
async def new_game(request: NewGameRequest = None):
    """新規ゲーム作成"""
    if request is None:
        request = NewGameRequest()
    state = manager.create_game(
        ai_type=request.ai_type,
        num_players=request.num_players
    )
    return state.to_dict()


@app.get("/api/models")
async def list_models():
    """利用可能なAIモデル一覧"""
    models = manager.list_available_models()
    return {
        "models": models,
        "has_ai": HAS_AI,
        "default_model": manager._default_model_path
    }


@app.post("/api/game/place")
async def place_card(action: PlaceAction):
    """カード配置"""
    try:
        state = manager.place_card(action.game_id, action.card_index, action.row)
        return state.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/game/discard")
async def discard_card(action: DiscardAction):
    """カード捨て"""
    try:
        state = manager.discard_card(action.game_id, action.card_index)
        return state.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/game/{game_id}")
async def get_game(game_id: str):
    """ゲーム状態取得"""
    state = manager.get_game(game_id)
    if not state:
        raise HTTPException(status_code=404, detail="ゲームが見つかりません")
    return state.to_dict()


@app.get("/api/health")
async def health():
    """ヘルスチェック"""
    return {
        "status": "ok",
        "version": "1.2.0",
        "games_active": len(manager.games),
        "has_ai": HAS_AI,
        "default_model": manager._default_model_path
    }


# ========== 新しいターンアクションAPI ==========

@app.post("/api/game/pending")
async def add_pending(action: PendingAction):
    """仮配置/仮捨てを追加"""
    try:
        state = manager.add_pending_action(
            action.game_id,
            action.card_idx,
            action.action_type,
            action.row
        )
        return state.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/game/undo")
async def undo(request: UndoRequest):
    """最後のアクションを取り消し"""
    try:
        state = manager.undo_last_action(request.game_id)
        return state.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/game/submit")
async def submit_turn(request: UndoRequest):
    """ターンを確定"""
    try:
        state = manager.submit_turn(request.game_id)
        return state.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ========== Entry Point ==========

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    print("=" * 50)
    print("OFC Pineapple AI - Web Application")
    print("=" * 50)
    print(f"URL: http://localhost:{port}")
    print()

    uvicorn.run(app, host=host, port=port)
