"""
OFC Pineapple AI - Web UI
FastAPIベースのブラウザ対戦インターフェース

起動方法:
    pip install fastapi uvicorn jinja2
    python src/python/web_ui.py

アクセス:
    http://localhost:8000
"""

import os
import sys
import json
import random
import uuid
from typing import Dict, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("[WebUI] FastAPI not installed. Run: pip install fastapi uvicorn")

try:
    import ofc_engine as ofc
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False
    print("[WebUI] ofc_engine not available. Using mock mode.")

try:
    from sb3_contrib import MaskablePPO
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False


# ========== Data Models ==========

@dataclass
class GameState:
    """ゲーム状態"""
    game_id: str
    phase: str = "waiting"  # waiting, initial, turn, showdown, complete
    current_player: int = 0
    player_board: List[List[int]] = field(default_factory=lambda: [[], [], []])
    ai_board: List[List[int]] = field(default_factory=lambda: [[], [], []])
    player_hand: List[int] = field(default_factory=list)
    used_cards: List[int] = field(default_factory=list)
    message: str = ""
    score: Optional[Dict] = None


class PlaceAction(BaseModel):
    """カード配置アクション"""
    game_id: str
    card_index: int
    row: str  # "top", "middle", "bottom"


# ========== Game Manager ==========

class GameManager:
    """ゲーム管理クラス"""

    def __init__(self, model_path: Optional[str] = None):
        self.games: Dict[str, GameState] = {}
        self.model = None
        self.model_path = model_path

        if model_path and HAS_MODEL and os.path.exists(model_path):
            try:
                self.model = MaskablePPO.load(model_path)
                print(f"[WebUI] Model loaded: {model_path}")
            except Exception as e:
                print(f"[WebUI] Failed to load model: {e}")

    def create_game(self) -> GameState:
        """新しいゲームを作成"""
        game_id = str(uuid.uuid4())[:8]
        state = GameState(game_id=game_id)

        # 初期手札を配布 (5枚)
        deck = list(range(52))
        random.shuffle(deck)

        state.player_hand = deck[:5]
        state.used_cards = deck[:10]  # 両プレイヤー分
        state.phase = "initial"
        state.message = "5枚のカードを配置してください"

        self.games[game_id] = state
        return state

    def place_card(self, game_id: str, card_index: int, row: str) -> GameState:
        """カードを配置"""
        if game_id not in self.games:
            raise ValueError("Game not found")

        state = self.games[game_id]

        if card_index < 0 or card_index >= len(state.player_hand):
            raise ValueError("Invalid card index")

        row_map = {"top": 0, "middle": 1, "bottom": 2}
        row_idx = row_map.get(row.lower())

        if row_idx is None:
            raise ValueError("Invalid row")

        # 行のカード上限チェック
        row_limits = [3, 5, 5]
        if len(state.player_board[row_idx]) >= row_limits[row_idx]:
            raise ValueError(f"{row} is full")

        # カードを配置
        card = state.player_hand.pop(card_index)
        state.player_board[row_idx].append(card)

        # 全カード配置完了チェック
        total_placed = sum(len(r) for r in state.player_board)

        if total_placed == 13:
            state.phase = "showdown"
            state = self._calculate_result(state)
        elif len(state.player_hand) == 0:
            # 次のターン (3枚配布)
            state = self._deal_next_cards(state)

        self.games[game_id] = state
        return state

    def _deal_next_cards(self, state: GameState) -> GameState:
        """次のカードを配布"""
        available = [c for c in range(52) if c not in state.used_cards]
        random.shuffle(available)

        state.player_hand = available[:3]
        state.used_cards.extend(available[:6])
        state.phase = "turn"
        state.message = "3枚から2枚を配置、1枚を捨てる"

        return state

    def _calculate_result(self, state: GameState) -> GameState:
        """結果を計算"""
        # 簡易的なファウル判定
        is_fouled = not self._is_valid_board(state.player_board)

        if is_fouled:
            state.score = {
                "player": 0,
                "ai": 6,  # スクープ
                "fouled": True,
                "message": "ファウル！ 役の強さの順序が正しくありません"
            }
        else:
            # 簡易的なスコア計算
            royalty = self._calculate_royalty(state.player_board)
            state.score = {
                "player": 1 + royalty,
                "ai": 0,
                "fouled": False,
                "royalty": royalty,
                "message": f"勝利！ ロイヤリティ: {royalty}"
            }

        state.phase = "complete"
        return state

    def _is_valid_board(self, board: List[List[int]]) -> bool:
        """ボードが有効か（役の順序が正しいか）チェック"""
        # 簡易実装: 各行のハイカードで比較
        def get_strength(cards: List[int]) -> int:
            if not cards:
                return 0
            ranks = [c % 13 for c in cards]
            return max(ranks)

        top_strength = get_strength(board[0])
        mid_strength = get_strength(board[1])
        bot_strength = get_strength(board[2])

        return bot_strength >= mid_strength >= top_strength

    def _calculate_royalty(self, board: List[List[int]]) -> int:
        """ロイヤリティを計算（簡易版）"""
        royalty = 0

        # Top: ペア以上
        top_ranks = [c % 13 for c in board[0]]
        if len(top_ranks) == 3:
            if top_ranks[0] == top_ranks[1] == top_ranks[2]:
                royalty += 10  # スリーカード
            elif top_ranks[0] == top_ranks[1] or top_ranks[1] == top_ranks[2] or top_ranks[0] == top_ranks[2]:
                if max(top_ranks) >= 4:  # 66以上
                    royalty += max(top_ranks) - 3

        return royalty

    def get_game(self, game_id: str) -> Optional[GameState]:
        """ゲーム状態を取得"""
        return self.games.get(game_id)


# ========== FastAPI App ==========

def create_app(model_path: Optional[str] = None) -> FastAPI:
    """FastAPIアプリを作成"""
    app = FastAPI(
        title="OFC Pineapple AI",
        description="Open-Face Chinese Poker Pineapple AI Web Interface",
        version="1.0.0"
    )

    manager = GameManager(model_path=model_path)

    # HTML テンプレート
    HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>OFC Pineapple AI</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 30px; font-size: 2.5em; }
        .game-board {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .row {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            align-items: center;
        }
        .row-label {
            width: 80px;
            font-weight: bold;
            color: #ffd700;
        }
        .cards { display: flex; gap: 8px; flex-wrap: wrap; }
        .card {
            width: 50px;
            height: 70px;
            background: #fff;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2em;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .card:hover { transform: translateY(-5px); box-shadow: 0 5px 20px rgba(0,0,0,0.3); }
        .card.empty {
            background: rgba(255,255,255,0.2);
            border: 2px dashed rgba(255,255,255,0.5);
        }
        .card.spade, .card.club { color: #000; }
        .card.heart, .card.diamond { color: #e63946; }
        .hand {
            background: rgba(255,215,0,0.2);
            border: 2px solid #ffd700;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .hand h3 { margin-bottom: 10px; color: #ffd700; }
        .message {
            text-align: center;
            padding: 15px;
            font-size: 1.2em;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .actions { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }
        button {
            background: #4361ee;
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
            transition: background 0.2s;
        }
        button:hover { background: #3a0ca3; }
        button.secondary { background: #6c757d; }
        .score-panel {
            background: linear-gradient(135deg, #f72585 0%, #7209b7 100%);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }
        .score-panel h2 { margin-bottom: 10px; }
        .selected { border: 3px solid #ffd700 !important; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🃏 OFC Pineapple AI</h1>

        <div id="app">
            <div class="message" id="message">ゲームを開始してください</div>

            <div class="game-board">
                <div class="row">
                    <span class="row-label">Top (3)</span>
                    <div class="cards" id="top-row"></div>
                </div>
                <div class="row">
                    <span class="row-label">Middle (5)</span>
                    <div class="cards" id="middle-row"></div>
                </div>
                <div class="row">
                    <span class="row-label">Bottom (5)</span>
                    <div class="cards" id="bottom-row"></div>
                </div>
            </div>

            <div class="hand" id="hand-section" style="display:none;">
                <h3>あなたの手札</h3>
                <div class="cards" id="hand"></div>
            </div>

            <div class="actions">
                <button onclick="newGame()">新しいゲーム</button>
                <button onclick="placeCard('top')" class="secondary">Topに配置</button>
                <button onclick="placeCard('middle')" class="secondary">Middleに配置</button>
                <button onclick="placeCard('bottom')" class="secondary">Bottomに配置</button>
            </div>
        </div>
    </div>

    <script>
        let gameId = null;
        let selectedCard = null;

        const suits = ['♠', '♥', '♦', '♣'];
        const ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K'];

        function cardToStr(cardId) {
            if (cardId >= 52) return '🃏';
            const rank = cardId % 13;
            const suit = Math.floor(cardId / 13);
            return suits[suit] + ranks[rank];
        }

        function cardClass(cardId) {
            const suit = Math.floor(cardId / 13);
            return ['spade', 'heart', 'diamond', 'club'][suit];
        }

        function renderCards(cards, containerId, clickable=false) {
            const container = document.getElementById(containerId);
            container.innerHTML = '';
            cards.forEach((card, idx) => {
                const div = document.createElement('div');
                div.className = 'card ' + cardClass(card);
                div.textContent = cardToStr(card);
                if (clickable) {
                    div.onclick = () => selectCard(idx, div);
                }
                container.appendChild(div);
            });
        }

        function renderEmptySlots(count, containerId) {
            const container = document.getElementById(containerId);
            for (let i = container.children.length; i < count; i++) {
                const div = document.createElement('div');
                div.className = 'card empty';
                container.appendChild(div);
            }
        }

        function selectCard(idx, element) {
            document.querySelectorAll('#hand .card').forEach(c => c.classList.remove('selected'));
            element.classList.add('selected');
            selectedCard = idx;
        }

        async function newGame() {
            const res = await fetch('/api/game/new', {method: 'POST'});
            const data = await res.json();
            gameId = data.game_id;
            updateUI(data);
        }

        async function placeCard(row) {
            if (!gameId || selectedCard === null) {
                alert('カードを選択してください');
                return;
            }

            const res = await fetch('/api/game/place', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({game_id: gameId, card_index: selectedCard, row: row})
            });

            if (!res.ok) {
                const err = await res.json();
                alert(err.detail);
                return;
            }

            const data = await res.json();
            selectedCard = null;
            updateUI(data);
        }

        function updateUI(state) {
            document.getElementById('message').textContent = state.message || '';

            renderCards(state.player_board[0], 'top-row');
            renderEmptySlots(3, 'top-row');
            renderCards(state.player_board[1], 'middle-row');
            renderEmptySlots(5, 'middle-row');
            renderCards(state.player_board[2], 'bottom-row');
            renderEmptySlots(5, 'bottom-row');

            const handSection = document.getElementById('hand-section');
            if (state.player_hand && state.player_hand.length > 0) {
                handSection.style.display = 'block';
                renderCards(state.player_hand, 'hand', true);
            } else {
                handSection.style.display = 'none';
            }

            if (state.score) {
                document.getElementById('message').innerHTML =
                    '<strong>' + state.score.message + '</strong><br>' +
                    'スコア: ' + state.score.player;
            }
        }
    </script>
</body>
</html>
    """

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTML_TEMPLATE

    @app.post("/api/game/new")
    async def new_game():
        state = manager.create_game()
        return asdict(state)

    @app.post("/api/game/place")
    async def place_card(action: PlaceAction):
        try:
            state = manager.place_card(action.game_id, action.card_index, action.row)
            return asdict(state)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/game/{game_id}")
    async def get_game(game_id: str):
        state = manager.get_game(game_id)
        if not state:
            raise HTTPException(status_code=404, detail="Game not found")
        return asdict(state)

    @app.get("/api/status")
    async def status():
        return {
            "version": "1.0.0",
            "model_loaded": manager.model is not None,
            "active_games": len(manager.games),
            "timestamp": datetime.now().isoformat()
        }

    return app


def main():
    import argparse

    parser = argparse.ArgumentParser(description="OFC Pineapple Web UI")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--model", type=str, help="Path to model file")
    args = parser.parse_args()

    if not HAS_FASTAPI:
        print("[Error] FastAPI is not installed")
        print("Run: pip install fastapi uvicorn")
        return

    print("=" * 60)
    print("OFC Pineapple AI - Web UI")
    print("=" * 60)
    print(f"URL: http://localhost:{args.port}")
    print(f"Model: {args.model or 'None'}")
    print()

    app = create_app(model_path=args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
