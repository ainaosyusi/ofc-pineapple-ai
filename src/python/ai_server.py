"""
OFC AI Inference Server
mix-poker-app のCPUプレイヤー用推論サーバー

起動:
    python src/python/ai_server.py --model models/phase9/p9_fl_mastery_150000000.zip --port 8765

エンドポイント:
    POST /predict - カード配置を推論
    GET /health - ヘルスチェック
"""

import argparse
import json
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# C++エンジンとモデル
import ofc_engine as ofc
from sb3_contrib import MaskablePPO

app = FastAPI(title="OFC AI Server", version="1.0.0")

# グローバルモデル
model: Optional[MaskablePPO] = None


# ========================================
# カード変換
# ========================================

RANK_MAP = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
            '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
SUIT_MAP = {'s': ofc.SPADE, 'h': ofc.HEART, 'd': ofc.DIAMOND, 'c': ofc.CLUB,
            '♠': ofc.SPADE, '♥': ofc.HEART, '♦': ofc.DIAMOND, '♣': ofc.CLUB}
RANK_NAMES = '23456789TJQKA'
SUIT_NAMES = 'shdc'


def card_str_to_ofc(card_str: str) -> ofc.Card:
    """カード文字列をofc.Cardに変換 (例: 'As' or 'A♠' → Card)"""
    if len(card_str) < 2:
        raise ValueError(f"Invalid card string: {card_str}")
    rank_char = card_str[0].upper()
    suit_char = card_str[1] if len(card_str) == 2 else card_str[1]

    if rank_char not in RANK_MAP:
        raise ValueError(f"Invalid rank: {rank_char}")
    if suit_char not in SUIT_MAP:
        raise ValueError(f"Invalid suit: {suit_char}")

    rank = list(RANK_MAP.keys()).index(rank_char)
    suit = SUIT_MAP[suit_char]
    return ofc.Card(suit, getattr(ofc, ['TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN',
                                        'EIGHT', 'NINE', 'TEN', 'JACK', 'QUEEN', 'KING', 'ACE'][rank]))


def ofc_card_to_str(card: ofc.Card) -> str:
    """ofc.Cardをカード文字列に変換"""
    rank_idx = card.rank()
    suit_idx = card.suit()
    return RANK_NAMES[rank_idx] + SUIT_NAMES[suit_idx]


def row_to_ofc(row: int) -> ofc.Row:
    """行番号をofc.Rowに変換 (0=TOP, 1=MIDDLE, 2=BOTTOM)"""
    return [ofc.TOP, ofc.MIDDLE, ofc.BOTTOM][row]


def row_name_to_idx(name: str) -> int:
    """行名を番号に変換"""
    return {'top': 0, 'middle': 1, 'bottom': 2}[name.lower()]


# ========================================
# リクエスト/レスポンスモデル
# ========================================

class BoardState(BaseModel):
    top: List[str] = []
    middle: List[str] = []
    bottom: List[str] = []


class PredictRequest(BaseModel):
    phase: str  # 'initial' | 'pineapple' | 'fantasyland'
    cards: List[str]  # 手札
    board: BoardState  # 自分のボード
    opponentBoards: List[BoardState] = []  # 相手のボード（観測可能な部分）


class Placement(BaseModel):
    card: str
    row: str  # 'top' | 'middle' | 'bottom'


class PredictResponse(BaseModel):
    placements: List[Placement]
    discard: Optional[str] = None


# ========================================
# 観測生成
# ========================================

def build_observation(req: PredictRequest) -> Dict[str, np.ndarray]:
    """mix-poker-appのリクエストから881次元の観測を生成"""
    NUM_CARDS = 54  # OFC NNは54枚（Joker含む）

    # 自分のボード
    my_board = np.zeros(3 * NUM_CARDS, dtype=np.int8)
    for i, row_cards in enumerate([req.board.top, req.board.middle, req.board.bottom]):
        for card_str in row_cards:
            try:
                card = card_str_to_ofc(card_str)
                my_board[i * NUM_CARDS + card.index] = 1
            except:
                pass

    # 手札
    my_hand = np.zeros(5 * NUM_CARDS, dtype=np.int8)
    for i, card_str in enumerate(req.cards[:5]):
        try:
            card = card_str_to_ofc(card_str)
            my_hand[i * NUM_CARDS + card.index] = 1
        except:
            pass

    # 相手ボード（最大2人）
    next_board = np.zeros(3 * NUM_CARDS, dtype=np.int8)
    prev_board = np.zeros(3 * NUM_CARDS, dtype=np.int8)

    for idx, opp_board in enumerate(req.opponentBoards[:2]):
        target = next_board if idx == 0 else prev_board
        for i, row_cards in enumerate([opp_board.top, opp_board.middle, opp_board.bottom]):
            for card_str in row_cards:
                try:
                    card = card_str_to_ofc(card_str)
                    target[i * NUM_CARDS + card.index] = 1
                except:
                    pass

    # その他の観測（簡略化）
    my_discards = np.zeros(NUM_CARDS, dtype=np.int8)
    unseen_prob = np.ones(NUM_CARDS, dtype=np.float32) / NUM_CARDS
    position_info = np.array([1, 0, 0], dtype=np.int8)  # ボタン位置

    # ゲーム状態
    street = 1 if req.phase == 'initial' else 2
    game_state = np.array([
        street,
        len(req.board.top), len(req.board.middle), len(req.board.bottom),
        0, 0, 0,  # next opponent
        0, 0, 0,  # prev opponent
        0, 0, 0, 0  # FL情報
    ], dtype=np.float32)

    return {
        'my_board': my_board,
        'my_hand': my_hand,
        'next_opponent_board': next_board,
        'prev_opponent_board': prev_board,
        'my_discards': my_discards,
        'unseen_probability': unseen_prob,
        'position_info': position_info,
        'game_state': game_state,
    }


def decode_action(action: int, cards: List[str], board: BoardState, phase: str) -> PredictResponse:
    """アクション番号をPlacementsに変換"""
    placements = []
    discard = None

    if phase == 'initial':
        # 初期配置: action = sum(row_i * 3^i) for i in 0..4
        temp = action
        rows = []
        for _ in range(5):
            rows.append(temp % 3)
            temp //= 3

        row_names = ['top', 'middle', 'bottom']
        for i, row_idx in enumerate(rows):
            if i < len(cards):
                placements.append(Placement(card=cards[i], row=row_names[row_idx]))

    else:  # pineapple
        # action = row1 + row2*3 + discard_idx*9
        row1 = action % 3
        row2 = (action // 3) % 3
        discard_idx = (action // 9) % 3

        row_names = ['top', 'middle', 'bottom']
        play_indices = [i for i in range(len(cards)) if i != discard_idx][:2]

        for i, play_idx in enumerate(play_indices):
            row_idx = row1 if i == 0 else row2
            placements.append(Placement(card=cards[play_idx], row=row_names[row_idx]))

        if discard_idx < len(cards):
            discard = cards[discard_idx]

    return PredictResponse(placements=placements, discard=discard)


def get_action_mask(cards: List[str], board: BoardState, phase: str) -> np.ndarray:
    """有効アクションのマスクを生成"""
    mask = np.zeros(243, dtype=np.int8)

    top_cap = 3 - len(board.top)
    mid_cap = 5 - len(board.middle)
    bot_cap = 5 - len(board.bottom)

    if phase == 'initial':
        for action in range(243):
            temp = action
            rows = []
            for _ in range(5):
                rows.append(temp % 3)
                temp //= 3

            top_count = rows.count(0)
            mid_count = rows.count(1)
            bot_count = rows.count(2)

            if top_count <= top_cap and mid_count <= mid_cap and bot_count <= bot_cap:
                mask[action] = 1
    else:
        for discard_idx in range(min(3, len(cards))):
            for placement_action in range(9):
                row1 = placement_action % 3
                row2 = placement_action // 3

                top_new = (1 if row1 == 0 else 0) + (1 if row2 == 0 else 0)
                mid_new = (1 if row1 == 1 else 0) + (1 if row2 == 1 else 0)
                bot_new = (1 if row1 == 2 else 0) + (1 if row2 == 2 else 0)

                if top_new <= top_cap and mid_new <= mid_cap and bot_new <= bot_cap:
                    action = discard_idx * 9 + row2 * 3 + row1
                    mask[action] = 1

    if mask.sum() == 0:
        mask[0] = 1

    return mask


# ========================================
# エンドポイント
# ========================================

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        obs = build_observation(req)
        mask = get_action_mask(req.cards, req.board, req.phase)

        action, _ = model.predict(obs, action_masks=mask, deterministic=True)

        response = decode_action(int(action), req.cards, req.board, req.phase)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# メイン
# ========================================

def main():
    global model

    parser = argparse.ArgumentParser(description="OFC AI Inference Server")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.zip)")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model = MaskablePPO.load(args.model)
    print("Model loaded successfully!")

    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
