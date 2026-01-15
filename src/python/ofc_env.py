"""
OFC Pineapple AI - Gymnasium Environment
強化学習用のOpenAI Gym互換環境
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

# C++エンジンをインポート
try:
    import ofc_engine as ofc
except ImportError:
    raise ImportError("ofc_engine module not found. Run 'python setup.py build_ext --inplace' first.")


class OFCPineappleEnv(gym.Env):
    """
    OFC Pineapple 強化学習環境
    
    ソリティアモード: 1人プレイで最大スコアを目指す
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    # カード枚数
    NUM_CARDS = 52
    NUM_RANKS = 13
    NUM_SUITS = 4
    
    # ボード構成
    TOP_SIZE = 3
    MID_SIZE = 5
    BOT_SIZE = 5
    TOTAL_SLOTS = 13
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        self.render_mode = render_mode
        
        # ゲームエンジン（ソリティアモード）
        self.engine = ofc.GameEngine(1)
        self.rng_seed = None
        
        # 現在のターン情報
        self.current_street = 0  # 0=初回(5枚), 1-4=通常ターン(3枚)
        
        # ============================================
        # Observation Space
        # ============================================
        # 自分のボード: 3行 x 52カード = 156次元
        # 現在の手札: 3枚 x 52カード = 156次元 (最大5枚だが3枚でパディング)
        # 残りデッキ: 52次元
        # ゲーム状態: 5次元 (ストリート番号、Top埋まり、Mid埋まり、Bot埋まり、FL中)
        
        self.observation_space = spaces.Dict({
            'board': spaces.MultiBinary(3 * self.NUM_CARDS),  # Top/Mid/Bot x 52
            'hand': spaces.MultiBinary(5 * self.NUM_CARDS),   # 最大5枚 x 52
            'used_cards': spaces.MultiBinary(self.NUM_CARDS), # 使用済みカード
            'game_state': spaces.Box(
                low=0, 
                high=np.array([5, 3, 5, 5, 1], dtype=np.float32),
                dtype=np.float32
            ),
        })
        
        # ============================================
        # Action Space (初回5枚配置用)
        # ============================================
        # 初回: 5枚を配置 → 各カードの配置先(0=Top, 1=Mid, 2=Bot)
        # 通常: 3枚から2枚選択 + 配置先
        #
        # シンプル化: 
        # 初回は全3^5=243パターン (ただし有効なのは制限あり)
        # 通常は C(3,2)*3*3 = 27パターン
        #
        # ここでは離散アクションとして設計
        # 初回: action[0-4] = 各カードの配置先 (0,1,2)
        # 通常: action = (card1_idx, card2_idx, row1, row2) をエンコード
        
        # 初回用: 3^5 = 243アクション
        # 通常用: 3 * 3 * 3 = 27 (card1, card2, discard) パターン
        # 統合: 最大243アクション
        self.action_space = spaces.Discrete(243)
        
        # 状態変数
        self.board = None
        self.hand = []
        self.done = False
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """環境をリセット"""
        super().reset(seed=seed)
        
        if seed is not None:
            self.rng_seed = seed
        else:
            self.rng_seed = np.random.randint(0, 2**32)
        
        # ゲームエンジンをリセット
        self.engine.reset()
        self.engine.start_new_game(self.rng_seed)
        
        self.current_street = 0
        self.done = False
        
        # 手札を取得
        self._update_hand()
        
        obs = self._get_observation()
        info = {'street': self.current_street}
        
        return obs, info
    
    def step(
        self, 
        action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        アクションを実行
        
        初回(street=0): action = 3進数エンコード (各カードの配置先)
        通常(street>0): action = 複合エンコード
        """
        if self.done:
            return self._get_observation(), 0.0, True, False, {}
        
        reward = 0.0
        info = {}
        
        if self.current_street == 0:
            # 初回5枚配置
            success = self._apply_initial_action(action)
        else:
            # 通常ターン
            success = self._apply_turn_action(action)
        
        if not success:
            # 無効なアクション → ペナルティ
            reward = -10.0
            info['invalid_action'] = True
        
        # ゲーム終了チェック
        phase = self.engine.phase()
        if phase == ofc.GamePhase.COMPLETE:
            self.done = True
            reward = self._calculate_final_reward()
            info['final_reward'] = reward
        else:
            # 手札更新
            self._update_hand()
        
        obs = self._get_observation()
        truncated = False
        
        return obs, reward, self.done, truncated, info
    
    def _apply_initial_action(self, action: int) -> bool:
        """初回5枚配置アクションを適用"""
        # actionを3進数でデコード: 各桁が各カードの配置先
        placements = []
        for i in range(5):
            row = action % 3
            action //= 3
            placements.append(row)
        
        # 配置先の容量チェック
        top_count = placements.count(0)
        mid_count = placements.count(1)
        bot_count = placements.count(2)
        
        if top_count > self.TOP_SIZE or mid_count > self.MID_SIZE or bot_count > self.BOT_SIZE:
            return False
        
        # アクション構築
        ps = self.engine.player(0)
        hand = ps.get_hand()
        
        initial_action = ofc.InitialAction()
        for i, (card, row) in enumerate(zip(hand, placements)):
            initial_action.set_placement(i, card, ofc.Row(row))
        
        success = self.engine.apply_initial_action(0, initial_action)
        
        if success:
            self.current_street = 1
        
        return success
    
    def _apply_turn_action(self, action: int) -> bool:
        """通常ターン（3枚から2枚選択）アクションを適用"""
        # action = card1 * 9 + row1 * 3 + row2
        # card1は配置するカードのインデックス(0,1,2から2枚選ぶ)
        # ここでは簡略化: action = combination_idx * 9 + row1 * 3 + row2
        
        # 組み合わせ: (0,1), (0,2), (1,2) の3パターン
        combinations = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]  # (place1, place2, discard)
        
        comb_idx = action // 9
        row_action = action % 9
        row1 = row_action // 3
        row2 = row_action % 3
        
        if comb_idx >= len(combinations):
            return False
        
        place1_idx, place2_idx, discard_idx = combinations[comb_idx]
        
        ps = self.engine.player(0)
        hand = ps.get_hand()
        
        if len(hand) < 3:
            return False
        
        # 配置先容量チェック
        board = ps.board
        if not board.can_place(ofc.Row(row1)) or not board.can_place(ofc.Row(row2)):
            return False
        
        # 同じ行に2枚配置する場合のチェック
        if row1 == row2:
            if board.remaining_slots(ofc.Row(row1)) < 2:
                return False
        
        turn_action = ofc.TurnAction()
        turn_action.set_placement(0, hand[place1_idx], ofc.Row(row1))
        turn_action.set_placement(1, hand[place2_idx], ofc.Row(row2))
        turn_action.discard = hand[discard_idx]
        
        success = self.engine.apply_turn_action(0, turn_action)
        
        if success:
            self.current_street += 1
        
        return success
    
    def _calculate_final_reward(self) -> float:
        """最終報酬を計算"""
        ps = self.engine.player(0)
        board = ps.board
        
        if board.is_foul():
            # ファウル: 大きなペナルティ
            return -20.0
        
        # ロイヤリティ
        royalty = board.calculate_royalties()
        
        # FL突入ボーナス
        fl_bonus = 5.0 if board.qualifies_for_fl() else 0.0
        
        return float(royalty) + fl_bonus
    
    def _update_hand(self):
        """手札を更新"""
        ps = self.engine.player(0)
        self.hand = ps.get_hand()
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """観測を取得"""
        ps = self.engine.player(0)
        board = ps.board
        
        # ボード状態 (各行のカードをone-hot)
        board_obs = np.zeros(3 * self.NUM_CARDS, dtype=np.int8)
        
        # Top
        top_mask = board.top_mask() if hasattr(board, 'top_mask') else 0
        for i in range(self.NUM_CARDS):
            if (top_mask >> i) & 1:
                board_obs[i] = 1
        
        # (簡略化: ボードマスクの代わりにカウントのみ使用)
        
        # 手札
        hand_obs = np.zeros(5 * self.NUM_CARDS, dtype=np.int8)
        for i, card in enumerate(self.hand[:5]):
            hand_obs[i * self.NUM_CARDS + card.index] = 1
        
        # 使用済みカード
        used_obs = np.zeros(self.NUM_CARDS, dtype=np.int8)
        
        # ゲーム状態
        game_state = np.array([
            self.current_street,
            board.count(ofc.TOP),
            board.count(ofc.MIDDLE),
            board.count(ofc.BOTTOM),
            1.0 if ps.in_fantasy_land else 0.0
        ], dtype=np.float32)
        
        return {
            'board': board_obs,
            'hand': hand_obs,
            'used_cards': used_obs,
            'game_state': game_state,
        }
    
    def render(self):
        """環境を描画"""
        if self.render_mode == 'human' or self.render_mode == 'ansi':
            ps = self.engine.player(0)
            print(ps.board.to_string())
            print(f"Hand: {[str(c) for c in self.hand]}")
            print(f"Street: {self.current_street}")
    
    def get_valid_actions(self) -> List[int]:
        """有効なアクションのリストを取得"""
        valid = []
        ps = self.engine.player(0)
        board = ps.board
        
        if self.current_street == 0:
            # 初回: 容量制約を満たすパターン
            for action in range(243):
                placements = []
                a = action
                for _ in range(5):
                    placements.append(a % 3)
                    a //= 3
                
                top_count = placements.count(0)
                mid_count = placements.count(1)
                bot_count = placements.count(2)
                
                if top_count <= self.TOP_SIZE and mid_count <= self.MID_SIZE and bot_count <= self.BOT_SIZE:
                    valid.append(action)
        else:
            # 通常ターン
            for action in range(27):
                comb_idx = action // 9
                row_action = action % 9
                row1 = row_action // 3
                row2 = row_action % 3
                
                if comb_idx >= 3:
                    continue
                
                # 容量チェック
                if not board.can_place(ofc.Row(row1)):
                    continue
                if row1 == row2:
                    if board.remaining_slots(ofc.Row(row1)) < 2:
                        continue
                elif not board.can_place(ofc.Row(row2)):
                    continue
                
                valid.append(action)
        
        return valid


# 環境を登録
gym.register(
    id='OFCPineapple-v0',
    entry_point='ofc_env:OFCPineappleEnv',
)


if __name__ == '__main__':
    # 簡単なテスト
    env = OFCPineappleEnv(render_mode='human')
    obs, info = env.reset(seed=42)
    
    print("=== Initial State ===")
    env.render()
    print()
    
    # 有効なアクションを取得
    valid_actions = env.get_valid_actions()
    print(f"Valid actions: {len(valid_actions)}")
    
    # ランダムにプレイ
    total_reward = 0
    for step in range(5):
        valid = env.get_valid_actions()
        if not valid:
            break
        action = np.random.choice(valid)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        print(f"\n=== Step {step + 1} ===")
        env.render()
        print(f"Reward: {reward}, Done: {done}")
        if done:
            break
    
    print(f"\n=== Total Reward: {total_reward} ===")
