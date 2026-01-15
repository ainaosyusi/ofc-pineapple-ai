"""
OFC Pineapple AI - Phase 1 Training Environment
ファウル回避特化型学習用環境

特徴:
- MaskablePPO対応（action_masks()実装）
- Phase 1報酬関数（ファウル回避特化）
- シングルプレイヤー（ソリティアモード）
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

# C++エンジンをインポート
try:
    import ofc_engine as ofc
except ImportError:
    raise ImportError("ofc_engine module not found. Run 'python setup.py build_ext --inplace' first.")


class OFCPhase1Env(gym.Env):
    """
    OFC Pineapple Phase 1 環境
    
    目的: ファウル回避の学習
    報酬: ファウル回避 +10, ファウル -10
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    # カード枚数
    NUM_CARDS = 54
    NUM_RANKS = 13
    NUM_SUITS = 4
    
    # ボード構成
    TOP_SIZE = 3
    MID_SIZE = 5
    BOT_SIZE = 5
    TOTAL_SLOTS = 13
    
    # 報酬設定（Phase 2）
    REWARD_VALID_BOARD = 0.0
    REWARD_FOUL = -30.0
    REWARD_INVALID_ACTION = -5.0
    
    def __init__(self, render_mode=None, reward_royalties=False):
        """
        Args:
            render_mode: 描画モード
            reward_royalties: ロイヤリティボーナスを有効にするか（Phase 2用）
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.reward_royalties = reward_royalties
        
        # ゲームエンジン（1人プレイ）
        self.engine = ofc.GameEngine(1)
        self.player_idx = 0
        self.rng_seed = None
        self.current_street = 0
        
        # アクション空間
        self.action_space = spaces.Discrete(243)
        
        # 観測空間
        self.observation_space = spaces.Dict({
            'board': spaces.MultiBinary(3 * self.NUM_CARDS),  # 3行 x 54カード
            'hand': spaces.MultiBinary(5 * self.NUM_CARDS),   # 最大5枚
            'used_cards': spaces.MultiBinary(self.NUM_CARDS),
            'game_state': spaces.Box(
                low=0,
                high=np.array([8, 3, 5, 5, 1], dtype=np.float32),
                dtype=np.float32
            ),
        })
        
        self._valid_actions_cache = None
    
    def reset(self, seed=None, options=None):
        """環境をリセット"""
        super().reset(seed=seed)
        
        if seed is not None:
            self.rng_seed = seed
        else:
            self.rng_seed = np.random.randint(0, 2**32)
        
        self.engine.reset()
        self.engine.start_new_game(self.rng_seed)
        self.current_street = 0
        self._valid_actions_cache = None
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action):
        """アクションを実行"""
        # 現在のフェーズを確認
        phase = self.engine.phase()
        if phase in [ofc.GamePhase.SHOWDOWN, ofc.GamePhase.COMPLETE]:
            obs = self._get_observation()
            reward, info = self._calculate_final_reward()
            return obs, reward, True, False, info

        # 無効アクションチェック
        valid_actions = self.get_valid_actions()
        if action not in valid_actions:
            # 無効アクション - 固定の大きなマイナス報酬を与えて終了させる
            obs = self._get_observation()
            return obs, self.REWARD_FOUL, True, False, {
                'invalid_action': True, 
                'fouled': True,
                'royalty': 0,
                'entered_fl': False
            }
        
        # アクション適用
        if self.current_street == 0:
            success = self._apply_initial_action(action)
        else:
            success = self._apply_turn_action(action)
        
        if not success:
            obs = self._get_observation()
            return obs, self.REWARD_FOUL, True, False, {
                'apply_failed': True, 
                'fouled': True,
                'royalty': 0,
                'entered_fl': False
            }

        # 内部状態の更新（エンジンのターンと同期）
        self.current_street = self.engine.current_turn()
        self._valid_actions_cache = None
        
        # ゲーム終了チェック (SHOWDOWN または COMPLETE なら終了)
        new_phase = self.engine.phase()
        done = new_phase in [ofc.GamePhase.SHOWDOWN, ofc.GamePhase.COMPLETE]
        
        reward = 0.0
        info = {}
        
        if done:
            reward, info = self._calculate_final_reward()
        
        obs = self._get_observation()
        return obs, reward, done, False, info
    
    def _apply_initial_action(self, action):
        """初回5枚配置"""
        placements = []
        a = action
        for _ in range(5):
            placements.append(a % 3)
            a //= 3
        
        ps = self.engine.player(self.player_idx)
        hand = ps.get_hand()
        
        initial_action = ofc.InitialAction()
        for i, (card, row) in enumerate(zip(hand, placements)):
            initial_action.set_placement(i, card, ofc.Row(row))
        
        return self.engine.apply_initial_action(self.player_idx, initial_action)
    
    def _apply_turn_action(self, action):
        """通常ターン（3枚から2枚選択）"""
        combinations = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
        
        comb_idx = action // 9
        row_action = action % 9
        row1 = row_action // 3
        row2 = row_action % 3
        
        if comb_idx >= len(combinations):
            return False
        
        place1_idx, place2_idx, discard_idx = combinations[comb_idx]
        
        ps = self.engine.player(self.player_idx)
        hand = ps.get_hand()
        
        if len(hand) < 3:
            return False
        
        turn_action = ofc.TurnAction()
        turn_action.set_placement(0, hand[place1_idx], ofc.Row(row1))
        turn_action.set_placement(1, hand[place2_idx], ofc.Row(row2))
        turn_action.discard = hand[discard_idx]
        
        return self.engine.apply_turn_action(self.player_idx, turn_action)
    
    # _check_street_progress メソッドを削除 (エンジンのターン管理に任せる)
    
    def _calculate_final_reward(self):
        """
        Phase 1 報酬関数
        
        Returns:
            (reward, info)
        """
        result = self.engine.result()
        ps = self.engine.player(self.player_idx)
        
        fouled = result.is_fouled(self.player_idx)
        royalty = result.get_royalty(self.player_idx)
        
        info = {
            'fouled': fouled,
            'royalty': royalty,
            'entered_fl': result.entered_fl(self.player_idx),
        }
        
        if fouled:
            reward = self.REWARD_FOUL
        else:
            reward = self.REWARD_VALID_BOARD
            if self.reward_royalties:
                reward += royalty  # Phase 2: ロイヤリティをそのまま加算
        
        return reward, info
    
    def _get_observation(self):
        """観測を取得"""
        ps = self.engine.player(self.player_idx)
        board = ps.board
        
        # ボード（3行 x 52カード）
        board_obs = np.zeros(3 * self.NUM_CARDS, dtype=np.int8)
        
        # マスクからビットを取得
        masks = [board.top_mask(), board.mid_mask(), board.bot_mask()]
        for row_idx, mask in enumerate(masks):
            for i in range(self.NUM_CARDS):
                if (mask >> i) & 1:
                    board_obs[row_idx * self.NUM_CARDS + i] = 1
        
        # 手札
        hand = ps.get_hand()
        hand_obs = np.zeros(5 * self.NUM_CARDS, dtype=np.int8)
        for i, card in enumerate(hand[:5]):
            hand_obs[i * self.NUM_CARDS + card.index] = 1
        
        # 使用済みカード (全プレイヤーの全マスクの論理和)
        all_mask = 0
        for i in range(self.engine.num_players()):
            p = self.engine.player(i)
            all_mask |= p.board.all_mask()
        
        used_obs = np.zeros(self.NUM_CARDS, dtype=np.int8)
        for i in range(self.NUM_CARDS):
            if (all_mask >> i) & 1:
                used_obs[i] = 1
        
        # ゲーム状態
        game_state = np.array([
            self.current_street,
            board.count(ofc.TOP),
            board.count(ofc.MIDDLE),
            board.count(ofc.BOTTOM),
            1.0 if ps.in_fantasy_land else 0.0,
        ], dtype=np.float32)
        
        return {
            'board': board_obs,
            'hand': hand_obs,
            'used_cards': used_obs,
            'game_state': game_state,
        }
    
    def get_valid_actions(self):
        """有効なアクションのリストを取得"""
        if self._valid_actions_cache is not None:
            return self._valid_actions_cache
        
        ps = self.engine.player(self.player_idx)
        board = ps.board
        valid = []
        
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
                
                if (top_count <= self.TOP_SIZE and 
                    mid_count <= self.MID_SIZE and 
                    bot_count <= self.BOT_SIZE):
                    valid.append(action)
        else:
            # 通常ターン
            remaining = {
                0: board.remaining_slots(ofc.TOP),
                1: board.remaining_slots(ofc.MIDDLE),
                2: board.remaining_slots(ofc.BOTTOM),
            }
            
            for action in range(27):
                comb_idx = action // 9
                row_action = action % 9
                row1 = row_action // 3
                row2 = row_action % 3
                
                if comb_idx >= 3:
                    continue
                
                if remaining[row1] < 1:
                    continue
                if row1 == row2:
                    if remaining[row1] < 2:
                        continue
                elif remaining[row2] < 1:
                    continue
                
                valid.append(action)
        
        self._valid_actions_cache = valid
        return valid
    
    def action_masks(self):
        """
        MaskablePPO用アクションマスク
        
        Returns:
            np.ndarray: 有効なアクションはTrue、無効はFalse
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        for action_id in self.get_valid_actions():
            mask[action_id] = True
        return mask
    
    def render(self):
        """環境を描画"""
        if self.render_mode in ['human', 'ansi']:
            ps = self.engine.player(self.player_idx)
            print("=" * 40)
            print(f"Street: {self.current_street}")
            print(ps.board.to_string())
            print(f"Hand: {[str(c) for c in ps.get_hand()]}")
            print("=" * 40)


if __name__ == '__main__':
    """テスト"""
    print("=== OFC Phase 1 Environment Test ===\n")
    
    env = OFCPhase1Env(render_mode='human')
    obs, info = env.reset(seed=42)
    
    print("Valid actions:", len(env.get_valid_actions()))
    print("Action mask sum:", env.action_masks().sum())
    
    total_reward = 0
    done = False
    step_count = 0
    
    while not done:
        valid = env.get_valid_actions()
        action = np.random.choice(valid) if valid else 0
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        if step_count > 20:
            break
    
    print(f"\nSteps: {step_count}")
    print(f"Total reward: {total_reward}")
    print(f"Info: {info}")
    env.render()
