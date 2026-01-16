"""
OFC Pineapple AI - 3-Max Multi-Agent Environment
PettingZoo互換の3人対戦環境 (Phase 5)

特徴:
- 3人プレイヤー (player_0, player_1, player_2)
- 相手の捨て札は見えない設計
- ポジション（ボタン）情報を観測に含む
- 自分の捨て札履歴を追跡
"""

import functools
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import AgentSelector

# C++エンジンをインポート
try:
    import ofc_engine as ofc
except ImportError:
    raise ImportError("ofc_engine module not found. Run 'python setup.py build_ext --inplace' first.")


def env(**kwargs):
    """環境のファクトリ関数"""
    env = OFC3MaxEnv(**kwargs)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class OFC3MaxEnv(AECEnv):
    """
    OFC Pineapple 3人対戦環境
    
    PettingZoo AECEnv準拠 - 3人交互ターン制
    """
    
    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "ofc_pineapple_3max_v0",
        "is_parallelizable": False,
    }
    
    # カード枚数
    NUM_CARDS = 54
    NUM_RANKS = 13
    NUM_SUITS = 4
    NUM_PLAYERS = 3
    
    # ボード構成
    TOP_SIZE = 3
    MID_SIZE = 5
    BOT_SIZE = 5
    TOTAL_SLOTS = 13
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        
        # プレイヤー設定（3人）
        self.possible_agents = ["player_0", "player_1", "player_2"]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}
        
        # ゲームエンジン（3人対戦）
        self.engine = ofc.GameEngine(self.NUM_PLAYERS)
        self.rng_seed = None
        
        # ターン情報
        self.current_street = 1
        
        # ボタン位置（ゲームごとに回転）
        self.button_position = 0
        
        # 各プレイヤーの捨て札履歴
        self.player_discards = {agent: np.zeros(self.NUM_CARDS, dtype=np.int8) 
                                for agent in self.possible_agents}
        
        # アクション空間（5枚配置: 3^5 = 243）
        self._action_spaces = {
            agent: spaces.Discrete(243) for agent in self.possible_agents
        }
        
        # 観測空間
        self._observation_spaces = {
            agent: spaces.Dict({
                'my_board': spaces.MultiBinary(3 * self.NUM_CARDS),          # 162
                'my_hand': spaces.MultiBinary(5 * self.NUM_CARDS),           # 270
                'next_opponent_board': spaces.MultiBinary(3 * self.NUM_CARDS), # 162
                'prev_opponent_board': spaces.MultiBinary(3 * self.NUM_CARDS), # 162
                'my_discards': spaces.MultiBinary(self.NUM_CARDS),           # 54
                'unseen_probability': spaces.Box(                            # 54
                    low=0.0, high=1.0, shape=(self.NUM_CARDS,), dtype=np.float32
                ),
                'position_info': spaces.MultiBinary(self.NUM_PLAYERS),       # 3 (one-hot)
                'game_state': spaces.Box(
                    low=0,
                    high=np.array([5, 3, 5, 5, 3, 5, 5, 3, 5, 5, 1], dtype=np.float32),
                    dtype=np.float32
                ),
            }) for agent in self.possible_agents
        }
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]
    
    def _get_relative_positions(self, player_idx):
        """ボタン基準の相対位置を取得"""
        next_idx = (player_idx + 1) % self.NUM_PLAYERS
        prev_idx = (player_idx - 1) % self.NUM_PLAYERS
        return next_idx, prev_idx
    
    def _get_position_from_button(self, player_idx):
        """ボタンからの相対位置を取得 (0=BTN, 1=SB, 2=BB)"""
        return (player_idx - self.button_position) % self.NUM_PLAYERS
    
    def _mask_to_array(self, mask, size=NUM_CARDS):
        """ビットマスクをnumpy配列に変換"""
        arr = np.zeros(size, dtype=np.int8)
        for i in range(size):
            if (mask >> i) & 1:
                arr[i] = 1
        return arr
    
    def _board_to_array(self, board):
        """ボードを3xNUM_CARDSの配列に変換"""
        arr = np.zeros(3 * self.NUM_CARDS, dtype=np.int8)
        masks = [board.top_mask(), board.mid_mask(), board.bot_mask()]
        for row_idx, mask in enumerate(masks):
            for i in range(self.NUM_CARDS):
                if (mask >> i) & 1:
                    arr[row_idx * self.NUM_CARDS + i] = 1
        return arr
    
    def reset(self, seed=None, options=None):
        """環境をリセット"""
        if seed is not None:
            self.rng_seed = seed
        else:
            self.rng_seed = np.random.randint(0, 2**32)
        
        # エージェント状態
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # 捨て札履歴をリセット
        self.player_discards = {agent: np.zeros(self.NUM_CARDS, dtype=np.int8) 
                                for agent in self.possible_agents}
        
        # ゲームエンジンをリセット
        self.engine.reset()
        self.engine.start_new_game(self.rng_seed)
        
        self.current_street = 1
        
        # ボタン設定
        if options and 'button_position' in options:
            self.button_position = options['button_position']
        
        # ターン管理（ボタンの左隣から開始）
        start_idx = (self.button_position + 1) % self.NUM_PLAYERS
        ordered_agents = [self.possible_agents[(start_idx + i) % self.NUM_PLAYERS] 
                          for i in range(self.NUM_PLAYERS)]
        
        self._agent_selector = AgentSelector(ordered_agents)
        self.agent_selection = self._agent_selector.next()
        
        # 観測を設定
        self.observations = {agent: self._get_observation(agent) for agent in self.agents}
    
    def step(self, action):
        """アクションを実行"""
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
        
        agent = self.agent_selection
        player_idx = self.agent_name_mapping[agent]
        
        # 報酬リセット
        self.rewards = {a: 0 for a in self.agents}
        
        # アクション適用
        if self.current_street == 1:
            success = self._apply_initial_action(player_idx, action)
        else:
            success = self._apply_turn_action(player_idx, action)
        
        if not success:
            self.rewards[agent] = -10.0
            self.infos[agent]['invalid_action'] = True
        
        # ゲーム終了チェック
        phase = self.engine.phase()
        if phase in [ofc.GamePhase.SHOWDOWN, ofc.GamePhase.COMPLETE]:
            self._calculate_final_rewards()
            for a in self.agents:
                self.terminations[a] = True
        
        # 累積報酬更新
        self._accumulate_rewards()
        
        # 観測更新
        for a in self.agents:
            self.observations[a] = self._get_observation(a)
        
        if not all(self.terminations.values()):
            prev_street = self.current_street
            self.current_street = self.engine.current_turn()
            
            if self.current_street > prev_street:
                start_idx = (self.button_position + 1) % self.NUM_PLAYERS
                ordered_agents = [self.possible_agents[(start_idx + i) % self.NUM_PLAYERS] 
                                  for i in range(self.NUM_PLAYERS)]
                self._agent_selector = AgentSelector(ordered_agents)
                self.agent_selection = self._agent_selector.next()
            else:
                self.agent_selection = self._agent_selector.next()
    
    def _apply_initial_action(self, player_idx, action):
        """初回5枚配置アクションを適用"""
        placements = []
        temp = action
        for i in range(5):
            row = temp % 3
            temp //= 3
            placements.append(row)
        
        # 配置先の容量チェック
        top_count = placements.count(0)
        mid_count = placements.count(1)
        bot_count = placements.count(2)
        
        if top_count > self.TOP_SIZE or mid_count > self.MID_SIZE or bot_count > self.BOT_SIZE:
            return False
        
        ps = self.engine.player(player_idx)
        hand = ps.get_hand()
        
        if len(hand) < 5:
            return False
        
        # InitialActionを作成
        initial_action = ofc.InitialAction()
        rows = [ofc.TOP, ofc.MIDDLE, ofc.BOTTOM]
        for i, row_idx in enumerate(placements):
            initial_action.set_placement(i, hand[i], rows[row_idx])
        
        return self.engine.apply_initial_action(player_idx, initial_action)
    
    def _apply_turn_action(self, player_idx, action):
        """通常ターン（3枚から2枚選択）アクションを適用"""
        ps = self.engine.player(player_idx)
        hand = ps.get_hand()
        
        if len(hand) < 3:
            return False
        
        # アクションをデコード
        discard_idx = action // 81
        remaining_action = action % 81
        
        if discard_idx >= len(hand):
            return False
        
        # 2枚の配置先
        placements = []
        temp = remaining_action
        for _ in range(2):
            row = temp % 3
            temp //= 3
            placements.append(row)
        
        # 容量チェック
        board = ps.board
        top_new = board.count(ofc.TOP) + placements.count(0)
        mid_new = board.count(ofc.MIDDLE) + placements.count(1)
        bot_new = board.count(ofc.BOTTOM) + placements.count(2)
        
        if top_new > self.TOP_SIZE or mid_new > self.MID_SIZE or bot_new > self.BOT_SIZE:
            return False
        
        # 捨て札を記録
        discarded_card = hand[discard_idx]
        agent = self.possible_agents[player_idx]
        self.player_discards[agent][discarded_card.index] = 1
        
        # TurnActionを作成
        turn_action = ofc.TurnAction()
        rows = [ofc.TOP, ofc.MIDDLE, ofc.BOTTOM]
        play_indices = [i for i in range(len(hand)) if i != discard_idx][:2]
        
        for i, play_idx in enumerate(play_indices):
            turn_action.set_placement(i, hand[play_idx], rows[placements[i]])
        turn_action.discard = discarded_card
        
        return self.engine.apply_turn_action(player_idx, turn_action)
    
    def _calculate_final_rewards(self):
        """3人分の最終報酬を計算（C++側で計算済み）"""
        result = self.engine.result()
        for i, agent in enumerate(self.possible_agents):
            self.rewards[agent] = float(result.get_score(i))
    
    def _get_observation(self, agent):
        """エージェントの観測を取得"""
        player_idx = self.agent_name_mapping[agent]
        ps = self.engine.player(player_idx)
        
        # 自分のボード
        my_board = self._board_to_array(ps.board)
        
        # 自分の手札
        my_hand = np.zeros(5 * self.NUM_CARDS, dtype=np.int8)
        hand = ps.get_hand()
        for i, card in enumerate(hand[:5]):
            my_hand[i * self.NUM_CARDS + card.index] = 1
        
        # 下家と上家のボード
        next_idx, prev_idx = self._get_relative_positions(player_idx)
        
        next_ps = self.engine.player(next_idx)
        next_board = self._board_to_array(next_ps.board)
        
        prev_ps = self.engine.player(prev_idx)
        prev_board = self._board_to_array(prev_ps.board)
        
        # 自分の捨て札
        my_discards = self.player_discards[agent].copy()
        
        # 見えないカードの確率を計算
        unseen_prob = self._calculate_unseen_probability(player_idx)
        
        # ポジション情報（one-hot）
        position = self._get_position_from_button(player_idx)
        position_info = np.zeros(self.NUM_PLAYERS, dtype=np.int8)
        position_info[position] = 1
        
        # ゲーム状態
        game_state = np.array([
            self.current_street,
            ps.board.count(ofc.TOP),
            ps.board.count(ofc.MIDDLE),
            ps.board.count(ofc.BOTTOM),
            next_ps.board.count(ofc.TOP),
            next_ps.board.count(ofc.MIDDLE),
            next_ps.board.count(ofc.BOTTOM),
            prev_ps.board.count(ofc.TOP),
            prev_ps.board.count(ofc.MIDDLE),
            prev_ps.board.count(ofc.BOTTOM),
            1 if ps.in_fantasy_land else 0
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
    
    def _calculate_unseen_probability(self, player_idx):
        """見えていないカードの残存確率を計算"""
        seen = np.zeros(self.NUM_CARDS, dtype=np.int8)
        
        # 自分のボード
        ps = self.engine.player(player_idx)
        all_mask = ps.board.all_mask()
        for i in range(self.NUM_CARDS):
            if (all_mask >> i) & 1:
                seen[i] = 1
        
        # 自分の手札
        for card in ps.get_hand():
            seen[card.index] = 1
        
        # 自分の捨て札
        agent = self.possible_agents[player_idx]
        seen = np.maximum(seen, self.player_discards[agent])
        
        # 相手のボード（公開情報）
        for i in range(self.NUM_PLAYERS):
            if i != player_idx:
                opponent = self.engine.player(i)
                opp_mask = opponent.board.all_mask()
                for j in range(self.NUM_CARDS):
                    if (opp_mask >> j) & 1:
                        seen[j] = 1
        
        # 確率計算
        unseen_count = self.NUM_CARDS - np.sum(seen)
        if unseen_count > 0:
            prob = 1.0 / unseen_count
            unseen_prob = np.where(seen == 0, prob, 0.0).astype(np.float32)
        else:
            unseen_prob = np.zeros(self.NUM_CARDS, dtype=np.float32)
        
        return unseen_prob
    
    def observe(self, agent):
        """エージェントの現在の観測を返す"""
        return self.observations[agent]
    
    def render(self):
        """環境を描画"""
        if self.render_mode == 'human':
            print("\n=== OFC 3-Max State ===")
            print(f"Street: {self.current_street}, Button: Player {self.button_position}")
            for i, agent in enumerate(self.possible_agents):
                ps = self.engine.player(i)
                print(f"\n{agent} (Pos: {self._get_position_from_button(i)}):")
                print(f"  Board: {ps.board.to_string()}")
    
    def get_valid_actions(self, agent):
        """有効なアクションのリストを取得"""
        player_idx = self.agent_name_mapping[agent]
        ps = self.engine.player(player_idx)
        hand = ps.get_hand()
        board = ps.board
        
        valid_actions = []
        
        if self.current_street == 1:
            # 初回5枚配置
            for action in range(243):
                placements = []
                temp = action
                for _ in range(5):
                    placements.append(temp % 3)
                    temp //= 3
                
                top_count = placements.count(0)
                mid_count = placements.count(1)
                bot_count = placements.count(2)
                
                if top_count <= self.TOP_SIZE and mid_count <= self.MID_SIZE and bot_count <= self.BOT_SIZE:
                    valid_actions.append(action)
        else:
            # 通常ターン
            for discard_idx in range(min(3, len(hand))):
                for placement_action in range(9):
                    row1 = placement_action % 3
                    row2 = placement_action // 3
                    
                    top_new = board.count(ofc.TOP) + [row1, row2].count(0)
                    mid_new = board.count(ofc.MIDDLE) + [row1, row2].count(1)
                    bot_new = board.count(ofc.BOTTOM) + [row1, row2].count(2)
                    
                    if top_new <= self.TOP_SIZE and mid_new <= self.MID_SIZE and bot_new <= self.BOT_SIZE:
                        action = discard_idx * 81 + placement_action
                        valid_actions.append(action)
        
        return valid_actions
    
    def action_masks(self, agent):
        """MaskablePPO用のアクションマスク"""
        valid = self.get_valid_actions(agent)
        mask = np.zeros(243, dtype=np.int8)
        for a in valid:
            mask[a] = 1
        return mask
    
    def rotate_button(self):
        """ボタンを時計回りに移動"""
        self.button_position = (self.button_position + 1) % self.NUM_PLAYERS


if __name__ == '__main__':
    """簡単なテスト"""
    print("=== OFC 3-Max Environment Test ===\n")
    
    env = OFC3MaxEnv(render_mode='human')
    env.reset(seed=42)
    
    print("Initial state:")
    env.render()
    
    step_count = 0
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            action = None
        else:
            valid_actions = env.get_valid_actions(agent)
            if valid_actions:
                action = valid_actions[0]
            else:
                action = 0
        
        env.step(action)
        step_count += 1
        
        if step_count > 30:
            break
    
    print("\nFinal state:")
    env.render()
    
    print("\nFinal rewards:")
    for agent in env.possible_agents:
        print(f"  {agent}: {env._cumulative_rewards.get(agent, 0)}")
