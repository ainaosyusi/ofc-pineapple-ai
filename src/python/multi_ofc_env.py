"""
OFC Pineapple AI - Multi-Agent Environment
PettingZoo互換の2人対戦環境
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
    DEBUG = True
except ImportError:
    raise ImportError("ofc_engine module not found. Run 'python setup.py build_ext --inplace' first.")


def env(**kwargs):
    """環境のファクトリ関数"""
    env = OFCMultiAgentEnv(**kwargs)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class OFCMultiAgentEnv(AECEnv):
    """
    OFC Pineapple 2人対戦環境
    
    PettingZoo AECEnv準拠 - 交互ターン制
    """
    
    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "ofc_pineapple_v0",
        "is_parallelizable": False,
    }
    
    # カード枚数
    NUM_CARDS = 54
    NUM_RANKS = 13
    NUM_SUITS = 4
    
    # ボード構成
    TOP_SIZE = 3
    MID_SIZE = 5
    BOT_SIZE = 5
    TOTAL_SLOTS = 13
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        
        # プレイヤー設定
        self.possible_agents = ["player_0", "player_1"]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}
        
        # ゲームエンジン（2人対戦）
        self.engine = ofc.GameEngine(2)
        self.rng_seed = None
        
        # ターン情報 (1-indexed: 1=初回, 2-5=通常)
        self.current_street = 1
        
        # アクション空間（既存環境と同じ）
        self._action_spaces = {
            agent: spaces.Discrete(243) for agent in self.possible_agents
        }
        
        # 観測空間（相手ボード情報を追加）
        self._observation_spaces = {
            agent: spaces.Dict({
                'my_board': spaces.MultiBinary(3 * self.NUM_CARDS),
                'opponent_board': spaces.MultiBinary(3 * self.NUM_CARDS),  # 相手の公開情報
                'hand': spaces.MultiBinary(5 * self.NUM_CARDS),
                'used_cards': spaces.MultiBinary(self.NUM_CARDS),
                'game_state': spaces.Box(
                    low=0,
                    high=np.array([5, 3, 5, 5, 1, 3, 5, 5], dtype=np.float32),
                    dtype=np.float32
                ),
                'probabilities': spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(8,),
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
        
        # ゲームエンジンをリセット
        self.engine.reset()
        self.engine.start_new_game(self.rng_seed)
        
        self.current_street = 1
        
        # ターン管理
        self._agent_selector = AgentSelector(self.agents)
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
            # 無効なアクション → ペナルティ
            self.rewards[agent] = -10.0
            self.infos[agent]['invalid_action'] = True
            # 無効な場合はゲームを強制終了させる等の処理が必要だが、
            # まずは正常系を安定させる
        
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
            # 次のエージェントを特定
            # Pineappleでは各ストリートで全プレイヤーが行動するまで待つ
            prev_street = self.current_street
            self.current_street = self.engine.current_turn()
            
            if self.current_street > prev_street:
                # ストリートが変わった＝全員のアクションが完了した
                self._agent_selector = AgentSelector(self.possible_agents)
                self.agent_selection = self._agent_selector.next()
            else:
                self.agent_selection = self._agent_selector.next()
    
    def _apply_initial_action(self, player_idx, action):
        """初回5枚配置アクションを適用"""
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
        
        ps = self.engine.player(player_idx)
        hand = ps.get_hand()
        
        initial_action = ofc.InitialAction()
        for i, (card, row) in enumerate(zip(hand, placements)):
            initial_action.set_placement(i, card, ofc.Row(row))
        
        return self.engine.apply_initial_action(player_idx, initial_action)
    
    def _apply_turn_action(self, player_idx, action):
        """通常ターン（3枚から2枚選択）アクションを適用"""
        combinations = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
        
        comb_idx = action // 9
        row_action = action % 9
        row1 = row_action // 3
        row2 = row_action % 3
        
        if comb_idx >= len(combinations):
            return False
        
        place1_idx, place2_idx, discard_idx = combinations[comb_idx]
        
        ps = self.engine.player(player_idx)
        hand = ps.get_hand()
        
        if len(hand) < 3:
            return False
        
        board = ps.board
        if not board.can_place(ofc.Row(row1)) or not board.can_place(ofc.Row(row2)):
            return False
        
        if row1 == row2:
            if board.remaining_slots(ofc.Row(row1)) < 2:
                return False
        
        turn_action = ofc.TurnAction()
        turn_action.set_placement(0, hand[place1_idx], ofc.Row(row1))
        turn_action.set_placement(1, hand[place2_idx], ofc.Row(row2))
        turn_action.discard = hand[discard_idx]
        
        return self.engine.apply_turn_action(player_idx, turn_action)
    
    def _check_round_complete(self):
        """ラウンドが完了したかチェック"""
        # 全プレイヤーがアクション完了したか確認
        for i in range(2):
            ps = self.engine.player(i)
            if self.current_street == 1:
                if ps.board.total_placed() < 5:
                    return False
            else:
                if ps.hand_count > 0:
                    return False
        
        # エンジンのターンが進行したか確認（または手動進行）
        new_street = self.engine.current_turn()
        if new_street > self.current_street:
            self.current_street = new_street
            return True
        
        return False
    
    def _calculate_final_rewards(self):
        """最終報酬を計算"""
        result = self.engine.result()
        
        for agent in self.agents:
            player_idx = self.agent_name_mapping[agent]
            score = float(result.get_score(player_idx))
            fouled = result.is_fouled(player_idx)
            
            # Phase 2 準拠の追加ペナルティ
            if fouled:
                score -= 24.0  # get_scoreが既に-6を返しているはずなので、計-30にする
            
            self.rewards[agent] = score
            self.infos[agent]['final_score'] = score
            self.infos[agent]['fouled'] = fouled
            self.infos[agent]['royalty'] = result.get_royalty(player_idx)
            self.infos[agent]['entered_fl'] = result.entered_fl(player_idx)
    
    def _get_observation(self, agent):
        """エージェントの観測を取得"""
        player_idx = self.agent_name_mapping[agent]
        opponent_idx = 1 - player_idx
        
        ps = self.engine.player(player_idx)
        opponent_ps = self.engine.player(opponent_idx)
        
        # 自分のボード
        my_board = np.zeros(3 * self.NUM_CARDS, dtype=np.int8)
        my_masks = [ps.board.top_mask(), ps.board.mid_mask(), ps.board.bot_mask()]
        for row_idx, mask in enumerate(my_masks):
            for i in range(self.NUM_CARDS):
                if (mask >> i) & 1:
                    my_board[row_idx * self.NUM_CARDS + i] = 1

        # 相手のボード（公開情報）
        opponent_board = np.zeros(3 * self.NUM_CARDS, dtype=np.int8)
        opp_masks = [opponent_ps.board.top_mask(), opponent_ps.board.mid_mask(), opponent_ps.board.bot_mask()]
        for row_idx, mask in enumerate(opp_masks):
            for i in range(self.NUM_CARDS):
                if (mask >> i) & 1:
                    opponent_board[row_idx * self.NUM_CARDS + i] = 1
        
        # 手札
        hand = ps.get_hand()
        hand_obs = np.zeros(5 * self.NUM_CARDS, dtype=np.int8)
        for i, card in enumerate(hand[:5]):
            hand_obs[i * self.NUM_CARDS + card.index] = 1
        
        # 使用済みカード (全プレイヤーの全マスクの論理和)
        all_mask = ps.board.all_mask() | opponent_ps.board.all_mask()
        used_obs = np.zeros(self.NUM_CARDS, dtype=np.int8)
        for i in range(self.NUM_CARDS):
            if (all_mask >> i) & 1:
                used_obs[i] = 1
        
        # ゲーム状態（自分 + 相手の情報）
        game_state = np.array([
            self.current_street,
            ps.board.count(ofc.TOP),
            ps.board.count(ofc.MIDDLE),
            ps.board.count(ofc.BOTTOM),
            1.0 if ps.in_fantasy_land else 0.0,
            opponent_ps.board.count(ofc.TOP),
            opponent_ps.board.count(ofc.MIDDLE),
            opponent_ps.board.count(ofc.BOTTOM),
        ], dtype=np.float32)

        # 確率情報の計算
        # 観測可能なカード: 自分のボード, 相手のボード, 自分の手札
        hand_mask = 0
        for card in hand:
            hand_mask |= (1 << card.index)
        
        visible_mask = ps.board.all_mask() | opponent_ps.board.all_mask() | hand_mask
        remaining_mask = 0xFFFFFFFFFFFFFFFF & ~visible_mask
        # 下位54ビットのみ有効にする
        remaining_mask &= ((1 << self.NUM_CARDS) - 1)
        remaining_count = self.NUM_CARDS - bin(visible_mask).count('1')
        
        # 自分の確率
        my_probs = [
            ofc.flush_probability(ps.board.mid_mask(), ps.board.count(ofc.MIDDLE), 5, remaining_mask, remaining_count),
            ofc.straight_probability(ps.board.mid_mask(), ps.board.count(ofc.MIDDLE), remaining_mask, remaining_count),
            ofc.flush_probability(ps.board.bot_mask(), ps.board.count(ofc.BOTTOM), 5, remaining_mask, remaining_count),
            ofc.straight_probability(ps.board.bot_mask(), ps.board.count(ofc.BOTTOM), remaining_mask, remaining_count),
        ]
        
        # 相手の確率（相手の視点ではなく、現在のプレイヤーから見た「相手が完成させる確率」）
        opp_probs = [
            ofc.flush_probability(opponent_ps.board.mid_mask(), opponent_ps.board.count(ofc.MIDDLE), 5, remaining_mask, remaining_count),
            ofc.straight_probability(opponent_ps.board.mid_mask(), opponent_ps.board.count(ofc.MIDDLE), remaining_mask, remaining_count),
            ofc.flush_probability(opponent_ps.board.bot_mask(), opponent_ps.board.count(ofc.BOTTOM), 5, remaining_mask, remaining_count),
            ofc.straight_probability(opponent_ps.board.bot_mask(), opponent_ps.board.count(ofc.BOTTOM), remaining_mask, remaining_count),
        ]
        
        probs_obs = np.array(my_probs + opp_probs, dtype=np.float32)
        
        return {
            'my_board': my_board,
            'opponent_board': opponent_board,
            'hand': hand_obs,
            'used_cards': used_obs,
            'game_state': game_state,
            'probabilities': probs_obs,
        }
    
    def observe(self, agent):
        """エージェントの現在の観測を返す"""
        return self.observations[agent]
    
    def render(self):
        """環境を描画"""
        if self.render_mode in ['human', 'ansi']:
            print("=" * 40)
            for agent in self.agents:
                player_idx = self.agent_name_mapping[agent]
                ps = self.engine.player(player_idx)
                print(f"\n{agent}:")
                print(ps.board.to_string())
                print(f"Hand: {[str(c) for c in ps.get_hand()]}")
            print(f"\nStreet: {self.current_street}")
            print("=" * 40)
    
    def get_valid_actions(self, agent):
        """有効なアクションのリストを取得"""
        if self.terminations[agent]:
            return []
        
        player_idx = self.agent_name_mapping[agent]
        ps = self.engine.player(player_idx)
        board = ps.board
        valid = []
        
        if self.current_street == 1:
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
                
                if not board.can_place(ofc.Row(row1)):
                    continue
                if row1 == row2:
                    if board.remaining_slots(ofc.Row(row1)) < 2:
                        continue
                elif not board.can_place(ofc.Row(row2)):
                    continue
                
                valid.append(action)
        
        return valid


if __name__ == '__main__':
    """簡単なテスト"""
    print("=== OFC Multi-Agent Environment Test ===\n")
    
    env = OFCMultiAgentEnv(render_mode='human')
    env.reset(seed=42)
    
    print("Initial state:")
    env.render()
    
    # ランダムにプレイ
    step_count = 0
    while env.agents:
        agent = env.agent_selection
        valid_actions = env.get_valid_actions(agent)
        
        if not valid_actions:
            action = 0  # ダミー
        else:
            action = np.random.choice(valid_actions)
        
        env.step(action)
        step_count += 1
        
        if step_count > 20:  # 安全のため
            break
    
    print("\nFinal state:")
    env.render()
    
    print("\nFinal rewards:")
    for agent in env.possible_agents:
        print(f"  {agent}: {env._cumulative_rewards[agent]}")
