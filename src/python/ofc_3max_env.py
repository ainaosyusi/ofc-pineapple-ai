"""
OFC Pineapple AI - 3-Max Multi-Agent Environment
PettingZoo互換の3人対戦環境 (Phase 8)

特徴:
- 3人プレイヤー (player_0, player_1, player_2)
- 相手の捨て札は見えない設計
- ポジション（ボタン）情報を観測に含む
- 自分の捨て札履歴を追跡
- Fantasy Land (FL) ターンをフル実装
  - FL突入プレイヤーは14-17枚を一度に配置
  - FantasySolverによる最適配置
  - 連続ゲームでFL状態を引き継ぎ
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
    
    def __init__(self, render_mode=None, enable_fl_turns=True, continuous_games=False):
        """
        OFC 3-Max環境の初期化

        Args:
            render_mode: 描画モード ('human' or None)
            enable_fl_turns: FLターンを有効にするか (Trueの場合、FL突入者は最適配置)
            continuous_games: 連続ゲームモード (FL状態を引き継ぐ)
        """
        super().__init__()

        self.render_mode = render_mode
        self.enable_fl_turns = enable_fl_turns
        self.continuous_games = continuous_games

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

        # FL状態の追跡（連続ゲーム用）
        self.fl_status = {agent: False for agent in self.possible_agents}
        self.fl_cards_count = {agent: 0 for agent in self.possible_agents}

        # FL統計
        self.fl_games_played = 0
        self.fl_entries_total = 0
        self.fl_stays_total = 0

        # アクション空間（5枚配置: 3^5 = 243）
        self._action_spaces = {
            agent: spaces.Discrete(243) for agent in self.possible_agents
        }

        # 観測空間 (FL手札スロット追加: 17枚分)
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
                    high=np.array([5, 3, 5, 5, 3, 5, 5, 3, 5, 5, 1, 17, 1, 1], dtype=np.float32),
                    dtype=np.float32
                ),  # 追加: FL手札枚数, FL中フラグ (next/prev)
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

        # FL状態の引き継ぎ処理
        if self.continuous_games and any(self.fl_cards_count[a] > 0 for a in self.possible_agents):
            # Ultimate Rules: FL枚数を指定して開始
            # QQ=14, KK=15, AA=16, Trips=17
            fl_cards_array = [self.fl_cards_count[agent] for agent in self.possible_agents]
            self.engine.start_with_fl_cards(self.rng_seed, fl_cards_array)
            self.fl_games_played += 1
        else:
            # 通常のゲーム開始
            self.engine.start_new_game(self.rng_seed)
            # FL状態をリセット
            self.fl_status = {agent: False for agent in self.possible_agents}
            self.fl_cards_count = {agent: 0 for agent in self.possible_agents}

        self.current_street = 1

        # ボタン設定
        if options and 'button_position' in options:
            self.button_position = options['button_position']
        elif self.continuous_games and hasattr(self, '_game_count'):
            # 連続ゲームモードでは自動的にボタンを回転
            self.rotate_button()

        # ゲームカウント（ボタン回転用）
        if not hasattr(self, '_game_count'):
            self._game_count = 0
        self._game_count += 1

        # FLプレイヤーの自動処理（enable_fl_turnsがTrueの場合）
        if self.enable_fl_turns:
            self._process_fl_players()

        # ターン管理（ボタンの左隣から開始）
        # FLプレイヤーはスキップ（既にボードが完成）
        start_idx = (self.button_position + 1) % self.NUM_PLAYERS
        ordered_agents = []
        for i in range(self.NUM_PLAYERS):
            agent = self.possible_agents[(start_idx + i) % self.NUM_PLAYERS]
            player_idx = self.agent_name_mapping[agent]
            ps = self.engine.player(player_idx)
            # FLプレイヤー以外を順序に追加
            if not ps.in_fantasy_land:
                ordered_agents.append(agent)

        # 全員がFLの場合は空のリスト（すぐにショーダウン）
        if not ordered_agents:
            ordered_agents = [self.possible_agents[0]]  # ダミー（即終了）

        self._agent_selector = AgentSelector(ordered_agents)
        self.agent_selection = self._agent_selector.next()

        # 観測を設定
        self.observations = {agent: self._get_observation(agent) for agent in self.agents}

    def _process_fl_players(self):
        """FLプレイヤーをFantasySolverで自動処理"""
        for i, agent in enumerate(self.possible_agents):
            ps = self.engine.player(i)
            fl_cards = ps.get_hand()  # FLの場合はFL手札が返される
            if ps.in_fantasy_land and len(fl_cards) >= 14:
                # FantasySolverで最適配置を取得
                try:
                    solution = ofc.solve_fantasy_land(fl_cards, already_in_fl=True)

                    # FLActionを作成
                    fl_action = ofc.FLAction()
                    rows = [ofc.TOP, ofc.MIDDLE, ofc.BOTTOM]

                    # Top (3枚)
                    for j, card in enumerate(solution.top[:3]):
                        fl_action.set_placement(j, card, ofc.TOP)

                    # Middle (5枚)
                    for j, card in enumerate(solution.mid[:5]):
                        fl_action.set_placement(3 + j, card, ofc.MIDDLE)

                    # Bottom (5枚)
                    for j, card in enumerate(solution.bot[:5]):
                        fl_action.set_placement(8 + j, card, ofc.BOTTOM)

                    # 捨て札
                    fl_action.discards = solution.discards

                    # アクション適用
                    self.engine.apply_fl_action(i, fl_action)

                    # FL統計を記録
                    self.fl_cards_count[agent] = len(fl_cards)
                except Exception as e:
                    print(f"[FL] Error processing FL for {agent}: {e}")
    
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
                # FLプレイヤーを除外してターン順を作成
                start_idx = (self.button_position + 1) % self.NUM_PLAYERS
                ordered_agents = []
                for i in range(self.NUM_PLAYERS):
                    agent = self.possible_agents[(start_idx + i) % self.NUM_PLAYERS]
                    player_idx = self.agent_name_mapping[agent]
                    ps = self.engine.player(player_idx)
                    # FLプレイヤー以外を順序に追加
                    if not ps.in_fantasy_land:
                        ordered_agents.append(agent)

                if ordered_agents:
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
        # 2枚の配置先
        placements = []
        temp = action 
        row1 = temp % 3
        row2 = (temp // 3) % 3
        discard_idx = (temp // 9) % 3
        placements = [row1, row2]
        
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
            score = float(result.get_score(i))
            royalty = result.get_royalty(i)
            entered_fl = result.entered_fl(i)
            stayed_fl = result.stayed_fl(i)
            fouled = result.is_fouled(i)

            # FL突入ボーナス (Phase 8用強化: Superhuman Strategy)
            # Top QQ+ でFLに入った瞬間に +15.0 の巨大な報酬を与える
            if entered_fl:
                score += 15.0
                self.fl_entries_total += 1

            # FL継続ボーナス (連続ゲームでFLを維持)
            if stayed_fl:
                score += 10.0
                self.fl_stays_total += 1

            self.rewards[agent] = score

            # 次のゲーム用にFL状態を保存（連続ゲームモード）
            if self.continuous_games:
                # FL突入またはFL継続した場合、次もFLで開始
                self.fl_status[agent] = entered_fl or stayed_fl

                # Ultimate Rules: TopハンドによりFL枚数を決定
                # QQ=14, KK=15, AA=16, Trips=17
                if entered_fl or stayed_fl:
                    ps = self.engine.player(i)
                    top_eval = ps.board.evaluate_top()
                    self.fl_cards_count[agent] = ofc.fantasy_land_cards(top_eval)
                else:
                    self.fl_cards_count[agent] = 0

            # 詳細情報を保存
            self.infos[agent] = {
                'score': float(result.get_score(i)),
                'royalty': royalty,
                'fouled': fouled,
                'entered_fl': entered_fl,
                'stayed_fl': stayed_fl,
                'was_in_fl': self.engine.player(i).in_fantasy_land,
                'fl_cards_next': self.fl_cards_count[agent],  # Ultimate Rules: 次ゲームのFL枚数
                'win': score > 0,
                'loss': score < 0,
                'draw': abs(score) < 0.001
            }
    
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
        
        # 下家と上家のボード (FLプレイヤーのボードはショーダウンまで隠蔽)
        next_idx, prev_idx = self._get_relative_positions(player_idx)
        
        next_ps = self.engine.player(next_idx)
        if next_ps.in_fantasy_land and self.engine.phase() not in [ofc.GamePhase.SHOWDOWN, ofc.GamePhase.COMPLETE]:
            next_board = np.zeros(3 * self.NUM_CARDS, dtype=np.int8)
        else:
            next_board = self._board_to_array(next_ps.board)
        
        prev_ps = self.engine.player(prev_idx)
        if prev_ps.in_fantasy_land and self.engine.phase() not in [ofc.GamePhase.SHOWDOWN, ofc.GamePhase.COMPLETE]:
            prev_board = np.zeros(3 * self.NUM_CARDS, dtype=np.int8)
        else:
            prev_board = self._board_to_array(prev_ps.board)
        
        # 自分の捨て札
        my_discards = self.player_discards[agent].copy()
        
        # 見えないカードの確率を計算
        unseen_prob = self._calculate_unseen_probability(player_idx)
        
        # ポジション情報（one-hot）
        position = self._get_position_from_button(player_idx)
        position_info = np.zeros(self.NUM_PLAYERS, dtype=np.int8)
        position_info[position] = 1
        
        # ゲーム状態（FL情報を追加）
        # FL手札枚数を取得（通常は0-5、FL中は14-17）
        hand = ps.get_hand()
        fl_hand_count = len(hand) if ps.in_fantasy_land else 0
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
            1 if ps.in_fantasy_land else 0,
            fl_hand_count,  # FL手札枚数
            1 if next_ps.in_fantasy_land else 0,  # 下家がFL中か
            1 if prev_ps.in_fantasy_land else 0,  # 上家がFL中か
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
                fl_status = "[FL]" if ps.in_fantasy_land else ""
                print(f"\n{agent} (Pos: {self._get_position_from_button(i)}) {fl_status}:")
                print(f"  Board: {ps.board.to_string()}")
                hand = ps.get_hand()
                if ps.in_fantasy_land and len(hand) >= 14:
                    fl_cards_str = ", ".join([c.to_string() for c in hand])
                    print(f"  FL Hand ({len(hand)}): {fl_cards_str}")

    def get_fl_stats(self):
        """FL統計を取得"""
        return {
            'fl_games_played': self.fl_games_played,
            'fl_entries_total': self.fl_entries_total,
            'fl_stays_total': self.fl_stays_total,
            'current_fl_status': {agent: self.fl_status[agent] for agent in self.possible_agents}
        }

    def reset_fl_stats(self):
        """FL統計をリセット"""
        self.fl_games_played = 0
        self.fl_entries_total = 0
        self.fl_stays_total = 0
    
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
                        # Compact 27-action space: row1 + row2*3 + discard*9
                        action = (discard_idx * 9) + (row2 * 3) + row1
                        valid_actions.append(action)
        
        if not valid_actions:
            print(f"[Env] Warning: No valid actions for {agent} at street {self.current_street}. Board: {board.total_placed()} cards.")
            # Fallback: just return the first possible if any, or 0
            valid_actions = [0]
            
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
    """FLサポート付きの環境テスト"""
    print("=== OFC 3-Max Environment Test (with FL Support) ===\n")

    # FL有効モードでテスト
    env = OFC3MaxEnv(render_mode='human', enable_fl_turns=True, continuous_games=True)
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

    print("\nFinal rewards and FL info:")
    for agent in env.possible_agents:
        info = env.infos.get(agent, {})
        print(f"  {agent}:")
        print(f"    Reward: {env._cumulative_rewards.get(agent, 0):.2f}")
        print(f"    Entered FL: {info.get('entered_fl', False)}")
        print(f"    Stayed FL: {info.get('stayed_fl', False)}")
        print(f"    Was in FL: {info.get('was_in_fl', False)}")

    print("\nFL Statistics:")
    fl_stats = env.get_fl_stats()
    print(f"  FL Games Played: {fl_stats['fl_games_played']}")
    print(f"  FL Entries Total: {fl_stats['fl_entries_total']}")
    print(f"  FL Stays Total: {fl_stats['fl_stays_total']}")

    # FL状態が引き継がれることをテスト
    if any(env.fl_status.values()):
        print("\n--- Testing FL Carryover ---")
        env.reset()  # FL状態を引き継いでリセット
        print("After reset with FL carryover:")
        env.render()
