import numpy as np
import ofc_engine as ofc
import random

class OFC3MaxSimulator:
    def __init__(self, models=None, initial_scores=None):
        self.num_players = 3
        self.scores = initial_scores if initial_scores else [200.0, 200.0, 200.0]
        self.button_idx = 0
        self.is_fantasy = [False] * self.num_players
        self.engine = ofc.GameEngine(self.num_players)
        self.models = models # List of agents (e.g., policy or random)
        
        # 統計用
        self.history = []
        self.hand_count = 0

    def solve_fl(self, cards, already_in_fl=True):
        """C++ソルバーを呼び出してFLの最適配置を得る"""
        solution = ofc.solve_fantasy_land(cards, already_in_fl)
        return solution

    def play_match(self, num_hands=100):
        print(f"=== Starting OFC 3-Max Match ({num_hands} hands) ===")
        for i in range(num_hands):
            self.hand_count += 1
            self.play_hand()
            
            # バストチェック
            if any(s <= 0 for s in self.scores):
                print(f"Match ended early at hand {self.hand_count} due to bust.")
                break
        
        print("\n=== Match Result ===")
        for i, s in enumerate(self.scores):
            print(f"Player {i}: {s:.1f} pts")

    def play_hand(self):
        # 1. FL判定 & ボタン処理
        has_fantasy_player = any(self.is_fantasy)
        if not has_fantasy_player:
            # 誰もFLでない場合のみボタンが移動 (JOPTルール)
            self.button_idx = (self.button_idx + 1) % self.num_players
        
        # 2. ゲーム開始 (FL状態をエンジンに渡す)
        seed = random.randint(0, 1000000)
        # start_with_fl(seed, [bool, bool, bool])
        self.engine.start_with_fl(seed, self.is_fantasy)
        
        # 3. FLプレイヤーの先行処理 (ソルバー使用)
        for i in range(self.num_players):
            if self.is_fantasy[i]:
                ps = self.engine.player(i)
                hand = ps.get_hand() # bindingsによりFL時には全枚数返る
                
                # ソルバーで最強配置を取得
                solution = self.solve_fl(hand, already_in_fl=True)
                
                # FLActionを作成して配置を設定
                action = ofc.FLAction()
                # Bot: 5 cards
                for j, card in enumerate(solution.bot):
                    action.set_placement(j, card, ofc.BOTTOM)
                # Mid: 5 cards
                for j, card in enumerate(solution.mid):
                    action.set_placement(j + 5, card, ofc.MIDDLE)
                # Top: 3 cards
                for j, card in enumerate(solution.top):
                    action.set_placement(j + 10, card, ofc.TOP)
                # Discards
                action.discards = solution.discards
                
                # エンジンに適用
                self.engine.apply_fl_action(i, action)

        # 4. 平場プレイヤーのターン進行
        while self.engine.phase() not in [ofc.GamePhase.SHOWDOWN, ofc.GamePhase.COMPLETE]:
            for i in range(self.num_players):
                pid = (self.button_idx + 1 + i) % self.num_players
                if self.is_fantasy[pid]: continue
                
                ps = self.engine.player(pid)
                if self.engine.phase() == ofc.GamePhase.INITIAL_DEAL:
                    if ps.board.total_placed() == 0:
                        action = self.get_agent_action(pid)
                        self.engine.apply_initial_action(pid, action)
                elif self.engine.phase() == ofc.GamePhase.TURN:
                    if ps.hand_count > 0:
                        action = self.get_agent_action(pid)
                        self.engine.apply_turn_action(pid, action)
            
            # 安全策: ループが回りすぎないようにチェック
            # (通常はエンジン側でフェーズが進むはず)

        # 5. ショーダウン
        if self.engine.phase() == ofc.GamePhase.SHOWDOWN:
            self.engine.calculate_scores()

        # 5. 精算とFL継続判定
        result = self.engine.result()
        hand_diffs = []
        next_fantasy = [False] * self.num_players

        for i in range(self.num_players):
            diff = float(result.get_score(i))
            self.scores[i] += diff
            hand_diffs.append(diff)
            
            # FL継続または新規突入
            # stayed_fl はそのプレイヤーが元々FLだった場合のみ True になる
            # entered_fl は平場から突入した場合に True
            next_fantasy[i] = result.stayed_fl(i) or result.entered_fl(i)

        self.is_fantasy = next_fantasy

        # ログ記録
        self.history.append({
            'hand': self.hand_count,
            'scores': self.scores.copy(),
            'diffs': hand_diffs,
            'fantasy': self.is_fantasy.copy(),
            'button': self.button_idx
        })
        
        # 簡易ログ表示
        print(f"Hand {self.hand_count:3} | Button: {self.button_idx} | FL: {next_fantasy} | Scores: {['%+d' % d for d in hand_diffs]}")

    def get_agent_action(self, pid):
        """エージェントのアクションを取得 (有効な配置を生成)"""
        ps = self.engine.player(pid)
        hand = ps.get_hand()
        board = ps.board
        
        # 現在のアクション内での仮配置枚数を追跡
        c_bot = board.count(ofc.BOTTOM)
        c_mid = board.count(ofc.MIDDLE)
        c_top = board.count(ofc.TOP)

        if self.engine.phase() == ofc.GamePhase.INITIAL_DEAL:
            action = ofc.InitialAction()
            for i in range(5):
                if c_bot < 5:
                    row = ofc.BOTTOM
                    c_bot += 1
                elif c_mid < 5:
                    row = ofc.MIDDLE
                    c_mid += 1
                else:
                    row = ofc.TOP
                    c_top += 1
                action.set_placement(i, hand[i], row)
            return action
        else:
            action = ofc.TurnAction()
            # 2枚配置、1枚捨て
            for i in range(2):
                if c_bot < 5:
                    row = ofc.BOTTOM
                    c_bot += 1
                elif c_mid < 5:
                    row = ofc.MIDDLE
                    c_mid += 1
                else:
                    row = ofc.TOP
                    c_top += 1
                action.set_placement(i, hand[i], row)
            action.discard = hand[2]
            return action

if __name__ == "__main__":
    # 3人対戦シチュエーションのシミュレーション実行
    sim = OFC3MaxSimulator()
    sim.play_match(20)
