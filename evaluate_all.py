import os
import sys
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# パス追加
sys.path.insert(0, os.getcwd())

from src.python.ofc_phase1_env import OFCPhase1Env
from src.python.multi_ofc_env import OFCMultiAgentEnv

class Phase3EvalEnv(OFCPhase1Env):
    """Phase 3モデルを評価するためのシングルエージェント用ラップ環境"""
    def __init__(self, include_probs=True):
        super().__init__()
        self.include_probs = include_probs
        self.multi_env = OFCMultiAgentEnv()
        
        # 観測空間の上書き
        obs_dict = {
            'my_board': self.multi_env.observation_space("player_0")['my_board'],
            'opponent_board': self.multi_env.observation_space("player_0")['opponent_board'],
            'hand': self.multi_env.observation_space("player_0")['hand'],
            'used_cards': self.multi_env.observation_space("player_0")['used_cards'],
            'game_state': self.multi_env.observation_space("player_0")['game_state'],
        }
        if include_probs:
            obs_dict['probabilities'] = self.multi_env.observation_space("player_0")['probabilities']
            
        from gymnasium import spaces
        self.observation_space = spaces.Dict(obs_dict)

    def reset(self, seed=None, options=None):
        self.multi_env.reset(seed=seed)
        obs = self.multi_env.observe("player_0")
        if not self.include_probs and 'probabilities' in obs:
            del obs['probabilities']
        return obs, {}

    def step(self, action):
        # プレイヤー0の行動
        self.multi_env.step(action)
        # プレイヤー1（相手）の行動はランダムで行う（評価用）
        while self.multi_env.agent_selection == "player_1" and not all(self.multi_env.terminations.values()):
            valid = self.multi_env.get_valid_actions("player_1")
            self.multi_env.step(np.random.choice(valid))
            
        done = all(self.multi_env.terminations.values())
        obs = self.multi_env.observe("player_0")
        if not self.include_probs and 'probabilities' in obs:
            del obs['probabilities']
            
        reward = self.multi_env._cumulative_rewards["player_0"] if done else 0
        info = self.multi_env.infos.get("player_0", {})
        return obs, reward, done, False, info

    def action_masks(self):
        valid = self.multi_env.get_valid_actions("player_0")
        mask = np.zeros(243, dtype=bool)
        for a in valid: mask[a] = True
        return mask

def evaluate_model(model_path, num_games=100):
    print(f"\n--- Evaluating: {os.path.basename(model_path)} ---")
    
    # モデルのロードを試み、環境を決定する
    env = None
    model = None
    
    # 候補となる環境
    envs_to_try = [
        ("Phase 1/2", lambda: OFCPhase1Env()),
        ("Phase 3 Standard", lambda: Phase3EvalEnv(include_probs=False)),
        ("Phase 3 Enhanced", lambda: Phase3EvalEnv(include_probs=True)),
    ]
    
    for name, env_fn in envs_to_try:
        try:
            test_env = env_fn()
            model = MaskablePPO.load(model_path, env=test_env)
            env = test_env
            print(f"Detected Environment: {name}")
            break
        except ValueError:
            continue
            
    if env is None:
        print(f"Error: Could not find matching environment for {model_path}")
        return None

    fouls = 0
    total_royalty = 0
    total_reward = 0
    success_count = 0
    
    for i in range(num_games):
        try:
            obs, info = env.reset()
            done = False
            while not done:
                mask = env.action_masks()
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
            
            if info.get('fouled', False):
                fouls += 1
            total_royalty += info.get('royalty', 0)
            total_reward += reward
            success_count += 1
        except Exception as e:
            if i == 0: # 最初の1回だけ表示してノイズを抑える
                print(f"Game {i} failed: {e}")
            pass
            
    if success_count == 0: return None
    
    return {
        'model': os.path.basename(model_path),
        'foul_rate': fouls / success_count,
        'avg_royalty': total_royalty / success_count,
        'avg_reward': total_reward / success_count,
        'success_rate': success_count / num_games
    }

if __name__ == "__main__":
    models = [
        "models/latest_phase1_model.zip",
        "models/ofc_phase2_20260115_005826_final.zip",
        "models/ofc_phase3_20260115_144209_final.zip",
        "models/enhanced_ppo_final.zip"
    ]
    
    results = []
    for m in models:
        if os.path.exists(m):
            res = evaluate_model(m, num_games=100)
            if res: results.append(res)
            
    print("\n" + "="*80)
    print(f"{'Model':<40} | {'Foul Rate':<10} | {'Avg Roy':<10} | {'Avg Rew':<10}")
    print("-"*80)
    for r in results:
        print(f"{r['model']:<40} | {r['foul_rate']:>9.1%} | {r['avg_royalty']:>9.2f} | {r['avg_reward']:>9.2f}")
    print("="*80)
