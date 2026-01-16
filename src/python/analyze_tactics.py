import os
import sys
import numpy as np
from sb3_contrib import MaskablePPO
from ofc_3max_env import OFC3MaxEnv
import ofc_engine as ofc
import torch
torch.distributions.Distribution.set_default_validate_args(False)

def analyze_model(model_path, num_hands=10):
    env = OFC3MaxEnv()
    model = MaskablePPO.load(model_path)
    
    print(f"--- Tactical Analysis: {model_path} ---")
    
    for h in range(num_hands):
        print(f"\nHand #{h+1}")
        env.reset()
        done = False
        
        while not done:
            agent = env.agent_selection
            if agent.startswith("player_"):
                # 観測とマスク取得
                obs = env.observe(agent)
                mask = env.action_masks(agent)
                
                # 推論
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                
                # 配置の可視化 (デコード)
                ps = env.engine.player(env.agent_name_mapping[agent])
                hand = ps.get_hand()
                
                if env.current_street == 0:
                    print(f"[{agent}] Initial Deal (5 cards): {' '.join([str(c) for c in hand])}")
                else:
                    print(f"[{agent}] Turn (Pick 2 from 3): {' '.join([str(c) for c in hand])}")
                
                # アクションの適用
                env.step(action)
                
                # 配置後のボードを表示
                print(f"Resulting Board:\n{ps.board.to_string()}")
                
            else:
                env.step(None) # 各種フラグ更新用
            
            # 手の終了チェック
            if all(env.terminations.values()):
                done = True
                
        # スコア表示
        res = env.engine.result()
        for i in range(3):
            print(f"Player {i} Result: Score={res.get_score(i)}, Foul={res.is_fouled(i)}, FL={res.entered_fl(i)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_tactics.py <model_path>")
    else:
        analyze_model(sys.argv[1])
