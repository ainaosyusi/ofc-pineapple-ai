"""
OFC Pineapple AI - Model Evaluation
モデルの精度（ファウル率、平均ロイヤリティ）を評価
"""

import argparse
import numpy as np
from sb3_contrib import MaskablePPO
from src.python.ofc_phase1_env import OFCPhase1Env

def evaluate(model_paths, num_games=1000):
    env = OFCPhase1Env()
    
    results = []
    
    for model_path in model_paths:
        print(f"Evaluating model: {model_path}")
        model = MaskablePPO.load(model_path, env=env)
        
        fouls = 0
        total_royalty = 0
        entered_fl = 0
        
        for _ in range(num_games):
            obs, info = env.reset()
            done = False
            while not done:
                action_masks = env.action_masks()
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
            
            if info['fouled']:
                fouls += 1
            total_royalty += info['royalty']
            if info['entered_fl']:
                entered_fl += 1
                
        results.append({
            'name': model_path.split('/')[-1],
            'foul_rate': fouls / num_games,
            'avg_royalty': total_royalty / num_games,
            'fl_rate': entered_fl / num_games
        })
    
    print("\nComparison Results:")
    print(f"{'Model':<40} | {'Foul Rate':<10} | {'Avg Royalty':<12} | {'FL Rate':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<40} | {r['foul_rate']:>9.2%} | {r['avg_royalty']:>12.2f} | {r['fl_rate']:>9.2%}")
    print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs='+', required=True, help="Path to model zip file(s)")
    parser.add_argument("--games", type=int, default=1000, help="Number of games to evaluate")
    args = parser.parse_args()
    
    evaluate(args.models, args.games)
