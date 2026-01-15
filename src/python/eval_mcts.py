"""
OFC Pineapple AI - MCTS Evaluation Script
MCTS エージェント vs Policy-only エージェントの対戦評価

使用例:
    python eval_mcts.py --model models/ofc_selfplay_20260115.zip --games 50
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from mcts_agent import MCTSAgent

try:
    from sb3_contrib import MaskablePPO
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False
    print("[eval_mcts] Warning: sb3_contrib not available")

try:
    from train_selfplay import SelfPlayEnv
    HAS_ENV = True
except ImportError:
    HAS_ENV = False
    print("[eval_mcts] Warning: SelfPlayEnv not available")


def evaluate_mcts_vs_policy(
    model_path: str = None,
    n_games: int = 50,
    simulations: int = 50,
    top_k: int = 3,
    policy_weight: float = 0.3,
):
    """
    MCTS エージェント vs Policy-only エージェントの対戦評価
    
    MCTS は Player 0 として、直感（Policy）+ 思考（Search）を使用
    Policy-only は Player 1 として、直感のみを使用
    """
    print("=" * 60)
    print("MCTS vs Policy-only Evaluation")
    print("=" * 60)
    print(f"Model: {model_path or 'Random'}")
    print(f"Games: {n_games}")
    print(f"Simulations per action: {simulations}")
    print(f"Top-K candidates: {top_k}")
    print(f"Policy weight: {policy_weight}")
    print()
    
    # モデル読み込み
    model = None
    if model_path and HAS_MODEL:
        try:
            model = MaskablePPO.load(model_path)
            print(f"[Model] Loaded: {model_path}")
        except Exception as e:
            print(f"[Model] Failed to load: {e}")
    
    if not HAS_ENV:
        print("[Error] SelfPlayEnv not available")
        return
    
    # 環境とエージェント
    env = SelfPlayEnv(seed=42)
    
    mcts_agent = MCTSAgent(
        model=model,
        top_k=top_k,
        policy_weight=policy_weight
    )
    
    # 統計
    mcts_wins = 0
    policy_wins = 0
    draws = 0
    mcts_scores = []
    mcts_fouls = 0
    policy_fouls = 0
    
    start_time = time.time()
    
    print("\nRunning games...")
    for game_i in range(n_games):
        obs, _ = env.reset()
        done = False
        
        while not done:
            # 現在のプレイヤーを確認
            current = env.env.agent_selection
            
            if current == "player_0":
                # MCTS プレイヤー
                action_mask = env.action_masks()
                action = mcts_agent.predict_with_search(
                    obs, action_mask, simulations=simulations
                )
            else:
                # Policy-only プレイヤー (対戦相手)
                # 注: SelfPlayEnv では対戦相手は自動処理されるが
                # ここでは明示的に処理
                pass
            
            obs, reward, done, truncated, info = env.step(action)
        
        # 結果集計
        final_score = reward
        mcts_scores.append(final_score)
        
        if final_score > 0:
            mcts_wins += 1
        elif final_score < 0:
            policy_wins += 1
        else:
            draws += 1
        
        # 進捗表示
        if (game_i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            games_per_sec = (game_i + 1) / elapsed
            win_rate = mcts_wins / (game_i + 1) * 100
            print(f"  [{game_i+1:3d}/{n_games}] MCTS: {mcts_wins} wins ({win_rate:.1f}%) | "
                  f"Policy: {policy_wins} wins | {games_per_sec:.1f} games/s")
    
    elapsed = time.time() - start_time
    
    # 結果表示
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Total games: {n_games}")
    print(f"Elapsed time: {elapsed:.1f}s")
    print()
    print(f"  MCTS Wins:     {mcts_wins} ({mcts_wins/n_games*100:.1f}%)")
    print(f"  Policy Wins:   {policy_wins} ({policy_wins/n_games*100:.1f}%)")
    print(f"  Draws:         {draws} ({draws/n_games*100:.1f}%)")
    print()
    print(f"  MCTS Avg Score: {np.mean(mcts_scores):.2f} ± {np.std(mcts_scores):.2f}")
    print()
    
    # 仮説検証
    if mcts_wins > policy_wins:
        print("✅ 仮説「弱モデル + MCTS > 強モデル」が支持されました！")
    elif mcts_wins < policy_wins:
        print("❌ Policy-only の方が強い結果となりました")
    else:
        print("➖ 引き分け")
    
    return {
        'mcts_wins': mcts_wins,
        'policy_wins': policy_wins,
        'draws': draws,
        'mcts_avg_score': np.mean(mcts_scores),
        'mcts_win_rate': mcts_wins / n_games * 100
    }


def main():
    parser = argparse.ArgumentParser(description="MCTS vs Policy Evaluation")
    parser.add_argument("--model", type=str, default=None, 
                       help="Path to trained model (.zip)")
    parser.add_argument("--games", type=int, default=50, 
                       help="Number of games to play")
    parser.add_argument("--simulations", type=int, default=50, 
                       help="Simulations per MCTS action")
    parser.add_argument("--top-k", type=int, default=3, 
                       help="Top-K candidates from policy")
    parser.add_argument("--policy-weight", type=float, default=0.3, 
                       help="Weight for policy vs simulation (0-1)")
    
    args = parser.parse_args()
    
    evaluate_mcts_vs_policy(
        model_path=args.model,
        n_games=args.games,
        simulations=args.simulations,
        top_k=args.top_k,
        policy_weight=args.policy_weight
    )


if __name__ == "__main__":
    main()
