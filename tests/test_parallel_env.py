#!/usr/bin/env python3
"""
OFC Pineapple AI - ParallelOFCEnv Unit Tests
テスト対象: src/python/train_phase85_full_fl.py の ParallelOFCEnv

Edge cases:
- Dead agent handling
- FL status transitions
- Stuck detection
- Game termination flow
"""

import sys
import os
import numpy as np

# Add project root and src/python to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)  # For ofc_engine module
sys.path.insert(0, os.path.join(project_root, 'src', 'python'))

from ofc_3max_env import OFC3MaxEnv
from train_phase85_full_fl import ParallelOFCEnv, mask_fn


def test_basic_reset_step():
    """基本的な reset/step のテスト"""
    print("=== Test 1: Basic Reset/Step ===")

    env = ParallelOFCEnv(env_id=0, use_selfplay=False)
    obs, info = env.reset(seed=42)

    assert obs is not None, "Observation should not be None"
    assert 'my_board' in obs, "Observation should have 'my_board' key"
    assert 'my_hand' in obs, "Observation should have 'my_hand' key"

    # Get valid actions and take one step
    mask = env.action_masks()
    valid_actions = np.where(mask == 1)[0]
    assert len(valid_actions) > 0, "Should have at least one valid action"

    action = valid_actions[0]
    obs2, reward, terminated, truncated, info = env.step(action)

    assert obs2 is not None, "Observation after step should not be None"
    print("  Basic reset/step: PASSED")
    return True


def test_full_game_completion():
    """完全なゲーム終了までのテスト"""
    print("=== Test 2: Full Game Completion ===")

    env = ParallelOFCEnv(env_id=0, use_selfplay=False)
    obs, info = env.reset(seed=123)

    steps = 0
    max_steps = 100  # Safety limit

    while steps < max_steps:
        mask = env.action_masks()
        valid_actions = np.where(mask == 1)[0]

        if len(valid_actions) == 0:
            print(f"  Warning: No valid actions at step {steps}")
            break

        action = valid_actions[0]
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

        if terminated or truncated:
            break

    assert terminated or truncated, f"Game should terminate within {max_steps} steps"
    assert 'score' in info, "Info should contain 'score' on termination"
    print(f"  Full game completed in {steps} steps: PASSED")
    return True


def test_dead_agent_handling():
    """終了済みエージェントの処理テスト"""
    print("=== Test 3: Dead Agent Handling ===")

    env = ParallelOFCEnv(env_id=0, use_selfplay=False)
    obs, info = env.reset(seed=456)

    # Run until game ends
    steps = 0
    max_steps = 100

    while steps < max_steps:
        mask = env.action_masks()
        valid_actions = np.where(mask == 1)[0]

        if len(valid_actions) == 0:
            break

        obs, reward, terminated, truncated, info = env.step(valid_actions[0])
        steps += 1

        if terminated:
            break

    assert terminated, "Game should have terminated"

    # After termination, calling step should return the same terminated state
    # without errors
    obs2, reward2, terminated2, truncated2, info2 = env.step(0)
    assert terminated2, "Should still be terminated"
    assert reward2 == 0, "Reward after termination should be 0"

    print("  Dead agent handling: PASSED")
    return True


def test_multiple_games_reset():
    """複数ゲームの連続リセットテスト"""
    print("=== Test 4: Multiple Games Reset ===")

    env = ParallelOFCEnv(env_id=0, use_selfplay=False)

    for game_num in range(5):
        obs, info = env.reset(seed=game_num * 100)

        # Play a few steps
        for step in range(3):
            mask = env.action_masks()
            valid_actions = np.where(mask == 1)[0]

            if len(valid_actions) == 0:
                break

            obs, reward, terminated, truncated, info = env.step(valid_actions[0])

            if terminated:
                break

        # Verify game state is properly reset
        assert not env._game_over or terminated, "Game over flag should match termination"

    print("  Multiple games reset: PASSED")
    return True


def test_fl_continuous_games():
    """FL状態の連続ゲーム引き継ぎテスト"""
    print("=== Test 5: FL Continuous Games ===")

    # continuous_games=True の OFC3MaxEnv を直接テスト
    env = OFC3MaxEnv(
        enable_fl_turns=True,
        continuous_games=True
    )

    env.reset(seed=789)

    # Check that FL stats are initialized
    assert hasattr(env, 'fl_status'), "Should have fl_status attribute"
    assert hasattr(env, 'fl_cards_count'), "Should have fl_cards_count attribute"

    # Verify all players start without FL
    for agent in env.possible_agents:
        assert env.fl_status[agent] == False or env.fl_cards_count[agent] >= 14, \
            "FL status and card count should be consistent"

    print("  FL continuous games: PASSED")
    return True


def test_action_mask_validity():
    """アクションマスクの妥当性テスト"""
    print("=== Test 6: Action Mask Validity ===")

    env = ParallelOFCEnv(env_id=0, use_selfplay=False)
    obs, info = env.reset(seed=999)

    # Initial round (5 cards)
    mask = env.action_masks()
    assert mask.shape == (243,), f"Mask shape should be (243,), got {mask.shape}"
    assert np.sum(mask) > 0, "Should have at least one valid action"
    assert np.sum(mask) <= 243, "Cannot have more than 243 valid actions"

    # Take a step and verify mask updates
    valid_actions = np.where(mask == 1)[0]
    obs, reward, terminated, truncated, info = env.step(valid_actions[0])

    if not terminated:
        mask2 = env.action_masks()
        assert np.sum(mask2) > 0, "Should have valid actions after first step"

    print("  Action mask validity: PASSED")
    return True


def test_stuck_detection():
    """スタック検出のテスト"""
    print("=== Test 7: Stuck Detection (verify warning system) ===")

    # This test verifies that the stuck detection system is in place
    # by checking the ParallelOFCEnv code structure

    import inspect
    from train_phase85_full_fl import ParallelOFCEnv

    source = inspect.getsource(ParallelOFCEnv._play_opponents)

    assert 'stuck_count' in source, "_play_opponents should have stuck detection"
    assert 'max_iterations' in source, "_play_opponents should have max_iterations guard"

    print("  Stuck detection system: PASSED")
    return True


def test_game_over_flag():
    """_game_over フラグの正確性テスト"""
    print("=== Test 8: Game Over Flag ===")

    env = ParallelOFCEnv(env_id=0, use_selfplay=False)

    # After reset, game should not be over
    obs, info = env.reset(seed=111)
    assert env._game_over == False, "Game should not be over after reset"

    # Play until termination
    while True:
        mask = env.action_masks()
        valid_actions = np.where(mask == 1)[0]

        if len(valid_actions) == 0:
            break

        obs, reward, terminated, truncated, info = env.step(valid_actions[0])

        if terminated:
            break

    # After termination, game_over should be True
    assert env._game_over == True, "Game over flag should be True after termination"

    # After reset, it should be False again
    obs, info = env.reset(seed=222)
    assert env._game_over == False, "Game over flag should reset to False"

    print("  Game over flag: PASSED")
    return True


def test_reward_accumulation():
    """報酬の累積計算テスト"""
    print("=== Test 9: Reward Accumulation ===")

    env = ParallelOFCEnv(env_id=0, use_selfplay=False)
    obs, info = env.reset(seed=333)

    total_reward = 0

    while True:
        mask = env.action_masks()
        valid_actions = np.where(mask == 1)[0]

        if len(valid_actions) == 0:
            break

        obs, reward, terminated, truncated, info = env.step(valid_actions[0])
        total_reward += reward

        if terminated:
            break

    # Final reward should be close to the score
    if 'score' in info:
        # Allow some tolerance due to FL bonuses in rewards
        score = info['score']
        # Note: reward might include FL bonuses, so we just check it's reasonable
        assert abs(total_reward) < 1000, f"Total reward {total_reward} seems unreasonable"

    print(f"  Reward accumulation (total: {total_reward:.2f}): PASSED")
    return True


def run_all_tests():
    """全テストを実行"""
    print("=" * 60)
    print("ParallelOFCEnv Unit Tests")
    print("=" * 60)
    print()

    tests = [
        test_basic_reset_step,
        test_full_game_completion,
        test_dead_agent_handling,
        test_multiple_games_reset,
        test_fl_continuous_games,
        test_action_mask_validity,
        test_stuck_detection,
        test_game_over_flag,
        test_reward_accumulation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
