# OFC Pineapple AI (強化版)

C++エンジンと MaskablePPO を活用した、Open-Face Chinese Poker (Pineapple バリアント) のための高度な強化学習プロジェクト。

## 概要

このプロジェクトは、戦略的な意思決定と最適なロイヤリティ（役ボーナス）獲得が可能な、世界レベルの OFC Pineapple AI を構築することを目的としています。シミュレーション用の高性能 C++ ゲームエンジンと、深層強化学習用の Stable-Baselines3 (MaskablePPO) を組み合わせています。

## 主な特徴

- **高性能 C++ エンジン**: ゲームロジックと役判定を C++ で実装し、学習時 1,600 FPS 以上の高速処理を実現。
- **MaskablePPO**: 有効アクションのみを選択する「アクション・マスキング」により、無効な配置による無駄な探索を排除。
- **確率特徴量**: フラッシュやストレートの完成確率をリアルタイムに計算し、AI の観測情報（Observation）に追加。
- **MCTS（モンテカルロ木探索）**: Policy 誘導型ロールアウトを用いた先読み戦略。
- **エンドゲーム・ソルバー**: 残り 5 枚以下の局面で全探索を行い、終盤の完璧なプレイを保証。
- **セルフプレイ構築**: 「最新モデル vs 過去モデル（Pool）」による継続的な自己対戦学習。
- **自動カリキュラム (Auto-Curriculum)**: 学習進捗を AI が自己分析し、Discord/Slack を通じて戦略的なアドバイスを通知するシステム。

## 技術スタック

- **コア**: C++, Python (pybind11)
- **強化学習**: Stable-Baselines3 (sb3-contrib)
- **インフラ**: AWS EC2 (m7i-flex.large), Docker, Docker Compose
- **モニタリング**: Discord/Slack Webhooks, Tensorboard

## プロジェクト構造

- `src/cpp/`: 高性能ゲームエンジンと評価ロジック。
- `src/python/`: 学習スクリプト、MCTS 実装、Gym 環境。
- `docs/research/`: 強化学習戦略や数学的モデルの深掘りレポート。
- `docs/blog/`: 開発の過程や技術的なまとめ（note などのブログ用）。

## 実行方法

1. **エンジンのビルド**:
   ```bash
   python setup.py build_ext --inplace
   ```
2. **ローカル学習の開始**:
   ```bash
   python src/python/train_enhanced_phase3.py --steps 1000000
   ```
3. **AWS へのデプロイ**:
   ```bash
   python src/python/deploy_enhanced.py
   ```

---
*Created by Advanced Agentic Coding Team*
