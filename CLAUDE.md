# CLAUDE.md — プロジェクト運営ルール

## 絶対ルール（違反したら即座に作業を停止して報告）

### 1. 不明点は必ず質問する
- ゲームルール、ポーカー用語、OFC 特有の仕様で **少しでも不確かな点があれば必ずユーザーに確認する**
- 「たぶんこうだろう」で進めない。V1 では ACE のランキングを間違えたまま全学習が進行した
- 特に: ハンド強度の比較、フォール判定条件、FL 入場/残留条件、ロイヤリティ計算

### 2. 問題を隠さない
- エラー、バグ、想定外の結果が出たら **即座にユーザーに報告する**
- 「表面的に動いているから大丈夫」と判断しない
- 原因が不明な場合は「原因不明」と正直に報告し、調査方針を提案する
- ログの異常値（Foul Rate 急上昇、FL Rate 低下など）は見逃さず報告する

### 3. 実機テストを最優先する
- 数値上の評価だけで「成功」と判断しない
- 新しいモデルが完成したら、必ず **mix-poker-app での対人テスト** を提案する
- 評価スクリプトの結果と実際のゲームプレイが一致するか検証する

### 4. Discord Webhook を必ず設定する
- 学習を開始する際、`DISCORD_WEBHOOK_URL` 環境変数を必ず設定する
- URL: `DISCORD_WEBHOOK_URL='https://discord.com/api/webhooks/1461388510140301315/Ofc9ok3IDLgRhFR7Oa2iA4-CHHL9uFvFbIPT9G9tbA5X5r-wG-XMbnkg7ubocLJJQ5Pf'`

---

## 研究プロセスのルール

### レポート形式の統一

全ての実験結果は **統一フォーマット** で記録する。テンプレートは `docs/templates/` にある。

| 場面 | 使うテンプレート |
|------|------------------|
| 新しい実験を開始 | `docs/templates/EXPERIMENT_TEMPLATE.md` |
| 学習の進捗報告 | `docs/templates/PROGRESS_REPORT_TEMPLATE.md` |

### 実験開始前の必須事項
1. **仮説を明記**: 何を検証するのか、期待する結果は何か
2. **ベースライン指標を記録**: 比較対象がなければ評価できない
3. **評価基準を事前に決定**: 何をもって成功/失敗と判断するか
4. **ユーザーと方針を合意**: 勝手に始めない

### 進捗レポートの自動生成
- 学習中は **100K ステップごと** にログを確認し、以下を記録:
  - Foul Rate / FL Entry Rate / FL Stay Rate / Mean Reward / FPS
- **1M ステップごと** に進捗レポートを `docs/experiments/` に保存する
- レポートには必ずグラフ（学習曲線）を含める

### 実験終了時
- 結果をテンプレートに従ってまとめる
- 成功/失敗の判定理由を明記
- 次のアクションを提案する

---

## ユーザーの学習支援ルール

このプロジェクトは **ユーザーの機械学習の知識・技能向上** も目的としている。

- 新しい手法やアルゴリズムを使う前に、**なぜそれを選ぶのか** を説明する
- 「こういう状況ではこういう手法がある」という選択肢を提示する
- ハイパーパラメータの意味と、変更した場合の影響を説明する
- 専門用語を使うときは、初出時に簡潔な説明を添える

---

## プロジェクト概要

Open-Face Chinese Poker (Pineapple) 3-Max AI
- 54枚デッキ（52枚 + Joker 2枚）
- 3人同時プレイ
- Fantasy Land（Ultimate Rules: QQ=14枚, KK=15枚, AA=16枚, Trips=17枚）

## アーキテクチャ

### C++ ゲームエンジン (`src/cpp/`)
- `evaluator.hpp` — ハンド評価（ACE=14 修正済み）
- `board.hpp` — Top(3)/Middle(5)/Bottom(5) 管理
- `game.hpp` — ゲーム進行、FL 処理
- `solver.hpp` — FL ブルートフォースソルバー
- `pybind/bindings.cpp` — Python バインディング

### Python 環境 (`src/python/`)
- `ofc_3max_env.py` — PettingZoo AECEnv（3人マルチエージェント）
- `ofc_env.py` — Gymnasium 単体環境
- `greedy_fl_solver.py` — FL 近似ソルバー（学習用）
- `notifier.py` — Discord 通知

## ビルド

```bash
python setup.py build_ext --inplace
python -c "import ofc_engine as ofc; print('OK')"
make test
```

## 既知の問題

### ACE=0 バグ（修正済み、再学習必要）
- evaluator.hpp で ACE の kicker を 0（最弱）として処理していた
- 修正済みだが、**既存の全モデル（Phase 9, 10）はバグ環境で学習済み**
- 再学習が必要。fine-tune は 85M ステップで失敗済み（局所最適に嵌る）

## V1 からの引き継ぎモデル

| ファイル | 説明 | 注意 |
|----------|------|------|
| `models/phase10_gcp/p10_fl_stay_150000000.zip` | Phase 10 最終 | ACE バグあり |
| `models/phase9/p9_fl_mastery_250000000.zip` | Phase 9 最終 | ACE バグあり |
| `models/onnx/ofc_ai_v2.onnx` | ONNX (mix-poker-app用) | ACE バグあり |

## ディレクトリ構成

```
OFC NN/
├── CLAUDE.md              # このファイル（運営ルール）
├── src/
│   ├── cpp/               # C++ ゲームエンジン
│   └── python/            # 環境、ソルバー、通知
├── docs/
│   ├── learning/          # 技術学習ノート
│   ├── templates/         # レポートテンプレート
│   └── experiments/       # 実験記録（V2以降）
├── models/                # 学習済みモデル
├── scripts/               # ユーティリティ
├── tests/                 # テスト
├── archive/v1/            # V1 プロジェクト全記録
├── setup.py               # C++ ビルド設定
└── requirements.txt       # Python 依存パッケージ
```

## 過去の記録

V1 プロジェクトの全記録は `archive/v1/PROJECT_HISTORY.md` にまとめてある。
Phase 1-12 の全指標、失敗記録、教訓が含まれる。新しい実験を始める前に参照すること。
