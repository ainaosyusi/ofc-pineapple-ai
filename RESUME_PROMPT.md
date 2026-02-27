# 移行先PCでClaude Codeに渡す初回プロンプト

以下をそのままコピペしてください。

---

## プロンプト本文

```
このプロジェクトは OFC Pineapple AI V2 — 3人同時プレイのOpen-Face Chinese Poker (Pineapple) を強化学習で解くAIです。別PCから移行してきたばかりです。

### まずやること（環境構築）

1. SETUP_GUIDE.md を読んで環境構築の手順を確認
2. 以下を順に実行:
   - python3 -m venv .venv && source .venv/bin/activate
   - pip install -r requirements-lock.txt
   - python setup.py build_ext --inplace
   - python -c "import ofc_engine as ofc; print('OK')"
   - python tests/test_evaluator_comprehensive.py  (124テスト全パス確認)
3. 環境構築完了後、テスト学習で動作確認:
   - python v2/train_v2.py --test-mode --reward-config C

### プロジェクトの現状

■ V1 の歴史 (archive/v1/PROJECT_HISTORY.md に全記録)
- 250M+ステップ学習したがACE=0バグ発覚 → 全モデル使用不可
- V2でゼロから再学習中

■ V2 で完了済み
- Phase 0: C++エンジン整備、報酬設計3案（A/B/C）
- Phase 1 Step 1: 報酬アブレーション → Config C（条件付きシェーピング）採用確定
  - 5M結果: Foul 33.2%, Score +5.86, FL追求差 +2.0%
- C++エンジン網羅テスト: 124/124パス

■ 次のタスク: Phase 1 Step 2（Self-Play プール拡張比較）
- Config C をゼロから再学習（GCPチェックポイントは削除済み）
- 5M到達後に D/E 比較分岐:
  - D（ベースライン）: pool=5, latest=0.8, rule_based=0%
  - E（拡張版）:      pool=15, latest=0.6, rule_based=10%
- 各 5M → 10M ステップ

■ 学習コマンド (ローカル)
Config C をゼロから:
  DISCORD_WEBHOOK_URL='https://discord.com/api/webhooks/1461388510140301315/Ofc9ok3IDLgRhFR7Oa2iA4-CHHL9uFvFbIPT9G9tbA5X5r-wG-XMbnkg7ubocLJJQ5Pf' \
  NUM_ENVS=4 python v2/train_v2.py --reward-config C --steps 5000000

### 重要なルール（CLAUDE.md に詳細）
- 不明点は必ず質問する（V1でACEランキングを間違えた前科あり）
- 問題を隠さない・即報告
- Discord Webhook を必ず設定（上記URL）
- レポートは docs/templates/ のテンプレートに従う
- ユーザーのML学習支援も兼ねる（手法の選択理由を説明する）

### ファイル構成の要点
- v2/train_v2.py       — 学習スクリプト本体（Step 2対応済み）
- v2/rule_based_agent.py — SafeAgent/AggressiveAgent
- src/cpp/              — C++ゲームエンジン
- src/python/           — 環境・ソルバー・通知
- CLAUDE.md             — プロジェクト運営ルール（必ず最初に読む）
- SETUP_GUIDE.md        — 環境構築手順
- ofc_v2_action_plan_20260226.md — スペシャリスト2名の合意方針

まず環境構築を完了させて、全テストがパスすることを確認してください。
```
