# OFC Pineapple AI - 現在の開発状況

## 最終更新: 2026-02-26

> **現在の状態**: V2 プロジェクト開始準備完了

---

## V1 プロジェクト結果（参考）

V1 の全記録は `archive/v1/PROJECT_HISTORY.md` にまとめてある。

### V1 最終性能（Phase 10, ACE バグあり環境での評価値）

| 指標 | Phase 10 (150M) |
|------|----------------|
| Foul Rate | 18.0% |
| FL Entry Rate | 33.6% |
| FL Stay Rate | 15.8% |
| Mean Score | +36.08 |

**重要**: これらの数値は ACE=0 バグ環境でのもの。正しいルールでの実力はこれより低い。

### V1 で判明した問題
1. ACE=0 evaluator バグ → 全モデルが間違ったルールで学習
2. fine-tune で矯正不可 → 85M ステップで失敗確認済み
3. 数値上の成績と実際のプレイ品質が乖離

---

## V2 方針

- 修正済み evaluator（ACE=14）でゼロから学習
- 学術的アプローチ: 仮説→実験→検証のサイクル
- 実機テスト（mix-poker-app）を各フェーズで実施
- ユーザーの ML 学習も目的に含む

## 保存済みモデル（V1、参考用）

| ファイル | 説明 | 備考 |
|----------|------|------|
| `models/phase10_gcp/p10_fl_stay_150000000.zip` | Phase 10 最終 | ACE バグあり |
| `models/phase9/p9_fl_mastery_250000000.zip` | Phase 9 最終 | ACE バグあり |
| `models/onnx/ofc_ai_v2.onnx` | ONNX | ACE バグあり |

---

*Last updated: 2026-02-26*
