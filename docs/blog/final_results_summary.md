# OFC Pineapple AI 開発プロジェクト：Phase 9 完了報告

**更新日**: 2026-02-07

## プロジェクトの軌跡

約3週間にわたる開発で、OFC Pineapple AIプロジェクトは「全くのゼロ」から「プロレベルのAI」まで進化しました。

250Mステップ（2億5千万手）の強化学習を通じて、Fantasy Landを積極的に狙い、低いファウル率を維持するAIを実現しました。

---

## Phase 9 最終成果

| 指標 | 初期状態 | Phase 9 (250M) | 人間プロ |
|:---|:---|---:|:---|
| **Foul Rate** | 87.5% | **16.8%** | 10-20% |
| **Mean Score** | -30.0 | **+12.66** | - |
| **FL Entry Rate** | 0% | **22.8%** | 15-30% |
| **FL Stay Rate** | 0% | **8.0%** | - |
| **Win Rate** | 0% | **75.8%** | - |

**結論**: Phase 9 AIは人間のプロプレイヤーと同等のレベルに到達しました。

---

## 開発フェーズの進化

```
Phase 1   →   Phase 4   →   Phase 7   →   Phase 8.5   →   Phase 9
ファウル       ジョーカー      並列学習       Ultimate       FL Mastery
回避学習       対応            (GCP)         Rules FL        完了 ✅
37.8%         25.1%          25.8%         22.0%          16.8%
```

### 主要マイルストーン

| Phase | Steps | 主な達成 |
|:---|---:|:---|
| Phase 1-2 | 10M | ファウル回避・役作り基礎 |
| Phase 4 | 20M | ジョーカー対応 (54枚デッキ) |
| Phase 5 | 30M | 3人対戦 (3-Max) 導入 |
| Phase 7 | 20M | GCP並列学習インフラ構築 |
| Phase 8 | 100M | Self-Play・Multi-Variant |
| Phase 8.5 | 100M | Ultimate Rules FL導入 |
| **Phase 9** | **250M** | **全目標達成 ✅** |

---

## 技術的成果

### 1. 高性能C++エンジン

- **Bitboard表現**: 64ビット整数でカード集合を表現し、O(1)で役判定
- **ジョーカー対応**: 54枚デッキ（ジョーカー2枚）でワイルドカード処理
- **Header-only設計**: pybind11でPythonから直接呼び出し可能
- **速度**: 学習時900-1000 FPS

### 2. Fantasy Land完全サポート

Ultimate Rulesを完全実装:
| Top Hand | Cards |
|:---|---:|
| QQ | 14 |
| KK | 15 |
| AA | 16 |
| Trips | 17 |

### 3. 報酬シェーピング

FL特化の報酬設計でPPO飽和を突破:
```python
fl_entry_bonus = 30.0   # FL Entry達成時
fl_stay_bonus = 60.0    # FL Stay達成時
```

### 4. ONNX統合

Node.jsからPython不要で推論:
- 変換: PyTorch → ONNX
- 推論速度: 10-50ms/手
- デプロイ先: mix-poker-app

---

## 学習インフラ

### クラウド構成

- **GCP**: e2-standard-8 (8 vCPU, 32GB RAM)
- **並列化**: SubprocVecEnv × 4環境
- **通知**: Discord Webhook (100kステップごと)
- **総学習時間**: 約100時間

### 自動化

- 自動チェックポイント保存 (1Mステップごと)
- 自動再開機能 (クラッシュ時)
- ディスク容量監視・クリーンアップ

---

## 得られた知見

1. **PPO Self-Playは110M以降で飽和する**
   - 報酬シェーピングで突破可能
   - FL Entry/Stay報酬が効果的

2. **FL Stay条件の数学的困難さ**
   - Trips on Top: Bot >= Mid >= Trips必要で非常に困難
   - Quads on Bottom: Joker活用が実用的

3. **ネットワークサイズの重要性**
   - 881次元入力に対し [512, 256, 128] MLPが必要
   - 小さいネットワークでは表現力不足

4. **Expert Iterationは難しい**
   - MCTS 100 simsでは品質不十分
   - Catastrophic Forgetting発生

---

## 今後の展望

### Phase 10: FL Stay向上 (進行中)

- 目標: FL Stay Rate 8% → 15%+
- 手法: greedy_fl_solver改修 + Fine-tuning
- 現在: 学習中 (~400k / 50M)

### 将来計画

1. **Phase 11: Hold'em AI**
   - RLCardライブラリ活用
   - mix-poker-appへの統合

2. **Phase 12: PLO/Stud AI**
   - マルチゲーム対応
   - 統一されたUI

---

## デプロイ状況

### mix-poker-app統合

OFC AI v1.1.0が `mix-poker-app` に統合済み:
- ONNX形式でNode.jsから直接推論
- リアルタイムでAIと対戦可能
- モバイル/Web対応

---

## 結論

OFC Pineapple AIプロジェクトは、250Mステップの強化学習を経て**プロレベルのAI**を実現しました。

- **Foul Rate 16.8%**: プロレベル達成
- **FL Entry 22.8%**: 積極的なFL狙い
- **Win Rate 75.8%**: 安定した強さ

このプロジェクトは、深層強化学習とドメイン知識の融合により、複雑なポーカーゲームをマスターできることを示しました。

**「AIにポーカーを教えることはできたか？」**
その答えは、Foul Rate 16.8%、FL Entry 22.8%という数字が物語っています。

---

*2026年2月7日 - OFC Pineapple AI 開発チーム*
