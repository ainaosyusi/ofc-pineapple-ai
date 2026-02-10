# OFC AI - 次のアクション

**最終更新: 2026-02-07**

---

## 現在の状態: Phase 10 学習中

- **Phase 9 (250M steps)** 完了 ✅
- 全目標達成: Foul 16.8%, FL Entry 22.8%, Win 75.8%
- mix-poker-app に OFCBot v1.1.0 統合済み
- **Phase 10 (FL Stay向上)** 学習中

---

## Phase 10: FL Stay向上 (進行中)

**目標**: FL Stay Rate 8% → 30%+

### greedy_fl_solver v3 改善結果

| 枚数 | Foul | FL Stay | 平均Royalty | 平均Score |
|:---:|:---:|:---:|:---:|:---:|
| 14枚 | 0% | **29%** | 8.7 | 13.0 |
| 15枚 | 0% | **32%** | 9.7 | 14.4 |
| 16枚 | 0% | **50%** | 13.1 | 20.6 |
| 17枚 | 0% | **66%** | 17.5 | 27.4 |

### v3 ソルバーの主要改善

1. **Flush検出・優先配置** (`_find_flushes()`)
   - 5枚以上の同一スートを検出
   - Bot に Flush を優先配置

2. **Straight検出・優先配置** (`_find_straights()`)
   - 連続5枚のランクを検出
   - Mid に Straight を配置可能

3. **高ロイヤリティ組み合わせ探索** (`_royalty_aware_search()`)
   - Trips on Top + Flush Bot + Straight Mid
   - 全5枚フラッシュ組み合わせを試行してストレートと両立

4. **FL Stay条件の明確化**
   - Trips on Top: Bot >= Mid >= Trips が必要（非常に困難）
   - Quads on Bottom: Joker + Trips が必要

### 報酬設計

- FL Entry報酬: 50ポイント
- FL Stay報酬: 100ポイント（Phase 9より増額）

### 実行中

```bash
# ローカルで学習中
nohup python3 src/python/train_phase10_fl_stay.py > training_phase10.log 2>&1 &

# 進捗確認
tail -30 training_phase10.log

# Discord通知: 100kステップごと
```

---

## 完了した実験

### Phase 9: FL Mastery (250M) ✅

| 指標 | 250M | 150M | 100M | 目標 |
|:---|---:|---:|---:|:---|
| Foul Rate | 16.8% | 17.6% | 22.0% | <20% ✅ |
| Mean Score | +12.66 | +12.58 | +8.43 | >+10 ✅ |
| FL Entry | 22.8% | 21.2% | 8.2% | >15% ✅ |
| FL Stay | 8.0% | 8.2% | 0.0% | >5% ✅ |
| Win Rate | 75.8% | 75.4% | 68.8% | >70% ✅ |

### Phase 8.5b: Self-Play (150M) ✅
### ExIt v1/v2: ❌ 失敗・放棄

---

## 保存済みモデル

| ファイル | ステップ | 説明 |
|:---|---:|:---|
| `models/phase9/p9_fl_mastery_250000000.zip` | 250M | Phase 9 最終 ⭐ |
| `models/phase9/p9_fl_mastery_150000000.zip` | 150M | Phase 9 中間 |
| `models/phase10/p10_fl_stay_100000.zip` | 100k | Phase 10 開始 |
| `models/onnx/ofc_ai.onnx` | - | ONNX (Node.js用) |
| `mix-poker-app/server/models/ofc_ai.onnx` | - | デプロイ済み |

---

## ロードマップ

| Phase | 状態 | 説明 |
|:---|:---|:---|
| Phase 9 | ✅ 完了 | FL Mastery (250M) |
| Phase 10 | ⏳ 進行中 | FL Stay向上 |
| Phase 11 | 🔲 計画中 | Hold'em AI |
| Phase 12 | 🔲 計画中 | PLO/Stud AI |

---

## 教訓

1. **PPO self-playは110M以降飽和** - 報酬シェーピングで突破可能
2. **ExItは高品質エキスパートが必須** - 100-800 MCTS simsでは不十分
3. **FL報酬シェーピングは有効** - FL Entry 8%→23%
4. **ONNX変換でNode.js統合可能** - Python不要、推論10-50ms
5. **GCPインスタンスは学習完了後すぐ停止** - コスト注意
6. **modulo条件は並列環境で動作しない** - 閾値ベースを使用
7. **FL Stay は数学的制約が厳しい**
   - Trips on Top には Mid も Trips+ が必須 (OFC: Bot >= Mid >= Top)
   - 素材不足の場合 Stay は不可能 (例: ペアのみでは Trips+Trips 不可)
8. **FL Solver は Flush/Straight 優先探索が重要** - ランダム探索では高ロイヤリティを見逃す

---

## FL Stay条件

FL Stayには以下のいずれかが必要:

1. **Trips on Top** (任意ランク)
   - **制約**: Mid も Trips 以上が必須 (OFC: Bot >= Mid >= Top)
   - 必要素材: 少なくとも2組のTrips素材 (または Mid に Full House+)
   - 数学的に非常に困難

2. **Quads on Bottom**
   - 4枚同じランクをBottomに配置
   - 必要素材: 4枚同ランク or Trips + Joker
   - Joker活用が実用的なパス

### v3 ソルバーの FL Stay 達成率

| 枚数 | FL Stay | 主なパターン |
|:---:|:---:|:---|
| 14枚 | 29% | Quads Bottom (Joker+Trips) |
| 15枚 | 32% | Quads Bottom |
| 16枚 | 50% | Trips Top + Full House Mid |
| 17枚 | 66% | Trips Top + Trips Mid |

---

*このファイルは `CLAUDE.md` で参照が強制されています。*
