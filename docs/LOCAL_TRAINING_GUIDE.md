# OFC AI ローカルPC学習セットアップガイド

**作成日: 2026-02-23**

---

## 概要

Open-Face Chinese Poker (Pineapple) 3-Max AIの強化学習環境を、ローカルPCでセットアップ・実行するための手順書。

### 現状

- **最新モデル**: Phase 10 (150M steps) — バグあり evaluator で学習済み
- **問題**: C++ evaluator に ACE=0（最弱）バグがあり、全既存モデルが影響を受けている
- **修正済み**: evaluator.hpp は ACE=14（最強）に修正済み
- **目標**: 修正済み evaluator でゼロから再学習（Phase 12）

### Phase 10 (バグあり evaluator) の性能参考値

| 指標 | 値 |
|------|-----|
| Foul Rate | 18.0% |
| FL Entry Rate | 33.6% |
| FL Stay Rate | 15.8% |
| Mean Score | +36.08 |

---

## 1. 必要なもの

### ハードウェア

- **CPU**: 4コア以上推奨（並列環境用）
- **RAM**: 8GB以上
- **GPU**: オプション（CPU学習で十分。GPU があれば速くなる可能性あり）
- **ストレージ**: 5GB以上空き

### ソフトウェア

- Python 3.9+
- C++ コンパイラ (g++ / clang++)、C++17対応
- Git

---

## 2. セットアップ手順

### 2.1 リポジトリ取得

```bash
git clone <リポジトリURL> ofc-training
cd ofc-training
```

または、ファイルをコピーする場合:
```bash
# 必要なディレクトリ
# src/         ← C++ & Python ソースコード（必須）
# models/      ← 学習済みモデル（任意、fine-tuneする場合必要）
# setup.py     ← C++ビルド設定（必須）
# requirements.txt（必須）
```

### 2.2 Python 仮想環境の作成

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# Windows: venv\Scripts\activate
```

### 2.3 依存パッケージのインストール

```bash
pip install --upgrade pip

# PyTorch (CPU版)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# PyTorch (CUDA版 - NVIDIA GPU がある場合)
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# その他の依存パッケージ
pip install stable-baselines3==2.7.1 sb3-contrib==2.7.1
pip install gymnasium==1.1.1 pettingzoo==1.25.0
pip install numpy pybind11 requests matplotlib
```

### 2.4 C++ エンジンのビルド

```bash
python setup.py build_ext --inplace
```

ビルド確認:
```bash
python -c "import ofc_engine as ofc; print('Engine loaded OK')"
```

### 2.5 動作確認テスト

```bash
# C++ ユニットテスト
make test
# → evaluator/board 関連テストが PASSED ならOK
# → deck 系4件の FAILED は既知の問題（54枚デッキ vs 52枚テスト）、無視してよい

# Python テスト
python tests/test_joker.py
```

---

## 3. 学習の実行

### 3.1 テストモード（まず最初にこれを実行）

```bash
NUM_ENVS=2 python src/python/train_phase12_ace_fix.py --test-mode --steps 10000
```

正常に動けば以下のような出力:
```
Creating 2 parallel environments...
Loading Phase 10 model: models/phase10_gcp/p10_fl_stay_150000000.zip
*** ACE修正evaluatorで再学習開始 ***
...
```

### 3.2 本番学習

```bash
# 環境変数設定
export NUM_ENVS=4                    # CPUコア数に応じて調整（推奨: コア数と同じ）
export DISCORD_WEBHOOK_URL='https://discord.com/api/webhooks/1461388510140301315/Ofc9ok3IDLgRhFR7Oa2iA4-CHHL9uFvFbIPT9G9tbA5X5r-wG-XMbnkg7ubocLJJQ5Pf'

# バックグラウンド実行
nohup python3 src/python/train_phase12_ace_fix.py --steps 200000000 > training.log 2>&1 &

# プロセス確認
ps aux | grep train_phase12
```

### 3.3 進捗確認

```bash
# 最新のステップログ
grep '\[Step' training.log | tail -5

# リアルタイム監視
tail -f training.log
```

### 3.4 学習の停止

```bash
pkill -f train_phase12
```

### 3.5 学習の再開

学習は 1M ステップごとにチェックポイントを自動保存します。
スクリプトを再実行すると、最新チェックポイントから自動的に再開されます。

```bash
# そのまま再実行するだけでOK
nohup python3 src/python/train_phase12_ace_fix.py --steps 200000000 > training.log 2>&1 &
```

チェックポイントの保存先: `models/phase12/p12_ace_fix_*.zip`

---

## 4. 学習スクリプトの仕組み

### train_phase12_ace_fix.py

| 項目 | 設定値 |
|------|--------|
| アルゴリズム | MaskablePPO (Stable-Baselines3) |
| ベースモデル | Phase 10 150M (`models/phase10_gcp/p10_fl_stay_150000000.zip`) |
| 並列環境数 | NUM_ENVS (デフォルト: 4) |
| 学習率 | 0.0001 |
| エントロピー係数 | 0.02 |
| FL Entry ボーナス | +50.0 |
| FL Stay ボーナス | +100.0 |
| FL Solver | greedy (高速近似) |
| チェックポイント間隔 | 1M ステップ |
| Discord 通知間隔 | 100K ステップ |

### ゼロから学習する場合

Phase 10 モデルからの fine-tune ではなく、ゼロから学習する場合は、
`train_phase12_ace_fix.py` の `BASE_MODEL` を変更するか、
`models/phase12/` ディレクトリ内のチェックポイントと
`models/phase10_gcp/p10_fl_stay_150000000.zip` を削除してください。

ゼロから学習する場合、`MaskablePPO` のインスタンスを新規作成する必要があります:
```python
# train_phase12_ace_fix.py の main() 内を修正
model = MaskablePPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    ent_coef=0.05,
    n_steps=8192,
    batch_size=256,
    n_epochs=4,
    gamma=0.999,
    verbose=1,
)
```

---

## 5. 監視すべき指標

学習ログに以下の指標が50Kステップごとに出力されます:

| 指標 | 目標 | 説明 |
|------|------|------|
| Foul Rate | < 20% | ボードがフォール（Bot < Mid < Top 違反）になる率 |
| FL Entry Rate | > 30% | Fantasy Land に入れる率 |
| FL Stay Rate | > 15% | Fantasy Land に入った後、再度 FL に残れる率 |
| Mean Reward | > +30 | 平均スコア（高いほど良い） |
| FPS | 参考値 | 学習速度（環境依存） |

### 期待される推移

1. **序盤 (0-10M)**: Foul Rate が高い（40-50%）。ACE修正に適応中
2. **中盤 (10-50M)**: Foul Rate が徐々に下がる。FL Entry が上がり始める
3. **後半 (50-150M)**: 指標が安定。Phase 10 相当（Foul 18%, FL Entry 34%）に近づく
4. **完了 (150-200M)**: 収束。改善が止まったら終了

**注意**: 50M ステップで Foul Rate が 40% 以上のまま改善しない場合、
fine-tune ではなくゼロからの学習に切り替えることを検討してください。

---

## 6. ACE バグの詳細（技術背景）

### 問題

C++ evaluator (`src/cpp/evaluator.hpp`) で ACE のランクが 0（最弱）として
kicker 比較に使用されていた。

```
バグあり: 2 < 3 < 4 < ... < K < A(=0 → 最弱)
修正後:   2 < 3 < 4 < ... < K < A(=14 → 最強)
```

### 影響

- AA が最弱ペア（22 より弱い）として評価されていた
- A-high フラッシュが最弱フラッシュとして評価されていた
- フォール判定の約 24.8% が誤判定（フォールなのにフォールと検出されない）
- 全既存モデル（Phase 9, 10）がこのバグありで学習済み

### 修正内容 (evaluator.hpp)

- `to_cmp_rank()`: ACE(0) → 14 に変換
- `RANK_DESC`: ループ順を ACE 先頭に変更
- `pack5()`: kicker を4ビットずつパック
- FLUSH: 実カードランクを kicker として保存
- 全関連ファイル (board.hpp, mcts.hpp, solver.hpp) を更新済み

---

## 7. ファイル構成

```
OFC NN/
├── src/
│   ├── cpp/                          # C++ ゲームエンジン
│   │   ├── evaluator.hpp             # ハンド評価（ACE修正済み）
│   │   ├── board.hpp                 # ボード管理
│   │   ├── game.hpp                  # ゲーム進行
│   │   ├── solver.hpp                # FL ソルバー
│   │   ├── mcts.hpp                  # MCTS
│   │   └── pybind/bindings.cpp       # Python バインディング
│   └── python/
│       ├── train_phase12_ace_fix.py  # 学習スクリプト ★
│       ├── ofc_3max_env.py           # 3人対戦環境
│       ├── greedy_fl_solver.py       # FL 近似ソルバー
│       └── notifier.py               # Discord 通知
├── models/
│   ├── phase10_gcp/                  # Phase 10 チェックポイント
│   │   └── p10_fl_stay_150000000.zip # ベースモデル ★
│   └── phase12/                      # Phase 12 出力先
├── setup.py                          # C++ ビルド設定
├── requirements.txt                  # Python 依存パッケージ
└── Makefile                          # C++ テスト用
```

---

## 8. トラブルシューティング

### C++ ビルドエラー

```
ModuleNotFoundError: No module named 'pybind11'
```
→ `pip install pybind11` を実行

### ModuleNotFoundError: No module named 'ofc_engine'

→ `python setup.py build_ext --inplace` を再実行

### ベースモデルが見つからない

```
FileNotFoundError: models/phase10_gcp/p10_fl_stay_150000000.zip
```
→ モデルファイルを適切なパスに配置する。
  または、ゼロから学習する場合はセクション4の手順を参照。

### FPS が極端に低い (< 50)

→ `NUM_ENVS=1` に下げてメモリ不足でないか確認。
  SubprocVecEnv は各環境がプロセスを生成するため、コア数以上に設定しない。

### Discord 通知が来ない

→ 環境変数 `DISCORD_WEBHOOK_URL` が設定されているか確認:
```bash
echo $DISCORD_WEBHOOK_URL
```

---

## 9. 学習完了後

### ONNX エクスポート

学習済みモデルを mix-poker-app で使用するには ONNX 変換が必要:

```bash
python scripts/export_onnx.py --model models/phase12/p12_ace_fix_XXXXX.zip --output models/onnx/ofc_ai_v3.onnx
```

### mix-poker-app への配置

```bash
cp models/onnx/ofc_ai_v3.onnx /path/to/mix-poker-app/server/models/ofc_ai.onnx
```

**重要**: Phase 12 モデルは ACE=最強の正しい evaluator で学習されているため、
mix-poker-app の `OFCScoring.ts` の `toTrainingRank` 関数を元に戻す必要がある
（ACE を 0 に変換する処理を削除する）。
