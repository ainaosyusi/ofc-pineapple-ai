# Bitboard によるカード表現 - 学習ノート

## 概要

Bitboardは、ボードゲームやカードゲームで使用される高効率なデータ表現手法です。
OFC Pineapple AIでは、52枚のカードを64ビット整数で表現します。

---

## 🎴 カードのビット表現

### ビット位置の計算

```
ビット位置 = スート × 13 + ランク

スート:
  0 = スペード (♠)
  1 = ハート (♥)
  2 = ダイヤ (♦)
  3 = クラブ (♣)

ランク:
  0 = A
  1 = 2
  2 = 3
  ...
  11 = Q
  12 = K
```

### 実装例

```cpp
enum Suit { SPADE = 0, HEART = 1, DIAMOND = 2, CLUB = 3 };
enum Rank { ACE = 0, TWO = 1, ..., KING = 12 };

using CardMask = uint64_t;

// カードをビットマスクに変換
constexpr CardMask card_to_mask(int suit, int rank) {
    return 1ULL << (suit * 13 + rank);
}

// 例: A♠ のマスク
CardMask ace_of_spades = card_to_mask(SPADE, ACE);  // = 1
```

---

## ⚡ ビット演算による高速判定

### ペア判定

```cpp
// 特定ランクのカード枚数をカウント
int count_rank(CardMask hand, int rank) {
    CardMask rank_mask = (1ULL << rank) | (1ULL << (rank + 13)) 
                       | (1ULL << (rank + 26)) | (1ULL << (rank + 39));
    return __builtin_popcountll(hand & rank_mask);
}
```

### フラッシュ判定

```cpp
// 同じスートのカードが5枚以上あるか
bool is_flush(CardMask hand) {
    CardMask suit_mask = 0x1FFF; // 下位13ビット
    for (int i = 0; i < 4; i++) {
        if (__builtin_popcountll(hand & (suit_mask << (i * 13))) >= 5) {
            return true;
        }
    }
    return false;
}
```

---

## 📊 メリット

| 項目 | 配列方式 | Bitboard方式 |
|-----|---------|-------------|
| メモリ使用量 | 52バイト以上 | 8バイト |
| フラッシュ判定 | O(n)ループ | O(1)ビット演算 |
| ペア判定 | ソート必要 | popcount一発 |

---

## 🔗 次のステップ

- [ ] ストレート判定の実装
- [ ] OFC用のルックアップテーブル作成
- [ ] 3枚役判定用の簡略化ロジック
