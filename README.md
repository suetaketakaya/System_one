# System One — LLM Routing Benchmark

> A local benchmark and implementation of the "System One + LLM" routing architecture using Ollama and lightweight models.

Ollama と極小 LLM を用いて、「System One（超高速判定器）」による LLM ルーティングをローカルで再現・検証するベンチマークプロジェクトです。

---

## 背景 (Background)

LLM アプリケーションでは、すべてのリクエストを大きなモデルに投げると**レイテンシ**と**コスト**が跳ね上がります。
人間の認知における二重過程理論（System 1 / System 2）になぞらえ、処理を 2 段構えに分離するのが本プロジェクトのアプローチです。

- **System One (Fast Router):** 極小モデルによる超高速な分類・判定（例: `qwen2.5:0.5b`）
- **System Two (Heavy LLM):** 複雑な推論が必要な場合のみ起動（例: `qwen2.5:7b` や Claude API）

単純な定型リクエストは System One の段階で即応答し、System Two の呼び出し自体を回避します。

```
            ┌──────────────────────┐
  入力 ───► │  System One (0.5B)   │──► 単純 ──► 即時応答
            │  分類 / ルーティング  │
            └──────────┬───────────┘
                       │ 複雑
                       ▼
            ┌──────────────────────┐
            │ System Two (7B / API)│──► 生成応答
            └──────────────────────┘
```

## 特徴 (Features)

- **超低レイテンシ:** 極小モデルによる分類処理の速度を実測
- **LLM オフロード率の可視化:** 重いテキスト生成処理を回避できた割合を算出
- **過学習を避けた評価:** 調整用(dev)と汎化測定用(holdout)にケースを分離
- **完全ローカル実行:** Ollama を活用し、API 課金なしで完結

## 環境構築 (Prerequisites)

- Python 3.8+
- [Ollama](https://ollama.com/)
- 使用モデルの Pull:

```bash
# System One（ルーター役 / 必須）
ollama pull qwen2.5:0.5b

# System Two（重い推論役 / 任意）
ollama pull qwen2.5:7b
```

## 使い方 (Usage)

リポジトリをクローンし、必要なライブラリをインストールして実行します。

```bash
git clone https://github.com/suetaketakaya/System_one.git
cd System_one

# 依存関係のインストール
pip install -r requirements.txt   # または: pip install ollama

# ベンチマークの実行
python benchmark.py
```

## オプション (Options)

```bash
python benchmark.py --no-system-two            # 判定のみ計測（System Two を起動しない）
python benchmark.py --heavy-model llama3.2     # System Two のモデルを差し替え
python benchmark.py --set holdout              # ホールドアウトセットのみで評価
python benchmark.py --repeat 5                 # 1 ケースあたりの判定試行回数（中央値を採用）
python benchmark.py --host http://localhost:11434
```

## 検証結果 (Benchmark Results)

Apple Silicon / macOS・全 110 ケース・各ケース 3 回試行（中央値採用）での実測値です。

| 測定項目 | 結果 |
| --- | --- |
| System One（ルーター役） | Qwen2.5:0.5B |
| System Two（重い推論役） | Qwen2.5:7B |
| リクエスト数 | 110 |
| 平均判定レイテンシ | 約 27.9 ms |
| p95 判定レイテンシ | 約 37.5 ms |
| 分類精度 | 100.0 % (110/110) |
| LLM オフロード率 | 50.0 % |
| 誤オフロード (COMPLEX→SIMPLE) | 0 件 |
| 誤エスカレーション (SIMPLE→COMPLEX) | 0 件 |
| System Two 平均レイテンシ | 約 6,286 ms |
| 総処理時間の削減率 | 49.8 % |

判定は System Two の約 1/225 の時間（約 0.03 秒 vs 約 6.3 秒）で完了し、
半数のリクエストで重いモデルの起動自体を回避できました。
実測値はハードウェア・モデル・プロンプトにより変動します。

## 評価の分割 (Dev / Holdout split)

プロンプト調整がテストセットへの過学習になっていないことを確かめるため、
ケースを `cases.py` で 2 つに分けています。

| セット | 件数 | 構成 | 用途 |
| --- | --- | --- | --- |
| `DEV_CASES` | 60 | SIMPLE 30 / COMPLEX 30、日英各 30 | プロンプト調整に使用 |
| `HOLDOUT_CASES` | 50 | SIMPLE 25 / COMPLEX 25、日 26 / 英 24 | **調整に一切使わない**。汎化性能の測定専用 |

```bash
python benchmark.py --set dev       # 調整用セット
python benchmark.py --set holdout   # 汎化性能の測定
python benchmark.py --set all       # 全 110 ケース（default）
```

### 改善前後 (Before / After)

few-shot 例を追加する前後のスコアです。**holdout は調整に使っていないため、
ここでの改善は過学習ではなく汎化性能の向上を意味します。**

| プロンプト | dev (60) | holdout (50) |
| --- | --- | --- |
| 改善前 | 93.3 % (56/60) | 92.0 % (46/50) |
| 改善後 | **100.0 % (60/60)** | **100.0 % (50/50)** |

改善前の誤分類は、dev・holdout いずれも **全件が英語の COMPLEX→SIMPLE（誤オフロード）**
に集中していました。

```
改善前に holdout で外した 4 件:
  [COMPLEX → SIMPLE] Explain how DNS resolution works end to end.
  [COMPLEX → SIMPLE] Refactor this nested conditional into guard-clause style.
  [COMPLEX → SIMPLE] Explain optimistic versus pessimistic locking, with examples.
  [COMPLEX → SIMPLE] Rewrite this recursive function iteratively and analyze the complexity.
```

`Explain` / `Refactor` / `Rewrite` という動詞は単体では単純に見えますが、実際には
多段の説明やコード書き換えを要します。そこで few-shot 例に次の 4 件を追加しました
（いずれも dev / holdout のどちらとも重複しない文です）。

| 追加した例 | ラベル | 狙い |
| --- | --- | --- |
| `Explain what a pointer is.` | A | 短い定義で済むものは A（動詞に引きずられない） |
| `Define latency.` | A | 同上 |
| `Refactor this function to remove duplication` | B | 既存コードの書き換えは B |
| `Explain how HTTPS certificate validation works, step by step` | B | 多段の仕組み説明は B |

### 効いたのは規則文ではなく few-shot 例だった

切り分けのため、規則文と few-shot 例を別々に変更して dev で比較しました。

| 変更内容 | dev 精度 |
| --- | --- |
| 変更なし（改善前） | 93.3 % |
| 規則文のみ書き換え | 93.3 %（**変化なし**） |
| few-shot 例のみ追加 | 100.0 % |
| 両方 | 100.0 % |

規則文に「refactoring」「multi-step explanation」を明記しても 0.5B モデルの挙動は
変わりませんでした。**極小モデルには抽象的な規則より具体例のほうが効く**という結果です。
そのため最終版では規則文は元のまま、few-shot 例のみを追加しています。

## 仕組み (How the router works)

極小モデルは長い指示文より「1 文字で答えさせる」ほうが安定するため、System One は
few-shot 付きの chat プロンプトで `A`（SIMPLE）/ `B`（COMPLEX）のみを出力させています。
`num_predict=2`・`temperature=0` で生成を最小化し、判定不能な出力は安全側（COMPLEX）に倒します。

ラベル付けの方針は次の通りです（`cases.py` 参照）。

- **SIMPLE** … 1〜2 文で答えきれる。挨拶・相槌・単一の事実・簡単な計算
- **COMPLEX** … コード生成・設計・比較検討・分析・多段の手順説明が必要

ケースを追加・変更する場合は `cases.py` を編集してください。few-shot 例
（`benchmark.py` の `ROUTER_SHOTS`）と重複させないことが、測定を成立させる条件です。

## ライセンス (License)

MIT
