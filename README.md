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
python benchmark.py --repeat 5                 # 1 ケースあたりの判定試行回数（中央値を採用）
python benchmark.py --host http://localhost:11434
```

## 検証結果 (Benchmark Results)

Apple Silicon / macOS・50 ケース（SIMPLE 25 / COMPLEX 25、日本語 26 / 英語 24）・
各ケース 3 回試行（中央値採用）での実測値です。

| 測定項目 | 結果 |
| --- | --- |
| System One（ルーター役） | Qwen2.5:0.5B |
| System Two（重い推論役） | Qwen2.5:7B |
| リクエスト数 | 50 |
| 平均判定レイテンシ | 約 23.2 ms |
| p95 判定レイテンシ | 約 31.6 ms |
| 分類精度 | 96.0 % (48/50) |
| LLM オフロード率 | 54.0 % |
| 誤オフロード (COMPLEX→SIMPLE) | 2 件 |
| 誤エスカレーション (SIMPLE→COMPLEX) | 0 件 |
| System Two 平均レイテンシ | 約 6,042 ms |
| 総処理時間の削減率 | 53.8 % |

### 内訳 (Breakdown)

| 区分 | 精度 | 平均判定レイテンシ |
| --- | --- | --- |
| SIMPLE 期待 | 100.0 % (25/25) | 26.3 ms |
| COMPLEX 期待 | 92.0 % (23/25) | 20.1 ms |
| 日本語 | 100.0 % (26/26) | 22.5 ms |
| 英語 | 91.7 % (22/24) | 24.0 ms |

判定は System Two の約 1/260 の時間（約 0.02 秒 vs 約 6.0 秒）で完了し、
半数強のリクエストで重いモデルの起動自体を回避できました。
実測値はハードウェア・モデル・プロンプトにより変動します。

### 既知の弱点 (Known limitations)

誤分類 2 件はいずれも **COMPLEX を SIMPLE と誤判定する「誤オフロード」方向**でした。
これは回答品質が落ちる危険側の誤りです（逆方向の誤エスカレーションは 0 件）。

```
[COMPLEX → SIMPLE] Refactor this callback-based code into async/await and explain why.
[COMPLEX → SIMPLE] Explain how TCP congestion control reacts to packet loss, step by step.
```

いずれも英語で、命令動詞（`Refactor` / `Explain`）そのものは単純に見えるが実際には
コード生成や多段の説明を要するケースです。日本語 26 件は 100% 正解しており、
現状の弱点は英語の一部の動詞パターンに集中しています。

> なお本リポジトリのプロンプトは、この 50 ケースに対してチューニングし直していません
> （テストセットへの過学習を避けるため）。上記は素の状態での測定値です。

## 仕組み (How the router works)

極小モデルは長い指示文より「1 文字で答えさせる」ほうが安定するため、System One は
few-shot 付きの chat プロンプトで `A`（SIMPLE）/ `B`（COMPLEX）のみを出力させています。
`num_predict=2`・`temperature=0` で生成を最小化し、判定不能な出力は安全側（COMPLEX）に倒します。
few-shot 例はテストケースと重複しないものを使用しています。

ケースを追加・変更する場合は `benchmark.py` の `CASES` を編集してください。

## ライセンス (License)

MIT
