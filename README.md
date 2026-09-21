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

## 検証結果のサンプル (Benchmark Results)

Apple Silicon / macOS・12 ケース（SIMPLE 6 / COMPLEX 6）・各ケース 3 回試行（中央値採用）での実測値です。

| 測定項目 | 結果 |
| --- | --- |
| System One（ルーター役） | Qwen2.5:0.5B |
| System Two（重い推論役） | Qwen2.5:7B |
| 平均判定レイテンシ | 約 20.4 ms |
| p95 判定レイテンシ | 約 26.3 ms |
| 分類精度 | 100.0 % |
| LLM オフロード率 | 50.0 % |
| System Two 平均レイテンシ | 約 5,976 ms |
| 総処理時間の削減率 | 49.8 % |

> 判定は System Two の約 1/293 の時間（約 0.02 秒 vs 約 6.0 秒）で完了し、半数のリクエストで重いモデルの起動自体を回避できました。
> 実測値はハードウェア・モデル・プロンプトにより変動します。

## 仕組み (How the router works)

極小モデルは長い指示文より「1 文字で答えさせる」ほうが安定するため、System One は
few-shot 付きの chat プロンプトで `A`（SIMPLE）/ `B`（COMPLEX）のみを出力させています。
`num_predict=2`・`temperature=0` で生成を最小化し、判定不能な出力は安全側（COMPLEX）に倒します。

## ライセンス (License)

MIT
