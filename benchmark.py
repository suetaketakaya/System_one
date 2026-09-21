#!/usr/bin/env python3
"""System One + LLM ルーティングのベンチマーク。

極小モデル（System One）でリクエストを SIMPLE / COMPLEX に分類し、
SIMPLE は即時応答、COMPLEX のみ重いモデル（System Two）に委譲する。
判定レイテンシ・分類精度・LLM オフロード率を実測する。

Usage:
    python benchmark.py
    python benchmark.py --no-system-two          # 判定のみ計測
    python benchmark.py --heavy-model llama3.2   # System Two を差し替え
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass

import ollama

ROUTER_MODEL = "qwen2.5:0.5b"
HEAVY_MODEL = "qwen2.5:7b"

# 極小モデルは長い説明文より「1 文字で答えさせる + few-shot」のほうが安定する。
ROUTER_SYSTEM = (
    "Route the request to a model tier. Reply with a single letter, A or B, and nothing else.\n"
    "A = greeting, thanks, small talk, one short factual question, or simple arithmetic.\n"
    "B = anything that needs writing code, designing, comparing, analyzing, planning steps, "
    "or a detailed explanation. If in doubt, answer B."
)

ROUTER_SHOTS: list[tuple[str, str]] = [
    ("おはよう", "A"),
    ("What is 7 times 8?", "A"),
    ("日本の首都は？", "A"),
    ("ありがとう", "A"),
    ("Write a Python script to parse CSV files", "B"),
    ("障害の原因を分析して再発防止策をまとめて", "B"),
    ("Compare PostgreSQL and MySQL for analytics", "B"),
    ("キャッシュ戦略を設計して", "B"),
]


def _router_messages() -> list[dict[str, str]]:
    messages = [{"role": "system", "content": ROUTER_SYSTEM}]
    for request, label in ROUTER_SHOTS:
        messages.append({"role": "user", "content": request})
        messages.append({"role": "assistant", "content": label})
    return messages


@dataclass(frozen=True)
class Case:
    request: str
    expected: str  # "SIMPLE" or "COMPLEX"
    lang: str      # "ja" or "en"


# few-shot 例と重複しないケースのみを採用している（リークを避けるため）。
CASES: list[Case] = [
    # --- SIMPLE / ja ---
    Case("こんにちは", "SIMPLE", "ja"),
    Case("今日は何曜日ですか？", "SIMPLE", "ja"),
    Case("2 + 2 は？", "SIMPLE", "ja"),
    Case("ありがとう、助かりました", "SIMPLE", "ja"),
    Case("東京の郵便番号の桁数は？", "SIMPLE", "ja"),
    Case("お疲れさまです", "SIMPLE", "ja"),
    Case("富士山の高さは？", "SIMPLE", "ja"),
    Case("1200 円の 10% はいくら？", "SIMPLE", "ja"),
    Case("またね", "SIMPLE", "ja"),
    Case("HTTP の 404 は何を意味する？", "SIMPLE", "ja"),
    Case("了解しました", "SIMPLE", "ja"),
    Case("1 マイルは何キロ？", "SIMPLE", "ja"),
    Case("JSON の正式名称は？", "SIMPLE", "ja"),

    # --- SIMPLE / en ---
    Case("Is Python a compiled language?", "SIMPLE", "en"),
    Case("good evening", "SIMPLE", "en"),
    Case("What's the capital of Australia?", "SIMPLE", "en"),
    Case("thanks a lot", "SIMPLE", "en"),
    Case("How many bytes are in a kilobyte?", "SIMPLE", "en"),
    Case("What does API stand for?", "SIMPLE", "en"),
    Case("Is 17 a prime number?", "SIMPLE", "en"),
    Case("Who wrote the novel 1984?", "SIMPLE", "en"),
    Case("see you tomorrow", "SIMPLE", "en"),
    Case("What port does HTTPS use?", "SIMPLE", "en"),
    Case("How many minutes are in a day?", "SIMPLE", "en"),
    Case("no problem, got it", "SIMPLE", "en"),

    # --- COMPLEX / ja ---
    Case("マイクロサービス構成でのイベント駆動設計の利点と欠点を比較して", "COMPLEX", "ja"),
    Case("この四半期の売上データからチャーンの原因を推論して施策を3つ提案して", "COMPLEX", "ja"),
    Case("SQL のクエリが遅い原因を段階的に切り分ける手順を書いて", "COMPLEX", "ja"),
    Case("新規プロダクトのオンボーディング改善ロードマップを作って", "COMPLEX", "ja"),
    Case("Kubernetes のオートスケール設定をレビューして改善案を出して", "COMPLEX", "ja"),
    Case("再帰を使わずに二分探索木を走査する Python コードを書いて", "COMPLEX", "ja"),
    Case("認証基盤を OAuth2 に移行する際のリスクと段取りを整理して", "COMPLEX", "ja"),
    Case("A/B テストの結果が有意かどうか判断する手順を説明して", "COMPLEX", "ja"),
    Case("レガシーコードのリファクタリング方針を優先度つきでまとめて", "COMPLEX", "ja"),
    Case("CI が不安定な原因を切り分けて恒久対策を提案して", "COMPLEX", "ja"),
    Case("マルチテナント DB のスキーマ設計を比較検討して", "COMPLEX", "ja"),
    Case("オンコール当番のローテーション方式を設計して", "COMPLEX", "ja"),
    Case("機械学習モデルの過学習を抑える手法を効果とコストで比較して", "COMPLEX", "ja"),

    # --- COMPLEX / en ---
    Case("Write a Python function that merges overlapping intervals.", "COMPLEX", "en"),
    Case("Explain the trade-offs between BFS and Dijkstra with complexity analysis.", "COMPLEX", "en"),
    Case("Design a rate limiter for a distributed API gateway.", "COMPLEX", "en"),
    Case("Refactor this callback-based code into async/await and explain why.", "COMPLEX", "en"),
    Case("Draft a migration plan from REST to GraphQL for a mobile backend.", "COMPLEX", "en"),
    Case("Analyze why our p99 latency regressed after the last deploy.", "COMPLEX", "en"),
    Case("Write unit tests covering the edge cases of a date parser.", "COMPLEX", "en"),
    Case("Compare vector databases for a RAG pipeline at 10M documents.", "COMPLEX", "en"),
    Case("Design a schema for a multi-region event log with ordering guarantees.", "COMPLEX", "en"),
    Case("Explain how TCP congestion control reacts to packet loss, step by step.", "COMPLEX", "en"),
    Case("Build a retry strategy with exponential backoff and jitter, with code.", "COMPLEX", "en"),
    Case("Evaluate whether we should adopt a monorepo, with the trade-offs.", "COMPLEX", "en"),
]


@dataclass
class Result:
    case: Case
    label: str
    router_ms: float
    heavy_ms: float | None  # None = System Two を起動しなかった


def classify(client: ollama.Client, model: str, request: str) -> tuple[str, float]:
    """System One による分類。(ラベル, 所要ミリ秒) を返す。"""
    messages = _router_messages() + [{"role": "user", "content": request}]

    start = time.perf_counter()
    response = client.chat(
        model=model,
        messages=messages,
        options={"temperature": 0.0, "num_predict": 2},
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    answer = response["message"]["content"].strip().upper()
    # 判定不能なら安全側（COMPLEX）に倒す。
    label = "SIMPLE" if answer.startswith("A") else "COMPLEX"
    return label, elapsed_ms


def invoke_system_two(client: ollama.Client, model: str, request: str) -> float:
    """System Two（重いモデル）を起動し、所要ミリ秒を返す。"""
    start = time.perf_counter()
    client.chat(
        model=model,
        messages=[{"role": "user", "content": request}],
        options={"temperature": 0.2, "num_predict": 128},
    )
    return (time.perf_counter() - start) * 1000


def warmup(client: ollama.Client, model: str) -> None:
    """初回ロードのコストを計測から除外するための空打ち。"""
    client.chat(
        model=model,
        messages=[{"role": "user", "content": "ping"}],
        options={"num_predict": 1},
    )


def _preview(text: str, width: int = 34) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def run(args: argparse.Namespace) -> int:
    client = ollama.Client(host=args.host)

    try:
        warmup(client, args.router_model)
    except Exception as exc:  # noqa: BLE001 - 環境不備を分かりやすく伝える
        print(f"[ERROR] ルーターモデル '{args.router_model}' を実行できません: {exc}", file=sys.stderr)
        print(
            f"        `ollama pull {args.router_model}` を実行し、Ollama が起動しているか確認してください。",
            file=sys.stderr,
        )
        return 1

    use_system_two = not args.no_system_two
    if use_system_two:
        try:
            warmup(client, args.heavy_model)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] System Two '{args.heavy_model}' が利用できないため、判定のみ計測します: {exc}\n")
            use_system_two = False

    results: list[Result] = []

    print(f"System One: {args.router_model}")
    print(f"System Two: {args.heavy_model if use_system_two else '(skipped)'}")
    print(f"試行回数  : {args.repeat} 回 / ケース")
    print("-" * 78)

    for case in CASES:
        latencies: list[float] = []
        label = "COMPLEX"
        for _ in range(args.repeat):
            label, ms = classify(client, args.router_model, case.request)
            latencies.append(ms)
        router_ms = statistics.median(latencies)

        heavy_ms: float | None = None
        if label == "COMPLEX" and use_system_two:
            heavy_ms = invoke_system_two(client, args.heavy_model, case.request)

        results.append(Result(case, label, router_ms, heavy_ms))

        mark = "OK  " if label == case.expected else "MISS"
        heavy_note = f" -> System Two {heavy_ms:7.0f} ms" if heavy_ms is not None else ""
        print(f"[{mark}] {case.lang} {label:<7} {router_ms:6.1f} ms  {_preview(case.request)}{heavy_note}")

    report(results, use_system_two)
    return 0


def report(results: list[Result], use_system_two: bool) -> None:
    router_latencies = [r.router_ms for r in results]
    offloaded = [r for r in results if r.label == "SIMPLE"]
    correct = [r for r in results if r.label == r.case.expected]

    # 誤りの向きで意味が変わる。
    #   誤オフロード: COMPLEX を SIMPLE と判定 -> 回答品質が落ちる（危険側）
    #   誤エスカレーション: SIMPLE を COMPLEX と判定 -> 無駄に重いモデルを起動（コスト側）
    wrong_offload = [r for r in results if r.case.expected == "COMPLEX" and r.label == "SIMPLE"]
    wrong_escalate = [r for r in results if r.case.expected == "SIMPLE" and r.label == "COMPLEX"]

    avg = statistics.mean(router_latencies)
    ordered = sorted(router_latencies)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    offload_rate = len(offloaded) / len(results) * 100
    accuracy = len(correct) / len(results) * 100

    print("-" * 78)
    print("\n## 検証結果 (Benchmark Results)\n")
    print("| 測定項目 | 結果 |")
    print("| --- | --- |")
    print(f"| リクエスト数 | {len(results)} |")
    print(f"| 平均判定レイテンシ | {avg:.1f} ms |")
    print(f"| p95 判定レイテンシ | {p95:.1f} ms |")
    print(f"| 分類精度 | {accuracy:.1f} % ({len(correct)}/{len(results)}) |")
    print(f"| LLM オフロード率 | {offload_rate:.1f} % |")
    print(f"| 誤オフロード (COMPLEX->SIMPLE) | {len(wrong_offload)} 件 |")
    print(f"| 誤エスカレーション (SIMPLE->COMPLEX) | {len(wrong_escalate)} 件 |")

    if use_system_two:
        heavy = [r.heavy_ms for r in results if r.heavy_ms is not None]
        if heavy:
            avg_heavy = statistics.mean(heavy)
            # 全件を System Two に投げた場合との比較（判定コストは両者に含める）。
            actual = sum(router_latencies) + sum(heavy)
            baseline = sum(router_latencies) + avg_heavy * len(results)
            saved = (1 - actual / baseline) * 100
            print(f"| System Two 平均レイテンシ | {avg_heavy:.1f} ms |")
            print(f"| 総処理時間の削減率 | {saved:.1f} % |")

    print("\n### 内訳 (Breakdown)\n")
    print("| 区分 | 精度 | 平均判定レイテンシ |")
    print("| --- | --- | --- |")
    for name, subset in _breakdowns(results):
        if not subset:
            continue
        ok = sum(1 for r in subset if r.label == r.case.expected)
        sub_avg = statistics.mean([r.router_ms for r in subset])
        print(f"| {name} | {ok / len(subset) * 100:.1f} % ({ok}/{len(subset)}) | {sub_avg:.1f} ms |")
    print()

    if wrong_offload or wrong_escalate:
        print("### 誤分類ケース (Misclassified)\n")
        for r in wrong_offload + wrong_escalate:
            print(f"- [{r.case.expected} -> {r.label}] {r.case.request}")
        print()


def _breakdowns(results: list[Result]) -> list[tuple[str, list[Result]]]:
    return [
        ("SIMPLE 期待", [r for r in results if r.case.expected == "SIMPLE"]),
        ("COMPLEX 期待", [r for r in results if r.case.expected == "COMPLEX"]),
        ("日本語", [r for r in results if r.case.lang == "ja"]),
        ("英語", [r for r in results if r.case.lang == "en"]),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="System One + LLM ルーティングのベンチマーク")
    parser.add_argument("--router-model", default=ROUTER_MODEL, help=f"System One のモデル (default: {ROUTER_MODEL})")
    parser.add_argument("--heavy-model", default=HEAVY_MODEL, help=f"System Two のモデル (default: {HEAVY_MODEL})")
    parser.add_argument("--repeat", type=int, default=3, help="1 ケースあたりの判定試行回数 (default: 3)")
    parser.add_argument("--no-system-two", action="store_true", help="System Two を起動せず判定のみ計測する")
    parser.add_argument("--host", default=None, help="Ollama のホスト (例: http://localhost:11434)")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
