"""ベンチマークのテストケース定義。

ラベル付けの方針:
  SIMPLE  … 1〜2 文で答えきれる。挨拶・相槌・単一の事実・簡単な計算。
  COMPLEX … コード生成・設計・比較検討・分析・多段の手順説明が必要。

DEV_CASES     … プロンプト調整に使ってよいセット。
HOLDOUT_CASES … 調整には一切使わないセット。汎化性能の測定専用。

ルーターの few-shot 例（benchmark.py の ROUTER_SHOTS）とは重複させないこと。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    request: str
    expected: str  # "SIMPLE" or "COMPLEX"
    lang: str      # "ja" or "en"


DEV_CASES: list[Case] = [
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
    Case("REST の冪等性とは？", "SIMPLE", "ja"),
    Case("git の HEAD とは何？", "SIMPLE", "ja"),

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
    Case("Explain what a semaphore is.", "SIMPLE", "en"),
    Case("Define idempotency.", "SIMPLE", "en"),
    Case("What is the default port for PostgreSQL?", "SIMPLE", "en"),

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
    Case("このコールバック地獄を async/await に書き換えて理由も説明して", "COMPLEX", "ja"),
    Case("TCP の輻輳制御がパケットロス時にどう動くか段階的に説明して", "COMPLEX", "ja"),

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
    Case("Refactor this loop into list comprehensions and explain the change.", "COMPLEX", "en"),
    Case("Explain how JVM garbage collection works, step by step.", "COMPLEX", "en"),
    Case("Rewrite this SQL query to avoid the N+1 problem.", "COMPLEX", "en"),
]


HOLDOUT_CASES: list[Case] = [
    # --- SIMPLE / ja ---
    Case("おはようございます", "SIMPLE", "ja"),
    Case("今何時ですか？", "SIMPLE", "ja"),
    Case("100 の平方根は？", "SIMPLE", "ja"),
    Case("助かりました、感謝します", "SIMPLE", "ja"),
    Case("日本の人口はおよそ何人？", "SIMPLE", "ja"),
    Case("CSS は何の略？", "SIMPLE", "ja"),
    Case("うん、それでいいよ", "SIMPLE", "ja"),
    Case("1 インチは何センチ？", "SIMPLE", "ja"),
    Case("TCP と UDP、コネクション型はどっち？", "SIMPLE", "ja"),
    Case("Linux の ls コマンドは何をする？", "SIMPLE", "ja"),
    Case("またあとで", "SIMPLE", "ja"),
    Case("光の速さは秒速およそ何キロ？", "SIMPLE", "ja"),
    Case("SSH のデフォルトポートは？", "SIMPLE", "ja"),

    # --- SIMPLE / en ---
    Case("hi there", "SIMPLE", "en"),
    Case("What year did the Berlin Wall fall?", "SIMPLE", "en"),
    Case("much appreciated", "SIMPLE", "en"),
    Case("What is the boiling point of water in Celsius?", "SIMPLE", "en"),
    Case("Define a race condition.", "SIMPLE", "en"),
    Case("What does CRUD stand for?", "SIMPLE", "en"),
    Case("Is 91 a prime number?", "SIMPLE", "en"),
    Case("What's 12 percent of 250?", "SIMPLE", "en"),
    Case("catch you later", "SIMPLE", "en"),
    Case("Which is idempotent, GET or POST?", "SIMPLE", "en"),
    Case("Explain what a mutex is.", "SIMPLE", "en"),
    Case("What file extension does a Python module use?", "SIMPLE", "en"),

    # --- COMPLEX / ja ---
    Case("検索機能のレスポンス改善案を計測方法込みでまとめて", "COMPLEX", "ja"),
    Case("マイグレーション失敗時のロールバック手順を設計して", "COMPLEX", "ja"),
    Case("在庫管理 API のエンドポイント設計をレビューして", "COMPLEX", "ja"),
    Case("同時実行数が増えた際のボトルネックを切り分ける手順を説明して", "COMPLEX", "ja"),
    Case("LRU キャッシュを Python で実装して", "COMPLEX", "ja"),
    Case("料金プランの変更が LTV に与える影響を分析して", "COMPLEX", "ja"),
    Case("テスト自動化の導入計画を段階的に作って", "COMPLEX", "ja"),
    Case("このバッチ処理を並列化する方針を比較検討して", "COMPLEX", "ja"),
    Case("監視アラートの設計方針をしきい値の根拠込みでまとめて", "COMPLEX", "ja"),
    Case("GraphQL の N+1 問題の回避策を実装例つきで説明して", "COMPLEX", "ja"),
    Case("権限管理を RBAC から ABAC へ移行する影響を整理して", "COMPLEX", "ja"),
    Case("デプロイ戦略をブルーグリーンとカナリアで比較して", "COMPLEX", "ja"),
    Case("ログ基盤のコスト最適化案を複数出して", "COMPLEX", "ja"),

    # --- COMPLEX / en ---
    Case("Implement a thread-safe LRU cache in Python.", "COMPLEX", "en"),
    Case("Explain how DNS resolution works end to end.", "COMPLEX", "en"),
    Case("Refactor this nested conditional into guard-clause style.", "COMPLEX", "en"),
    Case("Compare gRPC and REST for internal service communication.", "COMPLEX", "en"),
    Case("Design an idempotent webhook receiver.", "COMPLEX", "en"),
    Case("Investigate why our batch job's memory usage grows over time.", "COMPLEX", "en"),
    Case("Write integration tests for a payment flow with retries.", "COMPLEX", "en"),
    Case("Plan a zero-downtime schema migration for a large table.", "COMPLEX", "en"),
    Case("Explain optimistic versus pessimistic locking, with examples.", "COMPLEX", "en"),
    Case("Design a feature flag rollout strategy for a mobile app.", "COMPLEX", "en"),
    Case("Rewrite this recursive function iteratively and analyze the complexity.", "COMPLEX", "en"),
    Case("Assess whether we should move from REST polling to WebSockets.", "COMPLEX", "en"),
]


CASE_SETS: dict[str, list[Case]] = {
    "dev": DEV_CASES,
    "holdout": HOLDOUT_CASES,
    "all": DEV_CASES + HOLDOUT_CASES,
}
