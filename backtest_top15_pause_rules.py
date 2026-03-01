"""Backtest Precise TOP15 strategy with pause-after-hit rules to reduce drawdown."""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Dict, List

import pandas as pd

from precise_top15_predictor import PreciseTop15Predictor

TEST_PERIODS = 300
BASE_BET = 15
WIN_REWARD = 47
FIB_SEQUENCE = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
SAVE_DIR = "analysis"


@dataclass
class BacktestResult:
    mode: str
    description: str
    trigger_hits: int
    pause_length: int
    test_periods: int
    bet_periods: int
    pause_periods: int
    hits: int
    hit_rate: float
    total_bet: float
    total_win: float
    net_profit: float
    roi: float
    max_drawdown: float
    paused_hit_count: int
    pause_trigger_count: int


def run_backtest(strategy: Dict) -> Dict:
    """回测单个暂停策略"""
    mode = strategy["name"]
    trigger_hits = strategy.get("trigger_hits", 0)
    pause_length = strategy.get("pause_length", 0)
    description = strategy.get("description", "")

    df = pd.read_csv("data/lucky_numbers.csv", encoding="utf-8-sig")
    numbers = df["number"].tolist()

    if len(numbers) <= TEST_PERIODS:
        raise ValueError(f"数据期数不足，至少需要 {TEST_PERIODS + 1} 期，当前只有 {len(numbers)} 期")

    start_idx = len(numbers) - TEST_PERIODS
    predictor = PreciseTop15Predictor()

    balance = 0.0
    total_bet = 0.0
    total_win = 0.0
    fib_index = 0
    consecutive_hits = 0
    max_drawdown = 0.0
    min_balance = 0.0
    hits = 0
    pause_cooldown = 0
    pause_periods = 0
    paused_hit_count = 0
    pause_trigger_count = 0
    miss_streak = 0
    longest_miss_streak = 0

    details: List[Dict] = []

    for idx in range(start_idx, len(numbers)):
        history = numbers[:idx]
        actual = numbers[idx]
        date = df.iloc[idx]["date"] if "date" in df.columns else idx + 1

        predictions = predictor.predict(history)
        hit = actual in predictions

        if pause_cooldown > 0:
            pause_cooldown -= 1
            pause_periods += 1
            if hit:
                paused_hit_count += 1
            details.append(
                {
                    "date": date,
                    "period": idx + 1,
                    "status": "暂停",
                    "bet": 0,
                    "multiplier": 0,
                    "hit": hit,
                    "paused_hit": hit,
                    "balance": balance,
                    "max_drawdown": max_drawdown,
                }
            )
            continue

        multiplier = FIB_SEQUENCE[min(fib_index, len(FIB_SEQUENCE) - 1)]
        bet = BASE_BET * multiplier
        total_bet += bet

        if hit:
            win = WIN_REWARD * multiplier
            total_win += win
            profit = win - bet
            balance += profit
            hits += 1
            consecutive_hits += 1
            miss_streak = 0
            fib_index = 0
        else:
            win = 0
            profit = -bet
            balance += profit
            consecutive_hits = 0
            miss_streak += 1
            longest_miss_streak = max(longest_miss_streak, miss_streak)
            fib_index += 1

        if balance < min_balance:
            min_balance = balance
            max_drawdown = abs(min_balance)

        should_pause = False
        if pause_length > 0 and hit:
            if trigger_hits <= 1:
                should_pause = True
            elif consecutive_hits >= trigger_hits:
                should_pause = True

        if should_pause:
            pause_cooldown = pause_length
            pause_trigger_count += 1
            consecutive_hits = 0

        details.append(
            {
                "date": date,
                "period": idx + 1,
                "status": "投注",
                "bet": bet,
                "multiplier": multiplier,
                "hit": hit,
                "profit": profit,
                "balance": balance,
                "max_drawdown": max_drawdown,
                "miss_streak": miss_streak,
            }
        )

    bet_periods = TEST_PERIODS - pause_periods
    hit_rate = hits / bet_periods * 100 if bet_periods else 0.0
    roi = total_win / total_bet * 100 - 100 if total_bet else 0.0

    result = BacktestResult(
        mode=mode,
        description=description,
        trigger_hits=trigger_hits,
        pause_length=pause_length,
        test_periods=TEST_PERIODS,
        bet_periods=bet_periods,
        pause_periods=pause_periods,
        hits=hits,
        hit_rate=hit_rate,
        total_bet=total_bet,
        total_win=total_win,
        net_profit=balance,
        roi=roi,
        max_drawdown=max_drawdown,
        paused_hit_count=paused_hit_count,
        pause_trigger_count=pause_trigger_count,
    )

    detail_df = pd.DataFrame(details)
    os.makedirs(SAVE_DIR, exist_ok=True)
    detail_path = os.path.join(SAVE_DIR, f"top15_pause_details_{mode}.csv")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print(f"完成模式 {mode}，详情保存至 {detail_path}")

    return asdict(result)


def main() -> None:
    strategies = [
        {"name": "baseline", "trigger_hits": 0, "pause_length": 0, "description": "不暂停"},
        {"name": "pause1_after_single", "trigger_hits": 1, "pause_length": 1, "description": "命中1次停1期"},
        {"name": "pause2_after_single", "trigger_hits": 1, "pause_length": 2, "description": "命中1次停2期"},
        {"name": "pause1_after_double", "trigger_hits": 2, "pause_length": 1, "description": "连续中2次停1期"},
        {"name": "pause2_after_double", "trigger_hits": 2, "pause_length": 2, "description": "连续中2次停2期"},
        {"name": "pause1_after_triple", "trigger_hits": 3, "pause_length": 1, "description": "连续中3次停1期"},
    ]

    results = [run_backtest(strategy) for strategy in strategies]

    summary_df = pd.DataFrame(results)

    # 打印核心指标
    display_cols = [
        "mode",
        "description",
        "trigger_hits",
        "pause_length",
        "bet_periods",
        "pause_periods",
        "hits",
        "hit_rate",
        "total_bet",
        "total_win",
        "net_profit",
        "roi",
        "max_drawdown",
        "pause_trigger_count",
        "paused_hit_count",
    ]
    display_df = summary_df[display_cols].copy()
    display_df["hit_rate"] = display_df["hit_rate"].map(lambda x: f"{x:.2f}%")
    display_df["roi"] = display_df["roi"].map(lambda x: f"{x:+.2f}%")

    print("\nTOP15暂停规则回测结果 (最近300期)")
    print(display_df.to_string(index=False))

    ranking_df = summary_df.sort_values(by=["roi", "net_profit"], ascending=[False, False]).head(2)
    print("\n最优两种方案 (按ROI优先, 净利润次之):")
    best_display = ranking_df[
        ["mode", "description", "roi", "net_profit", "max_drawdown", "bet_periods", "pause_periods"]
    ].copy()
    best_display["roi"] = best_display["roi"].map(lambda x: f"{x:+.2f}%")
    print(best_display.to_string(index=False))

    summary_path = os.path.join(SAVE_DIR, "top15_pause_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n汇总已保存到 {summary_path}")


if __name__ == "__main__":
    main()
