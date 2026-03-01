"""Analyze Precise TOP15 prediction quality by position for the latest 300 periods."""

import os
from typing import List, Optional

import pandas as pd

from precise_top15_predictor import PreciseTop15Predictor

TEST_PERIODS = 300
PAYOUT_PER_HIT = 47  # 奖励：命中一个号码获得47元
BET_PER_NUMBER = 1   # 成本：每个号码投注1元
SAVE_DIR = "analysis"


def analyze_precise_top15_positions(
    test_periods: int = TEST_PERIODS,
    payout_per_hit: int = PAYOUT_PER_HIT,
    bet_per_number: int = BET_PER_NUMBER,
    save_dir: str = SAVE_DIR,
) -> None:
    """Compute per-position hit rates and ROI under truncated betting sizes."""

    df = pd.read_csv("data/lucky_numbers.csv", encoding="utf-8-sig")
    numbers: List[int] = df["number"].tolist()

    if len(numbers) <= test_periods:
        raise ValueError(f"数据期数不足，至少需要 {test_periods + 1} 期，当前只有 {len(numbers)} 期")

    predictor = PreciseTop15Predictor()
    start_idx = len(numbers) - test_periods

    position_hits = [0] * 15
    position_samples = [0] * 15
    topk_hits = [0] * 15

    detail_rows = []

    for idx in range(start_idx, len(numbers)):
        history = numbers[:idx]
        actual = numbers[idx]
        date = df.iloc[idx]["date"] if "date" in df.columns else idx + 1

        predictions = predictor.predict(history)
        predictions = predictions[:15]

        for pos in range(min(len(predictions), 15)):
            position_samples[pos] += 1
            if predictions[pos] == actual:
                position_hits[pos] += 1

        hit_position: Optional[int] = None
        if actual in predictions:
            hit_position = predictions.index(actual) + 1  # 1-based index

        for k in range(1, 16):
            if actual in predictions[:k]:
                topk_hits[k - 1] += 1

        detail_rows.append(
            {
                "date": date,
                "period_index": idx + 1,
                "actual_number": actual,
                "hit": bool(hit_position),
                "hit_position": hit_position,
                "predictions": predictions,
            }
        )

    position_summary = []
    for pos in range(15):
        samples = position_samples[pos] or 1
        hit_rate = position_hits[pos] / samples * 100
        position_summary.append(
            {
                "position": pos + 1,
                "samples": samples,
                "hits": position_hits[pos],
                "hit_rate": hit_rate,
            }
        )

    position_df = pd.DataFrame(position_summary)

    topk_summary = []
    for k in range(1, 16):
        hits = topk_hits[k - 1]
        cost = test_periods * k * bet_per_number
        reward = hits * payout_per_hit
        profit = reward - cost
        roi = profit / cost * 100 if cost > 0 else 0.0
        topk_summary.append(
            {
                "top_k": k,
                "hits": hits,
                "hit_rate": hits / test_periods * 100,
                "cost": cost,
                "reward": reward,
                "profit": profit,
                "roi": roi,
                "avg_profit_per_period": profit / test_periods,
            }
        )

    topk_df = pd.DataFrame(topk_summary)
    detail_df = pd.DataFrame(detail_rows)

    os.makedirs(save_dir, exist_ok=True)
    position_path = os.path.join(save_dir, "precise_top15_position_hit_rates.csv")
    topk_path = os.path.join(save_dir, "precise_top15_topk_roi.csv")
    detail_path = os.path.join(save_dir, "precise_top15_period_details.csv")

    position_df.to_csv(position_path, index=False, encoding="utf-8-sig")
    topk_df.to_csv(topk_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    display_positions = position_df.copy()
    display_positions["hit_rate"] = display_positions["hit_rate"].map(lambda x: f"{x:.2f}%")
    print("\n精准TOP15各位置命中统计 (最近300期)")
    print(display_positions.to_string(index=False))

    display_topk = topk_df.copy()
    display_topk["hit_rate"] = display_topk["hit_rate"].map(lambda x: f"{x:.2f}%")
    display_topk["roi"] = display_topk["roi"].map(lambda x: f"{x:+.2f}%")
    display_topk["avg_profit_per_period"] = display_topk["avg_profit_per_period"].map(
        lambda x: f"{x:+.2f}"
    )
    print("\nTOP-N投注命中与收益 (每期每号1元, 命中奖励47元)")
    print(
        display_topk[
            [
                "top_k",
                "hits",
                "hit_rate",
                "cost",
                "reward",
                "profit",
                "roi",
                "avg_profit_per_period",
            ]
        ].to_string(index=False)
    )

    print("\n数据已保存到:")
    print(f"  - {position_path}")
    print(f"  - {topk_path}")
    print(f"  - {detail_path}")


if __name__ == "__main__":
    analyze_precise_top15_positions()
