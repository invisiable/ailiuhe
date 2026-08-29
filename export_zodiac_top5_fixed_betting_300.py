"""Export fixed-stake Zodiac TOP5 profit details for a recent period."""

import argparse
from pathlib import Path

import pandas as pd

from zodiac_top5_audit import run_zodiac_top5_audit


DEFAULT_TEST_PERIODS = 300
DEFAULT_STAKE_PER_NUMBER = 123
NUMBERS_PER_ZODIAC = 4
TOP_N = 5
PAYOUT_MULTIPLIER = 47


def main() -> None:
    parser = argparse.ArgumentParser(description="导出生肖TOP5固定投注收益")
    parser.add_argument("--periods", type=int, default=DEFAULT_TEST_PERIODS)
    parser.add_argument(
        "--stake-per-number", type=int, default=DEFAULT_STAKE_PER_NUMBER
    )
    args = parser.parse_args()
    test_periods = args.periods
    stake_per_number = args.stake_per_number
    if test_periods <= 0:
        raise ValueError("回测期数必须大于0")
    if stake_per_number <= 0:
        raise ValueError("每个号码投注额必须大于0")

    source_path = Path("data/lucky_numbers.csv")
    output_dir = Path("预测结果")
    output_dir.mkdir(exist_ok=True)

    source = pd.read_csv(source_path, encoding="utf-8-sig")
    animals = source["animal"].astype(str).str.strip().tolist()
    result = run_zodiac_top5_audit(animals, test_periods=test_periods)

    period_stake = stake_per_number * NUMBERS_PER_ZODIAC * TOP_N
    hit_profit = stake_per_number * PAYOUT_MULTIPLIER - period_stake
    cumulative_profit = 0
    peak_profit = 0
    maximum_drawdown = 0
    detail_rows = []

    for sequence, row in enumerate(result["rows"], start=1):
        period_profit = hit_profit if row["hit"] else -period_stake
        cumulative_profit += period_profit
        peak_profit = max(peak_profit, cumulative_profit)
        drawdown = peak_profit - cumulative_profit
        maximum_drawdown = max(maximum_drawdown, drawdown)
        detail_rows.append(
            {
                "序号": sequence,
                "日期": str(source.iloc[row["index"]]["date"]),
                "实际生肖": row["actual"],
                "预测TOP5": "、".join(row["prediction"]),
                "是否命中": "命中" if row["hit"] else "未中",
                "本期投入": period_stake,
                "本期净收益": period_profit,
                "累计净收益": cumulative_profit,
                "当前连续不中": row["miss_streak"],
                "当前回撤": drawdown,
            }
        )

    total_stake = period_stake * test_periods
    total_profit = cumulative_profit
    roi = total_profit / total_stake * 100
    file_stem = f"生肖TOP5固定投注{stake_per_number}元_最近{test_periods}期收益"
    detail_path = output_dir / f"{file_stem}明细.csv"
    summary_path = output_dir / f"{file_stem}汇总.txt"
    pd.DataFrame(detail_rows).to_csv(
        detail_path, index=False, encoding="utf-8-sig"
    )

    summary_lines = [
        f"生肖TOP5固定投注方案｜最近{test_periods}期收益汇总",
        f"数据区间：{detail_rows[0]['日期']} 至 {detail_rows[-1]['日期']}",
        (
            f"投注规则：每个号码{stake_per_number}元，TOP5共20个号码，"
            f"每期投入{period_stake:,}元"
        ),
        f"单期命中净收益：+{hit_profit:,}元",
        f"单期未中净收益：-{period_stake:,}元",
        f"命中：{result['hits']}/{test_periods} = "
        f"{result['hit_rate'] * 100:.2f}%",
        f"最长连续不中：{result['max_miss_streak']}期",
        f"总投入：{total_stake:,}元",
        f"总净收益：{total_profit:+,}元",
        f"ROI：{roi:+.2f}%",
        f"最大回撤：{maximum_drawdown:,}元",
        f"期均净收益：{total_profit / test_periods:+.2f}元",
        f"期末累计收益：{cumulative_profit:+,}元",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8-sig")

    print("\n".join(summary_lines))
    print(f"明细文件：{detail_path.resolve()}")
    print(f"汇总文件：{summary_path.resolve()}")


if __name__ == "__main__":
    main()
