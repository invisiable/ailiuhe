"""自动回测并核对生肖TOP4投注预测是否正确

- 使用与 GUI 相同的集成生肖预测器 EnsembleZodiacPredictor
- 按照 GUI 中「生肖TOP4投注策略分析」的逻辑：
  对每一期使用之前所有期的生肖历史进行预测
- 打印指定区间的 TOP4 预测，并自动检查第354/355期是否相同

用法：
    python verify_zodiac_top4_backtest.py

可修改 main() 里的 start_period / end_period 查看其它区间
"""

import pandas as pd
from ensemble_zodiac_predictor import EnsembleZodiacPredictor


def backtest_top4(csv_file: str = "data/lucky_numbers.csv",
                  start_period: int = 350,
                  end_period: int = 360) -> None:
    """回测指定期数区间的 TOP4 预测，并检查第354/355期。"""
    # 读取数据
    df = pd.read_csv(csv_file, encoding="utf-8-sig")
    animals = [str(a).strip() for a in df["animal"].values]

    predictor = EnsembleZodiacPredictor()

    total_periods = len(df)
    print(f"数据总期数: {total_periods}\n")

    start_period = max(1, start_period)
    end_period = min(end_period, total_periods)

    # 打印指定区间的预测结果
    for period in range(start_period, end_period + 1):
        idx = period - 1  # 0-based
        if idx == 0:
            # 第一期开奖没有历史数据，跳过
            continue

        train_animals = animals[:idx]  # 用前 period-1 期做训练
        result = predictor.predict_from_history(train_animals, top_n=5, debug=False)
        top4 = result["top4"]
        actual_animal = animals[idx]
        date = df.iloc[idx]["date"]
        print(f"第{period:3d}期 {date} 实际:{actual_animal}  预测TOP4:{top4}")

    # 特别检查第354/355期
    print("\n特别检查: 第354期 与 第355期 的TOP4是否相同：")
    if total_periods >= 355:
        idx_354 = 354 - 1
        idx_355 = 355 - 1

        res_354 = predictor.predict_from_history(animals[:idx_354], top_n=5, debug=False)["top4"]
        res_355 = predictor.predict_from_history(animals[:idx_355], top_n=5, debug=False)["top4"]

        print("第354期TOP4:", res_354)
        print("第355期TOP4:", res_355)
        print("是否完全相同:", res_354 == res_355)
    else:
        print("数据不足355期，无法检查第354/355期")


def main() -> None:
    backtest_top4()


if __name__ == "__main__":
    main()
