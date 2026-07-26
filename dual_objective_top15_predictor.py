"""
双目标TOP15推理模型。

目标：
1. 固定输出15个数字；
2. 提高无前视滚动命中率；
3. 降低最长连续不中期数。

参数只使用最近400期中的前300期进行分段交叉验证选择，最后100期作为
独立验证集。连续5期不中时切换到100期冷号救援，命中后恢复常态模式。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from advanced_hybrid_top15_predictor import (
    AdaptiveHybridTop15Predictor,
    UltraTop15Predictor,
)
from precise_top15_predictor import PreciseTop15Predictor
from top15_predictor import Top15Predictor
from top15_statistical_predictor import Top15StatisticalPredictor


FEATURE_NAMES = [
    "基础TOP15",
    "精准模型全排名",
    "统计分布TOP15",
    "自适应混合TOP15",
    "Ultra TOP15",
    "近10期热度",
    "近10期冷度",
    "近20期热度",
    "近20期冷度",
    "近30期热度",
    "近30期冷度",
    "近50期热度",
    "近50期冷度",
    "近100期热度",
    "近100期冷度",
    "遗漏线性分",
    "遗漏10期峰值",
    "遗漏20期峰值",
]

# 在开发段3个连续100期折叠上，以“平均命中率-波动-最长连不中惩罚”选出。
NORMAL_WEIGHTS = np.array([
    0.03980889502488765,
    0.0008685525148789256,
    0.026816305506334452,
    0.08242245840891725,
    0.016939180156375588,
    0.08347394233153911,
    0.11698887816016061,
    0.068944394772974,
    0.0003683234003857011,
    0.005615195296687591,
    0.27868979655261483,
    0.009993873651955504,
    0.006951807518743998,
    0.08725949168448938,
    0.11921073973407509,
    0.053078157986743195,
    0.0004148226824916286,
    0.00215518461574554,
])


class DualObjectiveTop15Predictor:
    """常态集成评分 + 连续5期不中后的冷号救援。"""

    def __init__(self, rescue_threshold: int = 5) -> None:
        self.rescue_threshold = rescue_threshold
        self.reset()

    def reset(self) -> None:
        self.base = Top15Predictor()
        self.precise = PreciseTop15Predictor()
        self.statistical = Top15StatisticalPredictor()
        self.adaptive = AdaptiveHybridTop15Predictor()
        self.ultra = UltraTop15Predictor()
        self.consecutive_misses = 0

    @staticmethod
    def _membership(prediction) -> np.ndarray:
        selected = {int(number) for number in prediction}
        return np.array(
            [1.0 if number in selected else 0.0 for number in range(1, 50)]
        )

    @staticmethod
    def _frequency_features(history: list[int], window: int) -> tuple[np.ndarray, np.ndarray]:
        counts = Counter(history[-window:])
        hot = np.array([counts[number] for number in range(1, 50)], dtype=float)
        maximum = max(float(hot.max()), 1.0)
        hot /= maximum
        return hot, 1.0 - hot

    @staticmethod
    def _gap_features(history: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        recent = history[-100:]
        last_seen = {}
        for position, number in enumerate(recent):
            last_seen[int(number)] = position
        gaps = np.array(
            [len(recent) - 1 - last_seen.get(number, -1) for number in range(1, 50)],
            dtype=float,
        )
        return (
            np.clip(gaps, 0, 30) / 30.0,
            np.exp(-((gaps - 10.0) / 7.0) ** 2),
            np.exp(-((gaps - 20.0) / 10.0) ** 2),
        )

    def build_features(self, numbers) -> tuple[np.ndarray, list[int]]:
        history = [int(number) for number in numbers]
        if len(history) < 50:
            raise ValueError("双目标TOP15至少需要50期历史数据")

        base_top15 = self.base.predict(history)
        precise_ranking = self.precise.predict(history, k=49)
        precise_score = np.zeros(49, dtype=float)
        for rank, number in enumerate(precise_ranking):
            precise_score[int(number) - 1] = 1.0 - rank / 49.0

        features = [
            self._membership(base_top15),
            precise_score,
            self._membership(self.statistical.predict(history)),
            self._membership(self.adaptive.predict(history)),
            self._membership(self.ultra.predict(history)),
        ]
        for window in (10, 20, 30, 50, 100):
            features.extend(self._frequency_features(history, window))
        features.extend(self._gap_features(history))
        return np.array(features), precise_ranking

    @staticmethod
    def _select_top15(scores: np.ndarray) -> list[int]:
        # 与开发回测保持同一选择方式；先固定集合，再按分数和号码排序展示。
        selected_indexes = np.argpartition(scores, -15)[-15:]
        selected_indexes = sorted(
            selected_indexes, key=lambda index: (-scores[index], index)
        )
        return [int(index) + 1 for index in selected_indexes]

    def predict_with_details(self, numbers) -> dict:
        features, precise_ranking = self.build_features(numbers)
        normal_scores = np.tensordot(NORMAL_WEIGHTS, features, axes=(0, 0))

        if self.consecutive_misses >= self.rescue_threshold:
            scores = features[14]  # 近100期冷度
            mode = "冷号救援"
            reason = (
                f"已连续{self.consecutive_misses}期不中，达到"
                f"{self.rescue_threshold}期阈值，切换到近100期冷号覆盖。"
            )
        else:
            scores = normal_scores
            mode = "常态集成"
            reason = (
                "使用基础、统计、自适应模型，以及10/20/30/50/100期"
                "冷热度和遗漏特征的固定交叉验证权重。"
            )

        prediction = self._select_top15(scores)
        return {
            "top15": prediction,
            "mode": mode,
            "reason": reason,
            "consecutive_misses": self.consecutive_misses,
            "scores": {number: float(scores[number - 1]) for number in prediction},
            "precise_ranking": [int(number) for number in precise_ranking],
        }

    def predict(self, numbers) -> list[int]:
        return self.predict_with_details(numbers)["top15"]

    def update(self, prediction, actual_number: int) -> bool:
        actual = int(actual_number)
        selected = [int(number) for number in prediction]
        hit = actual in selected
        self.consecutive_misses = 0 if hit else self.consecutive_misses + 1
        self.precise.update_performance(selected, actual)
        return hit

    def walk_forward(self, df: pd.DataFrame, test_periods: int = 400) -> dict:
        numbers = df["number"].astype(int).tolist()
        periods = min(test_periods, len(numbers) - 50)
        if periods < 100:
            raise ValueError("至少需要150期数据进行双目标回测")

        start = len(numbers) - periods
        self.reset()
        records = []
        for index in range(start, len(numbers)):
            details = self.predict_with_details(numbers[:index])
            hit = self.update(details["top15"], numbers[index])
            records.append({
                "date": str(df.iloc[index]["date"]),
                "actual": numbers[index],
                "prediction": details["top15"],
                "hit": hit,
                "mode": details["mode"],
                "miss_streak_after": self.consecutive_misses,
            })

        return {
            "periods": periods,
            "records": records,
            "summary": summarize_records(records),
            "next_prediction": self.predict_with_details(numbers),
        }


def summarize_records(records: list[dict]) -> dict:
    hits = sum(bool(record["hit"]) for record in records)
    current = maximum = 0
    for record in records:
        current = 0 if record["hit"] else current + 1
        maximum = max(maximum, current)
    return {
        "periods": len(records),
        "hits": hits,
        "hit_rate": hits / len(records) * 100.0 if records else 0.0,
        "max_miss_streak": maximum,
        "ending_miss_streak": current,
    }


def _baseline_records(df: pd.DataFrame, predictor, precise: bool = False) -> list[dict]:
    numbers = df["number"].astype(int).tolist()
    start = len(numbers) - min(400, len(numbers) - 50)
    records = []
    for index in range(start, len(numbers)):
        prediction = (
            predictor.predict(numbers[:index], k=15)
            if precise else predictor.predict(numbers[:index])
        )
        hit = numbers[index] in prediction
        records.append({"hit": hit})
        if precise:
            predictor.update_performance(prediction, numbers[index])
    return records


def build_validation_report(
    data_path: str | Path = "data/lucky_numbers.csv",
    test_periods: int = 400,
) -> dict:
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    model = DualObjectiveTop15Predictor()
    result = model.walk_forward(df, test_periods=test_periods)
    records = result["records"]
    holdout_size = min(100, len(records))
    development = records[:-holdout_size]
    holdout = records[-holdout_size:]

    base_records = _baseline_records(df, Top15Predictor())
    precise_records = _baseline_records(df, PreciseTop15Predictor(), precise=True)

    def split_baseline(records: list[dict]) -> dict:
        return {
            "development": summarize_records(records[:-holdout_size]),
            "holdout": summarize_records(records[-holdout_size:]),
            "overall": summarize_records(records),
        }

    latest_date = pd.to_datetime(df.iloc[-1]["date"])
    return {
        "model": "双目标TOP15 v1.0",
        "data_through": latest_date.strftime("%Y-%m-%d"),
        "prediction_date": (latest_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        "development": summarize_records(development),
        "holdout": summarize_records(holdout),
        "overall": result["summary"],
        "records": records,
        "baselines": {
            "基础TOP15": split_baseline(base_records),
            "精准TOP15": split_baseline(precise_records),
        },
        "next_prediction": result["next_prediction"],
        "method": {
            "normal": "18项固定特征加权",
            "rescue": "连续5期不中后切换近100期冷号TOP15，命中后恢复",
            "selection": "开发段3×100期时间折叠选择参数，最后100期独立验证",
        },
        "disclaimer": "历史滚动回测不代表未来必然命中。",
    }


def format_walk_forward_details(records: list[dict]) -> str:
    """Format every walk-forward prediction for display in the running log."""
    summary = summarize_records(records)
    lines = [
        "",
        "=" * 118,
        f"双目标TOP15｜最近{summary['periods']}期逐期预测详情",
        (
            f"命中 {summary['hits']}/{summary['periods']} = "
            f"{summary['hit_rate']:.2f}%｜"
            f"最长连续不中 {summary['max_miss_streak']}期｜"
            f"期末连续不中 {summary['ending_miss_streak']}期"
        ),
        "=" * 118,
        f"{'序号':>4} {'日期':>12} {'实际':>4} {'结果':>5} {'预测模式':<10} {'连未中':>6}  预测号码",
        "-" * 118,
    ]
    for sequence, record in enumerate(records, start=1):
        status = "命中" if record["hit"] else "未中"
        prediction = " ".join(
            f"{int(number):02d}" for number in record["prediction"]
        )
        lines.append(
            f"{sequence:>4} {str(record['date']):>12} "
            f"{int(record['actual']):>4} {status:>5} "
            f"{str(record['mode']):<10} "
            f"{int(record['miss_streak_after']):>6}  {prediction}"
        )
    lines.append("=" * 118)
    return "\n".join(lines)


def format_report(report: dict) -> str:
    next_result = report["next_prediction"]
    lines = [
        "=" * 76,
        f"{report['model']}｜下一期 {report['prediction_date']}",
        "=" * 76,
        f"数据截止：{report['data_through']}",
        f"运行模式：{next_result['mode']}",
        f"当前连续不中：{next_result['consecutive_misses']}期",
        f"模式依据：{next_result['reason']}",
        "",
        "下一期TOP15：",
        " ".join(f"{number:02d}" for number in next_result["top15"]),
        "",
        "【开发段：前300期】",
        _format_summary(report["development"]),
        "【独立验证：最后100期】",
        _format_summary(report["holdout"]),
        "【完整400期】",
        _format_summary(report["overall"]),
        "",
        "【现有模型对比】",
    ]
    for name, summaries in report["baselines"].items():
        lines.append(
            f"{name}｜独立100期：{_format_summary(summaries['holdout'])}"
        )
        lines.append(
            f"{name}｜完整400期：{_format_summary(summaries['overall'])}"
        )
    lines.extend([
        "",
        f"说明：{report['disclaimer']}",
    ])
    return "\n".join(lines)


def _format_summary(summary: dict) -> str:
    return (
        f"{summary['hits']}/{summary['periods']}，命中率{summary['hit_rate']:.2f}%，"
        f"最长连续不中{summary['max_miss_streak']}期"
    )


def save_report(report: dict, output_dir: str | Path = "预测结果") -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    date = report["prediction_date"]
    text_path = output_dir / f"双目标TOP15预测_{date}.txt"
    json_path = output_dir / f"双目标TOP15预测_{date}.json"
    text_path.write_text(format_report(report), encoding="utf-8-sig")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return text_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="双目标TOP15预测与无前视验证")
    parser.add_argument("--data", default="data/lucky_numbers.csv")
    parser.add_argument("--periods", type=int, default=400)
    parser.add_argument("--output-dir", default="预测结果")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    report = build_validation_report(args.data, args.periods)
    print(format_report(report))
    if not args.no_save:
        for path in save_report(report, args.output_dir):
            print(f"已保存：{path}")


if __name__ == "__main__":
    main()
