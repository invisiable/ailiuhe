"""双目标 TOP20 与动态 12/25 TOP20 的固定数量融合模型。

融合只使用每期预测前可获得的数据。参数在最近 400 期的前 300 期
三段验证中选择，最后 100 期仅用于独立验证。
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from dual_objective_top15_predictor import DualObjectiveTop15Predictor
from precise_top15_predictor import PreciseTop15Predictor


TOP_N = 20
TEST_PERIODS = 400
HOLDOUT_PERIODS = 100
SHORT_WINDOW = 12
LONG_WINDOW = 25
DYNAMIC_SWITCH_MISSES = 3


def _select_top_k(scores: np.ndarray, top_n: int = TOP_N) -> list[int]:
    indexes = np.argpartition(scores, -top_n)[-top_n:]
    indexes = sorted(indexes, key=lambda index: (-scores[index], index))
    return [int(index) + 1 for index in indexes]


class DualObjectiveTop20Predictor(DualObjectiveTop15Predictor):
    """复用双目标评分和救援状态，仅把固定候选数扩展为 20。"""

    @staticmethod
    def _select_top15(scores: np.ndarray) -> list[int]:
        return _select_top_k(scores, TOP_N)


class DynamicTop20Predictor:
    """连续未中少于 3 期用 12 期窗口，否则使用 25 期窗口。"""

    def __init__(self) -> None:
        self.predictor = PreciseTop15Predictor()
        self.consecutive_misses = 0

    @property
    def window(self) -> int:
        return (
            LONG_WINDOW
            if self.consecutive_misses >= DYNAMIC_SWITCH_MISSES
            else SHORT_WINDOW
        )

    def predict(self, history: list[int]) -> tuple[list[int], int]:
        window = self.window
        prediction = self.predictor.predict(history[-window:], k=TOP_N)
        return [int(number) for number in prediction], window

    def update(self, prediction: list[int], actual: int) -> bool:
        hit = int(actual) in prediction
        self.predictor.update_performance(prediction, int(actual))
        self.consecutive_misses = 0 if hit else self.consecutive_misses + 1
        return hit


def fuse_rankings(
    dual_prediction: list[int],
    dynamic_prediction: list[int],
    dual_weight: float,
    consensus_bonus: float = 0.0,
) -> tuple[list[int], dict[int, float]]:
    """Fuse two TOP20 rankings and keep exactly 20 unique numbers."""
    dual_rank = {int(number): rank for rank, number in enumerate(dual_prediction)}
    dynamic_rank = {
        int(number): rank for rank, number in enumerate(dynamic_prediction)
    }
    scores = {}
    for number in set(dual_rank) | set(dynamic_rank):
        score = 0.0
        if number in dual_rank:
            score += dual_weight * (TOP_N - dual_rank[number]) / TOP_N
        if number in dynamic_rank:
            score += (1.0 - dual_weight) * (
                TOP_N - dynamic_rank[number]
            ) / TOP_N
        if number in dual_rank and number in dynamic_rank:
            score += consensus_bonus
        scores[number] = score

    prediction = [
        number
        for number, _ in sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )[:TOP_N]
    ]
    return prediction, scores


def summarize_flags(flags: list[bool]) -> dict:
    hits = sum(bool(flag) for flag in flags)
    current = maximum = 0
    for flag in flags:
        current = 0 if flag else current + 1
        maximum = max(maximum, current)
    return {
        "periods": len(flags),
        "hits": hits,
        "hit_rate": hits / len(flags) * 100.0 if flags else 0.0,
        "max_miss_streak": maximum,
        "ending_miss_streak": current,
    }


def _candidate_metrics(rows: list[dict], weight: float, bonus: float) -> dict:
    flags = []
    for row in rows:
        prediction, _ = fuse_rankings(
            row["dual_prediction"],
            row["dynamic_prediction"],
            weight,
            bonus,
        )
        flags.append(row["actual"] in prediction)

    fold_size = 100
    folds = [
        summarize_flags(flags[start:start + fold_size])
        for start in range(0, len(flags), fold_size)
    ]
    rates = np.array([fold["hit_rate"] for fold in folds], dtype=float)
    maximum = max(fold["max_miss_streak"] for fold in folds)
    selection_score = float(rates.mean() - 0.4 * rates.std() - 0.5 * maximum)
    return {
        "dual_weight": weight,
        "dynamic_weight": 1.0 - weight,
        "consensus_bonus": bonus,
        "selection_score": selection_score,
        "folds": folds,
        "overall": summarize_flags(flags),
    }


def select_fusion_parameters(development_rows: list[dict]) -> dict:
    """Choose parameters using development rows only."""
    candidates = []
    for weight in (0.3, 0.4, 0.5, 0.6, 0.7):
        for bonus in (0.0, 0.1, 0.25):
            candidates.append(_candidate_metrics(
                development_rows, weight, bonus
            ))
    return max(
        candidates,
        key=lambda item: (
            item["selection_score"],
            item["overall"]["hit_rate"],
            -item["overall"]["max_miss_streak"],
        ),
    )


def build_fused_top20_report(
    data_path: str | Path = "data/lucky_numbers.csv",
    test_periods: int = TEST_PERIODS,
) -> dict:
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    numbers = df["number"].astype(int).tolist()
    periods = min(int(test_periods), len(numbers) - 50)
    if periods < 200:
        raise ValueError("融合TOP20至少需要250期历史数据")

    start = len(numbers) - periods
    dual_model = DualObjectiveTop20Predictor()
    dynamic_model = DynamicTop20Predictor()
    base_rows = []

    for index in range(start, len(numbers)):
        history = numbers[:index]
        dual_details = dual_model.predict_with_details(history)
        dual_prediction = dual_details["top15"]
        dynamic_prediction, dynamic_window = dynamic_model.predict(history)
        actual = numbers[index]

        dual_hit = dual_model.update(dual_prediction, actual)
        dynamic_hit = dynamic_model.update(dynamic_prediction, actual)
        base_rows.append({
            "index": index,
            "date": str(df.iloc[index]["date"]),
            "actual": actual,
            "dual_prediction": dual_prediction,
            "dynamic_prediction": dynamic_prediction,
            "dual_hit": dual_hit,
            "dynamic_hit": dynamic_hit,
            "dual_mode": dual_details["mode"],
            "dynamic_window": dynamic_window,
        })

    holdout_size = min(HOLDOUT_PERIODS, periods // 4)
    development_rows = base_rows[:-holdout_size]
    selected = select_fusion_parameters(development_rows)
    dual_weight = selected["dual_weight"]
    bonus = selected["consensus_bonus"]

    records = []
    for row in base_rows:
        prediction, scores = fuse_rankings(
            row["dual_prediction"],
            row["dynamic_prediction"],
            dual_weight,
            bonus,
        )
        hit = row["actual"] in prediction
        records.append({
            **row,
            "prediction": prediction,
            "hit": hit,
            "overlap": len(
                set(row["dual_prediction"]) & set(row["dynamic_prediction"])
            ),
            "scores": {
                str(number): round(float(scores[number]), 6)
                for number in prediction
            },
        })

    dual_next_details = dual_model.predict_with_details(numbers)
    dynamic_next, next_dynamic_window = dynamic_model.predict(numbers)
    next_prediction, next_scores = fuse_rankings(
        dual_next_details["top15"],
        dynamic_next,
        dual_weight,
        bonus,
    )
    latest_date = pd.to_datetime(df.iloc[-1]["date"])

    return {
        "model": "双模型融合TOP20 v1.0",
        "data_through": latest_date.strftime("%Y-%m-%d"),
        "prediction_date": (latest_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        "parameters": selected,
        "development": summarize_flags(
            [record["hit"] for record in records[:-holdout_size]]
        ),
        "holdout": summarize_flags(
            [record["hit"] for record in records[-holdout_size:]]
        ),
        "overall": summarize_flags([record["hit"] for record in records]),
        "baselines": {
            "双目标TOP20": {
                "development": summarize_flags(
                    [row["dual_hit"] for row in base_rows[:-holdout_size]]
                ),
                "holdout": summarize_flags(
                    [row["dual_hit"] for row in base_rows[-holdout_size:]]
                ),
                "overall": summarize_flags(
                    [row["dual_hit"] for row in base_rows]
                ),
            },
            "动态12/25 TOP20": {
                "development": summarize_flags(
                    [row["dynamic_hit"] for row in base_rows[:-holdout_size]]
                ),
                "holdout": summarize_flags(
                    [row["dynamic_hit"] for row in base_rows[-holdout_size:]]
                ),
                "overall": summarize_flags(
                    [row["dynamic_hit"] for row in base_rows]
                ),
            },
        },
        "records": records,
        "next_prediction": {
            "top20": next_prediction,
            "dual_top20": dual_next_details["top15"],
            "dynamic_top20": dynamic_next,
            "overlap": sorted(
                set(dual_next_details["top15"]) & set(dynamic_next)
            ),
            "dual_mode": dual_next_details["mode"],
            "dynamic_window": next_dynamic_window,
            "scores": {
                str(number): round(float(next_scores[number]), 6)
                for number in next_prediction
            },
        },
        "disclaimer": "历史滚动回测不代表未来必然命中。",
    }


def _summary_text(summary: dict) -> str:
    return (
        f"{summary['hits']}/{summary['periods']}，"
        f"命中率{summary['hit_rate']:.2f}%，"
        f"最长连续不中{summary['max_miss_streak']}期"
    )


def format_fused_top20_report(report: dict) -> str:
    params = report["parameters"]
    next_result = report["next_prediction"]
    lines = [
        "=" * 82,
        f"{report['model']}｜下一期 {report['prediction_date']}",
        "=" * 82,
        f"数据截止：{report['data_through']}",
        (
            f"融合权重：双目标{params['dual_weight']:.0%}＋"
            f"动态12/25 {params['dynamic_weight']:.0%}"
        ),
        "参数规则：仅用前300期三段验证选择，最后100期不参与选参。",
        "",
        "【下一期融合TOP20】",
        " ".join(f"{number:02d}" for number in next_result["top20"]),
        f"双方原始TOP20重合号码：{next_result['overlap']}",
        f"双目标模式：{next_result['dual_mode']}",
        f"动态模型窗口：最近{next_result['dynamic_window']}期",
        "",
        f"开发段（前300期）：{_summary_text(report['development'])}",
        f"独立验证（后100期）：{_summary_text(report['holdout'])}",
        f"完整400期：{_summary_text(report['overall'])}",
        "",
        "【单模型对比】",
    ]
    for name, summaries in report["baselines"].items():
        lines.append(
            f"{name}｜400期：{_summary_text(summaries['overall'])}；"
            f"后100期：{_summary_text(summaries['holdout'])}"
        )
    lines.extend(["", report["disclaimer"]])
    return "\n".join(lines)


def save_fused_top20_outputs(
    report: dict,
    output_dir: str | Path = "预测结果",
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    date = report["prediction_date"]
    text_path = output_dir / f"双模型融合TOP20预测_{date}.txt"
    json_path = output_dir / f"双模型融合TOP20预测_{date}.json"
    csv_path = output_dir / f"双模型融合TOP20回测_{date}.csv"

    text_path.write_text(
        format_fused_top20_report(report), encoding="utf-8-sig"
    )
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame([
        {
            "date": record["date"],
            "actual": record["actual"],
            "hit": record["hit"],
            "overlap": record["overlap"],
            "dual_mode": record["dual_mode"],
            "dynamic_window": record["dynamic_window"],
            "prediction": " ".join(
                f"{number:02d}" for number in record["prediction"]
            ),
            "dual_prediction": " ".join(
                f"{number:02d}" for number in record["dual_prediction"]
            ),
            "dynamic_prediction": " ".join(
                f"{number:02d}" for number in record["dynamic_prediction"]
            ),
        }
        for record in report["records"]
    ]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return text_path, json_path, csv_path


if __name__ == "__main__":
    current_report = build_fused_top20_report()
    print(format_fused_top20_report(current_report))
    for output_path in save_fused_top20_outputs(current_report):
        print(f"已保存：{output_path}")
