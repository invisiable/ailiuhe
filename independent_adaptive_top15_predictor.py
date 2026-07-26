"""完全独立的分状态在线专家 TOP15。

本模型不导入、不调用双目标 TOP15。正常状态和连续未中恢复状态
分别维护专家权重，并混入 20% 全局权重抑制短样本过拟合。
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from precise_top15_predictor import PreciseTop15Predictor
from top20_dual_horizon_predictor import Top20DualHorizonPredictor


TOP_N = 15
TEST_PERIODS = 400
HOLDOUT_PERIODS = 100
EXPERT_NAMES = [
    "动态12/25",
    "精准25期",
    "双周期6/25",
    "固定15期全排名",
    "100期冷号",
    "最大遗漏",
]


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


def _cold_numbers(history: list[int]) -> list[int]:
    frequency = Counter(history[-100:])
    return sorted(
        range(1, 50), key=lambda number: (frequency[number], number)
    )[:TOP_N]


def _largest_gap_numbers(history: list[int]) -> list[int]:
    last_seen = {}
    for index, number in enumerate(history):
        last_seen[int(number)] = index
    return sorted(
        range(1, 50),
        key=lambda number: -(
            len(history) - 1 - last_seen.get(number, -1)
        ),
    )[:TOP_N]


def _rank_fusion(
    expert_predictions: list[list[int]],
    expert_weights: np.ndarray,
) -> tuple[list[int], dict[int, float]]:
    scores = np.zeros(49, dtype=float)
    for expert_index, prediction in enumerate(expert_predictions):
        for rank, number in enumerate(prediction):
            scores[int(number) - 1] += (
                expert_weights[expert_index] * (TOP_N - rank) / TOP_N
            )
    indexes = np.argpartition(scores, -TOP_N)[-TOP_N:]
    indexes = sorted(indexes, key=lambda index: (-scores[index], index))
    prediction = [int(index) + 1 for index in indexes]
    return prediction, {
        number: float(scores[number - 1]) for number in prediction
    }


def _updated_weights(
    weights: np.ndarray,
    expert_hits: np.ndarray,
    learning_rate: float,
) -> np.ndarray:
    updated = weights * np.exp(learning_rate * (expert_hits - 0.3))
    updated /= updated.sum()
    return updated


def _expert_predictions(
    history: list[int],
    dynamic_model: PreciseTop15Predictor,
    dynamic_misses: int,
    precise_25_model: PreciseTop15Predictor,
    dual_horizon_model: Top20DualHorizonPredictor,
) -> tuple[list[list[int]], int]:
    dynamic_window = 25 if dynamic_misses >= 3 else 12
    predictions = [
        dynamic_model.predict(history[-dynamic_window:], k=TOP_N),
        precise_25_model.predict(history[-25:], k=TOP_N),
        dual_horizon_model.predict(history, top_n=TOP_N),
        PreciseTop15Predictor().predict(history[-15:], k=49)[:TOP_N],
        _cold_numbers(history),
        _largest_gap_numbers(history),
    ]
    return [
        [int(number) for number in prediction]
        for prediction in predictions
    ], dynamic_window


def _build_expert_rows(df: pd.DataFrame, periods: int) -> tuple[list[dict], dict]:
    numbers = df["number"].astype(int).tolist()
    start = len(numbers) - periods
    dynamic_model = PreciseTop15Predictor()
    precise_25_model = PreciseTop15Predictor()
    dual_horizon_model = Top20DualHorizonPredictor()
    dynamic_misses = 0
    rows = []

    for index in range(start, len(numbers)):
        history = numbers[:index]
        actual = numbers[index]
        predictions, dynamic_window = _expert_predictions(
            history,
            dynamic_model,
            dynamic_misses,
            precise_25_model,
            dual_horizon_model,
        )
        dynamic_hit = actual in predictions[0]
        dynamic_model.update_performance(predictions[0], actual)
        precise_25_model.update_performance(predictions[1], actual)
        dynamic_misses = 0 if dynamic_hit else dynamic_misses + 1
        rows.append({
            "index": index,
            "date": str(df.iloc[index]["date"]),
            "actual": actual,
            "dynamic_window": dynamic_window,
            "expert_predictions": predictions,
        })

    next_predictions, next_window = _expert_predictions(
        numbers,
        dynamic_model,
        dynamic_misses,
        precise_25_model,
        dual_horizon_model,
    )
    return rows, {
        "expert_predictions": next_predictions,
        "dynamic_window": next_window,
    }


def _simulate(
    rows: list[dict],
    learning_rate: float,
    global_mix: float,
) -> dict:
    expert_count = len(EXPERT_NAMES)
    state_weights = [
        np.ones(expert_count, dtype=float) / expert_count,
        np.ones(expert_count, dtype=float) / expert_count,
    ]
    global_weights = np.ones(expert_count, dtype=float) / expert_count
    miss_streak = 0
    records = []

    for row in rows:
        state_index = 0 if miss_streak == 0 else 1
        effective_weights = (
            (1.0 - global_mix) * state_weights[state_index]
            + global_mix * global_weights
        )
        prediction, scores = _rank_fusion(
            row["expert_predictions"], effective_weights
        )
        hit = row["actual"] in prediction
        before_miss = miss_streak
        miss_streak = 0 if hit else miss_streak + 1
        expert_hits = np.array([
            1.0 if row["actual"] in expert_prediction else 0.0
            for expert_prediction in row["expert_predictions"]
        ])
        state_weights[state_index] = _updated_weights(
            state_weights[state_index], expert_hits, learning_rate
        )
        global_weights = _updated_weights(
            global_weights, expert_hits, learning_rate
        )
        records.append({
            **row,
            "prediction": prediction,
            "mode": "正常在线" if state_index == 0 else "恢复在线",
            "hit": hit,
            "miss_streak_before": before_miss,
            "miss_streak_after": miss_streak,
            "scores": {
                str(number): round(float(score), 6)
                for number, score in scores.items()
            },
            "expert_weights": {
                name: float(weight)
                for name, weight in zip(EXPERT_NAMES, effective_weights)
            },
        })
    return {
        "records": records,
        "state_weights": state_weights,
        "global_weights": global_weights,
        "ending_miss_streak": miss_streak,
    }


def select_parameters(development_rows: list[dict]) -> dict:
    """Small, regularized parameter search using development rows only."""
    candidates = []
    for learning_rate in (0.2, 0.5, 1.0):
        for global_mix in (0.2, 0.4):
            simulation = _simulate(
                development_rows, learning_rate, global_mix
            )
            flags = [record["hit"] for record in simulation["records"]]
            folds = [
                summarize_flags(flags[start:start + 100])
                for start in range(0, len(flags), 100)
            ]
            rates = np.array(
                [fold["hit_rate"] for fold in folds], dtype=float
            )
            maximum = max(
                fold["max_miss_streak"] for fold in folds
            )
            selection_score = float(
                rates.mean() - 0.4 * rates.std() - 2.0 * maximum
            )
            candidates.append({
                "learning_rate": learning_rate,
                "global_mix": global_mix,
                "selection_score": selection_score,
                "folds": folds,
                "development": summarize_flags(flags),
            })
    return max(
        candidates,
        key=lambda item: (
            item["selection_score"],
            item["development"]["hit_rate"],
            -item["development"]["max_miss_streak"],
        ),
    )


def build_independent_adaptive_report(
    data_path: str | Path = "data/lucky_numbers.csv",
    test_periods: int = TEST_PERIODS,
) -> dict:
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    periods = min(int(test_periods), len(df) - 50)
    if periods < 300:
        raise ValueError("独立自适应TOP15至少需要350期历史数据")

    rows, next_experts = _build_expert_rows(df, periods)
    holdout_size = min(HOLDOUT_PERIODS, periods // 4)
    selected = select_parameters(rows[:-holdout_size])
    simulation = _simulate(
        rows,
        selected["learning_rate"],
        selected["global_mix"],
    )
    records = simulation["records"]
    next_state = 0 if simulation["ending_miss_streak"] == 0 else 1
    next_weights = (
        (1.0 - selected["global_mix"])
        * simulation["state_weights"][next_state]
        + selected["global_mix"] * simulation["global_weights"]
    )
    next_prediction, next_scores = _rank_fusion(
        next_experts["expert_predictions"], next_weights
    )
    latest_date = pd.to_datetime(df.iloc[-1]["date"])

    return {
        "model": "独立自适应TOP15 v2.0",
        "data_through": latest_date.strftime("%Y-%m-%d"),
        "prediction_date": (latest_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        "parameters": selected,
        "development": summarize_flags([
            record["hit"] for record in records[:-holdout_size]
        ]),
        "holdout": summarize_flags([
            record["hit"] for record in records[-holdout_size:]
        ]),
        "overall": summarize_flags([
            record["hit"] for record in records
        ]),
        "records": records,
        "next_prediction": {
            "top15": next_prediction,
            "mode": "正常在线" if next_state == 0 else "恢复在线",
            "current_miss_streak": simulation["ending_miss_streak"],
            "dynamic_window": next_experts["dynamic_window"],
            "expert_weights": {
                name: float(weight)
                for name, weight in zip(EXPERT_NAMES, next_weights)
            },
            "scores": {
                str(number): round(float(next_scores[number]), 6)
                for number in next_prediction
            },
        },
        "method": (
            "完全独立的两状态在线专家模型；正常与恢复状态分别学习权重，"
            "并混入全局权重稳定短样本。模型不调用双目标TOP15。"
        ),
        "disclaimer": "历史滚动回测不代表未来必然命中。",
    }


def _summary_text(summary: dict) -> str:
    return (
        f"{summary['hits']}/{summary['periods']}，"
        f"命中率{summary['hit_rate']:.2f}%，"
        f"最长连续不中{summary['max_miss_streak']}期"
    )


def format_independent_walk_forward_details(records: list[dict]) -> str:
    summary = summarize_flags([record["hit"] for record in records])
    lines = [
        "",
        "=" * 118,
        f"独立自适应TOP15｜最近{summary['periods']}期逐期预测详情",
        (
            f"命中 {summary['hits']}/{summary['periods']} = "
            f"{summary['hit_rate']:.2f}%｜"
            f"最长连续不中 {summary['max_miss_streak']}期｜"
            f"期末连续不中 {summary['ending_miss_streak']}期"
        ),
        "=" * 118,
        f"{'序号':>4} {'日期':>12} {'实际':>4} {'结果':>5} "
        f"{'预测模式':<10} {'连未中':>6}  预测号码",
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


def format_independent_adaptive_report(report: dict) -> str:
    params = report["parameters"]
    next_result = report["next_prediction"]
    lines = [
        "=" * 82,
        f"{report['model']}｜下一期 {report['prediction_date']}",
        "=" * 82,
        f"数据截止：{report['data_through']}",
        f"下一期模式：{next_result['mode']}",
        f"当前连续未中：{next_result['current_miss_streak']}期",
        f"模型方法：{report['method']}",
        (
            f"开发段选参：学习率{params['learning_rate']}，"
            f"全局权重约束{params['global_mix']:.0%}"
        ),
        "",
        "【下一期TOP15】",
        " ".join(f"{number:02d}" for number in next_result["top15"]),
        "",
        f"开发段（前300期）：{_summary_text(report['development'])}",
        f"独立验证（后100期）：{_summary_text(report['holdout'])}",
        f"完整400期：{_summary_text(report['overall'])}",
        "",
        "【当前状态专家权重】",
    ]
    for name, weight in sorted(
        next_result["expert_weights"].items(),
        key=lambda item: -item[1],
    ):
        lines.append(f"{name}：{weight:.2%}")
    lines.extend(["", report["disclaimer"]])
    return "\n".join(lines)


def save_independent_adaptive_outputs(
    report: dict,
    output_dir: str | Path = "预测结果",
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    date = report["prediction_date"]
    text_path = output_dir / f"独立自适应TOP15预测_{date}.txt"
    json_path = output_dir / f"独立自适应TOP15预测_{date}.json"
    csv_path = output_dir / f"独立自适应TOP15回测_{date}.csv"
    text_path.write_text(
        format_independent_adaptive_report(report), encoding="utf-8-sig"
    )
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame([
        {
            "date": record["date"],
            "actual": record["actual"],
            "hit": record["hit"],
            "mode": record["mode"],
            "miss_streak_after": record["miss_streak_after"],
            "prediction": " ".join(
                f"{number:02d}" for number in record["prediction"]
            ),
        }
        for record in report["records"]
    ]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return text_path, json_path, csv_path


if __name__ == "__main__":
    current_report = build_independent_adaptive_report()
    print(format_independent_adaptive_report(current_report))
    for output_path in save_independent_adaptive_outputs(current_report):
        print(f"已保存：{output_path}")
