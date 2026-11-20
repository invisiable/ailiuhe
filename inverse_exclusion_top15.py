"""TOP9正向、尾数反向筛选和正向评分组成的固定15号预测。"""

from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np
import pandas as pd

from tail_digit_grid_predictor import TailDigitGridPredictor
from tail_digit_predictor import (
    TAIL_DIGIT_NUMBERS,
    TailDigitPredictor,
    TailDigitRotationPredictor,
)
from tail_digit_three_period_predictor import ThreePeriodTailTop3Predictor
from tail_digit_top3_predictor import TailDigitTop3Predictor
from top15_predictor import Top15Predictor
from top5_top10_balanced19 import get_zodiac_number_map
from zodiac_top9_predictor import ZODIAC_CYCLE_2026, ZodiacTop9Predictor


MODEL_NAME = "15反向排除预测"
TARGET_COUNT = 15
TAIL_START = 60
CALIBRATION = 100
NUMBER_METHODS = (
    "method_frequency_advanced",
    "method_zone_dynamic",
    "method_cyclic_pattern",
    "method_gap_prediction",
)


def _prepare_history(df: pd.DataFrame, test_periods: int) -> tuple[pd.DataFrame, int]:
    if isinstance(test_periods, bool) or not isinstance(test_periods, (int, np.integer)):
        raise ValueError("回测期数必须是正整数")
    if test_periods <= 0:
        raise ValueError("回测期数必须是正整数")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("历史数据必须是DataFrame")
    if not {"date", "number"}.issubset(df.columns):
        raise ValueError("历史数据必须包含date、number列")
    minimum = max(test_periods + 30, TAIL_START + 1)
    if len(df) < minimum:
        raise ValueError(f"回测最近{test_periods}期至少需要{minimum}条历史数据")

    history = df.copy(deep=True).reset_index(drop=True)
    try:
        dates = pd.to_datetime(history["date"], errors="raise").dt.normalize()
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("date列必须是有效日期") from exc
    if dates.isna().any():
        raise ValueError("date列不能包含缺失日期")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("日期必须严格按时间升序且不能重复")

    try:
        numbers = pd.to_numeric(history["number"], errors="raise")
        values = numbers.to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("number列必须是1到49之间的整数") from exc
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError("number列必须是1到49之间的整数")
    if not ((values >= 1) & (values <= 49)).all():
        raise ValueError("number列必须是1到49之间的整数")
    history["number"] = numbers.astype(int)
    history["date"] = dates

    animals = (
        history["animal"].astype("string").str.strip()
        if "animal" in history
        else pd.Series(pd.NA, index=history.index, dtype="string")
    )
    missing = animals.isna() | animals.eq("").fillna(False)
    invalid = sorted(set(animals[~missing]) - set(ZODIAC_CYCLE_2026))
    if invalid:
        raise ValueError(f"animal列包含无效生肖：{'、'.join(invalid)}")
    for index in history.index[missing]:
        mapping = get_zodiac_number_map(dates.iloc[index])
        number = int(history.iloc[index]["number"])
        animals.iloc[index] = next(zodiac for zodiac, nums in mapping.items() if number in nums)
    history["animal"] = animals.astype(str)
    return history, int(missing.sum())


def _tail_record(index: int, prediction, actual: int | None) -> dict:
    tails = [int(tail) for tail in prediction]
    if not tails or len(tails) != len(set(tails)) or any(tail < 0 or tail > 9 for tail in tails):
        raise ValueError("尾数模型必须返回不重复的0到9尾数")
    return {
        "index": index,
        "predicted_tails": tails,
        "hit": actual % 10 in tails if actual is not None else None,
        "covered_numbers": sum(len(TAIL_DIGIT_NUMBERS[tail]) for tail in tails),
    }


def _build_tail_records(history: pd.DataFrame, tail_trend_runner: Callable) -> dict:
    """Each state is updated after its draw; the final row is unsettled."""
    numbers = history["number"].tolist()
    models = {
        "base_TOP4": TailDigitPredictor(),
        "rotation_TOP4": TailDigitRotationPredictor(),
        "legacy_rotation_TOP3": TailDigitTop3Predictor(),
        "fixed_three_period_TOP3": ThreePeriodTailTop3Predictor(),
        "grid_TOP3": TailDigitGridPredictor(),
    }
    records = {name: [] for name in models}
    for index in range(TAIL_START, len(history) + 1):
        actual = numbers[index] if index < len(history) else None
        for name, model in models.items():
            prediction = (
                model.predict_with_details(numbers[:index])[0]
                if name == "fixed_three_period_TOP3"
                else model.predict(numbers[:index])
            )
            row = _tail_record(index, prediction, actual)
            records[name].append(row)
            if actual is not None:
                if name == "fixed_three_period_TOP3":
                    model.record_result(row["predicted_tails"], row["hit"], actual_tail=actual % 10)
                elif hasattr(model, "record_result"):
                    model.record_result(row["predicted_tails"], row["hit"])

    trend = tail_trend_runner(
        history.copy(deep=True),
        test_periods=len(history) - TAIL_START,
        hold_periods=3,
        top_n=3,
    )
    if len(trend["results"]) != len(history) - TAIL_START:
        raise ValueError("尾数走势回测返回的期数与历史数据不一致")
    trend_records = [
        _tail_record(TAIL_START + offset, row["predicted"], numbers[TAIL_START + offset])
        for offset, row in enumerate(trend["results"])
    ]
    trend_records.append(_tail_record(len(history), trend["next_plan"]["predicted"], None))
    records["GUI_tail_trend_TOP3"] = trend_records
    return records


def _exclusions_for(index: int, records: dict) -> tuple[list[int], dict[str, float]]:
    offset = index - TAIL_START
    if offset < CALIBRATION:
        return [], {}
    scores = {tail: 0.0 for tail in range(10)}
    weights = {}
    for name, rows in records.items():
        if rows[offset]["index"] != index:
            raise ValueError("尾数模型预测记录顺序不一致")
        deficits = []
        for window in (20, 100):
            sample = rows[offset - window:offset]
            if len(sample) != window or any(row["hit"] is None or row["index"] >= index for row in sample):
                raise ValueError("尾数模型历史表现记录不足或包含未开奖记录")
            deficits.append(sum(row["covered_numbers"] / 49 - int(row["hit"]) for row in sample) / window)
        weight = max(0.0, deficits[1]) if deficits[0] > 0 else 0.0
        weights[name] = weight
        for tail in rows[offset]["predicted_tails"]:
            scores[tail] += weight
    chosen = sorted((tail for tail in scores if scores[tail] > 0), key=lambda tail: (-scores[tail], tail))[:3]
    return chosen, weights


def _positive_scores(numbers: list[int], model: Top15Predictor) -> dict[int, float]:
    pattern = model.analyze_pattern(numbers)
    scores = {number: 0.0 for number in range(1, 50)}
    for method in NUMBER_METHODS:
        prediction = getattr(model, method)(pattern, 20)
        for rank, number in enumerate(prediction):
            scores[number] += 0.25 * (1 - rank / len(prediction))
    return scores


def _predict_for(index, target_date, numbers, animals, tail_records, zodiac_model, number_model):
    scores = zodiac_model._base_scores(animals[:index])
    top9 = [ZODIAC_CYCLE_2026[item] for item in np.argsort(-scores)[:9]]
    mapping = get_zodiac_number_map(target_date)
    pool = sorted({number for zodiac in top9 for number in mapping[zodiac]})
    excluded, weights = _exclusions_for(index, tail_records)
    filtered = [number for number in pool if number % 10 not in excluded]
    positive = _positive_scores(numbers[:index], number_model)
    prediction = sorted(filtered, key=lambda number: (-positive[number], number))[:TARGET_COUNT]
    if len(prediction) != TARGET_COUNT or len(set(prediction)) != TARGET_COUNT:
        raise ValueError("筛选后的候选池不足15个不重复号码")
    return {
        "date": pd.Timestamp(target_date).strftime("%Y-%m-%d"),
        "top9": top9,
        "excluded_tails": excluded,
        "top9_pool_count": len(pool),
        "filtered_pool_count": len(filtered),
        "prediction": [int(number) for number in prediction],
        "tail_calibration_ready": index - TAIL_START >= CALIBRATION,
        "inverse_tail_model_count": sum(weight > 0 for weight in weights.values()),
        "tail_model_weights": weights,
        "_top9_pool": pool,
        "_filtered_pool": filtered,
    }


def _miss_runs(records: list[dict]) -> list[dict]:
    runs = []
    start = None
    for position, row in enumerate(records):
        if not row["hit"] and start is None:
            start = position
        if row["hit"] and start is not None:
            runs.append({"length": position - start, "from": records[start]["date"], "through": records[position - 1]["date"], "closed": True})
            start = None
    if start is not None:
        runs.append({"length": len(records) - start, "from": records[start]["date"], "through": records[-1]["date"], "closed": False})
    return runs


def _summarize(records: list[dict]) -> dict:
    if not records:
        return {"periods": 0}
    hits = sum(row["hit"] for row in records)
    runs = _miss_runs(records)
    maximum = max((run["length"] for run in runs), default=0)
    return {
        "from": records[0]["date"], "through": records[-1]["date"],
        "periods": len(records), "hits": hits, "misses": len(records) - hits,
        "hit_rate": hits / len(records), "candidate_count": TARGET_COUNT,
        "max_consecutive_misses": maximum,
        "ending_miss_streak": runs[-1]["length"] if runs and not runs[-1]["closed"] else 0,
        "longest_miss_runs": [run for run in runs if run["length"] == maximum],
        "miss_run_distribution": dict(sorted(Counter(run["length"] for run in runs).items())),
        "tail_calibration_ready_periods": sum(row["tail_calibration_ready"] for row in records),
        "active_tail_filter_periods": sum(bool(row["excluded_tails"]) for row in records),
    }


def run_inverse_exclusion_top15_audit(
    df: pd.DataFrame,
    test_periods: int = 400,
    *,
    tail_trend_runner: Callable,
) -> dict:
    """Run a fresh prefix-only replay and a next-draw forecast, without file IO.

    ``tail_trend_runner`` supplies the application's existing pure tail trend
    calculation. It must return the usual ``results`` and ``next_plan`` fields.
    No Tk widgets, cached reports, or GUI imports are accessed here.
    """
    history, inferred = _prepare_history(df, test_periods)
    if not callable(tail_trend_runner):
        raise ValueError("必须提供可调用的尾数走势计算入口")
    numbers = history["number"].tolist()
    animals = history["animal"].tolist()
    dates = history["date"]
    tail_records = _build_tail_records(history, tail_trend_runner)
    zodiac_model = ZodiacTop9Predictor()
    number_model = Top15Predictor()
    records = []
    miss_streak = 0
    for index in range(len(history) - test_periods, len(history)):
        row = _predict_for(index, dates.iloc[index], numbers, animals, tail_records, zodiac_model, number_model)
        actual = numbers[index]
        row["actual"] = actual
        row["actual_animal"] = animals[index]
        row["hit"] = actual in row["prediction"]
        row["top9_hit"] = actual in row.pop("_top9_pool")
        row["filtered_pool_hit"] = actual in row.pop("_filtered_pool")
        miss_streak = 0 if row["hit"] else miss_streak + 1
        row["miss_streak_after"] = miss_streak
        row["history_through"] = dates.iloc[index - 1].strftime("%Y-%m-%d")
        records.append(row)

    next_date = dates.iloc[-1] + pd.Timedelta(days=1)
    next_prediction = _predict_for(len(history), next_date, numbers, animals, tail_records, zodiac_model, number_model)
    next_prediction.pop("_top9_pool")
    next_prediction.pop("_filtered_pool")
    summary = _summarize(records)
    return {
        "model": MODEL_NAME,
        "data_through": dates.iloc[-1].strftime("%Y-%m-%d"),
        "prediction_date": next_prediction["date"],
        "input_records": len(history), "inferred_animal_count": inferred,
        "summary": summary,
        "blocks100": [_summarize(records[index:index + 100]) for index in range(0, len(records), 100)],
        "records": records,
        "next_prediction": next_prediction,
        "warmup_periods": len(records) - summary["tail_calibration_ready_periods"],
    }


def _numbers_text(numbers) -> str:
    return " ".join(f"{int(number):02d}" for number in numbers)


def _tail_text(row: dict) -> str:
    if not row["tail_calibration_ready"]:
        return "预热不排尾"
    return ",".join(map(str, row["excluded_tails"])) if row["excluded_tails"] else "不排尾"


def format_inverse_exclusion_top15_report(report: dict, include_details: bool = True) -> str:
    """Format a concise summary first, followed by every requested draw row."""
    summary = report["summary"]
    next_prediction = report["next_prediction"]
    lines = [
        report["model"],
        "规则：生肖TOP9正向 → 尾数反向筛选 → 正向评分精简15个",
        f"历史数据：{report['input_records']}期，最新日期：{report['data_through']}",
        f"回测范围：{summary['from']} 至 {summary['through']}（最近{summary['periods']}期）",
        f"命中：{summary['hits']}/{summary['periods']}，成功率：{summary['hit_rate']:.2%}，未中：{summary['misses']}期",
        f"每期固定15个号码；最长连续不中：{summary['max_consecutive_misses']}期；当前连续不中：{summary['ending_miss_streak']}期",
        "",
        f"下期日期：{next_prediction['date']}",
        f"下期生肖TOP9：{'、'.join(next_prediction['top9'])}",
        f"下期反向排除尾数：{_tail_text(next_prediction)}；精简前候选池：{next_prediction['filtered_pool_count']}个",
        f"下期15个号码：{_numbers_text(next_prediction['prediction'])}",
        "",
        "每100期统计：",
    ]
    for block in report["blocks100"]:
        lines.append(
            f"{block['from']} 至 {block['through']}：命中{block['hits']}/{block['periods']}，"
            f"成功率{block['hit_rate']:.2%}，最长连续不中{block['max_consecutive_misses']}期"
        )
    if summary["longest_miss_runs"]:
        lines.append("最长连续不中区间（按历史记录统计）：")
        for run in summary["longest_miss_runs"]:
            suffix = "，尚未结束" if not run["closed"] else ""
            lines.append(f"{run['from']} 至 {run['through']}：{run['length']}期{suffix}")
    lines.append(
        f"尾数记录充足：{summary['tail_calibration_ready_periods']}/{summary['periods']}期；"
        f"实际排尾：{summary['active_tail_filter_periods']}期"
    )
    if report.get("warmup_periods", 0):
        lines.append(f"其中{report['warmup_periods']}期尾数表现记录不足100期，按规则预热不排尾。")
    if report.get("inferred_animal_count", 0):
        lines.append(f"已按各期日期的生肖号码映射补齐{report['inferred_animal_count']}条缺失生肖，仅用于本次计算。")
    if include_details:
        lines.extend(["", f"最近{summary['periods']}期预测详情：", "期号 日期 开奖 预测号码 结果 排尾 精简前候选池 连续不中"])
        for index, row in enumerate(report["records"], 1):
            lines.append(
                f"{index:03d} {row['date']} 实际={row['actual']:02d} "
                f"预测={_numbers_text(row['prediction'])} "
                f"结果={'命中' if row['hit'] else '未中'} 排尾={_tail_text(row)} "
                f"候选池={row['filtered_pool_count']} 连续不中={row['miss_streak_after']}"
            )
    return "\n".join(lines) + "\n"
