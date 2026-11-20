"""Compare fixed-five zodiac candidates with chronological configuration selection."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from top5_top10_balanced19 import get_zodiac_number_map
from zodiac_top5_calibrated_candidates import BASELINE_ID, build_calibrated_candidates
from zodiac_top5_online_candidates import build_online_candidates


TRAIN_WINDOW = 200
BLOCK_SIZE = 100
FIRST_CANDIDATE_INDEX = 40
VALID_ZODIACS = frozenset(("鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"))
STRATEGY_LABELS = {
    "baseline": "原TOP5反馈模型",
    "calibrated": "评分校准择优",
    "smoothed": "平滑统计择优",
    "online": "在线专家择优",
    "baseline_online": "原模型在线融合",
}


def _prepare_history(df, test_periods):
    if isinstance(test_periods, bool) or not isinstance(test_periods, (int, np.integer)) or test_periods <= 0:
        raise ValueError("回测期数必须是正整数")
    if not isinstance(df, pd.DataFrame) or not {"date", "animal"}.issubset(df.columns):
        raise ValueError("历史数据必须包含date、animal列")
    minimum = int(test_periods) + TRAIN_WINDOW + FIRST_CANDIDATE_INDEX
    if len(df) < minimum:
        raise ValueError(f"滚动验证最近{test_periods}期至少需要{minimum}条历史记录")
    history = df.copy(deep=True).reset_index(drop=True)
    try:
        dates = pd.to_datetime(history["date"], errors="raise").dt.normalize()
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("date列必须为有效日期") from error
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("日期必须有效、严格升序且不能重复")
    animals = history["animal"].astype("string").str.strip()
    if animals.isna().any() or not animals.isin(VALID_ZODIACS).all():
        raise ValueError("animal列必须为有效生肖，不能包含缺失值")
    history["date"] = dates
    history["animal"] = animals.astype(str)
    return history


def _summarize(records):
    flags = [bool(row["hit"]) for row in records]
    runs = []
    start = None
    for index, hit in enumerate(flags):
        if not hit and start is None:
            start = index
        if hit and start is not None:
            runs.append({"length": index - start, "from": records[start]["date"], "through": records[index - 1]["date"], "closed": True})
            start = None
    if start is not None:
        runs.append({"length": len(flags) - start, "from": records[start]["date"], "through": records[-1]["date"], "closed": False})
    maximum = max((run["length"] for run in runs), default=0)
    return {
        "periods": len(records), "hits": sum(flags), "misses": len(flags) - sum(flags),
        "hit_rate": sum(flags) / len(flags) if flags else 0.0,
        "max_miss_streak": maximum,
        "current_miss_streak": runs[-1]["length"] if runs and not runs[-1]["closed"] else 0,
        "longest_miss_runs": [run for run in runs if run["length"] == maximum],
        "miss_streak_distribution": dict(sorted(Counter(run["length"] for run in runs).items())),
    }


def _training_key(candidate, animals, start, end):
    hits = 0
    streak = maximum = 0
    for index in range(start, end):
        hit = animals[index] in candidate["predictions"][index]
        hits += int(hit)
        streak = 0 if hit else streak + 1
        maximum = max(maximum, streak)
    return -hits, maximum, candidate["id"] != BASELINE_ID, candidate["id"]


def _select(group, animals, index):
    start = index - TRAIN_WINDOW
    selected = min(group, key=lambda candidate: _training_key(candidate, animals, start, index))
    key = _training_key(selected, animals, start, index)
    return selected, {
        "selected_config": selected["id"], "parameters": selected["parameters"],
        "training_from_index": start, "training_through_index": index - 1,
        "training_hits": -key[0], "training_periods": TRAIN_WINDOW,
        "training_max_miss_streak": key[1],
    }


def build_zodiac_top5_comparison(df, test_periods=400):
    """Select configurations before each100-period block; never use that block's outcomes."""
    history = _prepare_history(df, test_periods)
    animals = history["animal"].tolist()
    dates = history["date"]
    size = len(history)
    start = size - int(test_periods)
    candidates = build_calibrated_candidates(animals, first_index=FIRST_CANDIDATE_INDEX)
    candidates += build_online_candidates(animals, first_index=FIRST_CANDIDATE_INDEX)
    by_id = {candidate["id"]: candidate for candidate in candidates}
    if len(by_id) != len(candidates) or BASELINE_ID not in by_id:
        raise ValueError("候选模型ID重复或缺少原模型")
    groups = {key: [candidate for candidate in candidates if candidate["family"] == key] for key in STRATEGY_LABELS if key != "baseline"}
    if any(not group for group in groups.values()):
        raise ValueError("缺少待比较的模型方案")
    for candidate in candidates:
        for index in range(start - TRAIN_WINDOW, size + 1):
            prediction = candidate["predictions"].get(index, [])
            if len(prediction) != 5 or len(set(prediction)) != 5 or not set(prediction).issubset(VALID_ZODIACS):
                raise ValueError("所有候选方案必须每期固定输出5个不重复生肖")

    strategies = {}
    next_date = dates.iloc[-1] + pd.Timedelta(days=1)
    next_mapping = get_zodiac_number_map(next_date)
    for key, label in STRATEGY_LABELS.items():
        records = []
        schedule = []
        selected = by_id[BASELINE_ID]
        selection = {"selected_config": BASELINE_ID, "training_from_index": None, "training_through_index": None}
        for index in range(start, size):
            if key != "baseline" and (index - start) % BLOCK_SIZE == 0:
                selected, selection = _select(groups[key], animals, index)
                schedule.append({**selection, "apply_from_index": index, "apply_through_index": min(index + BLOCK_SIZE, size) - 1})
            prediction = list(selected["predictions"][index])
            records.append({
                "index": index, "date": dates.iloc[index].strftime("%Y-%m-%d"),
                "actual": animals[index], "prediction": prediction,
                "hit": animals[index] in prediction,
                "selected_config": selection["selected_config"],
                "training_from_index": selection["training_from_index"],
                "training_through_index": selection["training_through_index"],
            })

        if key != "baseline" and int(test_periods) % BLOCK_SIZE == 0:
            selected, next_selection = _select(groups[key], animals, size)
        else:
            next_selection = dict(selection)
        next_prediction = list(selected["predictions"][size])
        strategies[key] = {
            "label": label, "summary": _summarize(records),
            "blocks100": [_summarize(records[position:position + BLOCK_SIZE]) for position in range(0, len(records), BLOCK_SIZE)],
            "records": records, "selection_schedule": schedule,
            "next_prediction": next_prediction,
            "next_numbers": sorted({number for zodiac in next_prediction for number in next_mapping[zodiac]}),
            "next_selected_config": selected["id"], "next_selection": next_selection,
        }

    baseline_rows = strategies["baseline"]["records"]
    for strategy in strategies.values():
        won = sum(row["hit"] and not base["hit"] for row, base in zip(strategy["records"], baseline_rows))
        lost = sum(base["hit"] and not row["hit"] for row, base in zip(strategy["records"], baseline_rows))
        strategy["baseline_comparison"] = {"won": won, "lost": lost, "tied": int(test_periods) - won - lost, "net_hits": won - lost}
    data_warnings = []
    if "number" in history:
        supplied_numbers = pd.to_numeric(history["number"], errors="coerce")
        for index, number in enumerate(supplied_numbers):
            if pd.notna(number) and 1 <= number <= 49 and number == int(number):
                mapping = get_zodiac_number_map(dates.iloc[index])
                expected = next(zodiac for zodiac, values in mapping.items() if int(number) in values)
                if animals[index] != expected:
                    data_warnings.append(
                        f"{dates.iloc[index]:%Y-%m-%d}：{int(number):02d}号记录为{animals[index]}，按日期映射为{expected}；本次保留原记录。"
                    )
    return {
        "model": "生肖TOP5优化对比", "test_periods": int(test_periods),
        "input_records": size, "data_through": dates.iloc[-1].strftime("%Y-%m-%d"),
        "prediction_date": next_date.strftime("%Y-%m-%d"),
        "from": dates.iloc[start].strftime("%Y-%m-%d"),
        "training_window": TRAIN_WINDOW, "block_size": BLOCK_SIZE,
        "candidate_count": len(candidates), "strategies": strategies,
        "data_warnings": data_warnings,
        "method": "每100期开始前，仅用此前200期选配置，按命中次数、最长连续不中、原参数优先及ID依次打破同分；配置在下一段100期固定，专家内部只用已开奖记录更新。所有方案每期固定5个生肖。原模型入口不自动替换。",
        "limitations": "候选设计已经参考当前历史数据；按时间回算不能替代新增数据上的前瞻验证。相关模型共享数据和组件，不能视为独立确认。",
    }


def format_zodiac_top5_comparison(report, include_details=True):
    strategies = report["strategies"]
    lines = [
        report["model"],
        f"回测：{report['from']} 至 {report['data_through']}，共{report['test_periods']}期；每期固定5个生肖。",
        f"选参：每{report['block_size']}期只用此前{report['training_window']}期；共{report['candidate_count']}个固定候选配置。",
        "原模型继续保留；新增方案用于对照，不按本段开奖结果事后换参数。",
        "", "方案｜命中｜成功率｜最长不中｜相对原模型新增命中/丢失命中",
    ]
    for strategy in strategies.values():
        summary = strategy["summary"]
        paired = strategy["baseline_comparison"]
        lines.append(f"{strategy['label']}｜{summary['hits']}/{summary['periods']}｜{summary['hit_rate']:.2%}｜{summary['max_miss_streak']}期｜+{paired['won']}/-{paired['lost']}")
    lines.extend(["", "每100期命中数（由早到近）："])
    for strategy in strategies.values():
        lines.append(f"{strategy['label']}：" + "｜".join(f"{block['hits']}/{block['periods']}" for block in strategy["blocks100"]))
    lines.extend(["", f"下一期：{report['prediction_date']}"])
    for strategy in strategies.values():
        lines.append(f"{strategy['label']}：{'、'.join(strategy['next_prediction'])}")
        lines.append("  对应号码：" + " ".join(f"{number:02d}" for number in strategy["next_numbers"]))
    lines.extend(["", report["limitations"]])
    if report.get("data_warnings"):
        lines.append("原始数据待核对：")
        lines.extend(report["data_warnings"])
    if include_details:
        lines.extend(["", f"最近{report['test_periods']}期逐期对比："])
        keys = list(strategies)
        for position in range(report["test_periods"]):
            first = strategies[keys[0]]["records"][position]
            details = []
            for key in keys:
                strategy = strategies[key]
                row = strategy["records"][position]
                details.append(f"{strategy['label']}={'、'.join(row['prediction'])}({'命中' if row['hit'] else '未中'}，配置={row['selected_config']})")
            lines.append(f"{position + 1:03d} {first['date']} 实际={first['actual']}｜" + "｜".join(details))
    return "\n".join(lines) + "\n"
