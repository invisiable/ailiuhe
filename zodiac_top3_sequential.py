"""Fixed-rule TOP3 forecasts using past-only calibration of the original ranking.

This reuses ZodiacSimpleSmart's full ranking, including its canonical TOP5
feedback. It does not train offline ML or search windows/parameters. The online
rank calibration still learns from past outcomes, and the reused source model's
parameters were historically designed. Rank frequencies update after settlement.
"""

from __future__ import annotations

from collections import Counter, deque
from numbers import Integral

import pandas as pd

from zodiac_simple_smart import ZodiacSimpleSmart


CALIBRATION_WINDOW = 200
RANK_PSEUDOCOUNT = 2
RANK_COUNT = 12
PREDICTION_SIZE = 3
ZODIACS = ("鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪")
VALID_ZODIACS = frozenset(ZODIACS)


def _prepare_history(df, test_periods):
    if isinstance(test_periods, bool) or not isinstance(test_periods, Integral) or test_periods <= 0:
        raise ValueError("回测期数必须是正整数")
    if not isinstance(df, pd.DataFrame) or not {"date", "animal"}.issubset(df.columns):
        raise ValueError("历史数据必须包含date、animal列")
    if not df.columns.is_unique:
        raise ValueError("历史数据不能含有重复列名")
    periods = int(test_periods)
    if len(df) < periods + 1:
        raise ValueError(f"回测最近{periods}期至少需要{periods + 1}条历史记录")
    history = df.copy(deep=True).reset_index(drop=True)
    try:
        dates = pd.to_datetime(history["date"], errors="raise").dt.normalize()
    except (ValueError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("date列必须为有效日期") from error
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("日期必须有效、严格升序且不能重复")
    animals = history["animal"].astype("string").str.strip()
    if animals.isna().any() or not animals.isin(VALID_ZODIACS).all():
        raise ValueError("animal列必须为有效生肖，不能包含缺失值")
    history["date"] = dates
    history["animal"] = animals.astype(str)
    return history


def _forecast(result, settled_ranks):
    ranking = [zodiac for zodiac, _ in result["all_scores"]]
    original_top5 = list(result["top5"])
    if len(ranking) != RANK_COUNT or set(ranking) != VALID_ZODIACS:
        raise ValueError("原模型必须输出12个不重复的有效生肖排名")
    if original_top5 != ranking[:5]:
        raise ValueError("原模型TOP5必须与完整排名的前5名一致")
    counts = Counter(settled_ranks)
    denominator = len(settled_ranks) + RANK_COUNT * RANK_PSEUDOCOUNT
    scores = [(counts[rank] + RANK_PSEUDOCOUNT) / denominator for rank in range(1, RANK_COUNT + 1)]
    selected = sorted(range(1, RANK_COUNT + 1), key=lambda rank: (-scores[rank - 1], rank))[:PREDICTION_SIZE]
    return {
        "original_ranking": ranking,
        "original_top5": original_top5,
        "original_first3": original_top5[:PREDICTION_SIZE],
        "prediction": [ranking[rank - 1] for rank in selected],
        "selected_ranks": selected,
        # Values are in original rank order 1..12, not selection order.
        "rank_scores": scores,
        "rank_counts": [counts[rank] for rank in range(1, RANK_COUNT + 1)],
        "calibration_count": len(settled_ranks),
        "full_calibration": len(settled_ranks) == CALIBRATION_WINDOW,
    }


def _summarize(rows, hit_key="hit"):
    runs = []
    start = None
    for position, row in enumerate(rows):
        if not row[hit_key] and start is None:
            start = position
        if row[hit_key] and start is not None:
            runs.append((start, position - 1))
            start = None
    if start is not None:
        runs.append((start, len(rows) - 1))
    maximum = max((end - begin + 1 for begin, end in runs), default=0)
    longest = [{
        "start_index": rows[begin]["index"], "end_index": rows[end]["index"],
        "from": rows[begin]["date"], "through": rows[end]["date"],
        "length": end - begin + 1,
    } for begin, end in runs if end - begin + 1 == maximum]
    hits = sum(bool(row[hit_key]) for row in rows)
    return {
        "periods": len(rows), "hits": hits, "misses": len(rows) - hits,
        "hit_rate": hits / len(rows) if rows else 0.0,
        "max_miss_streak": maximum,
        "current_miss_streak": len(rows) - start if start is not None else 0,
        "longest_miss_streaks": longest,
        "from": rows[0]["date"] if rows else None,
        "through": rows[-1]["date"] if rows else None,
    }


def _comparisons(rows):
    return {
        "original_top5": _summarize(rows, "original_top5_hit"),
        "original_first3": _summarize(rows, "original_first3_hit"),
    }


def build_zodiac_top3_sequential_report(df, test_periods=400):
    """Replay a single fixed rule, warming both feedback states from index 1.

    Only data before a forecast can influence that forecast. The last
    ``test_periods`` forecasts are reported, but earlier forecasts still settle
    the TOP5 feedback and the rolling rank frequencies. Missing calendar dates
    are not filled. The supplied valid zodiac labels are preserved.
    """
    history = _prepare_history(df, test_periods)
    animals = history["animal"].tolist()
    dates = history["date"]
    start_index = len(history) - int(test_periods)
    predictor = ZodiacSimpleSmart()
    selected_miss_streaks = {zodiac: 0 for zodiac in predictor.zodiacs}
    settled_ranks = deque(maxlen=CALIBRATION_WINDOW)
    rows = []

    for index in range(1, len(history)):
        result = predictor.predict_from_history(
            animals[:index], top_n=5, debug=False,
            selected_miss_streaks=selected_miss_streaks,
        )
        forecast = _forecast(result, settled_ranks)
        actual = animals[index]
        actual_rank = forecast["original_ranking"].index(actual) + 1
        if index >= start_index:
            rows.append({
                "index": index, "date": dates.iloc[index].strftime("%Y-%m-%d"),
                "actual": actual, **forecast, "actual_rank": actual_rank,
                "hit": actual in forecast["prediction"],
                "original_top5_hit": actual in forecast["original_top5"],
                "original_first3_hit": actual in forecast["original_first3"],
            })

        # Settle only after constructing this period's forecast. Keep the source
        # feedback TOP5-based, even when the calibrated TOP3 selects other ranks.
        for zodiac in predictor.zodiacs:
            if zodiac in forecast["original_top5"]:
                selected_miss_streaks[zodiac] = 0 if zodiac == actual else selected_miss_streaks[zodiac] + 1
            else:
                selected_miss_streaks[zodiac] = 0
        settled_ranks.append(actual_rank)

    next_result = predictor.predict_from_history(
        animals, top_n=5, debug=False,
        selected_miss_streaks=selected_miss_streaks,
    )
    next_forecast = {
        "index": len(history),
        "date": (dates.iloc[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        **_forecast(next_result, settled_ranks),
    }
    summary = _summarize(rows)
    incomplete = sum(not row["full_calibration"] for row in rows)
    return {
        "model": "固定TOP3排名在线校准",
        "rule_version": "rank200_plus2_top3_v1",
        "test_periods": int(test_periods), "input_records": len(history),
        "from": rows[0]["date"], "data_through": rows[-1]["date"],
        "prediction_date": next_forecast["date"],
        "calibration_window": CALIBRATION_WINDOW,
        "rank_pseudocount": RANK_PSEUDOCOUNT,
        "incomplete_calibration_periods": incomplete,
        "rows": rows, "summary": summary,
        "baselines": _comparisons(rows),
        "blocks100": [{
            "summary": _summarize(rows[offset:offset + 100]),
            "baselines": _comparisons(rows[offset:offset + 100]),
        } for offset in range(0, len(rows), 100)],
        "next_forecast": next_forecast,
        "next_prediction": next_forecast["prediction"],
        "method": "复用原TOP5反馈模型的12生肖排名；只统计此前最多200次已结算预测中实际生肖所处的名次。各名次频率=(次数+2)/(已结算次数+24)，选频率最高的3个名次；同分优先原名次较前者。每期固定3个，不跳期、不搜参，开奖后才更新。",
        "limitations": "这是原模型排名的在线校准，不是独立新信号；排名经验频率不等于下一期中奖概率。历史回算不能保证未来命中率。规则虽未按本次回测调参，但此前已经查看过这批历史，仍须新增数据作前瞻验证。",
    }


def format_zodiac_top3_sequential_report(report, include_details=True):
    """Return Chinese summary and, by default, every historical forecast row."""
    summary = report["summary"]
    lines = [
        report["model"],
        f"回测：{report['from']} 至 {report['data_through']}，共{report['test_periods']}条开奖记录；每期固定3个生肖。",
        report["method"],
        f"未满{report['calibration_window']}次已结算校准记录：{report['incomplete_calibration_periods']}期（详情中标注）。",
        "", "方案｜命中｜命中率｜最长连续不中",
    ]
    for label, result in [("在线校准TOP3", summary), ("原排名前3（保持TOP5反馈）", report["baselines"]["original_first3"]), ("原TOP5", report["baselines"]["original_top5"])]:
        lines.append(f"{label}｜{result['hits']}/{result['periods']}｜{result['hit_rate']:.2%}｜{result['max_miss_streak']}期")
    for run in summary["longest_miss_streaks"]:
        lines.append(f"TOP3最长不中：{run['from']} 至 {run['through']}，{run['length']}期。")
    lines.extend(["", "分段结果：日期｜期数｜校准TOP3｜原前3｜原TOP5"])
    for block in report["blocks100"]:
        result = block["summary"]
        first3 = block["baselines"]["original_first3"]
        top5 = block["baselines"]["original_top5"]
        lines.append(f"{result['from']} 至 {result['through']}｜{result['periods']}｜{result['hits']}（{result['hit_rate']:.2%}）｜{first3['hits']}（{first3['hit_rate']:.2%}）｜{top5['hits']}（{top5['hit_rate']:.2%}）")
    upcoming = report["next_forecast"]
    lines.extend([
        "", f"下一期 {upcoming['date']}：{'、'.join(upcoming['prediction'])}；对应原排名：{'、'.join(map(str, upcoming['selected_ranks']))}。",
        report["limitations"],
    ])
    if include_details:
        lines.extend(["", "逐期详情：日期｜校准TOP3｜原排名位置｜实际｜结果｜原前3/原TOP5｜校准样本数｜排名1至12的频率"])
        for row in report["rows"]:
            flags = f"{'中' if row['original_first3_hit'] else '未中'}/{'中' if row['original_top5_hit'] else '未中'}"
            count = str(row["calibration_count"]) + ("（不足200）" if not row["full_calibration"] else "")
            lines.append(f"{row['date']}｜{'、'.join(row['prediction'])}｜{','.join(map(str, row['selected_ranks']))}｜{row['actual']}｜{'中' if row['hit'] else '未中'}｜{flags}｜{count}｜{','.join(f'{value:.4f}' for value in row['rank_scores'])}")
    return "\n".join(lines)
