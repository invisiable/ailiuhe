"""Fixed-six zodiac forecasts with chronological replay and date-aware pools."""

from collections import Counter
from numbers import Integral

import numpy as np
import pandas as pd

from top5_top10_balanced19 import get_zodiac_number_map
from zodiac_top10_predictor import ZodiacTop10Predictor, ZODIAC_CYCLE_2026


WARMUP = 50
TOP_N = 6
BASELINE_PARAMETERS = {
    "version": "v1", "weights": {"cold15": .20, "cold30": .05, "mk150": .50, "gap": .10, "hot30": .15},
    "blend_threshold": 2, "hot_blend_ratio": .25, "score_decimals": None,
}
RATE_OPTIMIZED_PARAMETERS = {
    "version": "v2", "weights": {"cold15": .20, "cold30": 0.0, "mk150": .525, "gap": .075, "hot30": .20},
    "blend_threshold": 2, "hot_blend_ratio": .50, "score_decimals": 12,
}
OPTIMIZED_PARAMETERS = {
    **RATE_OPTIMIZED_PARAMETERS, "version": "v3",
    "weights": dict(RATE_OPTIMIZED_PARAMETERS["weights"]),
    "fallback_after_misses": 3, "fallback_kind": "markov150", "fallback_mix": 1.0,
}
OPTIMIZATION_DATA_THROUGH = "2026-11-16"
PARAMETER_SELECTED_DATE = "2026-11-17"
ROLLING_CONTROL_REFERENCE = {"data_through": "2026-11-16", "hits": 224, "periods": 400, "hit_rate": .56}


class FixedZodiacTop6Predictor:
    """Use existing multidimensional scores, but always return exactly six.

    Unlike the TOP9/TOP10 entry points, long misses never expand the set.
    Feedback affects the hot-score blend; labels come from actual history.
    """

    def __init__(self, parameters=None):
        source = OPTIMIZED_PARAMETERS if parameters is None else parameters
        self.parameters = {**source, "weights": dict(source["weights"])}
        self.scorer = ZodiacTop10Predictor()
        self.scorer.weights = dict(self.parameters["weights"])
        self.scorer.blend_threshold = self.parameters["blend_threshold"]
        self.scorer.hot_blend_ratio = self.parameters["hot_blend_ratio"]

    @property
    def consecutive_miss(self):
        return self.scorer.consecutive_miss

    def predict(self, animals):
        if animals is None or isinstance(animals, (str, bytes)):
            raise ValueError("历史生肖必须为序列")
        history = list(animals)
        history = [str(animal).strip() for animal in history]
        scores = self.scorer.score_from_animals(history)
        fallback_after = self.parameters.get("fallback_after_misses")
        if fallback_after is not None and self.consecutive_miss >= fallback_after:
            # Long-miss fallback replaces the hot blend, instead of stacking it.
            kind = self.parameters["fallback_kind"]
            base = self.scorer._base_scores(history)
            backups = {
                "base": lambda: base,
                "cold15": lambda: self.scorer._cold_scores(history, 15),
                "cold30": lambda: self.scorer._cold_scores(history, 30),
                "markov150": lambda: self.scorer._markov_scores(history, window=150),
                "gap": lambda: self.scorer._gap_scores(history),
                "hot30": lambda: self.scorer._hot_scores(history, 30),
                "cold15_hot30": lambda: .5 * self.scorer._cold_scores(history, 15) + .5 * self.scorer._hot_scores(history, 30),
                "markov_hot30": lambda: .5 * self.scorer._markov_scores(history, window=150) + .5 * self.scorer._hot_scores(history, 30),
            }
            if kind not in backups:
                raise ValueError("未知连失备用评分")
            ratio = self.parameters["fallback_mix"]
            scores = (1 - ratio) * base + ratio * backups[kind]()
            label = "150期转移" if kind == "markov150" else kind
            mode = "长连失切换：" + label
        else:
            mode = ("热号互补" if self.scorer.hot_blend_ratio > 0
                    and self.consecutive_miss >= self.scorer.blend_threshold else "基础多维")
        if self.parameters["score_decimals"] is not None:
            scores = np.round(scores, self.parameters["score_decimals"])
        ranking = np.argsort(-scores, kind="stable")
        return {
            "prediction": [ZODIAC_CYCLE_2026[int(index)] for index in ranking[:TOP_N]],
            "scores": {zodiac: float(scores[index]) for index, zodiac in enumerate(ZODIAC_CYCLE_2026)},
            "mode": mode,
            "miss_before": self.consecutive_miss,
        }

    def record_result(self, hit):
        self.scorer.record_result(bool(hit))


def _prepare_history(df):
    if not isinstance(df, pd.DataFrame) or not {"date", "number", "animal"}.issubset(df.columns):
        raise ValueError("历史数据必须包含date、number、animal列")
    history = df.copy(deep=True).reset_index(drop=True)
    try:
        dates = pd.to_datetime(history["date"], errors="raise").dt.normalize()
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError("date列必须为有效日期") from error
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("日期必须有效、严格升序且不能重复")
    animals = history["animal"].astype("string").str.strip()
    if animals.isna().any() or not animals.isin(ZODIAC_CYCLE_2026).all():
        raise ValueError("animal列必须为有效生肖，不能缺失")
    numbers = pd.to_numeric(history["number"], errors="coerce")
    if numbers.isna().any() or not (numbers.between(1, 49) & (numbers % 1 == 0)).all():
        raise ValueError("number列必须为1到49的整数")
    if len(history) <= WARMUP:
        raise ValueError(f"至少需要{WARMUP + 1}期数据，前{WARMUP}期用于预热")
    history["date"] = dates
    history["animal"] = animals.astype(str)
    history["number"] = numbers.astype(int)
    return history


def _summary(records):
    runs = []
    start = None
    for index, row in enumerate(records):
        if not row["hit"] and start is None:
            start = index
        if row["hit"] and start is not None:
            runs.append({"length": index - start, "from": records[start]["date"],
                         "through": records[index - 1]["date"], "closed": True})
            start = None
    if start is not None:
        runs.append({"length": len(records) - start, "from": records[start]["date"],
                     "through": records[-1]["date"], "closed": False})
    hits = sum(row["hit"] for row in records)
    maximum = max((run["length"] for run in runs), default=0)
    profit = peak = drawdown = 0
    for row in records:
        profit += 46 * int(row["hit"]) - 24
        peak = max(peak, profit)
        drawdown = max(drawdown, peak - profit)
    return {"periods": len(records), "hits": hits, "misses": len(records) - hits,
            "hit_rate": hits / len(records), "max_miss_streak": maximum,
            "current_miss_streak": runs[-1]["length"] if runs and not runs[-1]["closed"] else 0,
            "miss_streak_distribution": dict(sorted(Counter(run["length"] for run in runs).items())),
            "longest_miss_runs": [run for run in runs if run["length"] == maximum],
            "miss_runs_ge3": sum(run["length"] >= 3 for run in runs),
            "miss_runs_ge4": sum(run["length"] >= 4 for run in runs),
            "misses_in_runs_ge4": sum(run["length"] for run in runs if run["length"] >= 4),
            "fixed_stake_profit": profit, "fixed_stake_max_drawdown": drawdown,
            "number_coverage_baseline": sum(row["pool_size"] / 49 for row in records) / len(records),
            "from": records[0]["date"], "through": records[-1]["date"]}


def build_zodiac_top6_report(df, test_periods=400, parameters=None):
    if isinstance(test_periods, bool) or not isinstance(test_periods, Integral) or test_periods <= 0:
        raise ValueError("回测期数必须为正整数")
    history = _prepare_history(df)
    animals = history["animal"].tolist()
    periods = min(int(test_periods), len(history) - WARMUP)
    predictor = FixedZodiacTop6Predictor(parameters)
    baseline = FixedZodiacTop6Predictor(BASELINE_PARAMETERS)
    previous = FixedZodiacTop6Predictor(RATE_OPTIMIZED_PARAMETERS)
    all_records, baseline_records, previous_records = [], [], []
    for index in range(WARMUP, len(history)):
        forecast = predictor.predict(animals[:index])
        baseline_forecast = baseline.predict(animals[:index])
        previous_forecast = previous.predict(animals[:index])
        actual = animals[index]
        hit = actual in forecast["prediction"]
        mapping = get_zodiac_number_map(history.iloc[index]["date"])
        pool = sorted(number for zodiac in forecast["prediction"] for number in mapping[zodiac])
        actual_number = int(history.iloc[index]["number"])
        all_records.append({"index": index, "date": history.iloc[index]["date"].strftime("%Y-%m-%d"),
                            "history_count": index, "prediction": forecast["prediction"],
                            "scores": forecast["scores"], "mode": forecast["mode"],
                            "miss_before": forecast["miss_before"], "actual": actual,
                            "actual_number": actual_number, "hit": hit, "pool_size": len(pool),
                            "number_pool": pool, "number_hit": actual_number in pool,
                            "mapping_matches": actual_number in mapping[actual]})
        baseline_hit = actual in baseline_forecast["prediction"]
        baseline_records.append({"date": history.iloc[index]["date"].strftime("%Y-%m-%d"),
                                 "hit": baseline_hit,
                                 "pool_size": sum(len(mapping[zodiac]) for zodiac in baseline_forecast["prediction"])})
        previous_hit = actual in previous_forecast["prediction"]
        previous_records.append({"date": history.iloc[index]["date"].strftime("%Y-%m-%d"),
                                 "hit": previous_hit,
                                 "pool_size": sum(len(mapping[zodiac]) for zodiac in previous_forecast["prediction"])})
        predictor.record_result(hit)
        baseline.record_result(baseline_hit)
        previous.record_result(previous_hit)
    records = all_records[-periods:]
    next_forecast = predictor.predict(animals)
    next_date = history.iloc[-1]["date"] + pd.Timedelta(days=1)
    next_mapping = get_zodiac_number_map(next_date)
    next_details = [
        {"zodiac": zodiac, "score": next_forecast["scores"][zodiac], "numbers": next_mapping[zodiac]}
        for zodiac in next_forecast["prediction"]
    ]
    return {
        "scheme_id": "fixed_top6_multidimensional_hot_feedback_" + predictor.parameters["version"],
        "scheme_label": ("固定6肖：连失优化版v3" if predictor.parameters["version"] == "v3"
                         else "固定6肖：参数优化版" + predictor.parameters["version"]),
        "protocol": {"top_n": TOP_N, "warmup": WARMUP, "weights": dict(predictor.scorer.weights),
                     "blend_after_misses": predictor.scorer.blend_threshold,
                     "hot_blend_ratio": predictor.scorer.hot_blend_ratio,
                     "fallback_after_misses": predictor.parameters.get("fallback_after_misses"),
                     "fallback_kind": predictor.parameters.get("fallback_kind"),
                     "fallback_mix": predictor.parameters.get("fallback_mix"),
                     "expand_after_misses": False, "history_labels": "actual_animal_column",
                     "score_decimals": predictor.parameters["score_decimals"]},
        "total_history": len(history), "last_date": history.iloc[-1]["date"].strftime("%Y-%m-%d"),
        "last_number": int(history.iloc[-1]["number"]), "last_animal": animals[-1],
        "requested_periods": int(test_periods), "test_periods": periods,
        "summary": _summary(records),
        "baseline_summary": _summary(baseline_records[-periods:]),
        "previous_version_summary": _summary(previous_records[-periods:]),
        "previous_version_long600": _summary(previous_records[-600:]) if len(previous_records) >= 600 else None,
        "previous_version_recent100": _summary(previous_records[-100:]) if len(previous_records) >= 100 else None,
        "long_window600": _summary(all_records[-600:]) if len(all_records) >= 600 else None,
        "recent_windows": {str(window): _summary(records[-window:])
                           for window in (30, 100, 200, 300, 400) if window <= len(records)},
        "blocks": [_summary(records[offset:offset + 100]) for offset in range(0, periods, 100)],
        "records": records, "mapping_mismatches": sum(not row["mapping_matches"] for row in records),
        "optimization": {"parameter_selected_date": PARAMETER_SELECTED_DATE,
                         "optimization_data_through": OPTIMIZATION_DATA_THROUGH,
                         "target": .60, "selection_windows": [100, 200, 300, 400],
                         "objective": "shorter_miss_streaks_with_recent_hit_rate_at_least_60pct",
                         "rolling_control_reference": dict(ROLLING_CONTROL_REFERENCE),
                         "evaluation_type": "retrospective_parameter_search_not_independent_validation"},
        "new_data_summary": (_summary([row for row in all_records if row["date"] > OPTIMIZATION_DATA_THROUGH])
                             if any(row["date"] > OPTIMIZATION_DATA_THROUGH for row in all_records) else None),
        "next_date": next_date.strftime("%Y-%m-%d"), "next_prediction": next_forecast["prediction"],
        "next_mode": next_forecast["mode"], "next_miss_before": next_forecast["miss_before"],
        "next_details": next_details,
        "next_number_pool": sorted(number for item in next_details for number in item["numbers"]),
        "limitations": ["评分用于选出6肖，不是已校准的中奖概率，也不作为提高投注权重的依据。",
                        "参数使用截至2026-11-16的历史进行搜索；这些窗口属于优化样本回测，不是独立验证。",
                        "历史回放不代表未来命中率，60%是历史优化目标，不是未来成功率保证。",
                        "下一日期按最后记录的下一自然日展示，不代表已核实开奖安排。"]}


def format_zodiac_top6_report(report, include_details=False):
    summary = report["summary"]
    weights = report["protocol"]["weights"]
    fallback = report["protocol"]["fallback_after_misses"]
    blend_rule = (f"连失达到{report['protocol']['blend_after_misses']}期时混入{report['protocol']['hot_blend_ratio']:.0%}热号评分；"
                  if report["protocol"]["hot_blend_ratio"] > 0 else "关闭热号互补；")
    if fallback is not None:
        backup_label = "150期一阶转移评分" if report["protocol"]["fallback_kind"] == "markov150" else report["protocol"]["fallback_kind"]
        blend_rule += f"连失达到{fallback}期及以上时，改用{backup_label}（混合比例{report['protocol']['fallback_mix']:.0%}），不再叠加热号互补。"
    blend_rule += "始终只选6肖，连失再长也不扩展数量；切换评分不能保证下一期命中。"
    lines = ["生肖TOP6预测方案 · " + report["scheme_label"], "=" * 64,
             f"数据：{report['total_history']}期；截至{report['last_date']}，{report['last_number']}号 / {report['last_animal']}",
             f"下一日期参考：{report['next_date']}；模式：{report['next_mode']}；当前模型连失：{report['next_miss_before']}期",
             "", "下一期固定6肖：" + "、".join(report["next_prediction"]),
             "展示顺序只表示模型评分，不表示已验证的概率高低或投注权重。"]
    for index, row in enumerate(report["next_details"], 1):
        lines.append(f"{index}. {row['zodiac']}  评分{row['score']:.4f}  对应号码：" + "、".join(f"{number:02d}" for number in row["numbers"]))
    lines.extend([f"号码池（{len(report['next_number_pool'])}个）：" + "、".join(f"{number:02d}" for number in report["next_number_pool"]),
                  "", "方案规则：",
                  f"冷号15期×{weights['cold15']:g}＋冷号30期×{weights['cold30']:g}＋转移150期×{weights['mk150']:g}＋间隔×{weights['gap']:g}＋热号30期×{weights['hot30']:g}。",
                  blend_rule,
                  "历史使用原始生肖标签，号码按目标日期映射；逐期先预测、后核对结果，反馈从前50期预热后连续更新。",
                  "", f"最近{report['test_periods']}期回测：{summary['hits']}/{summary['periods']}，命中率{summary['hit_rate']:.2%}",
                  f"同窗口原版对照：{report['baseline_summary']['hits']}/{report['baseline_summary']['periods']} = {report['baseline_summary']['hit_rate']:.2%}。",
                  f"上一版v2对照：{report['previous_version_summary']['hit_rate']:.2%}，最长连失{report['previous_version_summary']['max_miss_streak']}期。",
                  "本结果为参数优化样本回测，不是独立验证；60%目标不代表未来每期或每个窗口均能达到60%。",
                  f"按当期号码覆盖计算的随机基线：{summary['number_coverage_baseline']:.2%}（6肖约覆盖一半号码）。",
                  f"最大连续不中：{summary['max_miss_streak']}期；当前窗口末尾连续不中：{summary['current_miss_streak']}期。",
                  f"≥3期连失段{summary['miss_runs_ge3']}次；≥4期连失段{summary['miss_runs_ge4']}次。",
                  f"按每期24元、命中总返46元测算：净收益{summary['fixed_stake_profit']:+d}元，最大资金回撤{summary['fixed_stake_max_drawdown']}元。",
                  "", "近期窗口："])
    for item in report["recent_windows"].values():
        lines.append(f"最近{item['periods']}期：{item['hits']}/{item['periods']} = {item['hit_rate']:.2%}，最大连失{item['max_miss_streak']}期。")
    if report["previous_version_recent100"]:
        old = report["previous_version_recent100"]
        lines.append(f"最近100期上一版v2对照：{old['hit_rate']:.2%}，最大连失{old['max_miss_streak']}期。")
    if report["long_window600"]:
        item = report["long_window600"]
        old = report["previous_version_long600"]
        lines.append(f"额外长窗口600期：{item['hits']}/{item['periods']} = {item['hit_rate']:.2%}，最大连失{item['max_miss_streak']}期；资金回撤{item['fixed_stake_max_drawdown']}元（v2为{old['fixed_stake_max_drawdown']}元）。")
        lines.append("历史最长连失不是未来上限；部分窗口的资金回撤或短连失可能变差，不能据此称所有风险都降低。")
    control = report["optimization"]["rolling_control_reference"]
    lines.append(f"研发数据上的另一种时间滚动选参对照：{control['hits']}/{control['periods']} = {control['hit_rate']:.2%}，未达到60%；不是本固定参数版的独立验证。")
    if report["new_data_summary"]:
        item = report["new_data_summary"]
        lines.append(f"优化数据截止日之后的新记录：{item['hits']}/{item['periods']} = {item['hit_rate']:.2%}。")
    else:
        lines.append("参数于2026-11-17确定、优化数据截至2026-11-16，目前尚无优化数据之后的新记录可验证未来表现。")
    lines.extend(["", "每100期分段："])
    for block in report["blocks"]:
        lines.append(f"{block['from']}～{block['through']}：{block['hits']}/{block['periods']} = {block['hit_rate']:.2%}")
    lines.append("连续不中分布（含末尾未结束段）：" + ("；".join(
        f"{length}期×{count}次" for length, count in summary["miss_streak_distribution"].items()) or "无"))
    for run in summary["longest_miss_runs"]:
        lines.append(f"最长连失：{run['from']}～{run['through']}，{run['length']}期" + ("（尚未结束）" if not run["closed"] else ""))
    if report["mapping_mismatches"]:
        lines.append(f"数据提示：有{report['mapping_mismatches']}期号码与生肖标签不符合项目日期映射，生肖命中率按原标签统计。")
    lines.extend(["", "等额测算：若按项目每肖4元、命中总返46元的假设，6肖共24元，命中净赚22元，未中亏24元。实际结算条款需核对。",
                  *report["limitations"]])
    if include_details:
        lines.extend(["", "逐期回测明细：", "日期 | 实际生肖/号码 | 预测6肖 | 结果 | 开奖前连失 | 模式"])
        for row in report["records"]:
            lines.append(f"{row['date']} | {row['actual']}/{row['actual_number']:02d} | {'、'.join(row['prediction'])} | "
                         f"{'命中' if row['hit'] else '未中'} | {row['miss_before']} | {row['mode']}")
    return "\n".join(lines) + "\n"
