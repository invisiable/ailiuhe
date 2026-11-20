"""Fixed16 numbers from the current TOP6 pool and TOP10 consensus scores."""

from numbers import Integral

import numpy as np
import pandas as pd

from dual_objective_top15_predictor import DualObjectiveTop15Predictor, NORMAL_WEIGHTS
from top5_top10_balanced19 import get_zodiac_number_map, _normalize, _summary
from zodiac_top6_scheme import FixedZodiacTop6Predictor, WARMUP, _prepare_history
from zodiac_top10_predictor import ZodiacTop10Predictor, ZODIAC_CYCLE_2026


TOP_N = 16
NUMBER_WEIGHT = .5
ZODIAC_WEIGHT = .5


def _select_numbers(top6, top10_scores, number_scores, target_date):
    if len(top6) != 6 or len(set(top6)) != 6 or not set(top6).issubset(ZODIAC_CYCLE_2026):
        raise ValueError("TOP6必须为6个不重复的有效生肖")
    top10_scores = np.asarray(top10_scores, dtype=float)
    number_scores = np.asarray(number_scores, dtype=float)
    if (top10_scores.shape != (12,) or number_scores.shape != (49,)
            or not np.isfinite(top10_scores).all() or not np.isfinite(number_scores).all()):
        raise ValueError("TOP10和号码评分必须分别为12及49个有限值")
    mapping = get_zodiac_number_map(target_date)
    number_to_zodiac = {number: zodiac for zodiac, values in mapping.items() for number in values}
    full_numbers = sorted({number for zodiac in top6 for number in mapping[zodiac]})
    normalized_numbers = _normalize(number_scores)
    normalized_zodiacs = dict(zip(ZODIAC_CYCLE_2026, _normalize(top10_scores)))
    combined = {number: float(NUMBER_WEIGHT * normalized_numbers[number - 1]
                              + ZODIAC_WEIGHT * normalized_zodiacs[number_to_zodiac[number]])
                for number in full_numbers}
    prediction = sorted(full_numbers, key=lambda number: (-combined[number], number))[:TOP_N]
    retained = {zodiac: [number for number in prediction if number_to_zodiac[number] == zodiac] for zodiac in top6}
    if len(prediction) != TOP_N or len(set(prediction)) != TOP_N:
        raise ValueError("未能生成固定16个唯一号码")
    top10 = [ZODIAC_CYCLE_2026[int(index)] for index in np.argsort(-top10_scores, kind="stable")[:10]]
    return {"prediction": prediction, "full_numbers": full_numbers, "combined_scores": combined,
            "top10": top10, "retained_numbers": retained,
            "retained_counts": {zodiac: len(values) for zodiac, values in retained.items()},
            "number_to_zodiac": number_to_zodiac}


def _number_scores(predictor, history):
    features, _ = predictor.build_features(history)
    rescue = predictor.consecutive_misses >= predictor.rescue_threshold
    active = features[14] if rescue else np.tensordot(NORMAL_WEIGHTS, features, axes=(0, 0))
    return active, predictor._select_top15(active), "冷号救援" if rescue else "常态评分"


def run_top6_top10_balanced16_audit(df, test_periods=400):
    """Predict before settlement, preserving each component's own feedback."""
    if isinstance(test_periods, bool) or not isinstance(test_periods, Integral) or test_periods <= 0:
        raise ValueError("回测期数必须为正整数")
    history = _prepare_history(df)
    periods = int(test_periods)
    if len(history) < periods + WARMUP:
        raise ValueError(f"完整回测{periods}期至少需要{periods + WARMUP}期数据")
    numbers = history["number"].tolist()
    animals = history["animal"].tolist()
    dates = history["date"]
    first_index = len(history) - periods
    top6_model = FixedZodiacTop6Predictor()
    top10_model = ZodiacTop10Predictor()
    number_model = DualObjectiveTop15Predictor()
    rows = []
    mismatches = []

    for index in range(1, len(history)):
        top10_before = top10_model.consecutive_miss
        top10_scores = top10_model.score_from_animals(animals[:index])
        top10 = [ZODIAC_CYCLE_2026[int(item)] for item in np.argsort(-top10_scores, kind="stable")[:10]]
        if index >= WARMUP:
            top6 = top6_model.predict(animals[:index])
            scores, base_top15, number_mode = _number_scores(number_model, numbers[:index])
            selected = _select_numbers(top6["prediction"], top10_scores, scores, dates.iloc[index])
            actual_number = numbers[index]
            label_hit = animals[index] in top6["prediction"]
            mapping_matches = selected["number_to_zodiac"][actual_number] == animals[index]
            if index >= first_index:
                row = {
                    "index": index, "date": dates.iloc[index].strftime("%Y-%m-%d"),
                    "history_count": index, "actual_number": actual_number, "actual_animal": animals[index],
                    "prediction": selected["prediction"], "prediction_count": TOP_N,
                    "hit": actual_number in selected["prediction"],
                    "full_top6_numbers": selected["full_numbers"], "full_top6_count": len(selected["full_numbers"]),
                    "full_top6_hit": actual_number in selected["full_numbers"],
                    "top6": list(top6["prediction"]), "top6_label_hit": label_hit, "top10": top10,
                    "top6_mode": top6["mode"], "top6_miss_before": top6["miss_before"],
                    "top10_miss_before": top10_before, "number_mode": number_mode,
                    "combined_scores": selected["combined_scores"], "retained_counts": selected["retained_counts"],
                    "mapping_matches": mapping_matches,
                }
                rows.append(row)
                if not mapping_matches:
                    mismatches.append({"date": row["date"], "number": actual_number,
                                       "animal": animals[index], "mapped_animal": selected["number_to_zodiac"][actual_number]})
            # Final16 performance does not replace any component's own feedback.
            top6_model.record_result(label_hit)
            number_model.update(base_top15, actual_number)
        top10_model.record_result(animals[index] in top10)

    next_date = dates.iloc[-1] + pd.Timedelta(days=1)
    next_top6 = top6_model.predict(animals)
    next_top10_scores = top10_model.score_from_animals(animals)
    next_scores, _, next_number_mode = _number_scores(number_model, numbers)
    selected = _select_numbers(next_top6["prediction"], next_top10_scores, next_scores, next_date)
    label_hits = sum(row["top6_label_hit"] for row in rows)
    report = {
        "scheme_id": "top6v3_top10_balanced_fixed16_v1", "scheme_label": "TOP6+TOP10精简16",
        "periods": periods, "target_count": TOP_N, "input_records": len(history),
        "data_through": dates.iloc[-1].strftime("%Y-%m-%d"),
        "protocol": {"number_weight": NUMBER_WEIGHT, "top10_weight": ZODIAC_WEIGHT,
                     "top6_version": top6_model.parameters["version"], "top6_parameters": top6_model.parameters,
                     "top6_warmup_index": WARMUP, "top10_warmup_index": 1,
                     "number_warmup_index": WARMUP, "top10_count": 10, "minimum_per_zodiac": 0},
        "rows": rows, "reduced": _summary(rows, "hit", "prediction_count"),
        "full_top6": _summary(rows, "full_top6_hit", "full_top6_count"),
        "top6_label_hits": label_hits, "top6_label_rate": label_hits / periods,
        "recent_windows": {str(window): _summary(rows[-window:], "hit", "prediction_count")
                           for window in (30, 100, 200, 300, 400) if window <= periods},
        "mapping_mismatches": mismatches,
        "next_prediction": selected["prediction"], "next_top6": next_top6["prediction"],
        "next_top10": selected["top10"], "next_target_date": next_date.strftime("%Y-%m-%d"),
        "next_full_top6_numbers": selected["full_numbers"], "next_retained_numbers": selected["retained_numbers"],
        "next_combined_scores": selected["combined_scores"], "next_top6_mode": next_top6["mode"],
        "next_number_mode": next_number_mode,
        "limitations": ["综合评分用于排序，不是校准后的中奖概率。",
                        "最终16号在TOP6号码池内选取，各肖保留数量由综合排序决定，不保证每肖均有号码。",
                        "16号命中率与完整TOP6生肖命中率不同，连续不中也分别统计。",
                        "各期特征只读取开奖前历史；TOP6及号码模型的既有参数使用过已查看历史，不属于独立验证。",
                        "下一期日期仅为最后记录的下一自然日，未核实实际开奖安排。"]}
    return report


def format_top6_top10_balanced16_report(report, include_details=False):
    reduced, full = report["reduced"], report["full_top6"]
    periods = report["periods"]
    lines = [report["scheme_label"], "=" * 72,
             f"数据截至{report['data_through']}，共{report['input_records']}期；回测最近{periods}期。",
             "", f"下一期日期参考：{report['next_target_date']}",
             f"下一期固定16号（综合分顺序）：{report['next_prediction']}",
             "按号码升序：" + "、".join(f"{number:02d}" for number in sorted(report["next_prediction"])),
             "TOP6：" + "、".join(report["next_top6"]), "TOP10：" + "、".join(report["next_top10"]),
             f"TOP6模式：{report['next_top6_mode']}；号码模型：{report['next_number_mode']}",
             f"完整TOP6号码池（{len(report['next_full_top6_numbers'])}号）：{report['next_full_top6_numbers']}",
             "各生肖实际保留号码："]
    for zodiac, values in report["next_retained_numbers"].items():
        lines.append(f"{zodiac}：{len(values)}号 {values}")
    lines.extend(["", "组合规则：TOP6提供候选池；号码模型评分和TOP10全排名生肖评分分别归一化，各占50%，池内固定取16号。",
                  "同分按号码升序；TOP10始终按前10肖反馈，三个组件各自更新状态。",
                  "", f"精简16：{reduced['hits']}/{periods} = {reduced['hit_rate']:.2%}；最长连续不中{reduced['max_miss_streak']}期。",
                  f"完整TOP6号码池：{full['hits']}/{periods} = {full['hit_rate']:.2%}；最长连续不中{full['max_miss_streak']}期。",
                  f"TOP6原生肖标签命中：{report['top6_label_hits']}/{periods} = {report['top6_label_rate']:.2%}。",
                  f"16个号码的等概率覆盖参考：{TOP_N}/49 = {TOP_N / 49:.2%}。",
                  "", "近期窗口："])
    for window, summary in report["recent_windows"].items():
        lines.append(f"最近{window}期：{summary['hits']}/{window} = {summary['hit_rate']:.2%}；最长连续不中{summary['max_miss_streak']}期。")
    lines.extend(["", "连续不中分布（含末尾未结束段）："])
    for label, summary in (("精简16", reduced), ("完整TOP6号码池", full)):
        distribution = summary["miss_streak_distribution"]
        text = "；".join(f"{length}期×{count}次" for length, count in distribution.items()) or "无"
        lines.append(label + "：" + text)
    lines.append("精简16最长连续不中区间：")
    by_index = {row["index"]: row for row in report["rows"]}
    for run in reduced["longest_miss_streaks"]:
        lines.append(f"{by_index[run['start_index']]['date']}～{by_index[run['end_index']]['date']}：{run['length']}期")
    if not reduced["longest_miss_streaks"]:
        lines.append("无")
    if report["mapping_mismatches"]:
        lines.append(f"数据提示：{len(report['mapping_mismatches'])}期号码与生肖标签不符合项目日期映射，16号和完整池命中均按实际号码核对。")
    lines.extend(["", *report["limitations"]])
    if include_details:
        lines.extend(["", "逐期回测明细：", "日期 | 实际号码/生肖 | 精简16 | 结果 | 完整池 | TOP6 | TOP10 | 保留数量"])
        for row in report["rows"]:
            lines.append(f"{row['date']} | {row['actual_number']:02d}/{row['actual_animal']} | {row['prediction']} | "
                         f"{'命中' if row['hit'] else '未中'} | {row['full_top6_count']}号 | "
                         f"{'、'.join(row['top6'])} | {'、'.join(row['top10'])} | {row['retained_counts']}")
    return "\n".join(lines) + "\n"
