"""Fixed 20-number subset of the original TOP6 V3 pool."""

from dual_objective_top15_predictor import DualObjectiveTop15Predictor
from top5_top10_balanced19 import _normalize, _summary, get_zodiac_number_map
from zodiac_top6_rank_fusion import FROZEN_V3_PARAMETERS
from zodiac_top6_scheme import _prepare_history, build_zodiac_top6_report


TARGET_COUNT = 20
MIN_PER_ZODIAC = 3


def _select_numbers(top6, date, previous_numbers):
    mapping = get_zodiac_number_map(date)
    groups = [mapping[zodiac] for zodiac in top6]
    # The locked 300/100 rule uses only the gap score centered at 20 draws.
    _, _, gap_peak20 = DualObjectiveTop15Predictor._gap_features(previous_numbers)
    scores = _normalize(gap_peak20)

    def rank(number):
        return (-scores[number - 1], number)

    chosen = {number for group in groups for number in sorted(group, key=rank)[:MIN_PER_ZODIAC]}
    remainder = [number for group in groups for number in group if number not in chosen]
    chosen.update(sorted(remainder, key=rank)[:TARGET_COUNT - len(chosen)])
    return sorted(chosen)


def build_top6_fixed20_report(df, test_periods=400):
    history = _prepare_history(df)
    top6 = build_zodiac_top6_report(history, test_periods=test_periods, parameters=FROZEN_V3_PARAMETERS)
    numbers = history["number"].tolist()
    rows = []
    for record in top6["records"]:
        index = record["index"]
        prediction = _select_numbers(record["prediction"], record["date"], numbers[:index])
        rows.append({
            "index": index, "date": record["date"], "prediction": prediction,
            "actual_number": record["actual_number"], "hit": record["actual_number"] in prediction,
            "count": len(prediction), "full_hit": record["number_hit"], "full_count": record["pool_size"],
        })

    windows = {}
    for periods in sorted({len(rows), min(100, len(rows))}):
        selected = rows[-periods:]
        windows[periods] = {
            "from": selected[0]["date"], "through": selected[-1]["date"],
            "reduced": _summary(selected, "hit", "count"),
            "full": _summary(selected, "full_hit", "full_count"),
        }
    next_date, next_top6 = top6["next_date"], top6["next_prediction"]
    return {
        "data_through": top6["last_date"], "next_date": next_date, "next_top6": next_top6,
        "next_prediction": _select_numbers(next_top6, next_date, numbers),
        "rows": rows, "windows": windows,
    }


def format_top6_fixed20_report(report):
    formatted_numbers = "、".join(f"{number:02d}" for number in report["next_prediction"])
    lines = [
        "TOP6精简20",
        f"数据截至 {report['data_through']} · 下期日期参考 {report['next_date']}",
        "原TOP6  " + "、".join(report["next_top6"]),
        "", f"下期20号  {formatted_numbers}",
        "每肖先保留遗漏最接近20期的3号，再按同一规则从余下号码补2号。", "",
        "同期号码命中对比",
    ]
    for periods, summary in report["windows"].items():
        lines.append(f"最近{periods}期（{summary['from']}～{summary['through']}）")
        for key, label in (("reduced", "精简20"), ("full", "原TOP6")):
            item = summary[key]
            lines.append(f"  {label}  {item['hits']}/{periods}（{item['hit_rate']:.2%}），"
                         f"最长连失{item['max_miss_streak']}期")
    lines.extend(["", "固定规则在原100期留出验证中未达到原TOP6的命中率和连失标准。"])
    return "\n".join(lines) + "\n"
