"""
每日动态择优投注方案。

统一使用 walk-forward（只看当期之前的数据）比较现有核心模型，并从长期、
近期均保持正收益的候选中选择近期命中率最高者。历史回测仅用于比较，
不代表未来结果。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from precise_top15_predictor import PreciseTop15Predictor
from dual_objective_top15_predictor import DualObjectiveTop15Predictor
from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy
from zodiac_simple_smart import ZodiacSimpleSmart


DISPLAY_ZODIAC_CYCLE_2026 = [
    "马", "蛇", "龙", "兔", "虎", "牛", "鼠", "猪", "狗", "鸡", "猴", "羊"
]
DISPLAY_ZODIAC_NUMBERS_2026 = {
    zodiac: [
        number for number in range(1, 50)
        if DISPLAY_ZODIAC_CYCLE_2026[(number - 1) % 12] == zodiac
    ]
    for zodiac in DISPLAY_ZODIAC_CYCLE_2026
}
PAYOUT_UNITS = 47.0


@dataclass
class ModelResult:
    model: str
    target: str
    periods: int
    hit_rate: float
    recent_100_hit_rate: float
    recent_30_hit_rate: float
    average_coverage: float
    random_baseline: float
    edge_over_random: float
    roi: float
    recent_100_roi: float
    max_miss_streak: int
    current_miss_streak: int
    streak_bucket: str
    conditional_samples: int
    conditional_hits: int
    conditional_hit_rate: float
    adjusted_conditional_hit_rate: float
    adjusted_conditional_roi: float
    streak_profile: dict
    next_selection: list
    next_numbers: list[int]


def _number_pool_from_zodiacs(zodiacs: Iterable[str]) -> list[int]:
    pool: set[int] = set()
    for zodiac in zodiacs:
        pool.update(DISPLAY_ZODIAC_NUMBERS_2026.get(str(zodiac).strip(), []))
    return sorted(pool)


def _max_miss_streak(records: list[dict]) -> int:
    current = maximum = 0
    for record in records:
        current = 0 if record["hit"] else current + 1
        maximum = max(maximum, current)
    return maximum


def _streak_bucket(streak: int) -> str:
    if streak <= 0:
        return "0期"
    if streak == 1:
        return "1期"
    if streak == 2:
        return "2期"
    if streak <= 4:
        return "3～4期"
    if streak <= 6:
        return "5～6期"
    return "7期以上"


def _streak_context(records: list[dict], overall_rate: float, next_cost: float) -> dict:
    """统计进入每一期之前的连不中状态，并估计当前状态的下一期命中率。"""
    streak = 0
    contexts = []
    for record in records:
        contexts.append((streak, bool(record["hit"])))
        streak = 0 if record["hit"] else streak + 1

    # 用20个等效样本向模型长期命中率收缩，降低小样本状态的偶然波动。
    prior_strength = 20.0
    profile = {}
    for bucket in ("0期", "1期", "2期", "3～4期", "5～6期", "7期以上"):
        values = [hit for before, hit in contexts if _streak_bucket(before) == bucket]
        bucket_samples = len(values)
        bucket_hits = sum(values)
        bucket_raw = bucket_hits / bucket_samples if bucket_samples else overall_rate
        bucket_adjusted = (
            (bucket_hits + overall_rate * prior_strength)
            / (bucket_samples + prior_strength)
            if bucket_samples else overall_rate
        )
        profile[bucket] = {
            "samples": bucket_samples,
            "hits": bucket_hits,
            "hit_rate": bucket_raw * 100.0,
            "adjusted_hit_rate": bucket_adjusted * 100.0,
        }

    current_bucket = _streak_bucket(streak)
    current = profile[current_bucket]
    samples = current["samples"]
    hits = current["hits"]
    raw_rate = current["hit_rate"] / 100.0
    adjusted_rate = (
        (hits + overall_rate * prior_strength) / (samples + prior_strength)
        if samples else overall_rate
    )
    adjusted_roi = (
        (adjusted_rate * PAYOUT_UNITS - next_cost) / next_cost * 100.0
        if next_cost else 0.0
    )
    return {
        "current_miss_streak": streak,
        "streak_bucket": current_bucket,
        "conditional_samples": samples,
        "conditional_hits": hits,
        "conditional_hit_rate": raw_rate * 100.0,
        "adjusted_conditional_hit_rate": adjusted_rate * 100.0,
        "adjusted_conditional_roi": adjusted_roi,
        "streak_profile": profile,
    }


def _summarize(
    name: str,
    target: str,
    records: list[dict],
    next_selection: list,
    next_numbers: list[int],
    population_size: int,
    next_cost: float,
) -> ModelResult:
    periods = len(records)
    recent_100 = records[-min(100, periods):]
    recent_30 = records[-min(30, periods):]

    def hit_rate(rows: list[dict]) -> float:
        return 100.0 * sum(row["hit"] for row in rows) / len(rows)

    def roi(rows: list[dict]) -> float:
        cost = sum(row["cost"] for row in rows)
        returns = sum(PAYOUT_UNITS for row in rows if row["hit"])
        return 100.0 * (returns - cost) / cost if cost else 0.0

    average_coverage = sum(row["coverage"] for row in records) / periods
    random_baseline = average_coverage / population_size * 100.0
    overall_hit_rate = hit_rate(records)
    streak_context = _streak_context(
        records, overall_hit_rate / 100.0, next_cost
    )

    return ModelResult(
        model=name,
        target=target,
        periods=periods,
        hit_rate=overall_hit_rate,
        recent_100_hit_rate=hit_rate(recent_100),
        recent_30_hit_rate=hit_rate(recent_30),
        average_coverage=average_coverage,
        random_baseline=random_baseline,
        edge_over_random=overall_hit_rate - random_baseline,
        roi=roi(records),
        recent_100_roi=roi(recent_100),
        max_miss_streak=_max_miss_streak(records),
        **streak_context,
        next_selection=next_selection,
        next_numbers=next_numbers,
    )


def _evaluate_number_model(
    name: str,
    numbers: list[int],
    start: int,
    factory: Callable,
    predict: Callable,
    settle: Callable,
) -> ModelResult:
    model = factory()
    records = []
    for index in range(start, len(numbers)):
        prediction = [int(value) for value in predict(model, numbers[:index])]
        actual = numbers[index]
        hit = actual in prediction
        records.append({"hit": hit, "cost": len(prediction), "coverage": len(prediction)})
        settle(model, prediction, actual, hit)

    next_numbers = [int(value) for value in predict(model, numbers)]
    return _summarize(
        name, "数字", records, next_numbers, next_numbers, 49, len(next_numbers)
    )


def _evaluate_animal_model(
    name: str,
    animals: list[str],
    start: int,
    factory: Callable,
    predict: Callable,
    settle: Callable,
) -> ModelResult:
    model = factory()
    records = []
    for index in range(start, len(animals)):
        prediction = [str(value).strip() for value in predict(model, animals[:index])]
        hit = animals[index] in prediction
        records.append({
            "hit": hit,
            "cost": len(prediction) * 4,
            "coverage": len(prediction),
        })
        settle(model, hit)

    next_zodiacs = [str(value).strip() for value in predict(model, animals)]
    next_numbers = _number_pool_from_zodiacs(next_zodiacs)
    return _summarize(
        name, "生肖", records, next_zodiacs, next_numbers, 12, len(next_zodiacs) * 4
    )


def benchmark_models(df: pd.DataFrame, test_periods: int = 400) -> list[ModelResult]:
    """以统一无前视口径回测核心模型。"""
    numbers = df["number"].astype(int).tolist()
    animals = df["animal"].astype(str).str.strip().tolist()
    if len(numbers) < 80:
        raise ValueError("至少需要80期数据才能生成每日动态方案")

    periods = min(test_periods, len(numbers) - 50)
    start = len(numbers) - periods
    results = [
        _evaluate_number_model(
            "双目标数字TOP15", numbers, start, DualObjectiveTop15Predictor,
            lambda model, history: model.predict(history),
            lambda model, prediction, actual, _hit: model.update(prediction, actual),
        ),
        _evaluate_number_model(
            "精准数字TOP20", numbers, start, PreciseTop15Predictor,
            lambda model, history: model.predict(history, k=20),
            lambda model, prediction, actual, _hit: model.update_performance(prediction, actual),
        ),
        _evaluate_animal_model(
            "四大生肖", animals, start, RecommendedZodiacTop4Strategy,
            lambda model, history: model.predict_top4(history)["top4"],
            _settle_zodiac_top4,
        ),
        _evaluate_animal_model(
            "五大生肖", animals, start, ZodiacSimpleSmart,
            lambda model, history: model.predict_from_history(
                history, top_n=5, debug=False
            )["top5"],
            lambda _model, _hit: None,
        ),
    ]
    return sorted(
        results,
        key=lambda result: result.adjusted_conditional_hit_rate,
        reverse=True,
    )


def _settle_zodiac_top4(model, hit: bool) -> None:
    model.update_performance(hit)
    model.check_and_switch_model()


def _select_daily_model(results: list[ModelResult]) -> tuple[ModelResult | None, str]:
    """
    优先比较各模型在“当前连不中区间”下的校正命中率。
    条件样本不足时，回退到长期/近期正收益规则。
    """
    state_eligible = [
        result for result in results
        if result.conditional_samples >= 8 and result.adjusted_conditional_roi > 0
    ]
    if state_eligible:
        selected = max(
            state_eligible,
            key=lambda result: (
                result.adjusted_conditional_hit_rate,
                result.conditional_samples,
                result.recent_100_hit_rate,
            ),
        )
        reason = (
            f"当前已连续{selected.current_miss_streak}期未中，属于"
            f"“{selected.streak_bucket}”状态；历史同状态"
            f"{selected.conditional_hits}/{selected.conditional_samples}期命中"
            f"（原始{selected.conditional_hit_rate:.2f}%），经小样本收缩后为"
            f"{selected.adjusted_conditional_hit_rate:.2f}%，在四类模型中最高。"
        )
        return selected, reason

    fallback = [
        result for result in results
        if result.roi > 0 and result.recent_100_roi > 0
    ]
    if not fallback:
        return None, "没有模型满足状态样本或正收益要求，今日建议空仓。"

    selected = max(fallback, key=lambda result: result.recent_100_hit_rate)
    return selected, (
        "当前连不中状态样本不足，已回退到长期及最近100期正收益过滤；"
        f"{selected.model}的最近100期命中率最高。"
    )


def build_daily_plan(
    data_path: str | Path = "data/lucky_numbers.csv",
    test_periods: int = 400,
) -> dict:
    data_path = Path(data_path)
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    results = benchmark_models(df, test_periods=test_periods)
    selected, reason = _select_daily_model(results)

    latest_date = pd.to_datetime(df.iloc[-1]["date"])
    plan_date = latest_date + timedelta(days=1)
    plan = {
        "plan_date": plan_date.strftime("%Y-%m-%d"),
        "data_through": latest_date.strftime("%Y-%m-%d"),
        "selection_rule": reason,
        "status": "BET" if selected else "SKIP",
        "selected_model": selected.model if selected else None,
        "target": selected.target if selected else None,
        "historical_hit_rate": selected.hit_rate if selected else None,
        "recent_100_hit_rate": selected.recent_100_hit_rate if selected else None,
        "recent_30_hit_rate": selected.recent_30_hit_rate if selected else None,
        "historical_roi": selected.roi if selected else None,
        "recent_100_roi": selected.recent_100_roi if selected else None,
        "max_miss_streak": selected.max_miss_streak if selected else None,
        "current_miss_streak": selected.current_miss_streak if selected else None,
        "streak_bucket": selected.streak_bucket if selected else None,
        "conditional_samples": selected.conditional_samples if selected else None,
        "conditional_hits": selected.conditional_hits if selected else None,
        "conditional_hit_rate": selected.conditional_hit_rate if selected else None,
        "adjusted_conditional_hit_rate": (
            selected.adjusted_conditional_hit_rate if selected else None
        ),
        "adjusted_conditional_roi": (
            selected.adjusted_conditional_roi if selected else None
        ),
        "selection": selected.next_selection if selected else [],
        "numbers": selected.next_numbers if selected else [],
        "stake_rule": _build_stake_rule(selected),
        "disclaimer": "命中率和ROI均为历史滚动回测结果，不是未来收益保证。",
        "benchmarks": [asdict(result) for result in results],
    }
    return plan


def _build_stake_rule(selected: ModelResult | None) -> str:
    if selected is None:
        return "空仓，不投注。"
    if selected.target == "生肖":
        total = len(selected.next_selection) * 4
        return (
            "每个生肖固定4单位，仅投1轮，不倍投、不追损；"
            f"本期总投入{total}单位。"
        )
    return (
        "逐号码等额1单位，仅投1轮，不倍投、不追损；"
        f"本期总投入{len(selected.next_numbers)}单位。"
    )


def format_plan(plan: dict) -> str:
    lines = [
        "=" * 72,
        f"每日动态择优投注方案｜{plan['plan_date']}",
        "=" * 72,
        f"数据截止：{plan['data_through']}",
        f"执行状态：{'执行' if plan['status'] == 'BET' else '空仓'}",
        f"选择模型：{plan['selected_model'] or '-'}",
        f"选择依据：{plan['selection_rule']}",
    ]
    if plan["status"] == "BET":
        selection_label = "推荐生肖" if plan["target"] == "生肖" else "推荐数字"
        lines.extend([
            "",
            f"{selection_label}：{'、'.join(str(value) for value in plan['selection'])}",
        ])
        if plan["target"] == "生肖":
            lines.extend([
                f"生肖对应号码（{len(plan['numbers'])}个，仅供核对）：",
                " ".join(f"{number:02d}" for number in plan["numbers"]),
            ])
        lines.extend([
            "",
            f"最近400期命中率：{plan['historical_hit_rate']:.2f}%",
            f"最近100期命中率：{plan['recent_100_hit_rate']:.2f}%",
            f"最近30期命中率：{plan['recent_30_hit_rate']:.2f}%",
            f"最近400期ROI：{plan['historical_roi']:+.2f}%",
            f"最近100期ROI：{plan['recent_100_roi']:+.2f}%",
            f"最大连续未中：{plan['max_miss_streak']}期",
            "",
            f"当前连续未中：{plan['current_miss_streak']}期",
            f"当前状态分段：{plan['streak_bucket']}",
            (
                f"同状态历史命中：{plan['conditional_hits']}/"
                f"{plan['conditional_samples']} = {plan['conditional_hit_rate']:.2f}%"
            ),
            f"校正后条件命中率：{plan['adjusted_conditional_hit_rate']:.2f}%",
            f"校正后条件ROI：{plan['adjusted_conditional_roi']:+.2f}%",
        ])
    lines.extend([
        "",
        f"投注规则：{plan['stake_rule']}",
        f"说明：{plan['disclaimer']}",
    ])
    return "\n".join(lines)


def save_outputs(plan: dict, output_dir: str | Path) -> tuple[Path, ...]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    date = plan["plan_date"]
    text_path = output_dir / f"每日最优投注方案_{date}.txt"
    json_path = output_dir / f"每日最优投注方案_{date}.json"
    benchmark_path = output_dir / f"模型统一回测_{date}.csv"
    streak_path = output_dir / f"模型连不中条件命中率_{date}.csv"

    text_path.write_text(format_plan(plan), encoding="utf-8-sig")
    json_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    benchmark_rows = []
    for item in plan["benchmarks"]:
        row = dict(item)
        row.pop("streak_profile", None)
        benchmark_rows.append(row)
    pd.DataFrame(benchmark_rows).to_csv(
        benchmark_path, index=False, encoding="utf-8-sig"
    )
    streak_rows = []
    for model in plan["benchmarks"]:
        for bucket, values in model["streak_profile"].items():
            streak_rows.append({
                "model": model["model"],
                "streak_bucket": bucket,
                "samples": values["samples"],
                "hits": values["hits"],
                "hit_rate": values["hit_rate"],
                "adjusted_hit_rate": values["adjusted_hit_rate"],
                "is_current": bucket == model["streak_bucket"],
            })
    pd.DataFrame(streak_rows).to_csv(
        streak_path, index=False, encoding="utf-8-sig"
    )
    return text_path, json_path, benchmark_path, streak_path


def main() -> None:
    parser = argparse.ArgumentParser(description="生成每日动态择优投注方案")
    parser.add_argument("--data", default="data/lucky_numbers.csv")
    parser.add_argument("--periods", type=int, default=400)
    parser.add_argument("--output-dir", default="预测结果")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    plan = build_daily_plan(args.data, args.periods)
    print(format_plan(plan))
    if not args.no_save:
        paths = save_outputs(plan, args.output_dir)
        print("\n已保存：")
        for path in paths:
            print(f"- {path}")


if __name__ == "__main__":
    main()
