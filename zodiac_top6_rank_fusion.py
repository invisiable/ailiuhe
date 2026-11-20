"""An independent fixed-six candidate:90% frozenv3 rank plus10% Finalv7 rank.

The v3 component settles its own six labels. The fusion separately tracks its
final result. Use settle(actual_label), not a final hit flag, after predict.
This module does not change the existing TOP6 default or combined-number entry.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from top5_top10_balanced19 import _summary, get_zodiac_number_map
from zodiac_final_predictor import ZodiacFinalPredictor
from zodiac_top6_scheme import FixedZodiacTop6Predictor, WARMUP, _prepare_history
from zodiac_top10_predictor import ZODIAC_CYCLE_2026


FROZEN_V3_PARAMETERS = {
    "version": "v3", "weights": {"cold15": .2, "cold30": 0., "mk150": .525, "gap": .075, "hot30": .2},
    "blend_threshold": 2, "hot_blend_ratio": .5, "score_decimals": 12,
    "fallback_after_misses": 3, "fallback_kind": "markov150", "fallback_mix": 1.,
}
FROZEN_FINAL_V7_WEIGHTS = {
    "ultra_cold": .30, "anti_hot": .15, "gap": .18, "rotation": .12,
    "absence_penalty": .08, "hot_momentum": .10, "diversity": .04, "similarity": .03,
}
SCHEME_ID = "fixed_top6_v3_90pct_finalv7_10pct_rank_v1"
LEGACY_WEIGHT = .1
ZODIACS = tuple(ZODIAC_CYCLE_2026)


def _labels(animals):
    if animals is None or isinstance(animals, (str, bytes)):
        raise ValueError("历史生肖必须为序列")
    try:
        history = tuple(str(animal).strip() for animal in animals)
    except TypeError as error:
        raise ValueError("历史生肖必须为序列") from error
    if not set(history).issubset(ZODIACS):
        raise ValueError("历史中含无效生肖")
    return history


class FixedZodiacTop6RankFusion:
    """Issue one forecast, then settle once with its actual zodiac label.

    Repeating predict with the same pending prefix is idempotent. A different
    prefix requires settling the pending result first, then appending exactly
    that actual label. Empty-prefix cold starts are supported; report replay
    uses the common50-period warmup for comparison with the existing model.
    """

    def __init__(self):
        self.base_predictor = FixedZodiacTop6Predictor(copy.deepcopy(FROZEN_V3_PARAMETERS))
        self.legacy_predictor = ZodiacFinalPredictor()
        self.legacy_predictor.weights = dict(FROZEN_FINAL_V7_WEIGHTS)
        self.final_misses = 0
        self._pending_history = None
        self._pending = None
        self._expected_history = None

    @property
    def consecutive_miss(self):
        return self.final_misses

    @property
    def base_consecutive_miss(self):
        return self.base_predictor.consecutive_miss

    def predict(self, animals):
        history = _labels(animals)
        if self._pending is not None:
            if history == self._pending_history:
                return copy.deepcopy(self._pending)
            raise RuntimeError("请先结算当前预测，再追加新的历史记录")
        if self._expected_history is not None and history != self._expected_history:
            raise ValueError("历史必须在上一预测前缀后恰好追加已结算的生肖")
        prefix = list(history)
        base = self.base_predictor.predict(prefix)
        base_ranking = sorted(ZODIACS, key=lambda zodiac: -base["scores"][zodiac])
        _, legacy_scores = self.legacy_predictor.predict_from_history(prefix, top_n=12)
        legacy_ranking = [zodiac for zodiac, _ in legacy_scores]
        if len(legacy_ranking) != 12 or set(legacy_ranking) != set(ZODIACS):
            raise ValueError("Final v7评分必须包含全部12生肖")
        base_values = {zodiac: (12 - index) / 78 for index, zodiac in enumerate(base_ranking)}
        legacy_values = {zodiac: (12 - index) / 78 for index, zodiac in enumerate(legacy_ranking)}
        scores = {zodiac: float(np.round((1 - LEGACY_WEIGHT) * base_values[zodiac]
                                         + LEGACY_WEIGHT * legacy_values[zodiac], 12)) for zodiac in ZODIACS}
        prediction = sorted(ZODIACS, key=lambda zodiac: -scores[zodiac])[:6]
        forecast = {
            "prediction": prediction, "scores": scores, "base_prediction": list(base["prediction"]),
            "base_ranking": base_ranking, "legacy_ranking": legacy_ranking,
            "base_miss_before": self.base_consecutive_miss, "final_miss_before": self.final_misses,
            "mode": "v3与Final v7排名融合（90%/10%）", "base_mode": base["mode"],
        }
        self._pending_history = history
        self._pending = copy.deepcopy(forecast)
        return copy.deepcopy(forecast)

    def settle(self, actual_label):
        if self._pending is None:
            raise RuntimeError("没有待结算的预测，不能重复结算")
        if not isinstance(actual_label, str) or actual_label.strip() not in ZODIACS:
            raise ValueError("实际结果必须为一个有效生肖标签，不能传入命中布尔值")
        actual = actual_label.strip()
        base_hit = actual in self._pending["base_prediction"]
        hit = actual in self._pending["prediction"]
        self.base_predictor.record_result(base_hit)
        self.final_misses = 0 if hit else self.final_misses + 1
        self._expected_history = self._pending_history + (actual,)
        self._pending_history = self._pending = None
        return {"actual": actual, "base_hit": base_hit, "hit": hit,
                "base_miss_after": self.base_consecutive_miss, "final_miss_after": self.final_misses}


def _window_summary(records, base=False):
    summary = _summary(records, "base_hit" if base else "hit", "base_pool_size" if base else "pool_size")
    by_index = {row["index"]: row for row in records}
    summary.update(periods=len(records), **{"from": records[0]["date"], "through": records[-1]["date"]})
    for run in summary["longest_miss_streaks"]:
        run["start_date"] = by_index[run["start_index"]]["date"]
        run["end_date"] = by_index[run["end_index"]]["date"]
    return summary


def build_rank_fusion_report(df, test_periods=400):
    """Replay all settled results from index50; reporting windows never reset it."""
    if isinstance(test_periods, bool) or not isinstance(test_periods, Integral) or test_periods <= 0:
        raise ValueError("回测期数必须为正整数")
    history = _prepare_history(df)
    periods = int(test_periods)
    if len(history) < periods + WARMUP:
        raise ValueError(f"完整回测{periods}期至少需要{periods + WARMUP}期数据")
    animals = history["animal"].tolist()
    model = FixedZodiacTop6RankFusion()
    all_records = []
    for index in range(WARMUP, len(history)):
        forecast = model.predict(animals[:index])
        actual = animals[index]
        mapping = get_zodiac_number_map(history.iloc[index]["date"])
        pool = sorted(number for zodiac in forecast["prediction"] for number in mapping[zodiac])
        base_pool_size = sum(len(mapping[zodiac]) for zodiac in forecast["base_prediction"])
        settlement = model.settle(actual)
        all_records.append({"index": index, "date": history.iloc[index]["date"].strftime("%Y-%m-%d"),
                            "history_count": index, "actual": actual, "actual_number": int(history.iloc[index]["number"]),
                            **forecast, "hit": settlement["hit"], "base_hit": settlement["base_hit"],
                            "number_pool": pool, "pool_size": len(pool), "base_pool_size": base_pool_size,
                            "number_hit": int(history.iloc[index]["number"]) in pool,
                            "mapping_matches": int(history.iloc[index]["number"]) in mapping[actual]})
    records = all_records[-periods:]
    next_forecast = model.predict(animals)
    next_date = history.iloc[-1]["date"] + pd.Timedelta(days=1)
    next_mapping = get_zodiac_number_map(next_date)
    return {
        "scheme_id": SCHEME_ID, "scheme_label": "生肖TOP6排名融合候选（v3 90%＋Final v7 10%）",
        "data_through": history.iloc[-1]["date"].strftime("%Y-%m-%d"), "input_records": len(history),
        "test_periods": periods, "records": records, "summary": _window_summary(records),
        "baseline_summary": _window_summary(records, base=True),
        "recent_windows": {str(window): _window_summary(records[-window:])
                           for window in (30, 100, 200, 300, 400, 600) if window <= len(records)},
        "baseline_recent_windows": {str(window): _window_summary(records[-window:], base=True)
                                    for window in (30, 100, 200, 300, 400, 600) if window <= len(records)},
        "long_window600": _window_summary(all_records[-600:]) if len(all_records) >= 600 else None,
        "baseline_long_window600": _window_summary(all_records[-600:], base=True) if len(all_records) >= 600 else None,
        "paired": {"added_hits": sum(row["hit"] and not row["base_hit"] for row in records),
                   "lost_hits": sum(row["base_hit"] and not row["hit"] for row in records)},
        "protocol": {"top_n": 6, "warmup": WARMUP, "v3_parameters": copy.deepcopy(FROZEN_V3_PARAMETERS),
                     "final_v7_weights": dict(FROZEN_FINAL_V7_WEIGHTS), "legacy_weight": LEGACY_WEIGHT,
                     "rank_values": "12..1 /78", "ties": "v3及融合：horse cycle；Final：保留原始rat cycle评分稳定排序；融合round12",
                     "feedback": "基础v3只按自己的6肖结算，最终融合另计命中与连失，不重置窗口状态"},
        "mapping_mismatches": sum(not row["mapping_matches"] for row in records),
        "next_date": next_date.strftime("%Y-%m-%d"), "next_prediction": next_forecast["prediction"],
        "next_base_prediction": next_forecast["base_prediction"], "next_forecast": next_forecast,
        "next_number_pool": sorted(number for zodiac in next_forecast["prediction"] for number in next_mapping[zodiac]),
        "limitations": ["这是独立候选模块，未替换既有TOP6默认或TOP6+TOP10号码方案。",
                        "评分只表示排名融合，不是校准后的中奖概率。",
                        "候选使用已查看历史进行选择，历史命中率不属于全新独立验证，也不能保证未来65%。",
                        "历史最大连失不是未来上限；下一日期仅按最后记录的下一自然日展示。",
                        "生肖命中按原始animal标签判断，号码池按目标日期映射，二者异常单独保留。"]}


def format_rank_fusion_report(report, include_details=False):
    lines = [report["scheme_label"], "=" * 72,
             f"数据截至{report['data_through']}，共{report['input_records']}期。",
             "窗口 | 融合命中 | 最长连失 | 原v3命中 | 原v3最长连失"]
    for window, summary in report["recent_windows"].items():
        base = report["baseline_recent_windows"][window]
        lines.append(f"{window} | {summary['hits']}/{window}={summary['hit_rate']:.2%} | {summary['max_miss_streak']} | "
                     f"{base['hits']}/{window}={base['hit_rate']:.2%} | {base['max_miss_streak']}")
    if report["long_window600"] and "600" not in report["recent_windows"]:
        summary, base = report["long_window600"], report["baseline_long_window600"]
        lines.append(f"600 | {summary['hits']}/600={summary['hit_rate']:.2%} | {summary['max_miss_streak']} | "
                     f"{base['hits']}/600={base['hit_rate']:.2%} | {base['max_miss_streak']}")
    lines.extend(["", f"下一日期参考：{report['next_date']}", "下一期6肖：" + "、".join(report["next_prediction"]),
                  f"配对比较：新增命中{report['paired']['added_hits']}次，丢失原命中{report['paired']['lost_hits']}次。",
                  f"标签与日期号码映射不符记录：{report['mapping_mismatches']}期。", *report["limitations"]])
    if include_details:
        lines.extend(["", "日期 | 实际 | 融合6肖 | 结果 | 原v3 6肖 | 原v3结果 | 基础/融合开奖前连失"])
        for row in report["records"]:
            lines.append(f"{row['date']} | {row['actual']} | {'、'.join(row['prediction'])} | {'中' if row['hit'] else '未中'} | "
                         f"{'、'.join(row['base_prediction'])} | {'中' if row['base_hit'] else '未中'} | "
                         f"{row['base_miss_before']}/{row['final_miss_before']}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "lucky_numbers.csv")
    parser.add_argument("--periods", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "预测结果")
    args = parser.parse_args()
    raw = args.data.read_bytes()
    report = build_rank_fusion_report(pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig"), args.periods)
    report["data_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"生肖TOP6排名融合候选_{report['data_through']}_{args.periods}期"
    stem.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    stem.with_suffix(".txt").write_text(format_rank_fusion_report(report, include_details=True), encoding="utf-8-sig")
    print(json.dumps({"report": str(stem.with_suffix('.txt')), "hits": report["summary"]["hits"],
                      "periods": report["test_periods"], "next_prediction": report["next_prediction"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
