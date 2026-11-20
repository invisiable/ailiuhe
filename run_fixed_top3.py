"""Replay the two frozen TOP3 rules and export their actual performance."""

from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path

import pandas as pd

from zodiac_top3_independent import build_independent_top3_report
from zodiac_top3_sequential import build_zodiac_top3_sequential_report


ROOT = Path(__file__).resolve().parent


def run(data_path, periods=400, output_dir="预测结果"):
    source = Path(data_path)
    source_bytes = source.read_bytes()
    history = pd.read_csv(BytesIO(source_bytes), encoding="utf-8-sig")
    sequential = build_zodiac_top3_sequential_report(history, test_periods=periods)
    independent = build_independent_top3_report(history, test_periods=periods)
    assert sequential["prediction_date"] == independent["next_date"]
    prediction_date = sequential["prediction_date"]
    configurations = {
        "rank_calibration": {
            "prior_settled_forecasts": 200,
            "pseudocount_per_rank": 2,
            "selection_count": 3,
            "uses_existing_top5_ranking": True,
        },
        "independent_counts": independent["protocol"],
    }
    report = {
        "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "code_sha256": {
            filename: hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
            for filename in ("zodiac_top3_sequential.py", "zodiac_top3_independent.py")
        },
        "prediction_date": prediction_date,
        "target_range": [0.4, 0.5],
        "target_is_not_a_guarantee": True,
        "observed_in_target_range": {
            "rank_calibration": 0.4 <= sequential["summary"]["hit_rate"] <= 0.5,
            "independent_counts": 0.4 <= independent["summary"]["hit_rate"] <= 0.5,
        },
        "rules": configurations,
        "rank_calibration": sequential,
        "independent_counts": independent,
        "evaluation_note": "两套规则事先固定、未根据本轮结果调整；现有数据已用于此前研究，因此不是全新盲测。无机器学习离线训练不等于不从历史做统计。",
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / f"固定规则TOP3对比_{prediction_date}_{periods}期"
    prefix.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    records = []
    for rank_row, independent_row in zip(sequential["rows"], independent["records"]):
        assert rank_row["index"] == independent_row["index"]
        assert rank_row["actual"] == independent_row["actual"]
        records.append({
            "date": rank_row["date"],
            "actual": rank_row["actual"],
            "rank_calibration_prediction": " ".join(rank_row["prediction"]),
            "rank_calibration_hit": rank_row["hit"],
            "selected_original_ranks": " ".join(map(str, rank_row["selected_ranks"])),
            "settled_calibration_count": rank_row["calibration_count"],
            "independent_prediction": " ".join(independent_row["prediction"]),
            "independent_hit": independent_row["hit"],
            "original_first3_prediction": " ".join(rank_row["original_first3"]),
            "original_first3_hit": rank_row["original_first3_hit"],
        })
    pd.DataFrame(records).to_csv(prefix.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    forecast = {
        "prediction_date": prediction_date,
        "data_through": sequential["data_through"],
        "input_sha256": report["input_sha256"],
        "code_sha256": report["code_sha256"],
        "rules": configurations,
        "predictions": {
            "rank_calibration": sequential["next_prediction"],
            "independent_counts": independent["next_prediction"],
        },
        "actual": None,
        "note": "尚未结算的下期预测；分数不代表真实开奖概率，历史回测不能保证未来命中率。",
    }
    (output / f"固定规则TOP3下期预测_{prediction_date}.json").write_text(
        json.dumps(forecast, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    for label, summary in (
        ("排名可靠度TOP3", sequential["summary"]),
        ("独立频率＋转移TOP3", independent["summary"]),
        ("原TOP5直接取前三", sequential["baselines"]["original_first3"]),
    ):
        print(f"{label}：{summary['hits']}/{summary['periods']} = {summary['hit_rate']:.2%}，最长不中{summary['max_miss_streak']}期")
    print(f"明细：{prefix.with_suffix('.csv')}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/lucky_numbers.csv")
    parser.add_argument("--periods", type=int, default=400)
    parser.add_argument("--output-dir", default="预测结果")
    args = parser.parse_args()
    run(args.data, args.periods, args.output_dir)
