"""
深入验证超越基准的方案:
1. 双窗口评分融合 36.50% (+0.25%)
2. 奇偶平衡约束 37.00% (+0.75%)

以及进一步组合优化
"""
import pandas as pd
import numpy as np
from collections import Counter
from precise_top15_predictor import PreciseTop15Predictor

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values.tolist()
test_periods = min(400, len(df) - 50)
start_idx = len(df) - test_periods


def backtest_detail(predict_fn, label):
    """回测并返回命中列表"""
    hits = []
    for i in range(start_idx, len(df)):
        hist = numbers[:i]
        actual = numbers[i]
        pred = predict_fn(hist, i)
        hits.append(actual in pred)
    total = sum(hits)
    rate = total / test_periods * 100
    # 最大连败
    streaks = []
    c = 0
    for h in hits:
        if not h:
            c += 1
        else:
            if c > 0:
                streaks.append(c)
            c = 0
    if c > 0:
        streaks.append(c)
    max_streak = max(streaks) if streaks else 0
    return total, rate, max_streak, hits


# === 奇偶平衡的变体测试 ===
print("=" * 70)
print("方案A: 奇偶平衡约束 - 不同配比")
print("=" * 70)


def odd_even_max(max_per_type):
    def predict(hist, idx):
        p = PreciseTop15Predictor()
        data = hist[max(0, len(hist) - 25):]
        pred = p.predict(data, k=35)
        result = []
        oi, ei = 0, 0
        for n in pred:
            if len(result) >= 15:
                break
            if n % 2 == 1 and oi < max_per_type:
                result.append(n)
                oi += 1
            elif n % 2 == 0 and ei < max_per_type:
                result.append(n)
                ei += 1
        return result[:15]
    return predict


for mx in [7, 8, 9, 10, 11, 12]:
    h, r, ms, _ = backtest_detail(odd_even_max(mx), f"奇偶各max{mx}")
    print(f"  奇偶各max{mx}: {h}/400 = {r:.2f}%, 最大连败{ms}")

# === 双窗口评分融合的变体 ===
print()
print("=" * 70)
print("方案B: 双窗口评分融合 - 不同窗口组合")
print("=" * 70)


def dual_window_score(w1, w2, wt1=0.4, wt2=0.6):
    def predict(hist, idx):
        p = PreciseTop15Predictor()
        d1 = hist[max(0, len(hist) - w1):]
        d2 = hist[max(0, len(hist) - w2):]
        p1 = p.predict(d1, k=25)
        p2 = p.predict(d2, k=25)
        scores = {}
        for rank, n in enumerate(p1):
            scores[n] = scores.get(n, 0) + wt1 * (1.0 - rank / 25)
        for rank, n in enumerate(p2):
            scores[n] = scores.get(n, 0) + wt2 * (1.0 - rank / 25)
        final = sorted(scores.items(), key=lambda x: -x[1])
        return [n for n, _ in final[:15]]
    return predict


combos = [(8, 25), (10, 25), (12, 25), (15, 25), (12, 30), (10, 30), (8, 30)]
for w1, w2 in combos:
    h, r, ms, _ = backtest_detail(dual_window_score(w1, w2), f"窗口{w1}+{w2}")
    print(f"  窗口{w1:>2}+{w2:>2} (权重0.4/0.6): {h}/400 = {r:.2f}%, 最大连败{ms}")

# === 组合方案: 奇偶平衡 + 双窗口 ===
print()
print("=" * 70)
print("方案C: 组合 - 双窗口评分 + 奇偶约束")
print("=" * 70)


def combo_dual_oddeven(w1, w2, wt1, wt2, max_oe):
    def predict(hist, idx):
        p = PreciseTop15Predictor()
        d1 = hist[max(0, len(hist) - w1):]
        d2 = hist[max(0, len(hist) - w2):]
        p1 = p.predict(d1, k=30)
        p2 = p.predict(d2, k=30)
        scores = {}
        for rank, n in enumerate(p1):
            scores[n] = scores.get(n, 0) + wt1 * (1.0 - rank / 30)
        for rank, n in enumerate(p2):
            scores[n] = scores.get(n, 0) + wt2 * (1.0 - rank / 30)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        # 奇偶约束
        result = []
        oi, ei = 0, 0
        for n, s in ranked:
            if len(result) >= 15:
                break
            if n % 2 == 1 and oi < max_oe:
                result.append(n)
                oi += 1
            elif n % 2 == 0 and ei < max_oe:
                result.append(n)
                ei += 1
        return result[:15]
    return predict


configs = [
    (12, 25, 0.4, 0.6, 8),
    (12, 25, 0.3, 0.7, 8),
    (12, 25, 0.5, 0.5, 8),
    (10, 25, 0.4, 0.6, 8),
    (8, 25, 0.4, 0.6, 8),
    (12, 25, 0.4, 0.6, 9),
    (12, 25, 0.4, 0.6, 10),
]
for w1, w2, wt1, wt2, mx in configs:
    h, r, ms, _ = backtest_detail(
        combo_dual_oddeven(w1, w2, wt1, wt2, mx),
        f"双窗口{w1}+{w2}({wt1}/{wt2})+奇偶max{mx}")
    print(f"  w={w1:>2}+{w2:>2} wt={wt1}/{wt2} 奇偶max{mx}: {h}/400 = {r:.2f}%, 连败{ms}")

# === 方案D: 单窗口25 + 奇偶约束(最简方案) ===
print()
print("=" * 70)
print("方案D: 基准w=25 + 奇偶约束 (最简优化)")
print("=" * 70)


def baseline_oddeven(max_oe):
    def predict(hist, idx):
        p = PreciseTop15Predictor()
        data = hist[max(0, len(hist) - 25):]
        pred = p.predict(data, k=35)
        result = []
        oi, ei = 0, 0
        for n in pred:
            if len(result) >= 15:
                break
            if n % 2 == 1 and oi < max_oe:
                result.append(n)
                oi += 1
            elif n % 2 == 0 and ei < max_oe:
                result.append(n)
                ei += 1
        return result[:15]
    return predict


h, r, ms, hits_base = backtest_detail(
    lambda hist, idx: PreciseTop15Predictor().predict(hist[max(0, len(hist)-25):], k=15),
    "基准w=25")
print(f"  基准w=25 (无约束):       {h}/400 = {r:.2f}%, 连败{ms}")

h, r, ms, hits_oe8 = backtest_detail(baseline_oddeven(8), "基准+奇偶max8")
print(f"  基准w=25 + 奇偶max8:    {h}/400 = {r:.2f}%, 连败{ms}")

h, r, ms, hits_oe9 = backtest_detail(baseline_oddeven(9), "基准+奇偶max9")
print(f"  基准w=25 + 奇偶max9:    {h}/400 = {r:.2f}%, 连败{ms}")

# 分段对比
print()
print("=" * 70)
print("奇偶max8 分段稳定性 vs 基准")
print("=" * 70)

seg = 50
for s in range(test_periods // seg):
    base_h = sum(hits_base[s*seg:(s+1)*seg])
    oe8_h = sum(hits_oe8[s*seg:(s+1)*seg])
    diff = oe8_h - base_h
    s_date = df.iloc[start_idx + s*seg]['date']
    e_date = df.iloc[start_idx + (s+1)*seg - 1]['date']
    mark = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
    print(f"  {s*seg+1:>3}-{(s+1)*seg:>3}期 ({s_date}~{e_date}): 基准{base_h} vs 奇偶{oe8_h} {mark}{abs(diff)}")

# 差异分析
both = sum(1 for a, b in zip(hits_base, hits_oe8) if a and b)
only_base = sum(1 for a, b in zip(hits_base, hits_oe8) if a and not b)
only_oe = sum(1 for a, b in zip(hits_base, hits_oe8) if not a and b)
neither = sum(1 for a, b in zip(hits_base, hits_oe8) if not a and not b)
print(f"\n差异分析:")
print(f"  两者都中: {both}")
print(f"  仅基准中: {only_base}")
print(f"  仅奇偶中: {only_oe}")
print(f"  都未中:   {neither}")
print(f"  净增: {only_oe - only_base}")
