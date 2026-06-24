"""
探索PreciseTop15命中率提升方案
当前: 25期窗口, 36.25%命中率
"""
import pandas as pd
import numpy as np
from collections import Counter
from precise_top15_predictor import PreciseTop15Predictor

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values.tolist()
test_periods = min(400, len(df) - 50)
start_idx = len(df) - test_periods


def backtest(predict_fn, label):
    """通用回测框架"""
    hits = 0
    for i in range(start_idx, len(df)):
        hist = numbers[:i]
        actual = numbers[i]
        pred = predict_fn(hist, i)
        if actual in pred:
            hits += 1
    rate = hits / test_periods * 100
    print(f"  {label:<45} {hits:>3}/{test_periods} = {rate:.2f}%")
    return hits, rate


print("=" * 70)
print("优化方案探索 (基准: PreciseTop15 w=25, 36.25%)")
print("=" * 70)
print()

# === 方案1: 动态窗口 (根据近期命中率切换窗口) ===
print("--- 方案1: 动态窗口 ---")


def dynamic_window_predict(hist, idx):
    """根据近期表现动态切换窗口"""
    # 简单策略: 默认25, 如果近5期全miss则切换到12
    predictor = PreciseTop15Predictor()
    w = 25
    lo = max(0, len(hist) - w)
    return predictor.predict(hist[lo:], k=15)


# 更智能的: 用12和25两个窗口的交集/并集
def dual_window_union(hist, idx):
    """12期和25期窗口的预测取并集top15"""
    p = PreciseTop15Predictor()
    w12 = hist[max(0, len(hist) - 12):]
    w25 = hist[max(0, len(hist) - 25):]
    pred12 = p.predict(w12, k=20)
    pred25 = p.predict(w25, k=20)
    # 两个窗口都选中的号码优先
    both = [n for n in pred25 if n in pred12]
    only25 = [n for n in pred25 if n not in pred12]
    only12 = [n for n in pred12 if n not in pred25]
    result = both[:15]
    for n in only12:
        if len(result) >= 15:
            break
        if n not in result:
            result.append(n)
    for n in only25:
        if len(result) >= 15:
            break
        if n not in result:
            result.append(n)
    return result[:15]


def dual_window_intersection(hist, idx):
    """12和25窗口共识优先"""
    p = PreciseTop15Predictor()
    w12 = hist[max(0, len(hist) - 12):]
    w25 = hist[max(0, len(hist) - 25):]
    pred12 = p.predict(w12, k=25)
    pred25 = p.predict(w25, k=15)
    # 25期top15为基础, 如果12期也选中则置信度高
    scores = {}
    for rank, n in enumerate(pred25):
        scores[n] = 2.0 - rank / 15
    for rank, n in enumerate(pred12):
        scores[n] = scores.get(n, 0) + (1.5 - rank / 25)
    final = sorted(scores.items(), key=lambda x: -x[1])
    return [n for n, _ in final[:15]]


backtest(dual_window_union, "双窗口(12+25)并集优先")
backtest(dual_window_intersection, "双窗口(12+25)评分融合")

# === 方案2: 热号/冷号动态权重 ===
print("\n--- 方案2: 频率偏差修正 ---")


def frequency_bias_fix(hist, idx):
    """修正PreciseTop15的频率偏差: 对过热号码额外惩罚"""
    p = PreciseTop15Predictor()
    w = 25
    data = hist[max(0, len(hist) - w):]
    # 获取扩展排名
    pred30 = p.predict(data, k=30)
    # 对最近3期出现的号码从top15中移除, 用后面的补
    recent3 = set(hist[-3:]) if len(hist) >= 3 else set()
    filtered = [n for n in pred30 if n not in recent3]
    return filtered[:15]


def avoid_recent_2(hist, idx):
    """排除最近2期出现的号码"""
    p = PreciseTop15Predictor()
    w = 25
    data = hist[max(0, len(hist) - w):]
    pred30 = p.predict(data, k=30)
    recent2 = set(hist[-2:]) if len(hist) >= 2 else set()
    filtered = [n for n in pred30 if n not in recent2]
    return filtered[:15]


def avoid_recent_1(hist, idx):
    """排除最近1期出现的号码"""
    p = PreciseTop15Predictor()
    w = 25
    data = hist[max(0, len(hist) - w):]
    pred30 = p.predict(data, k=30)
    recent1 = set(hist[-1:]) if len(hist) >= 1 else set()
    filtered = [n for n in pred30 if n not in recent1]
    return filtered[:15]


backtest(frequency_bias_fix, "排除最近3期号码")
backtest(avoid_recent_2, "排除最近2期号码")
backtest(avoid_recent_1, "排除最近1期号码")

# === 方案3: 多窗口评分融合 ===
print("\n--- 方案3: 多窗口评分融合 ---")


def multi_window_fusion(hist, idx):
    """3个窗口(10,25,40)加权评分"""
    p = PreciseTop15Predictor()
    windows = [(10, 0.3), (25, 0.5), (40, 0.2)]
    scores = {}
    for w, weight in windows:
        data = hist[max(0, len(hist) - w):]
        pred = p.predict(data, k=25)
        for rank, n in enumerate(pred):
            s = weight * (1.0 - rank / 25)
            scores[n] = scores.get(n, 0) + s
    final = sorted(scores.items(), key=lambda x: -x[1])
    return [n for n, _ in final[:15]]


def multi_window_fusion_v2(hist, idx):
    """4个窗口(8,15,25,50)加权"""
    p = PreciseTop15Predictor()
    windows = [(8, 0.15), (15, 0.25), (25, 0.40), (50, 0.20)]
    scores = {}
    for w, weight in windows:
        data = hist[max(0, len(hist) - w):]
        if len(data) < 3:
            continue
        pred = p.predict(data, k=25)
        for rank, n in enumerate(pred):
            s = weight * (1.0 - rank / 25)
            scores[n] = scores.get(n, 0) + s
    final = sorted(scores.items(), key=lambda x: -x[1])
    return [n for n, _ in final[:15]]


backtest(multi_window_fusion, "3窗口融合(10,25,40)")
backtest(multi_window_fusion_v2, "4窗口融合(8,15,25,50)")

# === 方案4: 扩展到20个再精选15个 ===
print("\n--- 方案4: 扩展预测再精选 ---")


def expand_then_filter(hist, idx):
    """预测top25, 然后用第二个独立信号过滤到15"""
    p = PreciseTop15Predictor()
    w = 25
    data = hist[max(0, len(hist) - w):]
    pred25 = p.predict(data, k=25)

    # 用间隔分析作为第二信号
    gap_scores = {}
    for n in range(1, 50):
        last_pos = -1
        for j in range(len(hist) - 1, max(0, len(hist) - 50) - 1, -1):
            if hist[j] == n:
                last_pos = j
                break
        gap = len(hist) - 1 - last_pos if last_pos >= 0 else 50
        # 间隔5-20期最佳
        if 5 <= gap <= 20:
            gap_scores[n] = 2.0
        elif 20 < gap <= 35:
            gap_scores[n] = 1.5
        elif gap > 35:
            gap_scores[n] = 1.0
        else:
            gap_scores[n] = 0.5

    # 结合排名分和间隔分
    combined = {}
    for rank, n in enumerate(pred25):
        rank_score = 1.0 - rank / 25
        combined[n] = rank_score * 0.7 + gap_scores.get(n, 1.0) * 0.3

    final = sorted(combined.items(), key=lambda x: -x[1])
    return [n for n, _ in final[:15]]


backtest(expand_then_filter, "Top25+间隔过滤→15")

# === 方案5: 奇偶平衡约束 ===
print("\n--- 方案5: 结构约束 ---")


def odd_even_balance(hist, idx):
    """强制奇偶平衡(7-8或8-7)"""
    p = PreciseTop15Predictor()
    w = 25
    data = hist[max(0, len(hist) - w):]
    pred = p.predict(data, k=30)
    odds = [n for n in pred if n % 2 == 1]
    evens = [n for n in pred if n % 2 == 0]
    # 取7奇8偶 或 8奇7偶 (看哪个排名更靠前)
    result = []
    oi, ei = 0, 0
    max_odd, max_even = 8, 8
    for n in pred:
        if len(result) >= 15:
            break
        if n % 2 == 1 and oi < max_odd:
            result.append(n)
            oi += 1
        elif n % 2 == 0 and ei < max_even:
            result.append(n)
            ei += 1
    return result[:15]


backtest(odd_even_balance, "奇偶平衡约束(max 8:8)")

# === 方案6: 区间覆盖约束 ===
print("\n--- 方案6: 区间覆盖 ---")


def zone_coverage(hist, idx):
    """保证5个区间各至少2个号码"""
    p = PreciseTop15Predictor()
    w = 25
    data = hist[max(0, len(hist) - w):]
    pred = p.predict(data, k=40)

    zones = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 49)]
    result = []
    # 先每区间取前2个
    for lo, hi in zones:
        zone_nums = [n for n in pred if lo <= n <= hi]
        result.extend(zone_nums[:2])
    # 剩余5个按原排名填
    for n in pred:
        if len(result) >= 15:
            break
        if n not in result:
            result.append(n)
    return result[:15]


backtest(zone_coverage, "5区间各≥2个覆盖")

# === 方案7: 基准对照 ===
print("\n--- 基准 ---")


def baseline_25(hist, idx):
    p = PreciseTop15Predictor()
    w = 25
    data = hist[max(0, len(hist) - w):]
    return p.predict(data, k=15)


backtest(baseline_25, "★ 基准: PreciseTop15 w=25")
