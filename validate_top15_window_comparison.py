# -*- coding: utf-8 -*-
"""
TOP15 预测窗口期对比验证
比较不同训练窗口（20/30/50/100/150/全量历史）对命中率的影响
"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from precise_top15_predictor import PreciseTop15Predictor

DATA      = 'data/lucky_numbers.csv'
TEST_N    = 300    # 回测期数
MIN_TRAIN = 30     # 最少历史数据才开始预测
BASE_BET  = 15
WIN_R     = 47

df      = pd.read_csv(DATA, encoding='utf-8-sig')
numbers = df['number'].values
n       = len(numbers)
start   = max(MIN_TRAIN, n - TEST_N)
print(f"\n数据总量: {n}期   回测期数: {n - start}期   ({df.iloc[start]['date']} ~ {df.iloc[-1]['date']})\n")

# ─── 窗口配置 ───────────────────────────────────────────────────────────────
WINDOWS = [
    ('全量历史(当前方式)', None),   # None = 用 i 前全部数据
    ('最近150期',         150),
    ('最近100期',         100),
    ('最近 50期',          50),
    ('最近 30期',          30),
    ('最近 20期',          20),
]

def run_window(window_size):
    """滚动回测一个窗口配置，返回每期详情列表"""
    predictor = PreciseTop15Predictor()
    records   = []
    for i in range(start, n):
        if window_size is None:
            train = numbers[:i]
        else:
            lo    = max(0, i - window_size)
            train = numbers[lo:i]
        if len(train) < 5:          # 训练数据太少则跳过
            continue
        preds = predictor.predict(train)
        actual = numbers[i]
        predictor.update_performance(preds, actual)
        records.append({
            'period': i - start + 1,
            'date':   df.iloc[i]['date'],
            'actual': actual,
            'hit':    actual in preds,
        })
    return records

def stats(records):
    hits    = sum(r['hit'] for r in records)
    total   = len(records)
    profit  = total_p = peak = maxdd = 0
    for r in records:
        if r['hit']: total_p += WIN_R - BASE_BET
        else:        total_p -= BASE_BET
        if total_p > peak: peak = total_p
        dd = peak - total_p
        if dd > maxdd: maxdd = dd
    roi = total_p / (total * BASE_BET) * 100 if total > 0 else 0
    hr  = hits / total * 100 if total > 0 else 0
    rr  = peak / maxdd if maxdd > 0 else float('inf')
    return total, hits, hr, total_p, roi, maxdd, rr

# ─── 逐期连续命中趋势（检测是否稳定） ───────────────────────────────────────
def rolling_hit_rate(records, window=30):
    """按30期滑动窗口的命中率序列"""
    hits = [1 if r['hit'] else 0 for r in records]
    return [sum(hits[max(0,i-window):i+1])/min(i+1, window) for i in range(len(hits))]

# ─── 执行所有窗口回测 ────────────────────────────────────────────────────────
results = {}
print("正在逐窗口滚动回测，请稍候…\n")
for label, ws in WINDOWS:
    print(f"  [{label}] 回测中…", end='', flush=True)
    recs = run_window(ws)
    results[label] = recs
    t, h, hr, tot, roi, dd, rr = stats(recs)
    print(f" 命中率 {hr:.1f}%  ROI {roi:+.2f}%  盈亏 {tot:+.0f}元  回撤 {dd:.0f}元")

# ─── 汇总对比表 ──────────────────────────────────────────────────────────────
print(f"\n{'='*84}")
print("窗口对比汇总（固投 15 元 / 命中返 47 元）")
print(f"{'='*84}")
print(f"  {'窗口配置':<18} {'期数':>5} {'命中':>5} {'命中率':>7} {'盈亏':>9} {'ROI':>9} {'最大回撤':>9} {'风险收益':>8}")
print("  " + "-"*79)
baseline_hr = None
for label, ws in WINDOWS:
    recs                       = results[label]
    t, h, hr, tot, roi, dd, rr = stats(recs)
    if baseline_hr is None: baseline_hr = hr
    delta = f"({hr - baseline_hr:+.1f}%)" if baseline_hr != hr else "(基准)"
    print(f"  {label:<18} {t:>5} {h:>5} {hr:>6.1f}%{delta:>8} {tot:>+9.0f}元 {roi:>+8.2f}% {dd:>9.0f}元 {rr:>8.2f}")

# ─── 按月度命中率细分（最优vs全量）──────────────────────────────────────────
def monthly_hit_rate(records):
    from collections import defaultdict
    import re
    monthly = defaultdict(lambda: [0, 0])
    for r in records:
        m = re.match(r'(\d{4})/(\d{1,2})/', str(r['date']))
        if m:
            key = f"{m.group(1)}/{int(m.group(2)):02d}"
            monthly[key][1] += 1
            if r['hit']: monthly[key][0] += 1
    return {k: (h, t, h/t*100) for k, (h, t) in sorted(monthly.items()) if t > 0}

# 找出命中率最高的窗口
best_label = max(WINDOWS, key=lambda lw: stats(results[lw[0]])[2])[0]
print(f"\n最优窗口: 【{best_label}】")

print(f"\n{'='*84}")
print(f"月度命中率对比：全量历史(当前方式) vs {best_label}")
print(f"{'='*84}")
full_monthly = monthly_hit_rate(results['全量历史(当前方式)'])
best_monthly = monthly_hit_rate(results[best_label])
all_months   = sorted(set(list(full_monthly) + list(best_monthly)))
print(f"  {'月份':<10} {'全量历史':>10} {'最优窗口':>10} {'差值':>8}")
print("  " + "-"*42)
for m in all_months:
    fh, ft, fhr = full_monthly.get(m, (0, 0, 0))
    bh, bt, bhr = best_monthly.get(m,  (0, 0, 0))
    diff = bhr - fhr
    mark = " ↑" if diff > 5 else (" ↓" if diff < -5 else "")
    print(f"  {m:<10} {fhr:>8.1f}%({ft}) {bhr:>8.1f}%({bt}) {diff:>+7.1f}%{mark}")

# ─── 最大连亏统计 ─────────────────────────────────────────────────────────
print(f"\n{'='*84}")
print("最大连亏对比")
print(f"{'='*84}")
for label, ws in WINDOWS:
    recs = results[label]
    cur_l = max_l = 0
    for r in recs:
        if not r['hit']: cur_l += 1; max_l = max(max_l, cur_l)
        else: cur_l = 0
    print(f"  {label:<20} 最大连亏: {max_l}期")

print(f"\n✅ 验证完成！\n")
