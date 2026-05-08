"""最终验证: v5.1 TOP15高ROI方案 - 模拟GUI实际行为"""
import pandas as pd
import numpy as np
from collections import Counter
from precise_top15_predictor import PreciseTop15Predictor

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values
TEST_PERIODS = 300
start_idx = len(df) - TEST_PERIODS

PREDICT_K = 15
TRAIN_WINDOW = 25

# 模拟GUI行为（含update_performance）
predictor = PreciseTop15Predictor()
results = []
for i in range(start_idx, len(df)):
    lo = max(0, i - TRAIN_WINDOW)
    train = numbers[lo:i]
    preds = predictor.predict(train, k=PREDICT_K)
    actual = int(numbers[i])
    hit = actual in preds
    predictor.update_performance(preds, actual)
    results.append({'hit': hit, 'actual': actual})

hits = sum(1 for r in results if r['hit'])
hit_rate = hits / TEST_PERIODS * 100

# Fibonacci + max=20 + 暂停1期
fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
BASE_COST = 15
WIN_REWARD = 47
MAX_MUL = 20

# 基础策略
fib_idx = 0; bal = 0; min_bal = 0; total_bet = 0; total_win = 0
sl = 0; msl = 0; mc = 0; c = 0; h10 = 0
for r in results:
    mul = min(fib[min(fib_idx, len(fib)-1)], MAX_MUL)
    bet = BASE_COST * mul; total_bet += bet
    if mul >= MAX_MUL: h10 += 1
    if r['hit']:
        w = WIN_REWARD * mul; total_win += w; bal += (w-bet); fib_idx = 0; sl = 0; c = 0
    else:
        bal -= bet; fib_idx += 1; sl += bet; c += 1
        if sl > msl: msl = sl
        if c > mc: mc = c
    if bal < min_bal: min_bal = bal

roi = (total_win - total_bet) / total_bet * 100

# 暂停策略
fib_idx2 = 0; bal2 = 0; min_bal2 = 0; tb2 = 0; tw2 = 0
sl2 = 0; msl2 = 0; mc2 = 0; c2 = 0; h102 = 0
paused = False; bp = 0; pp = 0
for r in results:
    if paused: paused = False; pp += 1; continue
    bp += 1
    mul = min(fib[min(fib_idx2, len(fib)-1)], MAX_MUL)
    bet = BASE_COST * mul; tb2 += bet
    if mul >= MAX_MUL: h102 += 1
    if r['hit']:
        w = WIN_REWARD * mul; tw2 += w; bal2 += (w-bet); fib_idx2 = 0; sl2 = 0; c2 = 0; paused = True
    else:
        bal2 -= bet; fib_idx2 += 1; sl2 += bet; c2 += 1
        if sl2 > msl2: msl2 = sl2
        if c2 > mc2: mc2 = c2
    if bal2 < min_bal2: min_bal2 = bal2

roi2 = (tw2 - tb2) / tb2 * 100
rr2 = bal2 / abs(min_bal2) if min_bal2 < 0 else 0

print(f"{'='*60}")
print(f"v5.1 TOP{PREDICT_K}高ROI方案 - 300期验证结果")
print(f"{'='*60}")
print(f"命中率: {hits}/{TEST_PERIODS} = {hit_rate:.1f}%")
print(f"赔付规则: 买{PREDICT_K}个数, 成本{BASE_COST}元/倍, 赔付{WIN_REWARD}元/倍, 净利{WIN_REWARD-BASE_COST}元/倍, 赔率{WIN_REWARD/BASE_COST:.2f}倍")

print(f"\n--- 基础策略（Fib max={MAX_MUL}，不暂停）---")
print(f"  净利润: {bal:.0f}元, ROI: {roi:.1f}%")
print(f"  总投入: {total_bet:.0f}元, 总奖金: {total_win:.0f}元")
print(f"  最大回撤: {abs(min_bal):.0f}元, 连续不中总额: {msl:.0f}元")
print(f"  最长连败: {mc}期, 触顶({MAX_MUL}倍): {h10}次")
print(f"  风险收益比: {bal/abs(min_bal):.2f}")

print(f"\n--- 暂停策略（Fib max={MAX_MUL}，命中1停1）---")
print(f"  投注期: {bp}, 暂停期: {pp}")
print(f"  净利润: {bal2:.0f}元, ROI: {roi2:.1f}%")
print(f"  总投入: {tb2:.0f}元, 总奖金: {tw2:.0f}元")
print(f"  最大回撤: {abs(min_bal2):.0f}元, 连续不中总额: {msl2:.0f}元")
print(f"  最长连败: {mc2}期, 触顶({MAX_MUL}倍): {h102}次")
print(f"  风险收益比: {rr2:.2f}")

# 对比各版本
print(f"\n{'='*60}")
print(f"各版本对比")
print(f"{'='*60}")
print(f"{'版本':<25} {'命中率':>8} {'ROI':>8} {'净利润':>8} {'回撤':>8} {'风险比':>8}")
print(f"{'-'*66}")
versions = [
    ("v4.0 TOP15 Fib10 停1", "35.7%", "21.7%", "1635", "779", "2.10"),
    ("v5.0 TOP23 Fib10 停1", "50.7%", "8.5%", "537", "883", "0.61"),
    (f"v5.1 TOP15 Fib{MAX_MUL} 基础", f"{hit_rate:.1f}%", f"{roi:.1f}%", f"{bal:.0f}", f"{abs(min_bal):.0f}", f"{bal/abs(min_bal):.2f}"),
    (f"v5.1 TOP15 Fib{MAX_MUL} 停1", f"{hit_rate:.1f}%", f"{roi2:.1f}%", f"{bal2:.0f}", f"{abs(min_bal2):.0f}", f"{rr2:.2f}"),
]
for v in versions:
    print(f"  {v[0]:<23} {v[1]:>8} {v[2]:>8} {v[3]:>8} {v[4]:>8} {v[5]:>8}")
