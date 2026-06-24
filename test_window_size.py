"""测试PreciseTop15Predictor不同滚动窗口的命中率"""
import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values.tolist()

test_periods = min(400, len(df) - 50)
start_idx = len(df) - test_periods

windows = [5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30, 35, 40, 50, 75, 100, 150, 200, 0]
# 0 = 全量历史

print('=' * 70)
print('PreciseTop15Predictor 滚动窗口对比 (400期回测)')
print('=' * 70)
print()

results = []
for w in windows:
    predictor = PreciseTop15Predictor()
    hits = 0
    for i in range(start_idx, len(df)):
        if w == 0:
            train_data = df.iloc[:i]['number'].values
        else:
            lo = max(0, i - w)
            train_data = df.iloc[lo:i]['number'].values

        if len(train_data) < 3:
            continue

        predictions = predictor.predict(train_data, k=15)
        actual = df.iloc[i]['number']
        hit = actual in predictions
        if hit:
            hits += 1
        predictor.update_performance(predictions, actual)

    rate = hits / test_periods * 100
    ev = rate / 100 * 32 - (1 - rate / 100) * 15
    profit = ev * test_periods
    label = "全量" if w == 0 else f"{w}期"
    results.append((w, label, hits, rate, ev, profit))

# 排序找最优
results.sort(key=lambda x: -x[3])
best_w = results[0][0]
best_rate = results[0][3]

# 按窗口大小显示
results.sort(key=lambda x: (x[0] == 0, x[0]))

header = f"  {'窗口':>6}  {'命中':>4}  {'命中率':>7}  {'每注EV':>7}  {'400期利润':>9}  {'备注'}"
print(header)
print(f"  {'-' * 60}")

for w, label, hits, rate, ev, profit in results:
    note = ""
    if w == 25:
        note = "<-- 当前设置"
    if w == best_w:
        note = "★ 最优" + (note and " " + note or "")
    bar = "█" * int(rate)
    print(f"  {label:>6}  {hits:>4}  {rate:>6.2f}%  {ev:>+6.2f}元  {profit:>+8.0f}元  {note}")

print()
best_label = "全量" if best_w == 0 else f"{best_w}期"
print(f"最优窗口: {best_label} (命中率 {best_rate:.2f}%)")
print()

# 分段验证最优窗口的稳定性
print('=' * 70)
print(f'最优窗口 {best_label} 的分段稳定性验证')
print('=' * 70)

predictor = PreciseTop15Predictor()
hit_list = []
for i in range(start_idx, len(df)):
    if best_w == 0:
        train_data = df.iloc[:i]['number'].values
    else:
        lo = max(0, i - best_w)
        train_data = df.iloc[lo:i]['number'].values
    if len(train_data) < 3:
        hit_list.append(False)
        continue
    predictions = predictor.predict(train_data, k=15)
    actual = df.iloc[i]['number']
    hit = actual in predictions
    hit_list.append(hit)
    predictor.update_performance(predictions, actual)

seg = 50
for s in range(test_periods // seg):
    seg_hits = sum(hit_list[s * seg:(s + 1) * seg])
    seg_rate = seg_hits / seg * 100
    bar = "█" * int(seg_rate / 2.5)
    s_date = df.iloc[start_idx + s * seg]['date']
    e_date = df.iloc[start_idx + (s + 1) * seg - 1]['date']
    print(f"  {s * seg + 1:>3}-{(s + 1) * seg:>3}期 ({s_date}~{e_date}): {seg_hits}/{seg} = {seg_rate:.0f}% {bar}")

# 连败分析
from collections import Counter
streaks = []
c = 0
for h in hit_list:
    if not h:
        c += 1
    else:
        if c > 0:
            streaks.append(c)
        c = 0
if c > 0:
    streaks.append(c)

print(f"\n最大连败: {max(streaks)}期")
dist = Counter(streaks)
for l in sorted(dist.keys()):
    print(f"  {l}期连败: {dist[l]}次")
