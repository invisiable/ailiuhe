"""蒸馏TOP15 300期回测 - 收益详情"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from distill_top15_predictor import DistillTop15Predictor
from zodiac_top9_predictor import NUM_TO_ZODIAC_2026, ZODIAC_NUMS_2026

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values.tolist()
total = len(df)

test_periods = min(300, total - 30)
start_idx = total - test_periods

predictor = DistillTop15Predictor()

BASE_COST = 15
WIN_REWARD = 47

hit_records = []
total_cost = 0
total_reward = 0
balance = 0
min_balance = 0
max_balance = 0
max_drawdown = 0
peak = 0

sep = '=' * 100
dash = '-' * 100

print(f'数据总量: {total}期, 回测: 最近{test_periods}期')
print(f'投注规则: 成本{BASE_COST}元/倍, 命中赔{WIN_REWARD}元/倍, 净利={WIN_REWARD-BASE_COST}元/倍')
print(sep)
print(f"{'期号':>5} {'日期':>12} {'实际':>4} {'结果':>4} {'本期收益':>8} {'累计收益':>8} {'保留':>4} {'补充':>4}")
print(dash)

for i in range(start_idx, total):
    hist = numbers[:i]
    actual = numbers[i]
    final_nums, details, top9_z = predictor.predict_with_details(hist, top_n=15)
    hit = actual in final_nums
    hit_records.append(hit)

    total_cost += BASE_COST
    period_profit = -BASE_COST
    if hit:
        total_reward += WIN_REWARD
        period_profit = WIN_REWARD - BASE_COST

    balance += period_profit
    peak = max(peak, balance)
    drawdown = peak - balance
    max_drawdown = max(max_drawdown, drawdown)
    min_balance = min(min_balance, balance)
    max_balance = max(max_balance, balance)

    pi = i - start_idx + 1
    mark = '✅' if hit else '❌'
    date_str = str(df.iloc[i]['date'])
    print(f'{pi:>5} {date_str:>12} {actual:>4} {mark:>4} {period_profit:>+8} {balance:>+8} {details["kept_count"]:>4} {details["supplement_count"]:>4}')

hits = sum(hit_records)
hit_rate = hits / test_periods * 100
net_profit = total_reward - total_cost
roi = net_profit / total_cost * 100

print(f'\n{sep}')
print('回测结果汇总')
print(sep)
print(f'命中: {hits}/{test_periods} = {hit_rate:.1f}%')
print(f'随机TOP15基线: 30.6%')
print(f'总投入: {total_cost}元')
print(f'总回报: {total_reward}元')
print(f'净利润: {net_profit:+d}元')
print(f'ROI: {roi:+.1f}%')
print(f'最大回撤: {max_drawdown}元')
print(f'最低余额: {min_balance}元')
print(f'最高余额: {max_balance}元')

# 分段统计
seg = 50
n_segs = test_periods // seg
print(f'\n分段统计(每{seg}期):')
for s in range(n_segs):
    seg_hits = sum(hit_records[s*seg:(s+1)*seg])
    seg_rate = seg_hits / seg * 100
    seg_profit = seg_hits * WIN_REWARD - seg * BASE_COST
    bar = '█' * int(seg_rate / 5) + '░' * (20 - int(seg_rate / 5))
    print(f'  {s*seg+1:>3}-{(s+1)*seg:>3}: {seg_hits}/{seg}={seg_rate:.0f}% {bar} 收益:{seg_profit:+d}元')

# 连续miss统计
max_miss = cur_miss = 0
streaks = []
c = 0
for h in hit_records:
    if not h:
        cur_miss += 1
        max_miss = max(max_miss, cur_miss)
        c += 1
    else:
        cur_miss = 0
        if c > 0:
            streaks.append(c)
        c = 0
if c > 0:
    streaks.append(c)
print(f'\n最大连续miss: {max_miss}期 (连续亏损: {max_miss * BASE_COST}元)')
ge2 = sum(1 for s in streaks if s >= 2)
ge3 = sum(1 for s in streaks if s >= 3)
ge5 = sum(1 for s in streaks if s >= 5)
print(f'≥2期连miss: {ge2}次, ≥3期: {ge3}次, ≥5期: {ge5}次')

# 盈亏平衡点
breakeven = BASE_COST / WIN_REWARD * 100
print(f'\n盈亏平衡命中率: {breakeven:.1f}%')
result_str = "盈利" if hit_rate > breakeven else "亏损"
print(f'实际命中率: {hit_rate:.1f}% ({result_str})')
