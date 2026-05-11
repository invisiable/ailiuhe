"""蒸馏TOP15 × Fibonacci投注策略 300期回测"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from distill_top15_predictor import DistillTop15Predictor
from zodiac_top9_predictor import NUM_TO_ZODIAC_2026

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values.tolist()
total = len(df)
test_periods = min(360, total - 30)
start_idx = total - test_periods

predictor = DistillTop15Predictor()

WIN_REWARD_PER = 47  # 命中赔47元/倍

# Fibonacci数列
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

fib_index = 0
hit_records = []
balance = 0
peak = 0
max_drawdown = 0
min_balance = 0
total_cost = 0
total_reward = 0

sep = '=' * 120
dash = '-' * 120

print(f'蒸馏TOP15 × Fibonacci投注策略 300期回测')
print(sep)
print(f'规则: 正常模式买15个号(成本15元/倍), 连miss≥2自动扩展到20个号(成本20元/倍)')
print(f'Fibonacci倍投: 不中→Fib序列递进, 命中→重置为1倍')
print(f'Fib序列: {FIB}')
print(f'命中赔付: 47元/倍')
print(sep)
print(f'{"期号":>5} {"日期":>12} {"实际":>4} {"结果":>4} {"K":>3} {"Fib倍":>5} {"单位成本":>8} {"投注额":>8} {"赔付":>8} {"本期收益":>8} {"累计收益":>8} {"回撤":>6}')
print(dash)

for i in range(start_idx, total):
    hist = numbers[:i]
    actual = numbers[i]
    
    current_k = predictor._get_current_k()
    final_nums, details, top9_z = predictor.predict_with_details(hist, top_n=current_k)
    hit = actual in final_nums
    hit_records.append(hit)
    
    # Fibonacci倍数
    fib_mul = min(FIB[min(fib_index, len(FIB) - 1)], 13)  # 最高13倍
    
    # 成本 = 号码数 × 倍数
    unit_cost = current_k  # 每倍成本
    bet_amount = unit_cost * fib_mul
    total_cost += bet_amount
    
    if hit:
        reward = WIN_REWARD_PER * fib_mul
        total_reward += reward
        period_profit = reward - bet_amount
        fib_index = 0  # 命中重置
    else:
        reward = 0
        period_profit = -bet_amount
        fib_index = min(fib_index + 1, len(FIB) - 1)  # 递进
    
    balance += period_profit
    peak = max(peak, balance)
    drawdown = peak - balance
    max_drawdown = max(max_drawdown, drawdown)
    min_balance = min(min_balance, balance)
    
    predictor.record_result(hit)
    
    pi = i - start_idx + 1
    mark = '✅' if hit else '❌'
    date_str = str(df.iloc[i]['date'])
    dd_str = f'{drawdown}' if drawdown > 0 else ''
    next_fib = min(FIB[min(fib_index, len(FIB)-1)], 13)
    print(f'{pi:>5} {date_str:>12} {actual:>4} {mark:>4} {current_k:>3} {fib_mul:>5}→{"重置" if hit else next_fib:>3} {unit_cost:>8} {bet_amount:>8} {reward:>8} {period_profit:>+8} {balance:>+8} {dd_str:>6}')

# 汇总
hits = sum(hit_records)
hit_rate = hits / test_periods * 100
net_profit = total_reward - total_cost
roi = net_profit / total_cost * 100

print(f'\n{sep}')
print(f'回测结果汇总')
print(sep)
print(f'命中: {hits}/{test_periods} = {hit_rate:.1f}%')
print(f'总投入: {total_cost}元')
print(f'总回报: {total_reward}元')
print(f'净利润: {net_profit:+d}元')
print(f'ROI: {roi:+.1f}%')
print(f'最大回撤: {max_drawdown}元')
print(f'最低余额: {min_balance}元')
print(f'最高余额: {peak}元')
print(f'最终余额: {balance}元')

# Fibonacci倍数分布
print(f'\nFibonacci倍数使用分布:')
fib_idx2 = 0
fib_usage = {}
predictor2 = DistillTop15Predictor()
for idx in range(test_periods):
    i = start_idx + idx
    hist = numbers[:i]
    actual = numbers[i]
    k = predictor2._get_current_k()
    final_nums, _, _ = predictor2.predict_with_details(hist, top_n=k)
    hit = actual in final_nums
    
    fm = min(FIB[min(fib_idx2, len(FIB) - 1)], 13)  # 最高13倍
    if fm not in fib_usage:
        fib_usage[fm] = [0, 0]
    fib_usage[fm][1] += 1
    if hit:
        fib_usage[fm][0] += 1
        fib_idx2 = 0
    else:
        fib_idx2 = min(fib_idx2 + 1, len(FIB) - 1)
    predictor2.record_result(hit)

for fm in sorted(fib_usage.keys()):
    h, t = fib_usage[fm]
    print(f'  {fm}倍: {t}期, 命中{h}/{t}={h/t*100:.1f}%')

# 连续miss
max_miss = cur_miss = 0
for h in hit_records:
    if not h:
        cur_miss += 1
        max_miss = max(max_miss, cur_miss)
    else:
        cur_miss = 0
print(f'\n最大连续miss: {max_miss}期')

# 分段统计
seg = 50
n_segs = test_periods // seg
print(f'\n分段统计(每{seg}期):')
predictor3 = DistillTop15Predictor()
fib_idx3 = 0
seg_costs = [0] * n_segs
seg_rewards = [0] * n_segs
seg_hits_list = [0] * n_segs

for idx in range(test_periods):
    i = start_idx + idx
    hist = numbers[:i]
    actual = numbers[i]
    k = predictor3._get_current_k()
    final_nums, _, _ = predictor3.predict_with_details(hist, top_n=k)
    hit = actual in final_nums
    
    fm = min(FIB[min(fib_idx3, len(FIB) - 1)], 13)  # 最高13倍
    bet = k * fm
    s = idx // seg
    if s < n_segs:
        seg_costs[s] += bet
        if hit:
            seg_rewards[s] += 47 * fm
            seg_hits_list[s] += 1
    
    if hit:
        fib_idx3 = 0
    else:
        fib_idx3 = min(fib_idx3 + 1, len(FIB) - 1)
    predictor3.record_result(hit)

for s in range(n_segs):
    seg_profit = seg_rewards[s] - seg_costs[s]
    seg_roi = seg_profit / seg_costs[s] * 100 if seg_costs[s] > 0 else 0
    print(f'  {s*seg+1:>3}-{(s+1)*seg:>3}: 命中{seg_hits_list[s]}/{seg} 投入{seg_costs[s]}元 回报{seg_rewards[s]}元 利润{seg_profit:+d}元 ROI{seg_roi:+.1f}%')
