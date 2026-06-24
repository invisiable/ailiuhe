# -*- coding: utf-8 -*-
"""
尾号预测器 - 持仓3期策略验证（最近300期）

策略规则：
  1. 生成一组预测（3个尾数 → 约15个号码）
  2. 该预测最多持续3期
  3. 若3期内命中 → 下一期重新生成
  4. 若3期全未命中 → 第4期重新生成

输出：每期详情 + 汇总统计
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import csv
from tail_digit_grid_predictor import TailDigitGridPredictor, TAIL_DIGIT_NUMBERS, number_to_tail

# ---------- 读取数据 ----------
draws = []
with open('data/lucky_numbers.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        draws.append({'date': row['date'], 'number': int(row['number']), 'animal': row['animal']})

TEST_PERIODS = 300
HOLD_PERIODS = 3   # 每组预测最多持续3期

start_idx = len(draws) - TEST_PERIODS

# ---------- 回测逻辑 ----------
predictor = TailDigitGridPredictor()

# 用前 start_idx 期预热
for d in draws[:start_idx]:
    pass  # predictor 是无状态模式，直接按滑窗调用

results = []
hits = 0          # TOP3尾数命中次数（单期）
hold_hits = 0     # 持仓内命中次数（一组至少命中1期）
groups_total = 0  # 总共生成多少组预测

current_tails = None   # 当前持仓的尾数预测
hold_remaining = 0     # 当前这组还剩几期可用
group_id = 0           # 当前组编号
group_hit_in_this_hold = False

for i in range(start_idx, len(draws)):
    hist_numbers = [d['number'] for d in draws[:i]]
    actual = draws[i]['number']
    actual_tail = number_to_tail(actual)

    # 需要重新生成预测（新的一组）
    if hold_remaining == 0:
        current_tails = predictor.predict(hist_numbers, top_n=3)
        hold_remaining = HOLD_PERIODS
        group_id += 1
        groups_total += 1
        group_hit_in_this_hold = False

    # 判断命中
    hit = actual_tail in current_tails
    is_new_group = (hold_remaining == HOLD_PERIODS)  # 本期是否是该组第1期

    # 本组内命中记录
    if hit:
        group_hit_in_this_hold = True

    # 本期统计
    predicted_numbers = []
    for t in current_tails:
        predicted_numbers.extend(TAIL_DIGIT_NUMBERS[t])

    results.append({
        'seq':      i - start_idx + 1,
        'date':     draws[i]['date'],
        'actual':   actual,
        'animal':   draws[i]['animal'],
        'tails':    list(current_tails),
        'numbers':  predicted_numbers,
        'hit':      hit,
        'group_id': group_id,
        'hold_pos': HOLD_PERIODS - hold_remaining + 1,   # 当前是组内第几期(1/2/3)
        'is_new':   is_new_group,
    })
    if hit:
        hits += 1

    hold_remaining -= 1

    # 命中后提前结束本组（下一期重新生成）
    if hit:
        hold_remaining = 0

    # 一组结束时统计是否有过命中
    if hold_remaining == 0 and group_hit_in_this_hold:
        hold_hits += 1

# 如果最后一组还没结束
if hold_remaining > 0 and group_hit_in_this_hold:
    hold_hits += 1
elif hold_remaining > 0 and not group_hit_in_this_hold:
    pass  # 未完成的最后一组不计入已完整统计

# ---------- 输出详情 ----------
print(f"{'期号':>4}  {'日期':<12} {'实际':>4} {'生肖':<4} {'预测尾数':<12} {'预测号码(~15个)':<45} {'组':>3} {'位':>3} {'命中'}")
print('-' * 100)

for r in results:
    tails_str = str(r['tails'])
    nums_str   = str(r['numbers'])
    mark = 'HIT' if r['hit'] else '---'
    new_mark = '*' if r['is_new'] else ' '
    print(f"{r['seq']:>4}  {str(r['date']):<12} {r['actual']:>4}  {r['animal']:<4} "
          f"{tails_str:<12} {nums_str:<45} {r['group_id']:>3}{new_mark} {r['hold_pos']:>3}  {mark}")

# ---------- 汇总 ----------
total = len(results)
rate_single = hits / total * 100
rate_baseline_single = 30.0  # 随机3/10尾数

# 完整组数（去掉最后一个未必完整的）
complete_groups = groups_total - (1 if hold_remaining > 0 else 0)
if complete_groups > 0:
    rate_group = hold_hits / complete_groups * 100
else:
    rate_group = 0.0

# 连续miss分析
max_miss = 0
cur_miss = 0
miss_streaks = []
for r in results:
    if r['hit']:
        if cur_miss > 0:
            miss_streaks.append(cur_miss)
        cur_miss = 0
    else:
        cur_miss += 1
if cur_miss > 0:
    miss_streaks.append(cur_miss)
max_miss = max(miss_streaks) if miss_streaks else 0

from collections import Counter
streak_dist = Counter(miss_streaks)

print()
print('=' * 100)
print(f"共 {total} 期  单期命中: {hits}/{total} = {rate_single:.1f}%  (随机基准 30.0%)")
print(f"共 {complete_groups} 完整组  组命中(至少1期): {hold_hits}/{complete_groups} = {rate_group:.1f}%  平均每组使用 {total/max(groups_total,1):.2f} 期")
print(f"最大连续miss: {max_miss} 期")
print(f"连续miss分布: {dict(sorted(streak_dist.items()))}")
