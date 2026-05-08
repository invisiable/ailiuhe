"""生肖TOP9预测器 - 300期完整验证详情输出"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from collections import Counter
from zodiac_top9_predictor import ZodiacTop9Predictor, NUM_TO_ZODIAC_2026, ZODIAC_NUMS_2026, validate_predictor, cross_validate

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].tolist()
total = len(numbers)
test_periods = 300
start = total - test_periods

predictor = ZodiacTop9Predictor()

hits = 0
hit_records = []
mode_records = []
miss_streaks = []
cur_miss = 0
total_cost = 0
total_reward = 0
details = []

print(f"{'='*120}")
print(f"生肖TOP9预测器 - 300期完整验证详情")
print(f"数据: {total}期, 测试: 最近{test_periods}期 (第{start+1}~{total}期)")
print(f"{'='*120}")
print(f"策略: cold15×0.20 + cold30×0.05 + MK150×0.50 + gap×0.10 + hot30×0.15")
print(f"反miss: L1(连miss≥2 blend热号) L2(连miss≥3 扩展TOP10)")
print(f"{'='*120}\n")

print(f"{'期号':>4} {'日期':>12} {'号码':>4} {'生肖':>4} {'预测生肖(TOP9)':^55} {'结果':>4} {'连miss':>6} {'模式':>20}")
print(f"{'-'*120}")

for pi in range(test_periods):
    i = start + pi
    hist = numbers[:i]
    actual_num = numbers[i]
    actual_z = NUM_TO_ZODIAC_2026[actual_num]
    date_str = str(df.iloc[i]['date'])
    
    preds, mode, scores = predictor.predict_with_details(hist, top_n=9)
    hit = actual_z in preds
    
    if hit:
        hits += 1
        if cur_miss > 0:
            miss_streaks.append(cur_miss)
        cur_miss = 0
    else:
        cur_miss += 1
    
    predictor.record_result(hit)
    hit_records.append(hit)
    
    bet_size = len(preds)
    total_cost += bet_size * 4
    if hit:
        total_reward += 46
    
    if "L2" in mode:
        mode_short = "扩展TOP10"
    elif "L1" in mode:
        mode_short = "热号blend"
    else:
        mode_short = "正常"
    mode_records.append(mode_short)
    
    mark = "✅" if hit else "❌"
    pred_str = ','.join(preds)
    miss_count = predictor.consecutive_miss
    
    print(f"{pi+1:>4} {date_str:>12} {actual_num:>4} {actual_z:>4} {pred_str:<55} {mark:>4} {miss_count:>6} {mode_short:>20}")

if cur_miss > 0:
    miss_streaks.append(cur_miss)

max_miss = max(miss_streaks) if miss_streaks else 0
hit_rate = hits / test_periods * 100
profit = total_reward - total_cost
roi = profit / total_cost * 100

print(f"\n{'='*120}")
print(f"汇总统计")
print(f"{'='*120}")
print(f"命中: {hits}/{test_periods} = {hit_rate:.1f}%")
print(f"随机基线: 75.0%, 提升: +{hit_rate-75.0:.1f}%")
print(f"最大连续miss: {max_miss}")
print(f"总投入: {total_cost}元, 总回报: {total_reward}元")
print(f"净利润: {profit:+d}元, ROI: {roi:+.1f}%")

# 分段统计
print(f"\n50期分段:")
for s in range(0, test_periods, 50):
    seg = hit_records[s:s+50]
    seg_h = sum(seg)
    seg_r = seg_h / len(seg) * 100
    bar = '█' * int(seg_r/5) + '░' * (20 - int(seg_r/5))
    print(f"  {s+1:>3}-{s+50:>3}: {seg_h}/{len(seg)} = {seg_r:.0f}% {bar}")

# 连续miss分布
streaks_all = []
c = 0
for h in hit_records:
    if not h: c += 1
    else:
        if c > 0: streaks_all.append(c)
        c = 0
if c > 0: streaks_all.append(c)
print(f"\n连续miss分布:")
for length in sorted(set(streaks_all)):
    cnt = streaks_all.count(length)
    print(f"  连续{length}期miss: {cnt}次")

# 模式统计
print(f"\n模式统计:")
for mode_name in sorted(set(mode_records)):
    total_m = mode_records.count(mode_name)
    hits_m = sum(1 for h, m in zip(hit_records, mode_records) if m == mode_name and h)
    print(f"  {mode_name}: {total_m}期, 命中{hits_m}/{total_m}={hits_m/total_m*100:.1f}%")

# 交叉验证
mean_cv, std_cv, cv_rates = cross_validate(numbers)
print(f"\n交叉验证: {mean_cv:.1f}% ± {std_cv:.1f}%")
for fi, r in enumerate(cv_rates):
    print(f"  Fold {fi+1}: {r:.1f}%")

print(f"\n{'='*120}")
