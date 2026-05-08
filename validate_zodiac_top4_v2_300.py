"""生肖TOP4 v2 - 最近300期详细验证"""
import pandas as pd
import numpy as np
import sys, io
sys.stdout.reconfigure(encoding='utf-8')
from zodiac_top4_v2_predictor import ZodiacTop4V2Predictor, NUM_TO_ZODIAC_2026, ZODIAC_NUMS_2026

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values.tolist()
T = len(df)
test = min(300, T - 20)
start = T - test

predictor = ZodiacTop4V2Predictor()
hits = 0
total_cost = 0
total_reward = 0
max_miss = 0
cur_miss = 0

out = io.StringIO()

def p(s=""):
    print(s)
    out.write(s + "\n")

p(f"生肖TOP4 v2 - 最近{test}期详细验证")
p(f"数据: {T}期, 测试区间: 第{start+1}期 ~ 第{T}期")
p("=" * 95)
p(f"{'序号':>4} {'日期':>12} {'号码':>4} {'实际':>4} {'预测TOP4':>20} {'结果':>4} {'模式':>4} {'累计':>8}")
p("-" * 95)

for i in range(start, T):
    hist = numbers[:i]
    actual_num = numbers[i]
    actual_z = NUM_TO_ZODIAC_2026[actual_num]
    
    mode = "切换" if predictor.consecutive_miss >= 2 else "正常"
    predicted = predictor.predict(hist, top_n=4)
    hit = actual_z in predicted
    predictor.record_result(hit)
    
    pi = i - start + 1
    if hit:
        hits += 1
        cur_miss = 0
    else:
        cur_miss += 1
        max_miss = max(max_miss, cur_miss)
    
    total_cost += 16
    if hit:
        total_reward += 46
    
    mark = "O" if hit else "X"
    rate = hits / pi * 100
    pred_str = ','.join(predicted)
    date_str = str(df.iloc[i]['date'])
    
    p(f"{pi:>4} {date_str:>12} {actual_num:>4} {actual_z:>4} {pred_str:>20} {mark:>4} {mode:>4} {rate:>7.1f}%")

profit = total_reward - total_cost
roi = profit / total_cost * 100

p("-" * 95)
p(f"总计: {hits}/{test} = {hits/test*100:.1f}%")
p(f"投入: {total_cost}元, 回报: {total_reward}元, 净利润: {profit:+d}元, ROI: {roi:+.1f}%")
p(f"最大连续miss: {max_miss}期")

# 分段
seg = 50
p(f"\n分段统计(每{seg}期):")
for s in range(test // seg):
    s0 = s * seg
    s1 = s0 + seg
    p2 = ZodiacTop4V2Predictor()
    sh = 0
    for i in range(start, start + s1):
        hist = numbers[:i]
        predicted = p2.predict(hist, top_n=4)
        actual_z = NUM_TO_ZODIAC_2026[numbers[i]]
        hit = actual_z in predicted
        p2.record_result(hit)
        if i >= start + s0:
            if hit: sh += 1
    p(f"  {s0+1:>3}-{s1:>3}: {sh}/{seg} = {sh/seg*100:.0f}%")

# 写入文件
with open('zodiac_top4_v2_detail.txt', 'w', encoding='utf-8-sig') as f:
    f.write(out.getvalue())
p("\n结果已保存到 zodiac_top4_v2_detail.txt")
