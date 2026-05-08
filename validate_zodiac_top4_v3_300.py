"""生肖TOP4 v3 - 300期详情输出"""
import sys, io
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from zodiac_top4_v3_predictor import (
    ZodiacTop4V3Predictor, NUM_TO_ZODIAC_2026, ZODIAC_CYCLE_2026
)

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].tolist()
dates = df['date'].tolist()
total = len(numbers)
test_periods = 300
start = total - test_periods

buf = io.StringIO()
def p(s=""):
    print(s)
    buf.write(s + "\n")

predictor = ZodiacTop4V3Predictor()
hits = 0
results = []

p(f"生肖TOP4 v3 - 最近{test_periods}期详细验证 (热号互补反miss)")
p(f"数据: {total}期, 测试区间: 第{start+1}期 ~ 第{total}期")
p("=" * 95)
p(f"  {'序号':>4}  {'日期':>12}  {'号码':>4}  {'实际':>4}  {'预测':>20}  {'结果':>4}  {'模式':>14}  {'累计':>6}")
p("-" * 95)

for pi in range(test_periods):
    i = start + pi
    hist = numbers[:i]
    actual = numbers[i]
    actual_z = NUM_TO_ZODIAC_2026[actual]
    date = dates[i]
    
    predicted, mode, scores = predictor.predict_with_details(hist, top_n=4)
    hit = actual_z in predicted
    
    if hit:
        hits += 1
    predictor.record_result(hit)
    
    hit_rate = hits / (pi + 1) * 100
    result_str = "O" if hit else "X"
    pred_str = ",".join(predicted)
    
    # 简化模式名
    if "L2" in mode:
        mode_short = "扩展TOP5"
    elif "L1" in mode:
        mode_short = "热号blend"
    else:
        mode_short = "正常"
    
    p(f"  {pi+1:>4}  {date:>12}  {actual:>4}  {actual_z:>4}  {pred_str:>20}  {result_str:>4}  {mode_short:>14}  {hit_rate:>5.1f}%")
    
    results.append({
        'hit': hit, 'mode': mode_short, 'bet_size': len(predicted),
        'consec_miss': predictor.consecutive_miss
    })

p("-" * 95)
p(f"总计: {hits}/{test_periods} = {hits/test_periods*100:.1f}%")

# 投注统计
total_bet = sum(r['bet_size'] * 4 for r in results)
total_win = sum(46 for r in results if r['hit'])
profit = total_win - total_bet
p(f"投入: {total_bet}元, 回报: {total_win}元, 净利润: {profit:+}元, ROI: {profit/total_bet*100:+.1f}%")

# 最大连miss
max_miss = max(r['consec_miss'] for r in results)
p(f"最大连续miss: {max_miss}期")

# 分段
p(f"\n分段统计(每50期):")
for s in range(6):
    seg = results[s*50:(s+1)*50]
    seg_hits = sum(1 for r in seg if r['hit'])
    seg_misses = []
    c = 0
    for r in seg:
        if not r['hit']: c += 1
        else:
            if c > 0: seg_misses.append(c)
            c = 0
    if c > 0: seg_misses.append(c)
    seg_max = max(seg_misses) if seg_misses else 0
    p(f"  {s*50+1:>4}-{(s+1)*50:>3}: {seg_hits}/50 = {seg_hits/50*100:>5.1f}%  最大连miss={seg_max}")

# 模式统计
from collections import Counter
mode_stats = Counter(r['mode'] for r in results)
p(f"\n模式统计:")
for mode, count in mode_stats.most_common():
    mode_hits = sum(1 for r in results if r['mode'] == mode and r['hit'])
    p(f"  {mode}: {count}期, 命中{mode_hits}/{count}={mode_hits/count*100:.1f}%")

# 与v2对比
p(f"\nv2→v3改进:")
p(f"  命中率: 46.7% → {hits/test_periods*100:.1f}%")
p(f"  最大连miss: 10 → {max_miss}")
p(f"  核心改进: 去掉有害的MK150切换, 用热号互补打破冷号陷阱")

with open('zodiac_top4_v3_detail.txt', 'w', encoding='utf-8-sig') as f:
    f.write(buf.getvalue())
p(f"\n结果已保存到 zodiac_top4_v3_detail.txt")
