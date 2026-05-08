"""最终验证: v5.0 TOP23扩展预测策略 - 模拟GUI实际行为"""
import pandas as pd
import numpy as np
from collections import Counter
from precise_top15_predictor import PreciseTop15Predictor

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values
TEST_PERIODS = 300
start_idx = len(df) - TEST_PERIODS

PREDICT_K = 23
TRAIN_WINDOW = 25

# 模拟GUI行为：不调用update_performance
predictor = PreciseTop15Predictor()
results = []
for i in range(start_idx, len(df)):
    lo = max(0, i - TRAIN_WINDOW)
    train = numbers[lo:i]
    preds = predictor.predict(train, k=PREDICT_K)
    actual = numbers[i]
    hit = actual in preds
    results.append({'hit': hit, 'actual': actual, 'date': df.iloc[i]['date']})

hits = sum(1 for r in results if r['hit'])
hit_rate = hits / TEST_PERIODS * 100

print(f"{'='*60}")
print(f"🏆 TOP{PREDICT_K} 扩展预测策略 v5.0 - 最终验证")
print(f"{'='*60}")
print(f"命中: {hits}/{TEST_PERIODS} = {hit_rate:.1f}%")
print(f"目标50%: {'✅ 达标!' if hit_rate >= 50 else '❌ 未达标'}")

# 斐波那契投注 + 暂停策略
fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
base_bet = 15
win_reward = 47
max_mul = 10

# --- 基础策略 ---
fib_idx = 0; bal = 0; min_bal = 0; total_bet = 0; total_win = 0
streak_l = 0; max_streak_l = 0; max_consec = 0; consec = 0; hit_10x = 0
for r in results:
    mul = min(fib[min(fib_idx, len(fib)-1)], max_mul)
    bet = base_bet * mul; total_bet += bet
    if mul >= 10: hit_10x += 1
    if r['hit']:
        w = win_reward * mul; total_win += w; bal += (w-bet); fib_idx = 0; streak_l = 0; consec = 0
    else:
        bal -= bet; fib_idx += 1; streak_l += bet; consec += 1
        if streak_l > max_streak_l: max_streak_l = streak_l
        if consec > max_consec: max_consec = consec
    if bal < min_bal: min_bal = bal

# --- 暂停策略 ---
fib_idx2 = 0; bal2 = 0; min_bal2 = 0; total_bet2 = 0; total_win2 = 0
streak_l2 = 0; max_streak_l2 = 0; max_consec2 = 0; consec2 = 0; hit_10x2 = 0
paused = False; bet_periods = 0; pause_periods = 0
for r in results:
    if paused:
        paused = False; pause_periods += 1; continue
    bet_periods += 1
    mul = min(fib[min(fib_idx2, len(fib)-1)], max_mul)
    bet = base_bet * mul; total_bet2 += bet
    if mul >= 10: hit_10x2 += 1
    if r['hit']:
        w = win_reward * mul; total_win2 += w; bal2 += (w-bet); fib_idx2 = 0; streak_l2 = 0; consec2 = 0; paused = True
    else:
        bal2 -= bet; fib_idx2 += 1; streak_l2 += bet; consec2 += 1
        if streak_l2 > max_streak_l2: max_streak_l2 = streak_l2
        if consec2 > max_consec2: max_consec2 = consec2
    if bal2 < min_bal2: min_bal2 = bal2

roi = (total_win - total_bet) / total_bet * 100
roi2 = (total_win2 - total_bet2) / total_bet2 * 100

print(f"\n{'='*60}")
print(f"v4.0(TOP15) vs v5.0(TOP{PREDICT_K}) 全面对比")
print(f"{'='*60}")
print(f"{'指标':20s} {'v4.0 TOP15':>12s} {'v5.0 TOP23':>12s} {'提升':>10s}")
print(f"{'-'*60}")
old_data = [
    ('命中率', '35.7%', f'{hit_rate:.1f}%', f'+{hit_rate-35.7:.1f}pp'),
    ('净利润(基础)', '2265元', f'{bal:.0f}元', f'+{(bal-2265)/2265*100:.0f}%'),
    ('净利润(暂停)', '1412元', f'{bal2:.0f}元', f'+{(bal2-1412)/1412*100:.0f}%'),
    ('ROI(基础)', '21.7%', f'{roi:.1f}%', f'+{roi-21.7:.1f}pp'),
    ('ROI(暂停)', '38.2%', f'{roi2:.1f}%', f'+{roi2-38.2:.1f}pp'),
    ('回撤-净值(基础)', '1242元', f'{abs(min_bal):.0f}元', f'-{(1242-abs(min_bal))/1242*100:.0f}%'),
    ('回撤-净值(暂停)', '393元', f'{abs(min_bal2):.0f}元', f'{(abs(min_bal2)-393)/393*100:+.0f}%'),
    ('连续不中总额(基础)', '729元', f'{max_streak_l:.0f}元', f'{(max_streak_l-729)/729*100:+.0f}%'),
    ('连续不中总额(暂停)', '747元', f'{max_streak_l2:.0f}元', f'{(max_streak_l2-747)/747*100:+.0f}%'),
    ('最长连败(基础)', '9期', f'{max_consec}期', f'{max_consec-9:+d}期'),
    ('最长连败(暂停)', '11期', f'{max_consec2}期', f'{max_consec2-11:+d}期'),
    ('触及10倍(基础)', '10次', f'{hit_10x}次', f'{hit_10x-10:+d}次'),
    ('触及10倍(暂停)', '6次', f'{hit_10x2}次', f'{hit_10x2-6:+d}次'),
]
for label, old_v, new_v, delta in old_data:
    print(f"  {label:18s} {old_v:>12s} {new_v:>12s} {delta:>10s}")

# 分段验证
print(f"\n{'='*60}")
print(f"分段命中率验证（每50期）")
print(f"{'='*60}")
all_above_40 = True
for seg in range(0, TEST_PERIODS, 50):
    seg_end = min(seg + 50, TEST_PERIODS)
    seg_hits = sum(1 for r in results[seg:seg_end] if r['hit'])
    seg_rate = seg_hits / (seg_end - seg) * 100
    bar = '█' * int(seg_rate / 2)
    status = '✅' if seg_rate >= 50 else ('⚠️' if seg_rate >= 45 else '❌')
    print(f"  {status} 第{seg+1:3d}-{seg_end:3d}期: {seg_hits}/{seg_end-seg} = {seg_rate:.1f}% {bar}")
    if seg_rate < 40:
        all_above_40 = False

print(f"\n  总体: {hits}/{TEST_PERIODS} = {hit_rate:.1f}%")
print(f"  目标达成: {'✅ 命中率 ≥ 50%' if hit_rate >= 50 else '❌ 命中率 < 50%'}")
print(f"  稳定性: {'✅ 所有分段 ≥ 40%' if all_above_40 else '⚠️ 存在低命中分段'}")
