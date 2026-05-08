"""最终验证: TOP23预测 + 斐波那契投注策略 v5.0"""
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

predictor = PreciseTop15Predictor()

# === 1. 命中率验证 ===
hits = 0
results = []
for i in range(start_idx, len(df)):
    lo = max(0, i - TRAIN_WINDOW)
    train = numbers[lo:i]
    preds = predictor.predict(train, k=PREDICT_K)
    actual = numbers[i]
    hit = actual in preds
    if hit:
        hits += 1
    predictor.update_performance(preds, actual)
    results.append({'hit': hit, 'actual': actual, 'preds': preds})

hit_rate = hits / TEST_PERIODS * 100
print(f"{'='*60}")
print(f"TOP{PREDICT_K} 扩展预测策略 v5.0 - 300期验证结果")
print(f"{'='*60}")
print(f"命中: {hits}/{TEST_PERIODS} = {hit_rate:.1f}%")
print(f"随机基准: {PREDICT_K}/49 = {PREDICT_K/49*100:.1f}%")
print(f"技巧溢价: +{hit_rate - PREDICT_K/49*100:.1f}%")
print(f"目标50%: {'✅ 达标!' if hit_rate >= 50 else '❌ 未达标'}")

# === 2. 斐波那契投注模拟 ===
fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
max_mul = 10
base_bet = 15
win_reward = 47

# 基础策略 (不含暂停)
fib_idx = 0
total_bet = 0
total_win = 0
balance = 0
min_balance = 0
max_streak_loss = 0
streak_loss = 0
max_consec = 0
consec = 0
hit_10x = 0

for r in results:
    mul = min(fib[min(fib_idx, len(fib)-1)], max_mul)
    bet = base_bet * mul
    total_bet += bet
    
    if mul >= 10:
        hit_10x += 1
    
    if r['hit']:
        win = win_reward * mul
        total_win += win
        balance += (win - bet)
        fib_idx = 0
        streak_loss = 0
        consec = 0
    else:
        balance -= bet
        fib_idx += 1
        streak_loss += bet
        consec += 1
        if streak_loss > max_streak_loss:
            max_streak_loss = streak_loss
        if consec > max_consec:
            max_consec = consec
    
    if balance < min_balance:
        min_balance = balance

roi = (total_win - total_bet) / total_bet * 100

print(f"\n{'='*60}")
print(f"基础策略（不含暂停）")
print(f"{'='*60}")
print(f"净利润: {balance:.0f}元")
print(f"总投入: {total_bet:.0f}元")
print(f"总奖金: {total_win:.0f}元")
print(f"ROI: {roi:.1f}%")
print(f"最大回撤(净值): {abs(min_balance):.0f}元")
print(f"连续不中总额: {max_streak_loss:.0f}元")
print(f"最长连败: {max_consec}期")
print(f"触及10倍: {hit_10x}次")

# 暂停策略 (命中1停1)
fib_idx = 0
total_bet_p = 0
total_win_p = 0
balance_p = 0
min_balance_p = 0
max_streak_loss_p = 0
streak_loss_p = 0
max_consec_p = 0
consec_p = 0
hit_10x_p = 0
paused = False
bet_periods = 0
pause_periods = 0

for r in results:
    if paused:
        paused = False
        pause_periods += 1
        continue
    
    bet_periods += 1
    mul = min(fib[min(fib_idx, len(fib)-1)], max_mul)
    bet = base_bet * mul
    total_bet_p += bet
    
    if mul >= 10:
        hit_10x_p += 1
    
    if r['hit']:
        win = win_reward * mul
        total_win_p += win
        balance_p += (win - bet)
        fib_idx = 0
        streak_loss_p = 0
        consec_p = 0
        paused = True  # 命中后暂停1期
    else:
        balance_p -= bet
        fib_idx += 1
        streak_loss_p += bet
        consec_p += 1
        if streak_loss_p > max_streak_loss_p:
            max_streak_loss_p = streak_loss_p
        if consec_p > max_consec_p:
            max_consec_p = consec_p
    
    if balance_p < min_balance_p:
        min_balance_p = balance_p

roi_p = (total_win_p - total_bet_p) / total_bet_p * 100
hit_rate_p = sum(1 for r in results if r['hit']) / bet_periods * 100 if bet_periods > 0 else 0

print(f"\n{'='*60}")
print(f"暂停策略（命中1停1期）")
print(f"{'='*60}")
print(f"投注期数: {bet_periods}, 暂停期数: {pause_periods}")
print(f"净利润: {balance_p:.0f}元")
print(f"总投入: {total_bet_p:.0f}元")
print(f"总奖金: {total_win_p:.0f}元")
print(f"ROI: {roi_p:.1f}%")
print(f"最大回撤(净值): {abs(min_balance_p):.0f}元")
print(f"连续不中总额: {max_streak_loss_p:.0f}元")
print(f"最长连败: {max_consec_p}期")
print(f"触及10倍: {hit_10x_p}次")

# === 3. 与旧版 TOP15 对比 ===
print(f"\n{'='*60}")
print(f"v4.0(TOP15) vs v5.0(TOP{PREDICT_K}) 对比")
print(f"{'='*60}")
print(f"{'指标':15s} {'TOP15 v4.0':>15s} {'TOP23 v5.0':>15s} {'变化':>12s}")
print(f"{'-'*60}")

# 旧版数据 (从之前的验证)
old = {'hit_rate': 35.7, 'roi': 21.7, 'drawdown': 1242, 'profit': 2265}
new = {'hit_rate': hit_rate, 'roi': roi, 'drawdown': abs(min_balance), 'profit': balance}

for label, old_v, new_v, fmt in [
    ('命中率', old['hit_rate'], new['hit_rate'], '.1f'),
    ('ROI', old['roi'], new['roi'], '.1f'),
    ('净利润', old['profit'], new['profit'], '.0f'),
    ('最大回撤', old['drawdown'], new['drawdown'], '.0f'),
]:
    delta = new_v - old_v
    pct = delta / abs(old_v) * 100 if old_v != 0 else 0
    print(f"  {label:13s} {old_v:>13{fmt}} {new_v:>13{fmt}} {pct:>+10.1f}%")

# === 4. 分段验证（每50期） ===
print(f"\n{'='*60}")
print(f"分段命中率验证（每50期）")
print(f"{'='*60}")
for seg in range(0, TEST_PERIODS, 50):
    seg_end = min(seg + 50, TEST_PERIODS)
    seg_hits = sum(1 for r in results[seg:seg_end] if r['hit'])
    seg_rate = seg_hits / (seg_end - seg) * 100
    bar = '█' * int(seg_rate / 2)
    print(f"  第{seg+1:3d}-{seg_end:3d}期: {seg_hits}/{seg_end-seg} = {seg_rate:.1f}% {bar}")

avg_rate = sum(sum(1 for r in results[seg:min(seg+50, TEST_PERIODS)] if r['hit']) / min(50, TEST_PERIODS-seg) for seg in range(0, TEST_PERIODS, 50)) / (TEST_PERIODS // 50 + (1 if TEST_PERIODS % 50 else 0)) * 100
print(f"\n  平均分段命中率: {avg_rate:.1f}%")
