# -*- coding: utf-8 -*-
"""相对基准率阈值分析 + 状态转移详情"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from validate_markov_top15 import rolling_backtest, _calculate_markov2_transition

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
records = rolling_backtest(df, test_periods=300)

base_rate = sum(r['hit'] for r in records) / len(records)
print(f"\n基准命中率: {base_rate*100:.1f}%   (理论: 15/49={15/49*100:.1f}%)\n")

BASE_BET = 15
WIN_R    = 47

def run(cond):
    bets = hits = total = peak = maxdd = 0
    for r in records:
        if cond(r):
            bets += 1
            if r['hit']: total += WIN_R - BASE_BET; hits += 1
            else: total -= BASE_BET
            if total > peak: peak = total
            dd = peak - total
            if dd > maxdd: maxdd = dd
    if bets == 0:
        return bets, 0, 0, 0, 0, 0
    return bets, hits/bets*100, total, total/(bets*BASE_BET)*100, maxdd, peak/maxdd if maxdd>0 else 0

print("── 相对基准率阈值策略 ─────────────────────────────────────────")
print(f"  {'阈值描述':<32} {'投注':>5} {'命中率':>7} {'盈亏':>8} {'ROI':>8} {'回撤':>7} {'风险收益':>8}")
print("  " + "-"*76)
for delta in [0, 2, 3, 5, 8, 10]:
    thresh = base_rate + delta/100
    bts, hr, tot, roi, dd, rr = run(lambda r, t=thresh: r['prob_markov'] > t)
    label = f"马尔可夫 > 基准+{delta:.0f}%({thresh*100:.1f}%)"
    print(f"  {label:<32} {bts:>4}期 {hr:>6.1f}% {tot:>+8.0f}元 {roi:>+7.2f}% {dd:>7.0f}元 {rr:>7.2f}")

# 固投全期基准
bts_all, hr_all, tot_all, roi_all, dd_all, rr_all = run(lambda r: True)
print(f"  {'基准：全部固投':<32} {bts_all:>4}期 {hr_all:>6.1f}% {tot_all:>+8.0f}元 {roi_all:>+7.2f}% {dd_all:>7.0f}元 {rr_all:>7.2f}")

# 马尔可夫状态转移 vs 实际（全量建模）
mk2 = _calculate_markov2_transition([r['hit'] for r in records])
names = {(False,False):'败-败', (True,False):'胜-败',
         (False,True): '败-胜', (True,True): '胜-胜'}

# 实际状态命中统计（滚动，每期用真实状态）
buckets = {k: [0,0] for k in names}
for i in range(2, len(records)):
    st = (records[i-2]['hit'], records[i-1]['hit'])
    buckets[st][1] += 1
    if records[i]['hit']:
        buckets[st][0] += 1

print(f"\n── 4种马尔可夫状态 vs 实际命中率 ─────────────────────────────────────")
print(f"  {'状态':<6} | {'预测概率':>8} | {'样本':>5} | {'实际命中率':>10} | {'vsBase':>8} | 建议")
print("  " + "-"*64)
for st in [(False,False),(True,False),(False,True),(True,True)]:
    h, t = buckets[st]
    _mh, _mt, mk_p = mk2.get(st, (0,0,0.0))
    actual = h/t*100 if t>0 else 0
    diff = actual - base_rate*100
    rec = "↑优先投" if diff > 3 else ("↓考虑跳" if diff < -3 else "→正常投")
    print(f"  {names[st]:<6} | {mk_p*100:>7.1f}% | {t:>5} | {actual:>9.1f}% | {diff:>+7.1f}% | {rec}")

print(f"\n  基准命中率: {base_rate*100:.1f}%  (理论15/49={15/49*100:.1f}%)\n")
