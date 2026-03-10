# -*- coding: utf-8 -*-
"""
TOP15 最近300期详细回测验证（训练窗口=25期）
验证当前GUI使用的25期滚动窗口模型的真实表现
"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from precise_top15_predictor import PreciseTop15Predictor

DATA        = 'data/lucky_numbers.csv'
TEST_N      = 300
MIN_TRAIN   = 10
TRAIN_WIN   = 25   # 与GUI保持一致
BASE_BET    = 15
WIN_R       = 47

df      = pd.read_csv(DATA, encoding='utf-8-sig')
n       = len(df)
start   = max(MIN_TRAIN, n - TEST_N)
total_periods = n - start

print(f"\n数据总量: {n}期   回测期数: {total_periods}期")
print(f"日期范围: {df.iloc[start]['date']} ~ {df.iloc[-1]['date']}")
print(f"训练窗口: 最近 {TRAIN_WIN} 期（滚动）\n")

predictor = PreciseTop15Predictor()
records = []

for i in range(start, n):
    lo    = max(0, i - TRAIN_WIN)
    train = df.iloc[lo:i]['number'].values
    if len(train) < 5:
        continue
    preds  = predictor.predict(train)
    actual = df.iloc[i]['number']
    predictor.update_performance(preds, actual)
    records.append({
        'seq':    i - start + 1,
        'date':   df.iloc[i]['date'],
        'actual': actual,
        'hit':    actual in preds,
        'top5':   preds[:5].tolist() if hasattr(preds, 'tolist') else list(preds)[:5],
    })

# ── 整体统计 ────────────────────────────────────────────────────────────────
hits     = sum(r['hit'] for r in records)
total    = len(records)
balance  = peak = maxdd = 0
bal_hist = []
for r in records:
    if r['hit']: balance += WIN_R - BASE_BET
    else:        balance -= BASE_BET
    if balance > peak: peak = balance
    dd = peak - balance
    if dd > maxdd: maxdd = dd
    bal_hist.append(balance)
roi = balance / (total * BASE_BET) * 100 if total > 0 else 0

print(f"{'='*72}")
print(f"  整体结果: 命中 {hits}/{total} 期  命中率 {hits/total*100:.1f}%")
print(f"  总盈亏: {balance:+.0f}元   ROI: {roi:+.2f}%   最大回撤: {maxdd:.0f}元")
print(f"{'='*72}\n")

# ── 月度统计 ────────────────────────────────────────────────────────────────────
from collections import defaultdict
import re

monthly = defaultdict(lambda: {'hits':0, 'total':0})
for r in records:
    m = re.match(r'(\d{4}/\d{1,2})', str(r['date']))
    if m:
        mon = m.group(1)
        monthly[mon]['total'] += 1
        if r['hit']: monthly[mon]['hits'] += 1

print(f"月度命中率明细：")
print(f"  {'月份':<10} {'命中/总期':>9} {'命中率':>7}")
print(f"  {'-'*30}")
for mon in sorted(monthly):
    h = monthly[mon]['hits']
    t = monthly[mon]['total']
    print(f"  {mon:<10} {h:>4}/{t:<4}    {h/t*100:>5.1f}%")

# ── 连亏统计 ───────────────────────────────────────────────────────────────────
print()
cur_l = max_l = 0
streak_counts = defaultdict(int)
for r in records:
    if not r['hit']:
        cur_l += 1
        max_l = max(max_l, cur_l)
    else:
        if cur_l > 0:
            streak_counts[cur_l] += 1
        cur_l = 0
if cur_l > 0:
    streak_counts[cur_l] += 1

print(f"最大连亏: {max_l} 期")
print(f"连亏分布:")
for k in sorted(streak_counts):
    print(f"  连亏{k}期: {streak_counts[k]}次")

# ── 最近30期详情 ───────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"最近30期详情（TOP5预测）：")
print(f"  {'序号':>4} {'日期':<12} {'实际':>4} {'TOP5预测':<30} {'结果':>4}")
print(f"  {'-'*58}")
for r in records[-30:]:
    top5_str = str(r['top5'])
    mark = 'HIT' if r['hit'] else '---'
    print(f"  {r['seq']:>4} {str(r['date']):<12} {r['actual']:>4}   {top5_str:<30} {mark:>4}")

print(f"\n回测完成！")
