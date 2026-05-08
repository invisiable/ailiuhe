"""v6.1 修复后完整详情输出（无未来函数）- 模拟GUI输出"""
import pandas as pd
from precise_top15_predictor import PreciseTop15Predictor

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values
TEST_PERIODS = 300; start_idx = len(df) - TEST_PERIODS
PREDICT_K = 15; TRAIN_WINDOW = 25; BASE_COST = 15; WIN_REWARD = 47; MAX_MUL = 10
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

CFG = {'pattern_len': 2, 'boost_factor': 2.5, 'reduce_factor': 0.4,
       'high_thresh': 0.35, 'low_thresh': 0.25, 'min_samples': 1, 'lookback': 12,
       'max_multiplier': 10, 'base_bet': 15, 'win_reward': 47}

# 生成预测+命中序列
predictor = PreciseTop15Predictor()
periods = []
for i in range(start_idx, len(df)):
    lo = max(0, i - TRAIN_WINDOW)
    train = numbers[lo:i]
    preds = predictor.predict(train, k=PREDICT_K)
    actual = int(numbers[i])
    date = df.iloc[i]['date']
    hit = actual in preds
    predictor.update_performance(preds, actual)
    periods.append({'date': date, 'actual': actual, 'preds': preds, 'hit': hit})

hits_total = sum(p['hit'] for p in periods)

# === 暂停策略模拟（修复版：无未来函数）===
fib_idx = 0; bal = 0; tb = 0; tw = 0; mb = 0
sl = 0; msl = 0; cm = 0; mcm = 0; hm = 0
recent = []; pstats = {}; pr = 0; bp = 0; sp = 0
pl = CFG['pattern_len']
detail_rows = []

for t, p in enumerate(periods):
    h = p['hit']
    period_num = t + 1
    
    if pr > 0:
        pr -= 1; sp += 1
        detail_rows.append({
            'period': period_num, 'date': p['date'], 'actual': p['actual'],
            'preds': p['preds'][:5], 'hit': h, 'mul': 0, 'bet': 0,
            'profit': 0, 'cum': bal, 'paused': True, 'fib': fib_idx,
            'markov_info': '暂停', 'pause_rem': pr
        })
        continue
    
    bp += 1
    
    # 步骤1: 先用历史stats算倍数(无未来函数)
    base = min(FIB[min(fib_idx, len(FIB)-1)], MAX_MUL)
    mul = base
    markov_info = "Fib基础"
    if len(recent) >= pl:
        pat = tuple(recent[-pl:])
        s = pstats.get(pat, [0, 0])
        pat_str = ''.join(['✓' if x else '✗' for x in pat])
        if s[1] >= CFG['min_samples']:
            r = s[0]/s[1]
            if r >= CFG['high_thresh']:
                mul = round(base * CFG['boost_factor'])
                markov_info = f"{pat_str}→{r:.0%}↑×{CFG['boost_factor']}"
            elif r <= CFG['low_thresh']:
                mul = max(1, round(base * CFG['reduce_factor']))
                markov_info = f"{pat_str}→{r:.0%}↓×{CFG['reduce_factor']}"
            else:
                markov_info = f"{pat_str}→{r:.0%}="
        else:
            markov_info = f"{pat_str}(样本{s[1]})"
    mul = min(max(1, mul), MAX_MUL)
    
    bet = BASE_COST * mul; tb += bet
    if mul >= MAX_MUL: hm += 1
    
    # 步骤2: 结算
    if h:
        w = WIN_REWARD * mul; tw += w
        profit = w - bet
        bal += profit; fib_idx = 0; sl = 0; cm = 0; pr = 1
    else:
        profit = -bet
        bal -= bet; fib_idx += 1; sl += bet; cm += 1
        if sl > msl: msl = sl
        if cm > mcm: mcm = cm
    if bal < mb: mb = bal
    
    # 步骤3: 结算后才更新stats
    if len(recent) >= pl:
        pat = tuple(recent[-pl:])
        if pat not in pstats: pstats[pat] = [0, 0]
        pstats[pat][1] += 1
        if h: pstats[pat][0] += 1
    
    # 步骤4: 更新recent
    recent.append(1 if h else 0)
    if len(recent) > CFG['lookback']: recent.pop(0)
    
    detail_rows.append({
        'period': period_num, 'date': p['date'], 'actual': p['actual'],
        'preds': p['preds'][:5], 'hit': h, 'mul': mul, 'bet': bet,
        'profit': profit, 'cum': bal, 'paused': False, 'fib': fib_idx if not h else 0,
        'markov_info': markov_info, 'pause_rem': pr
    })

dd = abs(mb); roi = (tw-tb)/tb*100 if tb > 0 else 0; rr = bal/dd if dd > 0 else 0

# === 输出 ===
print(f"{'='*130}")
print(f"v6.1 马尔可夫动态倍投 - 修复后300期详情（无未来函数）")
print(f"{'='*130}")
print(f"命中率: {hits_total}/{TEST_PERIODS} = {hits_total/TEST_PERIODS*100:.1f}%")
print(f"参数: 窗口={CFG['pattern_len']}, boost={CFG['boost_factor']}, reduce={CFG['reduce_factor']}, "
      f"高阈值={CFG['high_thresh']:.0%}, 低阈值={CFG['low_thresh']:.0%}, 样本>={CFG['min_samples']}")
print(f"数据验证: 倍数计算→结算→更新统计→更新历史（严格时序，无偷看）")
print()

# 汇总
print(f"【暂停策略汇总】")
print(f"  投注期: {bp}期, 暂停期: {sp}期, 总期: {TEST_PERIODS}期")
print(f"  ROI: {roi:.1f}%, 净利润: {bal:+.0f}元, 回撤: {dd:.0f}元, 连挂最大: {msl:.0f}元")
print(f"  风险收益比: {rr:.2f}, 触顶(10x): {hm}次, 最长连败: {mcm}期")
print()

# 每期详情
print(f"{'期号':<6}{'日期':<12}{'开奖':<6}{'预测TOP5':<22}{'命中':<4}{'Fib':>4}{'倍数':>6}{'投注':>8}{'盈亏':>10}{'累计':>10}{'暂停':<6}{'马尔可夫决策':<20}")
print(f"{'-'*120}")

for r in detail_rows:
    period = r['period']
    date = r['date']
    actual = r['actual']
    pred_str = str(r['preds'])
    hit_mark = '✓' if r['hit'] else '✗'
    mul = r['mul']
    bet = r['bet']
    profit = r['profit']
    cum = r['cum']
    paused = "暂停" if r['paused'] else ""
    fib = r['fib']
    mk = r['markov_info']
    
    if r['paused']:
        print(f"{period:<6}{date:<12}{actual:<6}{pred_str:<22}{hit_mark:<4}{'':>4}{'':>6}{'':>8}{'':>10}{cum:>+10.0f}  {paused:<6}{mk}")
    else:
        print(f"{period:<6}{date:<12}{actual:<6}{pred_str:<22}{hit_mark:<4}{fib:>4}{mul:>6}{bet:>8.0f}{profit:>+10.0f}{cum:>+10.0f}  {paused:<6}{mk}")

# 分段统计
print(f"\n{'='*130}")
print(f"分段统计（每50期）")
print(f"{'='*130}")
print(f"{'区间':<12}{'命中':>8}{'命中率':>8}{'净利润':>10}{'投注期':>8}{'暂停期':>8}")
print(f"{'-'*60}")

for seg_start in range(0, TEST_PERIODS, 50):
    seg_end = min(seg_start + 50, TEST_PERIODS)
    seg_rows = [r for r in detail_rows if seg_start < r['period'] <= seg_end]
    seg_hits = sum(1 for r in seg_rows if r['hit'] and not r['paused'])
    seg_profit = sum(r['profit'] for r in seg_rows)
    seg_bet = sum(1 for r in seg_rows if not r['paused'])
    seg_pause = sum(1 for r in seg_rows if r['paused'])
    seg_hit_total = sum(1 for r in seg_rows if r['hit'])
    print(f"  {seg_start+1}-{seg_end:<6} {seg_hit_total:>6}/{seg_end-seg_start} {seg_hit_total/(seg_end-seg_start)*100:>6.1f}% {seg_profit:>+9.0f}元 {seg_bet:>6}期 {seg_pause:>6}期")

# 马尔可夫模式统计
print(f"\n{'='*130}")
print(f"马尔可夫模式统计（最终状态）")
print(f"{'='*130}")
print(f"{'模式':<10}{'样本':>6}{'命中':>6}{'命中率':>8}{'决策':<12}")
print(f"{'-'*45}")
sorted_pats = sorted(pstats.items(), key=lambda x: x[1][1], reverse=True)
for pat, (h, t) in sorted_pats:
    pat_str = ''.join(['✓' if p else '✗' for p in pat])
    rate = h/t*100 if t > 0 else 0
    if rate >= 35:
        decision = f"加倍×{CFG['boost_factor']}"
    elif rate <= 25:
        decision = f"减倍×{CFG['reduce_factor']}"
    else:
        decision = "标准"
    print(f"  {pat_str:<8} {t:>6} {h:>6} {rate:>6.1f}% {decision}")

print(f"\n✅ 所有数据严格时序验证，无未来函数污染")
