"""
严格审计: 马尔可夫动态倍投是否存在未来函数(look-ahead bias)

审计要点:
1. pattern_stats 是否在知道当期结果之前就用了当期结果?
2. get_markov_multiplier() 查询的模式统计是否包含未来数据?
3. 倍数决定时刻 vs 结果揭晓时刻的先后顺序

模拟逐期执行，打印每步的数据状态
"""
import pandas as pd
from precise_top15_predictor import PreciseTop15Predictor

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values
PREDICT_K = 15; TRAIN_WINDOW = 25; MAX_MUL = 10
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

# 只取10期做详细审计
TEST_N = 15
start_idx = len(df) - TEST_N

predictor = PreciseTop15Predictor()
hit_seq = []
for i in range(start_idx, len(df)):
    lo = max(0, i - TRAIN_WINDOW)
    train = numbers[lo:i]
    preds = predictor.predict(train, k=PREDICT_K)
    actual = int(numbers[i])
    hit = actual in preds
    predictor.update_performance(preds, actual)
    hit_seq.append(hit)

print(f"审计序列 (最近{TEST_N}期): {['✓' if h else '✗' for h in hit_seq]}")
print()

# === 逐步模拟 process_period，打印每步状态 ===
CFG = {'pattern_len': 3, 'boost_factor': 2.5, 'reduce_factor': 0.4,
       'high_thresh': 0.40, 'low_thresh': 0.20, 'min_samples': 2, 'lookback': 12,
       'max_multiplier': 10, 'base_bet': 15, 'win_reward': 47}

recent = []
pattern_stats = {}
fib_idx = 0
pl = CFG['pattern_len']

print("="*100)
print("逐期审计: process_period 内部执行顺序")
print("="*100)

issues_found = 0

for t, hit in enumerate(hit_seq):
    print(f"\n--- 第{t+1}期 (实际结果: {'命中✓' if hit else '未中✗'}) ---")
    
    # 步骤1: 在process_period开头，先更新pattern_stats
    # 这里用的是 recent_results[-3:] 和当前的 hit
    print(f"  [步骤1] recent_results(更新前) = {['✓' if r else '✗' for r in recent]}")
    
    if len(recent) >= pl:
        pat = tuple(recent[-pl:])
        if pat not in pattern_stats:
            pattern_stats[pat] = [0, 0]
        pattern_stats[pat][1] += 1
        if hit:
            pattern_stats[pat][0] += 1
        pat_str = ''.join(['✓' if p else '✗' for p in pat])
        print(f"  [步骤1] ⚠️ 更新pattern_stats: 模式{pat_str} → 用当期hit={hit}更新 → stats={pattern_stats[pat]}")
        print(f"  [步骤1] ❌ 问题: 这里在计算倍数之前，就把当期结果(hit={hit})加入了统计!")
        print(f"         也就是说: 当期模式{pat_str}的统计已经包含了'当期是否命中'的信息")
        issues_found += 1
    else:
        print(f"  [步骤1] recent不足{pl}期，跳过模式更新")
    
    # 步骤2: 计算倍数 (get_markov_multiplier)
    base = min(FIB[min(fib_idx, len(FIB)-1)], MAX_MUL)
    mul = base
    markov_detail = "无调整(样本不足或recent不够)"
    
    if len(recent) >= pl:
        pat = tuple(recent[-pl:])
        stats = pattern_stats.get(pat, [0, 0])
        pat_str = ''.join(['✓' if p else '✗' for p in pat])
        if stats[1] >= CFG['min_samples']:
            rate = stats[0] / stats[1]
            if rate >= CFG['high_thresh']:
                mul = round(base * CFG['boost_factor'])
                markov_detail = f"加倍! 模式{pat_str} 命中率{rate:.0%}≥{CFG['high_thresh']:.0%} → base{base}×{CFG['boost_factor']}={mul}"
            elif rate <= CFG['low_thresh']:
                mul = max(1, round(base * CFG['reduce_factor']))
                markov_detail = f"减倍! 模式{pat_str} 命中率{rate:.0%}≤{CFG['low_thresh']:.0%} → base{base}×{CFG['reduce_factor']}={mul}"
            else:
                markov_detail = f"标准  模式{pat_str} 命中率{rate:.0%} → base{base}"
        else:
            markov_detail = f"样本不足 模式{pat_str} 样本{stats[1]}<{CFG['min_samples']}"
    
    mul = min(max(1, mul), MAX_MUL)
    print(f"  [步骤2] Fib基础={base} → 马尔可夫: {markov_detail} → 最终倍数={mul}")
    
    # 检查: 步骤2查询的pattern_stats是否已被步骤1用当期hit污染?
    if len(recent) >= pl:
        pat = tuple(recent[-pl:])
        # 步骤1更新的模式 和 步骤2查询的模式 是同一个模式!
        print(f"  [审计] ⚠️ 步骤1更新的模式 = 步骤2查询的模式 = {pat_str}")
        print(f"         步骤1已用当期hit更新了这个模式的统计")
        print(f"         步骤2查询时读到的是已被当期结果污染的统计")
        print(f"         → 这就是未来函数! 倍数决策使用了当期的命中结果!")
    
    # 步骤3: 结算 (使用hit)
    if hit:
        fib_idx = 0
    else:
        fib_idx += 1
    
    # 步骤4: 将结果加入recent
    recent.append(1 if hit else 0)
    if len(recent) > CFG['lookback']:
        recent.pop(0)
    print(f"  [步骤4] recent_results(更新后) = {['✓' if r else '✗' for r in recent]}")

print(f"\n{'='*100}")
print(f"审计结论")
print(f"{'='*100}")
if issues_found > 0:
    print(f"\n❌ 发现 {issues_found} 处未来函数问题!")
    print(f"\n问题根源:")
    print(f"  process_period(hit) 内部执行顺序:")
    print(f"    1. 用 hit 更新 pattern_stats  ← 问题在这里!")
    print(f"    2. 用 pattern_stats 计算倍数   ← 读到了被当期hit污染的数据")
    print(f"    3. 用 hit 结算盈亏")
    print(f"    4. 将 hit 加入 recent_results")
    print(f"\n  步骤1在步骤2之前执行，导致倍数计算时已经知道了当期结果。")
    print(f"  例如: 如果当期命中，步骤1会提高当前模式的命中率，")
    print(f"        然后步骤2查询这个被提高的命中率来决定倍数。")
    print(f"        这相当于'已知结果后再决定下注金额'。")
    print(f"\n修复方案:")
    print(f"  将步骤1(更新pattern_stats)移到步骤4(更新recent_results)之后，")
    print(f"  或者在步骤2(计算倍数)之后再执行步骤1。")
    print(f"  正确顺序: 计算倍数 → 结算 → 更新统计 → 更新recent")
else:
    print(f"\n✅ 未发现未来函数问题")

# === 量化影响 ===
print(f"\n{'='*100}")
print(f"量化影响: 修复前 vs 修复后 (300期)")
print(f"{'='*100}")

# 重新用300期数据
predictor2 = PreciseTop15Predictor()
hit_seq_300 = []
start_300 = len(df) - 300
for i in range(start_300, len(df)):
    lo = max(0, i - TRAIN_WINDOW)
    train = numbers[lo:i]
    preds = predictor2.predict(train, k=PREDICT_K)
    actual = int(numbers[i])
    hit = actual in preds
    predictor2.update_performance(preds, actual)
    hit_seq_300.append(hit)

BASE_COST = 15; WIN_REWARD = 47

def sim_buggy(hit_seq, cfg):
    """有bug版(当前代码): 先更新stats再算倍数"""
    fib_idx = 0; bal = 0; tb = 0; tw = 0; mb = 0; sl = 0; msl = 0
    recent = []; pstats = {}; pr = 0; hm = 0
    pl = cfg['pattern_len']
    for h in hit_seq:
        if pr > 0: pr -= 1; continue
        # BUG: 先用当期hit更新stats
        if len(recent) >= pl:
            pat = tuple(recent[-pl:])
            if pat not in pstats: pstats[pat] = [0, 0]
            pstats[pat][1] += 1
            if h: pstats[pat][0] += 1
        # 然后用污染后的stats算倍数
        base = min(FIB[min(fib_idx, len(FIB)-1)], MAX_MUL)
        mul = base
        if len(recent) >= pl:
            pat = tuple(recent[-pl:])
            s = pstats.get(pat, [0, 0])
            if s[1] >= cfg['min_samples']:
                r = s[0]/s[1]
                if r >= cfg['high_thresh']: mul = round(base * cfg['boost_factor'])
                elif r <= cfg['low_thresh']: mul = max(1, round(base * cfg['reduce_factor']))
        mul = min(max(1, mul), MAX_MUL)
        bet = BASE_COST * mul; tb += bet
        if mul >= MAX_MUL: hm += 1
        if h:
            w = WIN_REWARD * mul; tw += w; bal += (w-bet); fib_idx = 0; sl = 0; pr = 1
        else:
            bal -= bet; fib_idx += 1; sl += bet
            if sl > msl: msl = sl
        if bal < mb: mb = bal
        recent.append(1 if h else 0)
        if len(recent) > cfg['lookback']: recent.pop(0)
    dd = abs(mb); roi = (tw-tb)/tb*100 if tb > 0 else 0; rr = bal/dd if dd > 0 else 0
    return {'profit': bal, 'roi': roi, 'dd': dd, 'msl': msl, 'rr': rr, 'hm': hm}

def sim_fixed(hit_seq, cfg):
    """修复版: 先算倍数，再更新stats"""
    fib_idx = 0; bal = 0; tb = 0; tw = 0; mb = 0; sl = 0; msl = 0
    recent = []; pstats = {}; pr = 0; hm = 0
    pl = cfg['pattern_len']
    for h in hit_seq:
        if pr > 0: pr -= 1; continue
        # 正确: 先用干净的stats算倍数
        base = min(FIB[min(fib_idx, len(FIB)-1)], MAX_MUL)
        mul = base
        if len(recent) >= pl:
            pat = tuple(recent[-pl:])
            s = pstats.get(pat, [0, 0])
            if s[1] >= cfg['min_samples']:
                r = s[0]/s[1]
                if r >= cfg['high_thresh']: mul = round(base * cfg['boost_factor'])
                elif r <= cfg['low_thresh']: mul = max(1, round(base * cfg['reduce_factor']))
        mul = min(max(1, mul), MAX_MUL)
        bet = BASE_COST * mul; tb += bet
        if mul >= MAX_MUL: hm += 1
        if h:
            w = WIN_REWARD * mul; tw += w; bal += (w-bet); fib_idx = 0; sl = 0; pr = 1
        else:
            bal -= bet; fib_idx += 1; sl += bet
            if sl > msl: msl = sl
        if bal < mb: mb = bal
        # 修复: 先更新stats，再更新recent
        if len(recent) >= pl:
            pat = tuple(recent[-pl:])
            if pat not in pstats: pstats[pat] = [0, 0]
            pstats[pat][1] += 1
            if h: pstats[pat][0] += 1
        recent.append(1 if h else 0)
        if len(recent) > cfg['lookback']: recent.pop(0)
    dd = abs(mb); roi = (tw-tb)/tb*100 if tb > 0 else 0; rr = bal/dd if dd > 0 else 0
    return {'profit': bal, 'roi': roi, 'dd': dd, 'msl': msl, 'rr': rr, 'hm': hm}

cfg = {'pattern_len': 3, 'boost_factor': 2.5, 'reduce_factor': 0.4,
       'high_thresh': 0.40, 'low_thresh': 0.20, 'min_samples': 2, 'lookback': 12}

buggy = sim_buggy(hit_seq_300, cfg)
fixed = sim_fixed(hit_seq_300, cfg)

print(f"\n{'版本':<20} {'ROI':>8} {'净利润':>8} {'回撤':>8} {'连挂额':>8} {'风险比':>8}")
print(f"{'-'*60}")
print(f"  {'有bug(当前)':} {buggy['roi']:>7.1f}% {buggy['profit']:>+7.0f}元 {buggy['dd']:>6.0f}元 {buggy['msl']:>6.0f}元 {buggy['rr']:>7.2f}")
print(f"  {'修复后(正确)':} {fixed['roi']:>7.1f}% {fixed['profit']:>+7.0f}元 {fixed['dd']:>6.0f}元 {fixed['msl']:>6.0f}元 {fixed['rr']:>7.2f}")
print(f"\n差异:")
print(f"  ROI: {fixed['roi']-buggy['roi']:+.1f}%")
print(f"  利润: {fixed['profit']-buggy['profit']:+.0f}元")
print(f"  回撤: {fixed['dd']-buggy['dd']:+.0f}元")
print(f"  风险比: {fixed['rr']-buggy['rr']:+.2f}")
