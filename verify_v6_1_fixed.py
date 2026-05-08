"""最终验证: v6.1修复版（无未来函数）"""
import pandas as pd
from precise_top15_predictor import PreciseTop15Predictor

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values
TEST_PERIODS = 300; start_idx = len(df) - TEST_PERIODS
PREDICT_K = 15; TRAIN_WINDOW = 25; BASE_COST = 15; WIN_REWARD = 47; MAX_MUL = 10
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

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

CFG = {'pattern_len': 2, 'boost_factor': 2.5, 'reduce_factor': 0.4,
       'high_thresh': 0.35, 'low_thresh': 0.25, 'min_samples': 1, 'lookback': 12}

def sim(hit_seq, pause, cfg):
    fib_idx = 0; bal = 0; tb = 0; tw = 0; mb = 0; sl = 0; msl = 0; cm = 0; mcm = 0; hm = 0
    recent = []; pstats = {}; pr = 0; bp = 0; sp = 0
    pl = cfg['pattern_len']
    for h in hit_seq:
        if pr > 0: pr -= 1; sp += 1; continue
        bp += 1
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
            w = WIN_REWARD * mul; tw += w; bal += (w-bet); fib_idx = 0; sl = 0; cm = 0
            if pause >= 1: pr = pause
        else:
            bal -= bet; fib_idx += 1; sl += bet; cm += 1
            if sl > msl: msl = sl
            if cm > mcm: mcm = cm
        if bal < mb: mb = bal
        if len(recent) >= pl:
            pat = tuple(recent[-pl:])
            if pat not in pstats: pstats[pat] = [0, 0]
            pstats[pat][1] += 1
            if h: pstats[pat][0] += 1
        recent.append(1 if h else 0)
        if len(recent) > cfg['lookback']: recent.pop(0)
    dd = abs(mb); roi = (tw-tb)/tb*100 if tb > 0 else 0; rr = bal/dd if dd > 0 else 0
    return {'profit': bal, 'roi': roi, 'dd': dd, 'msl': msl, 'mcm': mcm, 'hm': hm, 'rr': rr, 'bp': bp, 'sp': sp}

fib_only = {'pattern_len': 2, 'boost_factor': 1.0, 'reduce_factor': 1.0,
    'high_thresh': 1.0, 'low_thresh': 0.0, 'min_samples': 999, 'lookback': 12}
r_fib = sim(hit_seq, 1, fib_only)
r_v61_base = sim(hit_seq, 0, CFG)
r_v61_pause = sim(hit_seq, 1, CFG)

hits = sum(hit_seq)
print(f"命中率: {hits}/{TEST_PERIODS} = {hits/TEST_PERIODS*100:.1f}%")
print(f"\n{'='*80}")
print(f"v6.1 最终验证（无未来函数，5400组网格搜索最优）")
print(f"{'='*80}")
print(f"{'版本':<28} {'ROI':>8} {'净利润':>8} {'回撤':>8} {'连挂额':>8} {'触顶':>6} {'风险比':>8}")
print(f"{'-'*80}")
for name, r in [
    ("纯Fib(停1) 基线", r_fib),
    ("v6.1 马尔可夫(优化) 基础", r_v61_base),
    ("v6.1 马尔可夫(优化) 停1", r_v61_pause),
]:
    print(f"  {name:<26} {r['roi']:>7.1f}% {r['profit']:>+7.0f}元 {r['dd']:>6.0f}元 {r['msl']:>6.0f}元 {r['hm']:>4}次 {r['rr']:>7.2f}")
print(f"\n马尔可夫(停1) vs 纯Fib(停1):")
print(f"  ROI: {r_v61_pause['roi']-r_fib['roi']:+.1f}%")
print(f"  利润: {r_v61_pause['profit']-r_fib['profit']:+.0f}元")
print(f"  回撤: {r_v61_pause['dd']-r_fib['dd']:+.0f}元")
print(f"  风险比: {r_v61_pause['rr']-r_fib['rr']:+.2f}")
print(f"\n✅ 所有数据均基于严格的时序验证，无未来函数污染")
