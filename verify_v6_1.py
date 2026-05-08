import pandas as pd
from precise_top15_predictor import PreciseTop15Predictor

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values
TEST_PERIODS = 300
start_idx = len(df) - TEST_PERIODS
PREDICT_K = 15; TRAIN_WINDOW = 25; BASE_COST = 15; WIN_REWARD = 47; MAX_MUL = 10

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

FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

# v6.1 params (from grid search)
CFG = {'pattern_len': 3, 'boost_factor': 2.5, 'reduce_factor': 0.4,
       'high_thresh': 0.40, 'low_thresh': 0.20, 'min_samples': 2, 'lookback': 12}

def sim(hit_seq, pause_mode, cfg):
    fib_idx = 0; bal = 0; tb = 0; tw = 0; mb = 0; sl = 0; msl = 0; cm = 0; mcm = 0; hm = 0
    recent = []; pstats = {}; pr = 0; bp = 0; sp = 0
    for h in hit_seq:
        if pr > 0: pr -= 1; sp += 1; continue
        bp += 1
        pl = cfg['pattern_len']
        if len(recent) >= pl:
            pat = tuple(recent[-pl:])
            if pat not in pstats: pstats[pat] = [0, 0]
            pstats[pat][1] += 1
            if h: pstats[pat][0] += 1
        base = min(FIB[min(fib_idx, len(FIB)-1)], MAX_MUL)
        mul = base
        if len(recent) >= pl:
            pat = tuple(recent[-pl:])
            s = pstats.get(pat, [0, 0])
            if s[1] >= cfg['min_samples']:
                r = s[0]/s[1]
                if r >= cfg['high_thresh']: mul = base * cfg['boost_factor']
                elif r <= cfg['low_thresh']: mul = base * cfg['reduce_factor']
        mul = min(max(1, round(mul)), MAX_MUL)
        bet = BASE_COST * mul; tb += bet
        if mul >= MAX_MUL: hm += 1
        if h:
            w = WIN_REWARD * mul; tw += w; bal += (w - bet); fib_idx = 0; sl = 0; cm = 0
            if pause_mode >= 1: pr = pause_mode
        else:
            bal -= bet; fib_idx += 1; sl += bet; cm += 1
            if sl > msl: msl = sl
            if cm > mcm: mcm = cm
        if bal < mb: mb = bal
        recent.append(1 if h else 0)
        if len(recent) > cfg['lookback']: recent.pop(0)
    dd = abs(mb); roi = (tw-tb)/tb*100 if tb > 0 else 0; rr = bal/dd if dd > 0 else 0
    return {'profit': bal, 'roi': roi, 'dd': dd, 'msl': msl, 'mcm': mcm, 'hm': hm, 'rr': rr, 'bp': bp, 'sp': sp, 'tb': tb, 'tw': tw}

# v6.0 baseline
v60 = sim(hit_seq, 1, {'pattern_len': 3, 'boost_factor': 1.5, 'reduce_factor': 0.7,
    'high_thresh': 0.50, 'low_thresh': 0.25, 'min_samples': 3, 'lookback': 12})
# v6.1 optimized
v61 = sim(hit_seq, 1, CFG)
# v6.1 no pause
v61b = sim(hit_seq, 0, CFG)

hits = sum(hit_seq)
print(f"命中率: {hits}/{TEST_PERIODS} = {hits/TEST_PERIODS*100:.1f}%")
print(f"\n{'='*70}")
print(f"版本对比")
print(f"{'='*70}")
print(f"{'版本':<25} {'ROI':>8} {'净利润':>8} {'回撤':>8} {'连挂额':>8} {'触顶':>6} {'风险比':>8}")
print(f"{'-'*75}")
for name, r in [("v6.0 Markov(旧参数) 停1", v60), ("v6.1 Markov(优化) 基础", v61b), ("v6.1 Markov(优化) 停1", v61)]:
    print(f"  {name:<23} {r['roi']:>7.1f}% {r['profit']:>+7.0f}元 {r['dd']:>6.0f}元 {r['msl']:>6.0f}元 {r['hm']:>4}次 {r['rr']:>7.2f}")
print(f"\nv6.0→v6.1 提升: ROI {v61['roi']-v60['roi']:+.1f}%, 回撤 {v61['dd']-v60['dd']:+.0f}元, 风险比 {v61['rr']-v60['rr']:+.2f}")
