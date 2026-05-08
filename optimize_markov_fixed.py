"""修复未来函数后，重新网格搜索最优参数"""
import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor
from itertools import product

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values
TEST_PERIODS = 300
start_idx = len(df) - TEST_PERIODS
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

print(f"命中: {sum(hit_seq)}/{TEST_PERIODS} = {sum(hit_seq)/TEST_PERIODS*100:.1f}%")

def sim_correct(hit_seq, pause_mode, cfg):
    """修复版: 先算倍数 → 结算 → 再更新stats → 更新recent"""
    fib_idx = 0; bal = 0; tb = 0; tw = 0; mb = 0; sl = 0; msl = 0; cm = 0; mcm = 0; hm = 0
    recent = []; pstats = {}; pr = 0; bp = 0
    pl = cfg['pattern_len']
    for h in hit_seq:
        if pr > 0: pr -= 1; continue
        bp += 1
        # 步骤1: 先用干净的stats算倍数
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
        # 步骤2: 结算
        if h:
            w = WIN_REWARD * mul; tw += w; bal += (w-bet); fib_idx = 0; sl = 0; cm = 0
            if pause_mode >= 1: pr = pause_mode
        else:
            bal -= bet; fib_idx += 1; sl += bet; cm += 1
            if sl > msl: msl = sl
            if cm > mcm: mcm = cm
        if bal < mb: mb = bal
        # 步骤3: 更新stats (结算之后!)
        if len(recent) >= pl:
            pat = tuple(recent[-pl:])
            if pat not in pstats: pstats[pat] = [0, 0]
            pstats[pat][1] += 1
            if h: pstats[pat][0] += 1
        # 步骤4: 更新recent
        recent.append(1 if h else 0)
        if len(recent) > cfg.get('lookback', 12): recent.pop(0)
    dd = abs(mb); roi = (tw-tb)/tb*100 if tb > 0 else 0; rr = bal/dd if dd > 0 else 0
    return {'profit': bal, 'roi': roi, 'dd': dd, 'msl': msl, 'mcm': mcm, 'hm': hm, 'rr': rr, 'bp': bp, 'tb': tb, 'tw': tw}

# 基线: 纯Fib(无马尔可夫)
pure_fib = sim_correct(hit_seq, 1, {'pattern_len': 3, 'boost_factor': 1.0, 'reduce_factor': 1.0,
    'high_thresh': 1.0, 'low_thresh': 0.0, 'min_samples': 999, 'lookback': 12})
print(f"\n纯Fib基线(停1): ROI={pure_fib['roi']:.1f}%, 利润={pure_fib['profit']:.0f}, 回撤={pure_fib['dd']:.0f}, 风险比={pure_fib['rr']:.2f}")

# 网格搜索
results = []
param_grid = list(product(
    [2, 3, 4],                   # pattern_len
    [1.3, 1.5, 1.8, 2.0, 2.5, 3.0],  # boost
    [0.3, 0.4, 0.5, 0.6, 0.7],        # reduce
    [0.35, 0.40, 0.45, 0.50, 0.55],   # high_thresh
    [0.15, 0.20, 0.25, 0.30],         # low_thresh
    [1, 2, 3],                          # min_samples
))

print(f"\n总参数组合: {len(param_grid)}")

for pl, bf, rf, ht, lt, ms in param_grid:
    cfg = {'pattern_len': pl, 'boost_factor': bf, 'reduce_factor': rf,
           'high_thresh': ht, 'low_thresh': lt, 'min_samples': ms, 'lookback': 12}
    r = sim_correct(hit_seq, 1, cfg)
    results.append({'pl': pl, 'bf': bf, 'rf': rf, 'ht': ht, 'lt': lt, 'ms': ms,
                    'params': f"窗{pl} b{bf} r{rf} 高{ht:.0%} 低{lt:.0%} s{ms}", **r})

# TOP20 by ROI
by_roi = sorted(results, key=lambda x: x['roi'], reverse=True)
print(f"\n{'='*110}")
print(f"TOP20 按ROI（修复后真实数据）")
print(f"{'='*110}")
print(f"{'#':<4} {'参数':<40} {'ROI':>7} {'利润':>8} {'回撤':>7} {'连挂':>7} {'风险比':>7} {'触顶':>5}")
print(f"{'-'*85}")
for i, r in enumerate(by_roi[:20], 1):
    m = " ⭐" if i <= 3 else ""
    print(f"  {i:<3} {r['params']:<38} {r['roi']:>6.1f}% {r['profit']:>+7.0f} {r['dd']:>6.0f} {r['msl']:>6.0f} {r['rr']:>6.2f} {r['hm']:>4}{m}")

# TOP10 by 风险比
by_rr = sorted(results, key=lambda x: x['risk_reward'] if 'risk_reward' in x else x['rr'], reverse=True)
print(f"\n{'='*110}")
print(f"TOP10 按风险收益比")
print(f"{'='*110}")
for i, r in enumerate(by_rr[:10], 1):
    m = " ⭐" if i <= 3 else ""
    print(f"  {i:<3} {r['params']:<38} {r['roi']:>6.1f}% {r['profit']:>+7.0f} {r['dd']:>6.0f} {r['msl']:>6.0f} {r['rr']:>6.2f} {r['hm']:>4}{m}")

# 综合评分
max_roi = max(r['roi'] for r in results); min_roi = min(r['roi'] for r in results)
max_rr = max(r['rr'] for r in results); min_rr = min(r['rr'] for r in results)
max_dd = max(r['dd'] for r in results); min_dd = min(r['dd'] for r in results)
for r in results:
    rs = (r['roi']-min_roi)/(max_roi-min_roi) if max_roi>min_roi else 0
    rrs = (r['rr']-min_rr)/(max_rr-min_rr) if max_rr>min_rr else 0
    dds = 1-(r['dd']-min_dd)/(max_dd-min_dd) if max_dd>min_dd else 0
    r['score'] = rs*0.4 + rrs*0.3 + dds*0.3

by_score = sorted(results, key=lambda x: x['score'], reverse=True)
print(f"\n{'='*110}")
print(f"综合TOP10 (ROI 40% + 风险比 30% + 低回撤 30%)")
print(f"{'='*110}")
print(f"{'#':<4} {'分数':>5} {'参数':<40} {'ROI':>7} {'利润':>8} {'回撤':>7} {'风险比':>7}")
print(f"{'-'*80}")
for i, r in enumerate(by_score[:10], 1):
    m = " 🏆" if i <= 3 else ""
    print(f"  {i:<3} {r['score']:.3f} {r['params']:<38} {r['roi']:>6.1f}% {r['profit']:>+7.0f} {r['dd']:>6.0f} {r['rr']:>6.2f}{m}")

best = by_score[0]
print(f"\n{'='*110}")
print(f"🏆 最优方案 vs 纯Fib基线")
print(f"{'='*110}")
print(f"  纯Fib(停1):   ROI={pure_fib['roi']:.1f}%, 利润={pure_fib['profit']:+.0f}元, 回撤={pure_fib['dd']:.0f}元, 风险比={pure_fib['rr']:.2f}")
print(f"  最优马尔可夫:  ROI={best['roi']:.1f}%, 利润={best['profit']:+.0f}元, 回撤={best['dd']:.0f}元, 风险比={best['rr']:.2f}")
print(f"  提升:         ROI{best['roi']-pure_fib['roi']:+.1f}%, 利润{best['profit']-pure_fib['profit']:+.0f}元, 回撤{best['dd']-pure_fib['dd']:+.0f}元, 风险比{best['rr']-pure_fib['rr']:+.2f}")
print(f"\n最优参数: 窗口={best['pl']}, boost={best['bf']}, reduce={best['rf']}, "
      f"高阈值={best['ht']:.0%}, 低阈值={best['lt']:.0%}, 样本>={best['ms']}")
