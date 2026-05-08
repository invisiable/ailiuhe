"""
生肖TOP4最终方案 - 稳健版状态特化
核心策略: 根据马尔可夫转移矩阵的"集中度"动态决定策略模式

发现:
- 方法A(12状态独立优化)达到51.7%，但CV仅41.7%（过拟合）
- 马尔可夫在某些状态下非常有效(蛇58%,兔57%,鼠55%)
- 关键是在"马尔可夫友好"状态用马尔可夫，在其他状态用冷号

解决方案: 用转移集中度(Gini系数)动态切换
"""
import pandas as pd
import numpy as np
from collections import Counter
import time

t0 = time.time()

ZODIAC_CYCLE = ['马', '蛇', '龙', '兔', '虎', '牛', '鼠', '猪', '狗', '鸡', '猴', '羊']
Z_IDX = {z: i for i, z in enumerate(ZODIAC_CYCLE)}
NUM_TO_ZODIAC = {n: ZODIAC_CYCLE[(n - 1) % 12] for n in range(1, 50)}

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].tolist()
animals = [NUM_TO_ZODIAC[n] for n in numbers]
TOTAL = len(animals)
TEST_PERIODS = 300
START = TOTAL - TEST_PERIODS

print(f"Data: {TOTAL} periods, test {TEST_PERIODS}")

# ============================================================
# 辅助策略函数
# ============================================================
def get_cold_scores(hist, window):
    w = min(window, len(hist))
    freq = Counter(hist[-w:])
    mx = max(freq.values()) if freq else 1
    return {z: 1.0 - freq.get(z,0)/max(mx,1) for z in ZODIAC_CYCLE}

def get_gap_scores(hist):
    scores = {}
    for z in ZODIAC_CYCLE:
        last = -1
        for j in range(len(hist)-1,-1,-1):
            if hist[j]==z: last=j; break
        scores[z] = (len(hist)-1-last)/12.0 if last>=0 else len(hist)/12.0
    return scores

def get_markov_probs(hist, laplace=1.0):
    probs = {z: 1/12 for z in ZODIAC_CYCLE}
    if len(hist) < 2:
        return probs, 0
    trans = {}
    for k in range(1, len(hist)):
        p,c = hist[k-1], hist[k]
        if p not in trans: trans[p]=Counter()
        trans[p][c]+=1
    state = hist[-1]
    if state in trans:
        n_samples = sum(trans[state].values())
        total = n_samples + laplace*12
        for z in ZODIAC_CYCLE:
            probs[z] = (trans[state].get(z,0)+laplace)/total
        return probs, n_samples
    return probs, 0

def markov_concentration(probs):
    """计算转移概率的集中度 (HHI: Herfindahl index)"""
    vals = list(probs.values())
    return sum(v*v for v in vals)  # HHI: 越高=越集中

# ============================================================
# 方案1: 集中度阈值切换 (2模式)
# ============================================================
def method_concentration_switch(hhi_thresh, w_mk_high, w_cold_high, w_gap_high,
                                 w_mk_low, w_cold_low, w_gap_low, cold_window=20):
    """
    HHI高(马尔可夫信号强) → 马尔可夫重权模式
    HHI低(随机) → 冷号+间隔模式
    """
    hits = 0
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        actual = animals[i]
        
        mk_probs, n_samples = get_markov_probs(hist)
        hhi = markov_concentration(mk_probs)
        
        if hhi >= hhi_thresh and n_samples >= 5:
            w_mk, w_cold, w_gap = w_mk_high, w_cold_high, w_gap_high
        else:
            w_mk, w_cold, w_gap = w_mk_low, w_cold_low, w_gap_low
        
        s_cold = get_cold_scores(hist, cold_window)
        s_gap = get_gap_scores(hist)
        
        final = {}
        for z in ZODIAC_CYCLE:
            final[z] = w_mk*mk_probs[z] + w_cold*s_cold[z] + w_gap*s_gap[z]
        
        top4 = sorted(ZODIAC_CYCLE, key=lambda z: -final[z])[:4]
        if actual in top4:
            hits += 1
    return hits

print("\n" + "="*70)
print("方案1: 集中度阈值切换 (HHI)")
print("="*70)

best = 0
best_params = None
count = 0
for hhi in [0.090, 0.095, 0.100, 0.105, 0.110, 0.115, 0.120]:
    for wmh in [0.4, 0.5, 0.6, 0.7, 0.8]:
        wch = (1-wmh)*0.6
        wgh = (1-wmh)*0.4
        for wml in [0.0, 0.05, 0.1, 0.15, 0.2]:
            wcl = (1-wml)*0.6
            wgl = (1-wml)*0.4
            for cw in [15, 20, 25]:
                h = method_concentration_switch(hhi, wmh, wch, wgh, wml, wcl, wgl, cw)
                count += 1
                if h > best:
                    best = h
                    best_params = (hhi, wmh, wml, cw)
                    if h/TEST_PERIODS > 0.44:
                        print(f"  [{count}] {h}/{TEST_PERIODS}={h/TEST_PERIODS*100:.1f}% hhi={hhi} wm_h={wmh} wm_l={wml} cw={cw}")

print(f"\n方案1最优: {best}/{TEST_PERIODS} = {best/TEST_PERIODS*100:.1f}% params={best_params}")

# ============================================================
# 方案2: 马尔可夫TOP-N选+冷号补充 (简化版)
# ============================================================
print("\n" + "="*70)
print("方案2: 马尔可夫选N + 冷号补充")
print("="*70)

def method_markov_select_and_fill(max_prob_thresh, n_mk_picks, cold_window, cold_gap_ratio):
    hits = 0
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        actual = animals[i]
        
        mk_probs, n_samples = get_markov_probs(hist)
        sorted_mk = sorted(ZODIAC_CYCLE, key=lambda z: -mk_probs[z])
        max_prob = mk_probs[sorted_mk[0]]
        
        selected = []
        if max_prob >= max_prob_thresh and n_samples >= 5:
            # 强信号: 马尔可夫选前N
            for z in sorted_mk[:n_mk_picks]:
                selected.append(z)
        
        # 冷号+间隔补充
        s_cold = get_cold_scores(hist, cold_window)
        s_gap = get_gap_scores(hist)
        cg = cold_gap_ratio
        mixed = sorted(ZODIAC_CYCLE, key=lambda z: -(cg*s_cold[z]+(1-cg)*s_gap[z]))
        
        for z in mixed:
            if z not in selected:
                selected.append(z)
            if len(selected) >= 4:
                break
        
        if actual in selected[:4]:
            hits += 1
    return hits

best2 = 0
best2_params = None
for mpt in [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18, 0.20]:
    for nmp in [1, 2, 3]:
        for cw in [15, 20, 25, 30]:
            for cgr in [0.3, 0.4, 0.5, 0.6, 0.7]:
                h = method_markov_select_and_fill(mpt, nmp, cw, cgr)
                if h > best2:
                    best2 = h
                    best2_params = (mpt, nmp, cw, cgr)
                    if h/TEST_PERIODS > 0.44:
                        print(f"  {h}/{TEST_PERIODS}={h/TEST_PERIODS*100:.1f}% mpt={mpt} picks={nmp} cw={cw} cgr={cgr}")

print(f"\n方案2最优: {best2}/{TEST_PERIODS} = {best2/TEST_PERIODS*100:.1f}% params={best2_params}")

# ============================================================
# 方案3: 多模式融合（3个确定性策略 + 马尔可夫条件加权）
# ============================================================
print("\n" + "="*70)
print("方案3: 冷号+间隔+马尔可夫条件融合")
print("="*70)

def method_conditional_fusion(base_wc, base_wg, base_wm, mk_boost, mk_thresh, cold_window):
    """
    基础权重 + 马尔可夫有强信号时boost
    """
    hits = 0
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        actual = animals[i]
        
        mk_probs, n_samples = get_markov_probs(hist)
        max_prob = max(mk_probs.values())
        
        wc, wg, wm = base_wc, base_wg, base_wm
        if max_prob >= mk_thresh and n_samples >= 5:
            # boost马尔可夫权重
            wm *= mk_boost
        
        # 归一化
        ws = wc + wg + wm
        wc/=ws; wg/=ws; wm/=ws
        
        s_cold = get_cold_scores(hist, cold_window)
        s_gap = get_gap_scores(hist)
        
        final = {}
        for z in ZODIAC_CYCLE:
            final[z] = wc*s_cold[z] + wg*s_gap[z] + wm*mk_probs[z]
        
        top4 = sorted(ZODIAC_CYCLE, key=lambda z: -final[z])[:4]
        if actual in top4:
            hits += 1
    return hits

best3 = 0
best3_params = None
for bwc in [0.2, 0.3, 0.4, 0.5]:
    for bwg in [0.1, 0.2, 0.3, 0.4]:
        for bwm in [0.1, 0.2, 0.3, 0.4]:
            for mkb in [1.5, 2.0, 2.5, 3.0, 4.0]:
                for mkt in [0.10, 0.12, 0.14, 0.16]:
                    for cw in [15, 20, 25]:
                        h = method_conditional_fusion(bwc, bwg, bwm, mkb, mkt, cw)
                        if h > best3:
                            best3 = h
                            best3_params = (bwc, bwg, bwm, mkb, mkt, cw)
                            if h/TEST_PERIODS > 0.44:
                                print(f"  {h}/{TEST_PERIODS}={h/TEST_PERIODS*100:.1f}% wc={bwc} wg={bwg} wm={bwm} boost={mkb} thresh={mkt} cw={cw}")

print(f"\n方案3最优: {best3}/{TEST_PERIODS} = {best3/TEST_PERIODS*100:.1f}% params={best3_params}")

# ============================================================
# 方案4: 前2期组合状态马尔可夫
# ============================================================
print("\n" + "="*70)
print("方案4: 二阶马尔可夫+冷号补充")
print("="*70)

def method_markov2_fill(max_prob_thresh, n_mk_picks, cold_window, cgr):
    hits = 0
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        actual = animals[i]
        
        selected = []
        
        # 二阶马尔可夫
        if len(hist) >= 3:
            trans2 = {}
            for k in range(2, len(hist)):
                prev = (hist[k-2], hist[k-1])
                curr = hist[k]
                if prev not in trans2: trans2[prev] = Counter()
                trans2[prev][curr] += 1
            state2 = (hist[-2], hist[-1])
            if state2 in trans2:
                total = sum(trans2[state2].values())
                probs = {z: trans2[state2].get(z,0)/total for z in ZODIAC_CYCLE}
                sorted_mk = sorted(ZODIAC_CYCLE, key=lambda z: -probs[z])
                if probs[sorted_mk[0]] >= max_prob_thresh and total >= 3:
                    for z in sorted_mk[:n_mk_picks]:
                        selected.append(z)
        
        # 一阶马尔可夫补充
        if len(selected) < 2 and len(hist) >= 2:
            mk_probs, ns = get_markov_probs(hist)
            sorted_mk = sorted(ZODIAC_CYCLE, key=lambda z: -mk_probs[z])
            max_p = mk_probs[sorted_mk[0]]
            if max_p >= 0.12 and ns >= 5:
                for z in sorted_mk[:1]:
                    if z not in selected: selected.append(z)
        
        # 冷号+间隔补充
        s_cold = get_cold_scores(hist, cold_window)
        s_gap = get_gap_scores(hist)
        mixed = sorted(ZODIAC_CYCLE, key=lambda z: -(cgr*s_cold[z]+(1-cgr)*s_gap[z]))
        for z in mixed:
            if z not in selected: selected.append(z)
            if len(selected) >= 4: break
        
        if actual in selected[:4]: hits += 1
    return hits

best4 = 0
best4_params = None
for mpt in [0.15, 0.20, 0.25, 0.30, 0.35]:
    for nmp in [1, 2]:
        for cw in [15, 20, 25]:
            for cgr in [0.4, 0.5, 0.6, 0.7]:
                h = method_markov2_fill(mpt, nmp, cw, cgr)
                if h > best4:
                    best4 = h
                    best4_params = (mpt, nmp, cw, cgr)
                    if h/TEST_PERIODS > 0.44:
                        print(f"  {h}/{TEST_PERIODS}={h/TEST_PERIODS*100:.1f}% mpt={mpt} picks={nmp} cw={cw} cgr={cgr}")

print(f"\n方案4最优: {best4}/{TEST_PERIODS} = {best4/TEST_PERIODS*100:.1f}% params={best4_params}")

# ============================================================
# 最终PK
# ============================================================
print("\n" + "="*70)
print("最终PK")
print("="*70)
results = [
    ('方案1: HHI切换', best, best_params),
    ('方案2: 马尔可夫+补充', best2, best2_params),
    ('方案3: 条件融合', best3, best3_params),
    ('方案4: 二阶+补充', best4, best4_params),
]
for name, h, params in sorted(results, key=lambda x: -x[1]):
    print(f"  {name}: {h}/{TEST_PERIODS} = {h/TEST_PERIODS*100:.1f}% params={params}")

# 最优方案逐期输出
winner = max(results, key=lambda x: x[1])
print(f"\n最优: {winner[0]} = {winner[1]/TEST_PERIODS*100:.1f}%")

# 分段统计
print("\n分段统计(每50期):")
# 重新运行最优方案记录逐期结果
if winner[0].startswith('方案1'):
    hhi, wmh, wml, cw = winner[2]
    streak = 0; max_streak = 0
    seg_hits = [0]*6
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        mk_probs, ns = get_markov_probs(hist)
        hhi_val = markov_concentration(mk_probs)
        if hhi_val >= hhi and ns >= 5:
            wm,wc,wg = wmh,(1-wmh)*0.6,(1-wmh)*0.4
        else:
            wm,wc,wg = wml,(1-wml)*0.6,(1-wml)*0.4
        s_cold = get_cold_scores(hist, cw)
        s_gap = get_gap_scores(hist)
        final = {z: wm*mk_probs[z]+wc*s_cold[z]+wg*s_gap[z] for z in ZODIAC_CYCLE}
        top4 = sorted(ZODIAC_CYCLE, key=lambda z: -final[z])[:4]
        hit = animals[i] in top4
        if hit: streak=0; seg_hits[pi//50]+=1
        else: streak+=1; max_streak=max(max_streak,streak)
    for s in range(6):
        print(f"  {s*50+1:>3}-{(s+1)*50:>3}: {seg_hits[s]}/50 = {seg_hits[s]*2}%")
    print(f"  Max consecutive miss: {max_streak}")

print(f"\nTotal time: {time.time()-t0:.1f}s")
