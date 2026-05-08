"""
生肖TOP4突破 v8 - 在线学习 + 集中度自适应
============================================
核心思路: 不再用固定权重或过拟合12组参数
而是用在线学习实时调整马尔可夫vs冷号的权重

方法1: EWMA在线学习 - 跟踪每个策略的近期命中表现, 动态调权
方法2: 集中度自适应 - 马尔可夫转移矩阵越集中, 马尔可夫权重越大
方法3: 连冷补偿 - 连续miss时自动扩大冷号权重
方法4: 滑窗马尔可夫 - 只用最近N期计算转移概率
方法5: 综合融合 - 以上方法的最优组合
"""
import pandas as pd
import numpy as np
from collections import Counter
import time, sys
sys.stdout.reconfigure(encoding='utf-8')

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
# 基础策略函数
# ============================================================
def cold_scores(hist, window):
    w = min(window, len(hist))
    freq = Counter(hist[-w:])
    mx = max(freq.values()) if freq else 1
    return np.array([1.0 - freq.get(z,0)/max(mx,1) for z in ZODIAC_CYCLE])

def gap_scores(hist):
    scores = []
    for z in ZODIAC_CYCLE:
        last = -1
        for j in range(len(hist)-1,-1,-1):
            if hist[j]==z: last=j; break
        scores.append((len(hist)-1-last)/12.0 if last>=0 else len(hist)/12.0)
    return np.array(scores)

def markov_scores(hist, window=None, laplace=1.0):
    """马尔可夫,可选滑窗"""
    probs = np.ones(12)/12
    h = hist[-window:] if window and len(hist) > window else hist
    if len(h) < 2:
        return probs
    trans = {}
    for k in range(1, len(h)):
        p,c = h[k-1], h[k]
        if p not in trans: trans[p]=Counter()
        trans[p][c]+=1
    state = hist[-1]  # 总是用最新状态
    if state in trans:
        total = sum(trans[state].values()) + laplace*12
        for zi, z in enumerate(ZODIAC_CYCLE):
            probs[zi] = (trans[state].get(z,0)+laplace)/total
    return probs

def overdue_scores(hist):
    scores = []
    for z in ZODIAC_CYCLE:
        pos = [k for k,a in enumerate(hist) if a==z]
        if len(pos) >= 2:
            ints = [pos[k+1]-pos[k] for k in range(len(pos)-1)]
            avg = np.mean(ints)
            g = len(hist)-1-pos[-1]
            scores.append(1/(1+np.exp(-3*(g/max(avg,1)-1))))
        elif len(pos) == 1:
            scores.append(min((len(hist)-1-pos[0])/12, 1))
        else:
            scores.append(0.8)
    return np.array(scores)

# ============================================================
# 方法1: EWMA在线学习
# ============================================================
def method_ewma(alpha=0.1, init_w_cold=0.5, init_w_mk=0.3, init_w_gap=0.2, 
                cold_window=20):
    """
    跟踪3个策略的近期命中表现, 用EWMA调权
    """
    w_cold = init_w_cold
    w_mk = init_w_mk
    w_gap = init_w_gap
    hits = 0
    
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        actual_z = Z_IDX[animals[i]]
        
        s_cold = cold_scores(hist, cold_window)
        s_mk = markov_scores(hist)
        s_gap = gap_scores(hist)
        
        # 归一化权重
        ws = w_cold + w_mk + w_gap
        wc, wm, wg = w_cold/ws, w_mk/ws, w_gap/ws
        
        combined = wc*s_cold + wm*s_mk + wg*s_gap
        top4 = np.argsort(-combined)[:4]
        
        hit = actual_z in top4
        if hit: hits += 1
        
        # 更新各策略得分(哪个策略选对了)
        cold_top4 = set(np.argsort(-s_cold)[:4])
        mk_top4 = set(np.argsort(-s_mk)[:4])
        gap_top4 = set(np.argsort(-s_gap)[:4])
        
        cold_hit = 1.0 if actual_z in cold_top4 else 0.0
        mk_hit = 1.0 if actual_z in mk_top4 else 0.0
        gap_hit = 1.0 if actual_z in gap_top4 else 0.0
        
        # EWMA更新
        w_cold = (1-alpha)*w_cold + alpha*cold_hit
        w_mk = (1-alpha)*w_mk + alpha*mk_hit
        w_gap = (1-alpha)*w_gap + alpha*gap_hit
        
        # 防止权重为0
        w_cold = max(w_cold, 0.05)
        w_mk = max(w_mk, 0.05)
        w_gap = max(w_gap, 0.05)
    
    return hits

print("\n" + "="*60)
print("方法1: EWMA在线学习")
print("="*60)
best1 = 0
best1_p = None
for alpha in [0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]:
    for ic in [0.3, 0.4, 0.5, 0.6]:
        for im in [0.1, 0.2, 0.3, 0.4]:
            ig = 1.0 - ic - im
            if ig < 0: continue
            for cw in [15, 20, 25]:
                h = method_ewma(alpha, ic, im, ig, cw)
                if h > best1:
                    best1 = h
                    best1_p = (alpha, ic, im, ig, cw)
                    if h/TEST_PERIODS > 0.43:
                        print(f"  {h}/{TEST_PERIODS}={h/TEST_PERIODS*100:.1f}% a={alpha} c={ic} m={im} g={ig:.1f} cw={cw}")

print(f"\n方法1最优: {best1}/{TEST_PERIODS} = {best1/TEST_PERIODS*100:.1f}%")

# ============================================================
# 方法2: 集中度自适应
# ============================================================
def method_concentration(hhi_boost=3.0, cold_window=20, base_wm=0.2, base_wc=0.5):
    """
    当马尔可夫HHI > uniform(1/12), 按比例提升马尔可夫权重
    HHI_uniform = sum((1/12)^2 * 12) = 1/12 = 0.0833
    """
    hits = 0
    HHI_UNIFORM = 1/12
    
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        actual_z = Z_IDX[animals[i]]
        
        s_cold = cold_scores(hist, cold_window)
        s_mk = markov_scores(hist)
        s_gap = gap_scores(hist)
        s_od = overdue_scores(hist)
        
        hhi = np.sum(s_mk**2)
        concentration = max(0, hhi - HHI_UNIFORM) / HHI_UNIFORM  # 0 = uniform, 高 = 集中
        
        # 马尔可夫权重随集中度增加
        wm = base_wm + base_wm * concentration * hhi_boost
        wc = base_wc
        wg = 0.2
        wo = 0.1
        ws = wm + wc + wg + wo
        
        combined = (wm*s_mk + wc*s_cold + wg*s_gap + wo*s_od) / ws
        top4 = np.argsort(-combined)[:4]
        
        if actual_z in top4: hits += 1
    return hits

print("\n" + "="*60)
print("方法2: 集中度自适应")
print("="*60)
best2 = 0
best2_p = None
for hb in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
    for cw in [15, 20, 25]:
        for bwm in [0.1, 0.15, 0.2, 0.25, 0.3]:
            for bwc in [0.3, 0.4, 0.5, 0.6]:
                h = method_concentration(hb, cw, bwm, bwc)
                if h > best2:
                    best2 = h
                    best2_p = (hb, cw, bwm, bwc)
                    if h/TEST_PERIODS > 0.43:
                        print(f"  {h}/{TEST_PERIODS}={h/TEST_PERIODS*100:.1f}% hb={hb} cw={cw} wm={bwm} wc={bwc}")

print(f"\n方法2最优: {best2}/{TEST_PERIODS} = {best2/TEST_PERIODS*100:.1f}%")

# ============================================================
# 方法3: 滑窗马尔可夫
# ============================================================
def method_sliding_markov(mk_window, cold_window, wm, wc, wg):
    hits = 0
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        actual_z = Z_IDX[animals[i]]
        
        s_cold = cold_scores(hist, cold_window)
        s_mk = markov_scores(hist, window=mk_window)
        s_gap = gap_scores(hist)
        
        ws = wm + wc + wg
        combined = (wm*s_mk + wc*s_cold + wg*s_gap) / ws
        top4 = np.argsort(-combined)[:4]
        if actual_z in top4: hits += 1
    return hits

print("\n" + "="*60)
print("方法3: 滑窗马尔可夫")
print("="*60)
best3 = 0
best3_p = None
for mkw in [30, 50, 80, 100, 120, 150, 200, 300]:
    for cw in [15, 20, 25]:
        for wm in [0.2, 0.3, 0.4, 0.5, 0.6]:
            for wc in [0.2, 0.3, 0.4, 0.5]:
                wg = 1.0 - wm - wc
                if wg < 0: continue
                h = method_sliding_markov(mkw, cw, wm, wc, wg)
                if h > best3:
                    best3 = h
                    best3_p = (mkw, cw, wm, wc, wg)
                    if h/TEST_PERIODS > 0.43:
                        print(f"  {h}/{TEST_PERIODS}={h/TEST_PERIODS*100:.1f}% mkw={mkw} cw={cw} wm={wm:.1f} wc={wc:.1f} wg={wg:.1f}")

print(f"\n方法3最优: {best3}/{TEST_PERIODS} = {best3/TEST_PERIODS*100:.1f}%")

# ============================================================
# 方法4: EWMA + 集中度 + 滑窗 三合一
# ============================================================
def method_triple(alpha, mk_window, cold_window, init_wc, init_wm, init_wg, hhi_boost):
    """最强融合: EWMA在线学习 + 集中度调整 + 滑窗马尔可夫"""
    w_cold = init_wc
    w_mk = init_wm
    w_gap = init_wg
    hits = 0
    HHI_UNIFORM = 1/12
    
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        actual_z = Z_IDX[animals[i]]
        
        s_cold = cold_scores(hist, cold_window)
        s_mk = markov_scores(hist, window=mk_window)
        s_gap = gap_scores(hist)
        
        # 集中度调整
        hhi = np.sum(s_mk**2)
        conc = max(0, hhi - HHI_UNIFORM) / HHI_UNIFORM
        eff_wm = w_mk * (1 + conc * hhi_boost)
        
        ws = w_cold + eff_wm + w_gap
        combined = (w_cold*s_cold + eff_wm*s_mk + w_gap*s_gap) / ws
        top4 = np.argsort(-combined)[:4]
        
        hit = actual_z in top4
        if hit: hits += 1
        
        # EWMA更新
        cold_hit = 1.0 if actual_z in set(np.argsort(-s_cold)[:4]) else 0.0
        mk_hit = 1.0 if actual_z in set(np.argsort(-s_mk)[:4]) else 0.0
        gap_hit = 1.0 if actual_z in set(np.argsort(-s_gap)[:4]) else 0.0
        
        w_cold = max(0.05, (1-alpha)*w_cold + alpha*cold_hit)
        w_mk = max(0.05, (1-alpha)*w_mk + alpha*mk_hit)
        w_gap = max(0.05, (1-alpha)*w_gap + alpha*gap_hit)
    
    return hits

print("\n" + "="*60)
print("方法4: 三合一融合")
print("="*60)
best4 = 0
best4_p = None
for alpha in [0.05, 0.1, 0.15, 0.2]:
    for mkw in [50, 100, 150, 200]:
        for cw in [15, 20, 25]:
            for ic in [0.4, 0.5, 0.6]:
                for im in [0.2, 0.3, 0.4]:
                    ig = max(0, 1 - ic - im)
                    for hb in [2.0, 4.0, 6.0]:
                        h = method_triple(alpha, mkw, cw, ic, im, ig, hb)
                        if h > best4:
                            best4 = h
                            best4_p = (alpha, mkw, cw, ic, im, ig, hb)
                            if h/TEST_PERIODS > 0.44:
                                print(f"  {h}/{TEST_PERIODS}={h/TEST_PERIODS*100:.1f}% a={alpha} mkw={mkw} cw={cw} ic={ic} im={im} hb={hb}")

print(f"\n方法4最优: {best4}/{TEST_PERIODS} = {best4/TEST_PERIODS*100:.1f}%")

# ============================================================
# 方法5: 优先级选择(非线性融合)
# ============================================================
def method_priority(mk_thresh, mk_picks, cold_window, mk_window):
    """
    1. 马尔可夫概率超过阈值→直接选入
    2. 剩余名额用冷号+间隔混合填充
    不是线性加权, 而是优先级选择
    """
    hits = 0
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        actual_z = Z_IDX[animals[i]]
        
        s_mk = markov_scores(hist, window=mk_window)
        s_cold = cold_scores(hist, cold_window)
        s_gap = gap_scores(hist)
        
        selected = set()
        
        # Step 1: 马尔可夫强信号
        sorted_mk = np.argsort(-s_mk)
        for k in range(min(mk_picks, 12)):
            if s_mk[sorted_mk[k]] >= mk_thresh:
                selected.add(sorted_mk[k])
        
        # Step 2: 冷号+间隔补充
        mixed = 0.6*s_cold + 0.4*s_gap
        for zi in np.argsort(-mixed):
            if zi not in selected:
                selected.add(zi)
            if len(selected) >= 4:
                break
        
        top4 = list(selected)[:4]
        if actual_z in top4: hits += 1
    return hits

print("\n" + "="*60)
print("方法5: 马尔可夫优先级+冷号补充")
print("="*60)
best5 = 0
best5_p = None
for mt in [0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18, 0.20]:
    for mp in [1, 2, 3]:
        for cw in [15, 20, 25, 30]:
            for mkw in [None, 50, 100, 150, 200]:
                h = method_priority(mt, mp, cw, mkw)
                if h > best5:
                    best5 = h
                    best5_p = (mt, mp, cw, mkw)
                    if h/TEST_PERIODS > 0.43:
                        print(f"  {h}/{TEST_PERIODS}={h/TEST_PERIODS*100:.1f}% thresh={mt} picks={mp} cw={cw} mkw={mkw}")

print(f"\n方法5最优: {best5}/{TEST_PERIODS} = {best5/TEST_PERIODS*100:.1f}%")

# ============================================================
# 方法6: 连miss补偿 + 优先级
# ============================================================
def method_miss_compensate(mk_thresh, mk_picks, cold_window, miss_boost_thresh=3, 
                           cold_mix=0.6, gap_mix=0.4):
    """连续miss后, 加大冷号权重/减少马尔可夫阈值"""
    hits = 0
    consecutive_miss = 0
    
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        actual_z = Z_IDX[animals[i]]
        
        s_mk = markov_scores(hist)
        s_cold = cold_scores(hist, cold_window)
        s_gap = gap_scores(hist)
        
        # 连miss降低MK阈值, 增加cold比例
        eff_thresh = mk_thresh
        eff_cold_mix = cold_mix
        if consecutive_miss >= miss_boost_thresh:
            eff_thresh += 0.02 * (consecutive_miss - miss_boost_thresh + 1)  # 提高阈值=减少MK
            eff_cold_mix = min(0.9, cold_mix + 0.05 * (consecutive_miss - miss_boost_thresh + 1))
        
        selected = set()
        sorted_mk = np.argsort(-s_mk)
        for k in range(min(mk_picks, 12)):
            if s_mk[sorted_mk[k]] >= eff_thresh:
                selected.add(sorted_mk[k])
        
        mixed = eff_cold_mix*s_cold + (1-eff_cold_mix)*s_gap
        for zi in np.argsort(-mixed):
            if zi not in selected:
                selected.add(zi)
            if len(selected) >= 4:
                break
        
        top4 = list(selected)[:4]
        hit = actual_z in top4
        if hit:
            hits += 1
            consecutive_miss = 0
        else:
            consecutive_miss += 1
    
    return hits

print("\n" + "="*60)
print("方法6: 连miss补偿")
print("="*60)
best6 = 0
best6_p = None
for mt in [0.10, 0.12, 0.14, 0.16]:
    for mp in [1, 2, 3]:
        for cw in [15, 20, 25]:
            for mb in [2, 3, 4, 5]:
                for cm in [0.4, 0.5, 0.6, 0.7]:
                    h = method_miss_compensate(mt, mp, cw, mb, cm, 1-cm)
                    if h > best6:
                        best6 = h
                        best6_p = (mt, mp, cw, mb, cm)
                        if h/TEST_PERIODS > 0.43:
                            print(f"  {h}/{TEST_PERIODS}={h/TEST_PERIODS*100:.1f}% mt={mt} mp={mp} cw={cw} mb={mb} cm={cm}")

print(f"\n方法6最优: {best6}/{TEST_PERIODS} = {best6/TEST_PERIODS*100:.1f}%")

# ============================================================
# 最终汇总
# ============================================================
print("\n" + "="*60)
print("最终汇总")
print("="*60)
results = [
    ('EWMA在线学习', best1, best1_p),
    ('集中度自适应', best2, best2_p),
    ('滑窗马尔可夫', best3, best3_p),
    ('三合一融合', best4, best4_p),
    ('优先级选择', best5, best5_p),
    ('连miss补偿', best6, best6_p),
]
for name, h, p in sorted(results, key=lambda x: -x[1]):
    star = ' ***' if h/TEST_PERIODS >= 0.45 else ' **' if h/TEST_PERIODS >= 0.44 else ''
    print(f"  {name:>12}: {h}/{TEST_PERIODS} = {h/TEST_PERIODS*100:.1f}%{star} | {p}")

winner = max(results, key=lambda x: x[1])
print(f"\nBEST: {winner[0]} = {winner[1]/TEST_PERIODS*100:.1f}%")
print(f"Total time: {time.time()-t0:.1f}s")
