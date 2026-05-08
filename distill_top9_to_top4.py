"""
蒸馏方案: TOP9先过滤 → 再从9个中选TOP4
============================================
思路: TOP9有85%命中率(12→9), 相当于排除了3个最不可能的生肖
     在这9个"高质量候选"中再选TOP4, 理论上比直接从12个中选TOP4更准

测试多种第二阶段(Stage2)策略:
- S2-A: 直接取TOP9的前4名 (最简单)
- S2-B: 用v3权重(cold15+cold30+mk150)在9个中重排
- S2-C: 用不同窗口的MK在9个中重排 (互补信息)
- S2-D: 多阶段融合 (TOP9分数 × Stage2分数)
- S2-E: 对比方向 - 用短窗口特征在9个中重排
- S2-F: 热号+MK短窗口 组合

对照组: 直接TOP4 v3 (48.0%)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import io
import pandas as pd
import numpy as np
from collections import Counter

ZODIAC_CYCLE_2026 = ['马', '蛇', '龙', '兔', '虎', '牛', '鼠', '猪', '狗', '鸡', '猴', '羊']
NUM_TO_ZODIAC_2026 = {n: ZODIAC_CYCLE_2026[(n - 1) % 12] for n in range(1, 50)}
ZODIAC_NUMS_2026 = {}
for z in ZODIAC_CYCLE_2026:
    ZODIAC_NUMS_2026[z] = sorted([n for n in range(1, 50) if NUM_TO_ZODIAC_2026[n] == z])

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].tolist()
total = len(numbers)
test_periods = 300
start = total - test_periods

# ===== 评分函数 =====
def cold_scores(animals, window):
    w = min(window, len(animals))
    freq = Counter(animals[-w:])
    mx = max(freq.values()) if freq else 1
    return np.array([1.0 - freq.get(z, 0) / max(mx, 1) for z in ZODIAC_CYCLE_2026])

def hot_scores(animals, window):
    w = min(window, len(animals))
    freq = Counter(animals[-w:])
    mx = max(freq.values()) if freq else 1
    return np.array([freq.get(z, 0) / max(mx, 1) for z in ZODIAC_CYCLE_2026])

def gap_scores(animals):
    scores = []
    for z in ZODIAC_CYCLE_2026:
        last = -1
        for j in range(len(animals) - 1, -1, -1):
            if animals[j] == z:
                last = j
                break
        gap = (len(animals) - 1 - last) if last >= 0 else len(animals)
        scores.append(gap / 12.0)
    return np.array(scores)

def markov_scores(animals, window=None, laplace=1.0):
    probs = np.ones(12) / 12
    h = animals[-window:] if window and len(animals) > window else animals
    if len(h) < 2: return probs
    trans = {}
    for k in range(1, len(h)):
        p, c = h[k-1], h[k]
        if p not in trans: trans[p] = Counter()
        trans[p][c] += 1
    state = animals[-1]
    if state in trans:
        t = sum(trans[state].values()) + laplace * 12
        for zi, z in enumerate(ZODIAC_CYCLE_2026):
            probs[zi] = (trans[state].get(z, 0) + laplace) / t
    return probs

def rotation_scores(animals, window=20):
    scores = []
    for z in ZODIAC_CYCLE_2026:
        positions = [j for j, a in enumerate(animals[-window:]) if a == z]
        if len(positions) >= 2:
            gaps = [positions[k+1] - positions[k] for k in range(len(positions)-1)]
            avg_gap = np.mean(gaps)
            last_gap = window - positions[-1] if positions else window
            score = 1.0 - abs(last_gap - avg_gap) / max(avg_gap, 1)
            score = max(0, score)
        else:
            score = 0.5
        scores.append(score)
    return np.array(scores)

# ===== Stage1: TOP9评分 =====
def stage1_top9_scores(animals):
    """TOP9的最优权重"""
    return (0.20 * cold_scores(animals, 15) +
            0.05 * cold_scores(animals, 30) +
            0.50 * markov_scores(animals, 150) +
            0.10 * gap_scores(animals) +
            0.15 * hot_scores(animals, 30))

# ===== Stage2: 从9个中选4个的多种策略 =====
def s2_top_rank(s1_scores, s1_idx, animals):
    """S2-A: 直接取TOP9的前4名"""
    return s1_scores[s1_idx]  # 保持原排序

def s2_v3_static(s1_scores, s1_idx, animals):
    """S2-B: 用v3静态权重重排"""
    return (0.30 * cold_scores(animals, 15) +
            0.10 * cold_scores(animals, 30) +
            0.60 * markov_scores(animals, 150))[s1_idx]

def s2_short_mk(s1_scores, s1_idx, animals):
    """S2-C: 用短窗口MK重排 (与stage1的MK150互补)"""
    mk50 = markov_scores(animals, 50)
    mk80 = markov_scores(animals, 80)
    c10 = cold_scores(animals, 10)
    return (0.40 * mk50 + 0.30 * mk80 + 0.30 * c10)[s1_idx]

def s2_fusion(s1_scores, s1_idx, animals):
    """S2-D: 融合 stage1分数 × stage2分数"""
    s2 = (0.30 * cold_scores(animals, 15) +
          0.10 * cold_scores(animals, 30) +
          0.60 * markov_scores(animals, 150))
    # 归一化后相乘
    s1_norm = s1_scores / (s1_scores.sum() + 1e-10)
    s2_norm = s2 / (s2.sum() + 1e-10)
    return (s1_norm * s2_norm)[s1_idx]

def s2_short_features(s1_scores, s1_idx, animals):
    """S2-E: 短窗口特征 (近期趋势敏感)"""
    c8 = cold_scores(animals, 8)
    c12 = cold_scores(animals, 12)
    mk50 = markov_scores(animals, 50)
    g = gap_scores(animals)
    return (0.25 * c8 + 0.15 * c12 + 0.35 * mk50 + 0.25 * g)[s1_idx]

def s2_hot_mk_short(s1_scores, s1_idx, animals):
    """S2-F: 热号+短MK (在冷号pool中挑热的)"""
    h20 = hot_scores(animals, 20)
    h10 = hot_scores(animals, 10)
    mk60 = markov_scores(animals, 60)
    return (0.30 * h20 + 0.20 * h10 + 0.50 * mk60)[s1_idx]

def s2_gap_heavy(s1_scores, s1_idx, animals):
    """S2-G: 重间隔 (回归到期)"""
    g = gap_scores(animals)
    mk100 = markov_scores(animals, 100)
    return (0.55 * g + 0.45 * mk100)[s1_idx]

def s2_diversity(s1_scores, s1_idx, animals):
    """S2-H: 多样性 - 避免选同质化的生肖"""
    mk80 = markov_scores(animals, 80)
    c15 = cold_scores(animals, 15)
    rot = rotation_scores(animals, 25)
    return (0.30 * mk80 + 0.35 * c15 + 0.35 * rot)[s1_idx]

strategies_s2 = {
    'S2-A 直接前4名': s2_top_rank,
    'S2-B v3静态重排': s2_v3_static,
    'S2-C 短窗口MK': s2_short_mk,
    'S2-D 融合相乘': s2_fusion,
    'S2-E 短窗口特征': s2_short_features,
    'S2-F 热号+短MK': s2_hot_mk_short,
    'S2-G 重间隔': s2_gap_heavy,
    'S2-H 多样性': s2_diversity,
}

# ===== 对照组: 直接TOP4 =====
def direct_top4_v3(animals):
    """直接v3 TOP4 (不经过TOP9过滤)"""
    return (0.30 * cold_scores(animals, 15) +
            0.10 * cold_scores(animals, 30) +
            0.60 * markov_scores(animals, 150))

def direct_top4_top9w(animals):
    """直接用TOP9权重选TOP4"""
    return stage1_top9_scores(animals)

# ===== 预计算 =====
all_data = []
for pi in range(test_periods):
    i = start + pi
    hist_animals = [NUM_TO_ZODIAC_2026[n] for n in numbers[:i]]
    actual_z = NUM_TO_ZODIAC_2026[numbers[i]]
    all_data.append((hist_animals, actual_z))

def analyze_hits(hit_list):
    streaks = []
    c = 0
    for h in hit_list:
        if not h: c += 1
        else:
            if c > 0: streaks.append(c)
            c = 0
    if c > 0: streaks.append(c)
    max_miss = max(streaks) if streaks else 0
    ge4 = sum(1 for s in streaks if s >= 4)
    ge6 = sum(1 for s in streaks if s >= 6)
    return max_miss, ge4, ge6

buf = io.StringIO()
def p(s=""):
    print(s)
    buf.write(s + "\n")

p("=" * 100)
p("蒸馏方案验证: TOP9(85%)过滤 → 从9个中选TOP4")
p(f"数据: {total}期, 测试: 最近{test_periods}期")
p("=" * 100)

# ===== 对照组 =====
p(f"\n{'='*100}")
p("对照组: 直接TOP4 (不经过TOP9过滤)")
p(f"{'='*100}")

for ctrl_name, ctrl_fn in [("v3静态(c15+c30+mk150)", direct_top4_v3),
                             ("TOP9权重直选TOP4", direct_top4_top9w)]:
    hits = 0
    hit_list = []
    for pi in range(test_periods):
        animals, actual_z = all_data[pi]
        scores = ctrl_fn(animals)
        top4_idx = np.argsort(-scores)[:4]
        top4 = [ZODIAC_CYCLE_2026[i] for i in top4_idx]
        hit = actual_z in top4
        if hit: hits += 1
        hit_list.append(hit)
    
    rate = hits / test_periods * 100
    mm, g4, g6 = analyze_hits(hit_list)
    cost = test_periods * 4 * 4
    roi = (hits * 46 - cost) / cost * 100
    p(f"  {ctrl_name}: {hits}/300 = {rate:.1f}% maxMiss={mm} ≥4miss={g4}次 ≥6miss={g6}次 ROI={roi:+.1f}%")

# ===== 蒸馏方案 =====
p(f"\n{'='*100}")
p("蒸馏方案: Stage1 TOP9过滤 → Stage2 在9个中选4个")
p(f"{'='*100}")

p(f"\n  {'策略':<25} {'命中':>8} {'命中率':>8} {'maxMiss':>8} {'≥4miss':>7} {'≥6miss':>7} {'ROI':>8} {'vs对照':>8}")
p(f"  {'-'*90}")

all_results = []

for s2_name, s2_fn in strategies_s2.items():
    hits = 0
    hit_list = []
    top9_miss = 0  # TOP9本身miss的次数
    
    for pi in range(test_periods):
        animals, actual_z = all_data[pi]
        
        # Stage 1: TOP9过滤
        s1_scores = stage1_top9_scores(animals)
        top9_idx = np.argsort(-s1_scores)[:9]
        top9_set = set(top9_idx)
        
        actual_idx = ZODIAC_CYCLE_2026.index(actual_z)
        if actual_idx not in top9_set:
            top9_miss += 1
        
        # Stage 2: 从9个中选4个
        s2_scores = s2_fn(s1_scores, top9_idx, animals)
        # s2_scores是top9_idx位置的分数, 从中选top4
        top4_in_9 = np.argsort(-s2_scores)[:4]
        top4_idx = top9_idx[top4_in_9]
        top4 = [ZODIAC_CYCLE_2026[i] for i in top4_idx]
        
        hit = actual_z in top4
        if hit: hits += 1
        hit_list.append(hit)
    
    rate = hits / test_periods * 100
    mm, g4, g6 = analyze_hits(hit_list)
    cost = test_periods * 4 * 4
    roi = (hits * 46 - cost) / cost * 100
    diff = rate - 45.3  # vs v3静态TOP4的45.3%
    mark = " ★" if rate > 48.0 else ""
    p(f"  {s2_name:<25} {hits:>4}/300 {rate:>7.1f}% {mm:>7} {g4:>6} {g6:>6} {roi:>+7.1f}% {diff:>+7.1f}%{mark}")
    
    all_results.append((s2_name, hits, rate, mm, g4, g6, roi, top9_miss))

# ===== 蒸馏+反miss =====
p(f"\n{'='*100}")
p("蒸馏+反miss: 连续miss时调整Stage2策略")
p(f"{'='*100}")

# 找出最优的S2策略
best_s2_name = max(all_results, key=lambda x: x[2])[0]
best_s2_fn = strategies_s2[best_s2_name]
p(f"\n  基于最优Stage2: {best_s2_name}")

for blend_t in [2, 3]:
    for expand_t in [3, 4, 5]:
        if expand_t <= blend_t: continue
        
        consec = 0
        hits = 0
        hit_list = []
        sizes = []
        
        for pi in range(test_periods):
            animals, actual_z = all_data[pi]
            
            # Stage 1: TOP9过滤
            s1_scores = stage1_top9_scores(animals)
            top9_idx = np.argsort(-s1_scores)[:9]
            
            # Stage 2: 从9个中选
            s2_scores = best_s2_fn(s1_scores, top9_idx, animals)
            
            # 反miss: blend热号
            if consec >= blend_t:
                h = hot_scores(animals, 30)
                h_in_9 = h[top9_idx]
                s2_scores = 0.75 * s2_scores + 0.25 * h_in_9
            
            # 选几个
            if consec >= expand_t:
                n = 5
            else:
                n = 4
            
            top_in_9 = np.argsort(-s2_scores)[:n]
            top_idx = top9_idx[top_in_9]
            top = [ZODIAC_CYCLE_2026[i] for i in top_idx]
            
            hit = actual_z in top
            if hit:
                hits += 1
                consec = 0
            else:
                consec += 1
            hit_list.append(hit)
            sizes.append(n)
        
        rate = hits / test_periods * 100
        mm, g4, g6 = analyze_hits(hit_list)
        avg_n = np.mean(sizes)
        cost = sum(s * 4 for s in sizes)
        roi = (hits * 46 - cost) / cost * 100
        
        p(f"  blend@{blend_t} expand→5@{expand_t}: {hits}/300={rate:.1f}% mm={mm} ≥4miss={g4} avg={avg_n:.2f} ROI={roi:+.1f}%")

# ===== 全穷举: S2权重搜索 =====
p(f"\n{'='*100}")
p("精细搜索: 蒸馏框架下的Stage2最优权重 (在TOP9内选TOP4)")
p(f"{'='*100}")

best_distill = (0, "", None)
distill_results = []

for c15w in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
    for mkw in [0.20, 0.30, 0.40, 0.50, 0.60]:
        for gapw in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]:
            for hotw in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
                for mk_win in [50, 80, 100, 150]:
                    remain = 1.0 - c15w - mkw - gapw - hotw
                    if remain < -0.01 or remain > 0.30:
                        continue
                    c30w = max(remain, 0)
                    
                    hits = 0
                    hit_list = []
                    for pi in range(test_periods):
                        animals, actual_z = all_data[pi]
                        
                        # Stage 1
                        s1_scores = stage1_top9_scores(animals)
                        top9_idx = np.argsort(-s1_scores)[:9]
                        
                        # Stage 2
                        s2_full = (c15w * cold_scores(animals, 15) +
                                   c30w * cold_scores(animals, 30) +
                                   mkw * markov_scores(animals, mk_win) +
                                   gapw * gap_scores(animals) +
                                   hotw * hot_scores(animals, 30))
                        s2_in_9 = s2_full[top9_idx]
                        top4_in_9 = np.argsort(-s2_in_9)[:4]
                        top4_idx = top9_idx[top4_in_9]
                        top4 = [ZODIAC_CYCLE_2026[i] for i in top4_idx]
                        
                        hit = actual_z in top4
                        if hit: hits += 1
                        hit_list.append(hit)
                    
                    rate = hits / test_periods * 100
                    if rate > best_distill[0]:
                        mm2, _, _ = analyze_hits(hit_list)
                        label = f"c15={c15w} c30={c30w:.2f} mk{mk_win}={mkw} gap={gapw} hot={hotw}"
                        best_distill = (rate, label, (hits, mm2))
                    if rate >= 50:
                        mm2, _, _ = analyze_hits(hit_list)
                        distill_results.append((rate, hits, mm2, c15w, c30w, mkw, gapw, hotw, mk_win))

distill_results.sort(key=lambda x: (-x[0], x[2]))
p(f"\n  蒸馏框架Stage2最优: {best_distill[1]}")
p(f"    命中率={best_distill[0]:.1f}% ({best_distill[2][0]}/300) maxMiss={best_distill[2][1]}")
p(f"\n  蒸馏TOP4 ≥50%的组合 (前10):")
for rate, hits, mm, c15w, c30w, mkw, gapw, hotw, mkwin in distill_results[:10]:
    p(f"    {rate:.1f}% ({hits}/300) mm={mm} c15={c15w} c30={c30w:.2f} mk{mkwin}={mkw} gap={gapw} hot={hotw}")

# ===== 蒸馏最优 + 反miss =====
if distill_results:
    p(f"\n--- 蒸馏最优 + 反miss ---")
    br = distill_results[0]
    b_c15, b_c30, b_mk, b_gap, b_hot, b_mkwin = br[3], br[4], br[5], br[6], br[7], br[8]
    
    for blend_t in [2, 3]:
        for expand_t in [3, 4, 5]:
            if expand_t <= blend_t: continue
            
            consec = 0
            hits = 0
            hit_list = []
            sizes = []
            
            for pi in range(test_periods):
                animals, actual_z = all_data[pi]
                
                s1_scores = stage1_top9_scores(animals)
                top9_idx = np.argsort(-s1_scores)[:9]
                
                s2_full = (b_c15 * cold_scores(animals, 15) +
                           b_c30 * cold_scores(animals, 30) +
                           b_mk * markov_scores(animals, b_mkwin) +
                           b_gap * gap_scores(animals) +
                           b_hot * hot_scores(animals, 30))
                s2_in_9 = s2_full[top9_idx]
                
                if consec >= blend_t:
                    h = hot_scores(animals, 30)[top9_idx]
                    s2_in_9 = 0.75 * s2_in_9 + 0.25 * h
                
                n = 5 if consec >= expand_t else 4
                top_in_9 = np.argsort(-s2_in_9)[:n]
                top_idx = top9_idx[top_in_9]
                top = [ZODIAC_CYCLE_2026[i] for i in top_idx]
                
                hit = actual_z in top
                if hit:
                    hits += 1
                    consec = 0
                else:
                    consec += 1
                hit_list.append(hit)
                sizes.append(n)
            
            rate = hits / test_periods * 100
            mm2, g4, _ = analyze_hits(hit_list)
            avg_n = np.mean(sizes)
            cost = sum(s * 4 for s in sizes)
            roi = (hits * 46 - cost) / cost * 100
            
            if rate >= 48:
                p(f"  blend@{blend_t} expand@{expand_t}: {hits}/300={rate:.1f}% mm={mm2} ≥4miss={g4} avg={avg_n:.2f} ROI={roi:+.1f}%")

# ===== 最终对比 =====
p(f"\n{'='*100}")
p("最终对比总结")
p(f"{'='*100}")
p(f"  直接TOP4 v3(基线):              45.3% (136/300) maxMiss=10")
p(f"  直接TOP4 v3+反miss(基线):       48.0% (144/300) maxMiss=7")
if best_distill[0] > 0:
    p(f"  蒸馏TOP9→TOP4(最优Stage2):     {best_distill[0]:.1f}% ({best_distill[2][0]}/300) maxMiss={best_distill[2][1]}")
    diff = best_distill[0] - 48.0
    if diff > 0:
        p(f"  ✅ 蒸馏方案提升: +{diff:.1f}%")
    else:
        p(f"  {'⚠️' if diff > -2 else '❌'} 蒸馏方案差异: {diff:+.1f}%")
p(f"\n  TOP9过滤miss次数: {all_results[0][7]}/300 = {all_results[0][7]/3:.1f}% (这些期无论如何不可能命中)")
p(f"  蒸馏理论上限: {(300-all_results[0][7])/300*100:.1f}% (TOP9命中的期数)")

with open('distill_top9_to_top4.txt', 'w', encoding='utf-8-sig') as f:
    f.write(buf.getvalue())
p(f"\n结果已保存到 distill_top9_to_top4.txt")
