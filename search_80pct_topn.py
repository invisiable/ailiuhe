"""
系统性搜索: 达到80%命中率需要最少多少个生肖
基于v3核心算法(静态组合+热号互补), 测试TOP4~TOP10
同时测试多种策略组合, 找到最优方案
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
    """轮换评分: 基于出现间隔周期性"""
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

# 预计算所有期数据
all_data = []
for pi in range(test_periods):
    i = start + pi
    hist_animals = [NUM_TO_ZODIAC_2026[n] for n in numbers[:i]]
    actual_z = NUM_TO_ZODIAC_2026[numbers[i]]
    all_data.append((hist_animals, actual_z))

# ===== 策略定义 =====
def static_combo(animals):
    """v3基础: cold15+cold30+mk150"""
    return 0.30 * cold_scores(animals, 15) + 0.10 * cold_scores(animals, 30) + 0.60 * markov_scores(animals, 150)

def enhanced_combo(animals):
    """增强: 静态+gap+rotation"""
    s = static_combo(animals)
    g = gap_scores(animals)
    r = rotation_scores(animals, 20)
    return 0.60 * s + 0.25 * g + 0.15 * r

def multi_markov(animals):
    """多窗口马尔可夫融合"""
    mk100 = markov_scores(animals, 100)
    mk150 = markov_scores(animals, 150)
    mk_full = markov_scores(animals)
    return 0.3 * mk100 + 0.4 * mk150 + 0.3 * mk_full

def comprehensive(animals):
    """综合: 静态+间隔+轮换+多MK"""
    s = static_combo(animals)
    g = gap_scores(animals)
    mk = multi_markov(animals)
    r = rotation_scores(animals, 25)
    return 0.40 * s + 0.20 * g + 0.25 * mk + 0.15 * r

def cold_heavy(animals):
    """重冷号: 多窗口冷号"""
    c10 = cold_scores(animals, 10)
    c20 = cold_scores(animals, 20)
    c40 = cold_scores(animals, 40)
    c60 = cold_scores(animals, 60)
    mk = markov_scores(animals, 150)
    return 0.15 * c10 + 0.15 * c20 + 0.10 * c40 + 0.10 * c60 + 0.50 * mk

def hot_cold_blend(animals):
    """冷热融合"""
    c = cold_scores(animals, 20)
    h = hot_scores(animals, 30)
    mk = markov_scores(animals, 150)
    g = gap_scores(animals)
    return 0.25 * c + 0.15 * h + 0.35 * mk + 0.25 * g

strategies = {
    'v3静态': static_combo,
    '增强(静态+gap+rot)': enhanced_combo,
    '多窗口MK': multi_markov,
    '综合全维度': comprehensive,
    '重冷号多窗口': cold_heavy,
    '冷热MK融合': hot_cold_blend,
}

def analyze_miss(hit_list):
    streaks = []
    c = 0
    for h in hit_list:
        if not h: c += 1
        else:
            if c > 0: streaks.append(c)
            c = 0
    if c > 0: streaks.append(c)
    max_miss = max(streaks) if streaks else 0
    return max_miss

buf = io.StringIO()
def p(s=""):
    print(s)
    buf.write(s + "\n")

p("=" * 100)
p("系统性搜索: 达到80%命中率需要最少多少个生肖")
p(f"数据: {total}期, 测试: 最近{test_periods}期")
p("=" * 100)

# ===== 随机基线 =====
p(f"\n--- 随机基线 ---")
for n in range(4, 11):
    baseline = n / 12 * 100
    p(f"  TOP{n}: 随机={baseline:.1f}%")

# ===== 各策略 × 各TOP-N =====
p(f"\n{'='*100}")
p("各策略 × 各TOP-N 命中率 (300期验证)")
p(f"{'='*100}")

all_results = []

for strat_name, strat_fn in strategies.items():
    p(f"\n  策略: {strat_name}")
    p(f"  {'TOP-N':>6} {'命中':>8} {'命中率':>8} {'随机':>7} {'提升':>7} {'maxMiss':>8} {'ROI':>8}")
    p(f"  {'-'*58}")
    
    for top_n in range(4, 11):
        hits = 0
        hit_list = []
        for pi in range(test_periods):
            hist_animals, actual_z = all_data[pi]
            scores = strat_fn(hist_animals)
            top_idx = np.argsort(-scores)[:top_n]
            top = [ZODIAC_CYCLE_2026[i] for i in top_idx]
            hit = actual_z in top
            if hit: hits += 1
            hit_list.append(hit)
        
        rate = hits / test_periods * 100
        baseline = top_n / 12 * 100
        lift = rate - baseline
        max_miss = analyze_miss(hit_list)
        
        # ROI: 每个生肖4元, 命中46元
        cost = test_periods * top_n * 4
        reward = hits * 46
        roi = (reward - cost) / cost * 100
        
        mark = " ★80%" if rate >= 80 else (" ★75%" if rate >= 75 else "")
        p(f"  TOP{top_n:>2}  {hits:>4}/300 {rate:>7.1f}% {baseline:>6.1f}% {lift:>+6.1f}% {max_miss:>7} {roi:>+7.1f}%{mark}")
        
        all_results.append({
            'strategy': strat_name,
            'top_n': top_n,
            'hits': hits,
            'rate': rate,
            'baseline': baseline,
            'lift': lift,
            'max_miss': max_miss,
            'roi': roi,
        })

# ===== 带反miss机制的版本 =====
p(f"\n{'='*100}")
p("带反miss机制 (连miss时blend热号+扩展)")
p(f"{'='*100}")

for base_n in range(5, 10):
    for strat_name, strat_fn in [('v3静态', static_combo), ('综合全维度', comprehensive)]:
        for blend_thresh in [2, 3]:
            for expand_thresh in [3, 4]:
                if expand_thresh <= blend_thresh:
                    continue
                
                consec = 0
                hits = 0
                hit_list = []
                sizes = []
                
                for pi in range(test_periods):
                    hist_animals, actual_z = all_data[pi]
                    scores = strat_fn(hist_animals)
                    
                    if consec >= blend_thresh:
                        h = hot_scores(hist_animals, 30)
                        scores = 0.75 * scores + 0.25 * h
                    
                    sorted_idx = np.argsort(-scores)
                    
                    if consec >= expand_thresh:
                        n = base_n + 1
                    else:
                        n = base_n
                    
                    top = [ZODIAC_CYCLE_2026[i] for i in sorted_idx[:n]]
                    hit = actual_z in top
                    if hit:
                        hits += 1
                        consec = 0
                    else:
                        consec += 1
                    hit_list.append(hit)
                    sizes.append(n)
                
                rate = hits / test_periods * 100
                max_miss = analyze_miss(hit_list)
                avg_n = np.mean(sizes)
                cost = sum(s * 4 for s in sizes)
                roi = (hits * 46 - cost) / cost * 100
                
                if rate >= 75:
                    p(f"  {strat_name} TOP{base_n}+反miss(blend@{blend_thresh},expand@{expand_thresh}): {hits}/300={rate:.1f}% maxMiss={max_miss} avg={avg_n:.1f} ROI={roi:+.1f}%")

# ===== 总结: 按TOP-N分组找最优 =====
p(f"\n{'='*100}")
p("各TOP-N最优策略 (按命中率排序)")
p(f"{'='*100}")
p(f"  {'TOP-N':>6} {'最优策略':<25} {'命中率':>8} {'提升vs随机':>12} {'maxMiss':>8} {'ROI':>8}")
p(f"  {'-'*75}")

for n in range(4, 11):
    n_results = [r for r in all_results if r['top_n'] == n]
    best = max(n_results, key=lambda x: x['rate'])
    mark = " ←达标" if best['rate'] >= 80 else ""
    p(f"  TOP{n:>2}  {best['strategy']:<25} {best['rate']:>7.1f}% {best['lift']:>+11.1f}% {best['max_miss']:>7} {best['roi']:>+7.1f}%{mark}")

# 找到达到80%的最小N
target = 80
min_n_80 = None
best_at_80 = None
for n in range(4, 11):
    n_results = [r for r in all_results if r['top_n'] == n]
    best = max(n_results, key=lambda x: x['rate'])
    if best['rate'] >= target and min_n_80 is None:
        min_n_80 = n
        best_at_80 = best

p(f"\n{'='*100}")
if min_n_80:
    p(f"✅ 达到{target}%命中率的最少生肖数: TOP{min_n_80}")
    p(f"   最优策略: {best_at_80['strategy']}")
    p(f"   命中率: {best_at_80['rate']:.1f}% (随机基线{best_at_80['baseline']:.1f}%)")
    p(f"   每期投入: {min_n_80*4}元, ROI: {best_at_80['roi']:+.1f}%")
else:
    p(f"❌ TOP10以内无法达到{target}%命中率")
    # 找最接近的
    for n in range(10, 3, -1):
        n_results = [r for r in all_results if r['top_n'] == n]
        best = max(n_results, key=lambda x: x['rate'])
        p(f"   最佳TOP{n}: {best['rate']:.1f}% ({best['strategy']})")
        if best['rate'] >= 75:
            break
p(f"{'='*100}")

# 保存
with open('search_80pct_topn.txt', 'w', encoding='utf-8-sig') as f:
    f.write(buf.getvalue())
p("\n结果已保存到 search_80pct_topn.txt")
