"""
寻找与静态组合互补的策略
目标: 在静态组合miss的160期中,找出命中率最高的策略
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from collections import Counter

ZODIAC_CYCLE_2026 = ['马', '蛇', '龙', '兔', '虎', '牛', '鼠', '猪', '狗', '鸡', '猴', '羊']
Z_IDX = {z: i for i, z in enumerate(ZODIAC_CYCLE_2026)}
NUM_TO_ZODIAC_2026 = {n: ZODIAC_CYCLE_2026[(n - 1) % 12] for n in range(1, 50)}

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].tolist()
total = len(numbers)
test_periods = 300
start = total - test_periods


def cold_scores(animals, window):
    w = min(window, len(animals))
    freq = Counter(animals[-w:])
    mx = max(freq.values()) if freq else 1
    return np.array([1.0 - freq.get(z, 0) / max(mx, 1) for z in ZODIAC_CYCLE_2026])

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
    if len(h) < 2:
        return probs
    trans = {}
    for k in range(1, len(h)):
        p, c = h[k - 1], h[k]
        if p not in trans:
            trans[p] = Counter()
        trans[p][c] += 1
    state = animals[-1]
    if state in trans:
        total_cnt = sum(trans[state].values()) + laplace * 12
        for zi, z in enumerate(ZODIAC_CYCLE_2026):
            probs[zi] = (trans[state].get(z, 0) + laplace) / total_cnt
    return probs

def hot_scores(animals, window):
    """热号: 出现越多得分越高 (与冷号相反)"""
    w = min(window, len(animals))
    freq = Counter(animals[-w:])
    mx = max(freq.values()) if freq else 1
    return np.array([freq.get(z, 0) / max(mx, 1) for z in ZODIAC_CYCLE_2026])

def markov2_scores(animals, window=None, laplace=1.0):
    """二阶马尔可夫"""
    probs = np.ones(12) / 12
    h = animals[-window:] if window and len(animals) > window else animals
    if len(h) < 3:
        return probs
    trans = {}
    for k in range(2, len(h)):
        state = (h[k-2], h[k-1])
        c = h[k]
        if state not in trans:
            trans[state] = Counter()
        trans[state][c] += 1
    state = (animals[-2], animals[-1])
    if state in trans:
        total_cnt = sum(trans[state].values()) + laplace * 12
        for zi, z in enumerate(ZODIAC_CYCLE_2026):
            probs[zi] = (trans[state].get(z, 0) + laplace) / total_cnt
    return probs

def diversity_scores(animals, window=30):
    """多样性: 偏好最近较少出现的+有一定间隔的"""
    cold = cold_scores(animals, window)
    g = gap_scores(animals)
    return 0.6 * cold + 0.4 * g

def rotation_scores(animals, window=20):
    """轮换: 基于出现间隔的周期性"""
    scores = []
    for z in ZODIAC_CYCLE_2026:
        positions = [j for j, a in enumerate(animals[-window:]) if a == z]
        if len(positions) >= 2:
            gaps = [positions[k+1] - positions[k] for k in range(len(positions)-1)]
            avg_gap = np.mean(gaps)
            last_gap = window - positions[-1] if positions else window
            # 如果距上次出现接近平均间隔,得分高
            score = 1.0 - abs(last_gap - avg_gap) / max(avg_gap, 1)
            score = max(0, score)
        else:
            score = 0.5
        scores.append(score)
    return np.array(scores)

# ===== 静态组合基准 =====
def static_predict(animals):
    cold15 = cold_scores(animals, 15)
    cold30 = cold_scores(animals, 30)
    mk150 = markov_scores(animals, window=150)
    return 0.30 * cold15 + 0.10 * cold30 + 0.60 * mk150

# 收集静态组合每期的命中情况
static_hits = []
all_animals = []
for pi in range(test_periods):
    i = start + pi
    hist_animals = [NUM_TO_ZODIAC_2026[n] for n in numbers[:i]]
    actual_z = NUM_TO_ZODIAC_2026[numbers[i]]
    all_animals.append((hist_animals, actual_z))
    
    s = static_predict(hist_animals)
    top4_idx = np.argsort(-s)[:4]
    top4 = [ZODIAC_CYCLE_2026[idx] for idx in top4_idx]
    hit = actual_z in top4
    static_hits.append(hit)

static_miss_indices = [i for i, h in enumerate(static_hits) if not h]
print(f"静态组合: {sum(static_hits)}/300 = {sum(static_hits)/300*100:.1f}%")
print(f"静态组合miss数: {len(static_miss_indices)}")

# ===== 测试各策略在静态miss期的表现 =====
strategies = {
    'cold10': lambda a: cold_scores(a, 10),
    'cold20': lambda a: cold_scores(a, 20),
    'cold40': lambda a: cold_scores(a, 40),
    'cold60': lambda a: cold_scores(a, 60),
    'hot10': lambda a: hot_scores(a, 10),
    'hot20': lambda a: hot_scores(a, 20),
    'hot30': lambda a: hot_scores(a, 30),
    'gap': lambda a: gap_scores(a),
    'mk_full': lambda a: markov_scores(a),
    'mk150': lambda a: markov_scores(a, 150),
    'mk100': lambda a: markov_scores(a, 100),
    'mk50': lambda a: markov_scores(a, 50),
    'mk200': lambda a: markov_scores(a, 200),
    'mk2_full': lambda a: markov2_scores(a),
    'mk2_150': lambda a: markov2_scores(a, 150),
    'diversity30': lambda a: diversity_scores(a, 30),
    'diversity50': lambda a: diversity_scores(a, 50),
    'rotation20': lambda a: rotation_scores(a, 20),
    'rotation30': lambda a: rotation_scores(a, 30),
    # 组合策略
    'cold10+mk_full': lambda a: 0.3 * cold_scores(a, 10) + 0.7 * markov_scores(a),
    'hot20+mk150': lambda a: 0.3 * hot_scores(a, 20) + 0.7 * markov_scores(a, 150),
    'gap+mk_full': lambda a: 0.4 * gap_scores(a) + 0.6 * markov_scores(a),
    'gap+cold10': lambda a: 0.5 * gap_scores(a) + 0.5 * cold_scores(a, 10),
    'diversity+mk': lambda a: 0.4 * diversity_scores(a, 30) + 0.6 * markov_scores(a),
    'rotation+mk': lambda a: 0.3 * rotation_scores(a, 20) + 0.7 * markov_scores(a),
    'hot10+gap': lambda a: 0.5 * hot_scores(a, 10) + 0.5 * gap_scores(a),
    'cold10+gap+mk': lambda a: 0.2 * cold_scores(a, 10) + 0.3 * gap_scores(a) + 0.5 * markov_scores(a),
}

print(f"\n各策略在静态miss时的命中率 (共{len(static_miss_indices)}期miss):")
print(f"{'策略':<25} {'miss期命中':>10} {'miss命中率':>10} {'全量命中':>10} {'全量命中率':>10} {'互补增益':>10}")
print("-" * 85)

results = []
for name, strat in strategies.items():
    miss_hits = 0
    all_hits = 0
    for pi in range(test_periods):
        hist_animals, actual_z = all_animals[pi]
        s = strat(hist_animals)
        top4_idx = np.argsort(-s)[:4]
        top4 = [ZODIAC_CYCLE_2026[idx] for idx in top4_idx]
        hit = actual_z in top4
        if hit:
            all_hits += 1
        if pi in static_miss_indices and hit:
            miss_hits += 1
    
    miss_rate = miss_hits / len(static_miss_indices) * 100
    all_rate = all_hits / test_periods * 100
    # 互补增益 = 如果替换时使用此策略,新增命中数
    gain = miss_hits  # 静态miss时此策略能命中的数量
    results.append((name, miss_hits, miss_rate, all_hits, all_rate, gain))

results.sort(key=lambda x: -x[2])  # 按miss期命中率排序

for name, mh, mr, ah, ar, g in results:
    print(f"  {name:<23} {mh:>8} {mr:>9.1f}% {ah:>8} {ar:>9.1f}% {g:>8}")

# ===== TOP1互补策略详细分析 =====
best_name = results[0][0]
print(f"\n最佳互补策略: {best_name}")
print(f"静态miss时命中: {results[0][1]}/{len(static_miss_indices)} = {results[0][2]:.1f}%")

# ===== 模拟最优融合: 静态+互补=====
print(f"\n{'='*60}")
print(f"模拟融合方案")
print(f"{'='*60}")

# 方案A: 纯静态 + 连miss>=N时扩展TOP5
for threshold in [2, 3, 4]:
    consec = 0
    hits = 0
    sizes = []
    hit_list = []
    for pi in range(test_periods):
        hist_animals, actual_z = all_animals[pi]
        s = static_predict(hist_animals)
        sorted_idx = np.argsort(-s)
        n = 5 if consec >= threshold else 4
        top = [ZODIAC_CYCLE_2026[idx] for idx in sorted_idx[:n]]
        hit = actual_z in top
        if hit:
            hits += 1
            consec = 0
        else:
            consec += 1
        hit_list.append(hit)
        sizes.append(n)
    
    streaks = []
    c = 0
    for h in hit_list:
        if not h:
            c += 1
        else:
            if c > 0: streaks.append(c)
            c = 0
    if c > 0: streaks.append(c)
    max_miss = max(streaks) if streaks else 0
    ge6 = sum(1 for s in streaks if s >= 6)
    
    print(f"  扩展@miss>={threshold}: {hits}/300 = {hits/300*100:.1f}% max_miss={max_miss} ≥6miss={ge6}次 avg_size={np.mean(sizes):.1f}")

# 方案B: 静态 + 连miss时切换到TOP1互补策略
best_strat = strategies[best_name]
for threshold in [2, 3, 4]:
    consec = 0
    hits = 0
    hit_list = []
    for pi in range(test_periods):
        hist_animals, actual_z = all_animals[pi]
        if consec >= threshold:
            s = best_strat(hist_animals)
        else:
            s = static_predict(hist_animals)
        top4_idx = np.argsort(-s)[:4]
        top4 = [ZODIAC_CYCLE_2026[idx] for idx in top4_idx]
        hit = actual_z in top4
        if hit:
            hits += 1
            consec = 0
        else:
            consec += 1
        hit_list.append(hit)
    
    streaks = []
    c = 0
    for h in hit_list:
        if not h:
            c += 1
        else:
            if c > 0: streaks.append(c)
            c = 0
    if c > 0: streaks.append(c)
    max_miss = max(streaks) if streaks else 0
    ge6 = sum(1 for s in streaks if s >= 6)
    
    print(f"  切换{best_name}@miss>={threshold}: {hits}/300 = {hits/300*100:.1f}% max_miss={max_miss} ≥6miss={ge6}次")

# 方案C: 静态 + 连miss时TOP5 + 使用互补策略排第5
for threshold in [2, 3]:
    consec = 0
    hits = 0
    hit_list = []
    for pi in range(test_periods):
        hist_animals, actual_z = all_animals[pi]
        static = static_predict(hist_animals)
        
        if consec >= threshold:
            # 取静态TOP4 + 互补策略中不在TOP4的最佳1个
            top4_idx = np.argsort(-static)[:4]
            top4 = set(top4_idx)
            alt = best_strat(hist_animals)
            # 找互补策略中排名最高但不在top4的
            alt_sorted = np.argsort(-alt)
            extra = None
            for idx in alt_sorted:
                if idx not in top4:
                    extra = idx
                    break
            if extra is not None:
                final = list(top4_idx) + [extra]
            else:
                final = list(top4_idx) + [np.argsort(-static)[4]]
            top = [ZODIAC_CYCLE_2026[idx] for idx in final]
        else:
            top4_idx = np.argsort(-static)[:4]
            top = [ZODIAC_CYCLE_2026[idx] for idx in top4_idx]
        
        hit = actual_z in top
        if hit:
            hits += 1
            consec = 0
        else:
            consec += 1
        hit_list.append(hit)
    
    streaks = []
    c = 0
    for h in hit_list:
        if not h:
            c += 1
        else:
            if c > 0: streaks.append(c)
            c = 0
    if c > 0: streaks.append(c)
    max_miss = max(streaks) if streaks else 0
    ge6 = sum(1 for s in streaks if s >= 6)
    
    print(f"  TOP4+互补扩展@miss>={threshold}: {hits}/300 = {hits/300*100:.1f}% max_miss={max_miss} ≥6miss={ge6}次")

# 方案D: 双策略融合得分(静态+互补加权)
for w in [0.1, 0.2, 0.3]:
    hits = 0
    hit_list = []
    for pi in range(test_periods):
        hist_animals, actual_z = all_animals[pi]
        s1 = static_predict(hist_animals)
        s2 = best_strat(hist_animals)
        combined = (1-w) * s1 + w * s2
        top4_idx = np.argsort(-combined)[:4]
        top4 = [ZODIAC_CYCLE_2026[idx] for idx in top4_idx]
        hit = actual_z in top4
        if hit: hits += 1
        hit_list.append(hit)
    
    streaks = []
    c = 0
    for h in hit_list:
        if not h:
            c += 1
        else:
            if c > 0: streaks.append(c)
            c = 0
    if c > 0: streaks.append(c)
    max_miss = max(streaks) if streaks else 0
    ge6 = sum(1 for s in streaks if s >= 6)
    
    print(f"  融合{1-w:.0%}静态+{w:.0%}互补: {hits}/300 = {hits/300*100:.1f}% max_miss={max_miss} ≥6miss={ge6}次")

# 方案E: 连miss时融合比例动态增加
for base_w in [0.1, 0.15, 0.2]:
    consec = 0
    hits = 0
    hit_list = []
    for pi in range(test_periods):
        hist_animals, actual_z = all_animals[pi]
        s1 = static_predict(hist_animals)
        s2 = best_strat(hist_animals)
        # 动态权重: miss越多,互补策略权重越大
        w = min(base_w + consec * 0.05, 0.5)
        combined = (1-w) * s1 + w * s2
        top4_idx = np.argsort(-combined)[:4]
        top4 = [ZODIAC_CYCLE_2026[idx] for idx in top4_idx]
        hit = actual_z in top4
        if hit:
            hits += 1
            consec = 0
        else:
            consec += 1
        hit_list.append(hit)
    
    streaks = []
    c = 0
    for h in hit_list:
        if not h:
            c += 1
        else:
            if c > 0: streaks.append(c)
            c = 0
    if c > 0: streaks.append(c)
    max_miss = max(streaks) if streaks else 0
    ge6 = sum(1 for s in streaks if s >= 6)
    
    print(f"  动态融合base={base_w}: {hits}/300 = {hits/300*100:.1f}% max_miss={max_miss} ≥6miss={ge6}次")

# 方案F: 原始模型做互补(miss时切到原始模型)
from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy
for threshold in [2, 3, 4]:
    orig = RecommendedZodiacTop4Strategy(use_emergency_backup=False)
    consec = 0
    hits = 0
    hit_list = []
    for pi in range(test_periods):
        hist_animals, actual_z = all_animals[pi]
        if consec >= threshold:
            result = orig.predict_top4(hist_animals)
            top = result['top4']
        else:
            s = static_predict(hist_animals)
            top4_idx = np.argsort(-s)[:4]
            top = [ZODIAC_CYCLE_2026[idx] for idx in top4_idx]
        hit = actual_z in top
        if hit:
            hits += 1
            consec = 0
        else:
            consec += 1
        hit_list.append(hit)
        orig.update_performance(hit)
    
    streaks = []
    c = 0
    for h in hit_list:
        if not h:
            c += 1
        else:
            if c > 0: streaks.append(c)
            c = 0
    if c > 0: streaks.append(c)
    max_miss = max(streaks) if streaks else 0
    ge6 = sum(1 for s in streaks if s >= 6)
    
    print(f"  切换原始模型@miss>={threshold}: {hits}/300 = {hits/300*100:.1f}% max_miss={max_miss} ≥6miss={ge6}次")
