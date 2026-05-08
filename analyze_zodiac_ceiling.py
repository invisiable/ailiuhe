"""
生肖TOP4 - 天花板分析 + 动态突破策略
"""
import pandas as pd
import numpy as np
from collections import Counter
import time

t0 = time.time()

ZODIAC_CYCLE = ['马', '蛇', '龙', '兔', '虎', '牛', '鼠', '猪', '狗', '鸡', '猴', '羊']
Z_IDX = {z: i for i, z in enumerate(ZODIAC_CYCLE)}
NUM_TO_ZODIAC = {n: ZODIAC_CYCLE[(n - 1) % 12] for n in range(1, 50)}
ZODIAC_NUMS = {}
for z in ZODIAC_CYCLE:
    ZODIAC_NUMS[z] = sorted([n for n in range(1, 50) if NUM_TO_ZODIAC[n] == z])

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].tolist()
animals = [NUM_TO_ZODIAC[n] for n in numbers]
TOTAL = len(animals)
TEST_PERIODS = 300
START = TOTAL - TEST_PERIODS

# ============================================================
# Part 1: Oracle (天花板) 分析
# ============================================================
print("=" * 70)
print("Part 1: 多策略组合的理论天花板")
print("=" * 70)

def get_cold_top4(hist, w): 
    freq = Counter(hist[-min(w,len(hist)):])
    mx = max(freq.values()) if freq else 1
    return sorted(ZODIAC_CYCLE, key=lambda z: freq.get(z,0))[:4]

def get_gap_top4(hist):
    gaps = {}
    for z in ZODIAC_CYCLE:
        last = -1
        for j in range(len(hist)-1,-1,-1):
            if hist[j]==z: last=j; break
        gaps[z] = len(hist)-1-last if last>=0 else len(hist)
    return sorted(ZODIAC_CYCLE, key=lambda z: -gaps[z])[:4]

def get_markov_top4(hist):
    if len(hist)<2: return ZODIAC_CYCLE[:4]
    trans = {}
    for k in range(1,len(hist)):
        p,c = hist[k-1],hist[k]
        if p not in trans: trans[p]=Counter()
        trans[p][c]+=1
    state = hist[-1]
    if state not in trans: return ZODIAC_CYCLE[:4]
    return sorted(ZODIAC_CYCLE, key=lambda z: -trans[state].get(z,0))[:4]

def get_overdue_top4(hist):
    scores = {}
    for z in ZODIAC_CYCLE:
        pos = [k for k,a in enumerate(hist) if a==z]
        if len(pos)>=2:
            ints = [pos[k+1]-pos[k] for k in range(len(pos)-1)]
            avg = np.mean(ints)
            gap = len(hist)-1-pos[-1]
            scores[z] = gap/max(avg,1)
        else:
            scores[z] = 0.5
    return sorted(ZODIAC_CYCLE, key=lambda z: -scores[z])[:4]

def get_num_zodiac_top4(nums_hist, w=30):
    w = min(w,len(nums_hist))
    freq = Counter(nums_hist[-w:])
    mx = max(freq.values()) if freq else 1
    zscores = {}
    for z in ZODIAC_CYCLE:
        ns = ZODIAC_NUMS[z]
        zscores[z] = sum(1-freq.get(n,0)/max(mx,1) for n in ns)/len(ns)
    return sorted(ZODIAC_CYCLE, key=lambda z: -zscores[z])[:4]

# 所有策略
strat_funcs = [
    ('冷号5',  lambda i: get_cold_top4(animals[:i], 5)),
    ('冷号10', lambda i: get_cold_top4(animals[:i], 10)),
    ('冷号15', lambda i: get_cold_top4(animals[:i], 15)),
    ('冷号20', lambda i: get_cold_top4(animals[:i], 20)),
    ('冷号25', lambda i: get_cold_top4(animals[:i], 25)),
    ('冷号30', lambda i: get_cold_top4(animals[:i], 30)),
    ('冷号40', lambda i: get_cold_top4(animals[:i], 40)),
    ('冷号50', lambda i: get_cold_top4(animals[:i], 50)),
    ('间隔',   lambda i: get_gap_top4(animals[:i])),
    ('马尔可夫', lambda i: get_markov_top4(animals[:i])),
    ('过期',   lambda i: get_overdue_top4(animals[:i])),
    ('数字30', lambda i: get_num_zodiac_top4(numbers[:i], 30)),
    ('数字50', lambda i: get_num_zodiac_top4(numbers[:i], 50)),
]
N = len(strat_funcs)

# 计算每期每个策略的命中情况
hit_matrix = np.zeros((TEST_PERIODS, N), dtype=int)
for pi in range(TEST_PERIODS):
    i = START + pi
    actual = animals[i]
    for si, (name, func) in enumerate(strat_funcs):
        top4 = func(i)
        if actual in top4:
            hit_matrix[pi, si] = 1

# 单策略命中率
print("\n单策略命中率:")
for si, (name, _) in enumerate(strat_funcs):
    rate = hit_matrix[:, si].sum() / TEST_PERIODS
    print(f"  {name:>8}: {rate*100:.1f}%")

# Oracle: 至少1个策略命中
oracle_hit = (hit_matrix.sum(axis=1) > 0).sum()
print(f"\nOracle (任意策略命中): {oracle_hit}/{TEST_PERIODS} = {oracle_hit/TEST_PERIODS*100:.1f}%")

# 分析不可覆盖期数
zero_hit = (hit_matrix.sum(axis=1) == 0).sum()
print(f"完全不可覆盖: {zero_hit}/{TEST_PERIODS} = {zero_hit/TEST_PERIODS*100:.1f}%")
print(f"→ 理论天花板: {(TEST_PERIODS-zero_hit)/TEST_PERIODS*100:.1f}%")

# 各策略独有命中（别的策略都不中，只有它中）
print("\n策略独有命中:")
for si, (name, _) in enumerate(strat_funcs):
    unique = 0
    for pi in range(TEST_PERIODS):
        if hit_matrix[pi, si] == 1 and hit_matrix[pi].sum() == 1:
            unique += 1
    if unique > 0:
        print(f"  {name:>8}: {unique}期")

# 策略相关性矩阵
print("\n策略相关性 (top pairs):")
pairs = []
for s1 in range(N):
    for s2 in range(s1+1, N):
        both = ((hit_matrix[:,s1]==1) & (hit_matrix[:,s2]==1)).sum()
        either = ((hit_matrix[:,s1]==1) | (hit_matrix[:,s2]==1)).sum()
        jaccard = both / max(either, 1)
        pairs.append((strat_funcs[s1][0], strat_funcs[s2][0], jaccard, both, either))
pairs.sort(key=lambda x: x[2])  # 低相关(互补)在前
print("  最互补:")
for p in pairs[:5]:
    print(f"    {p[0]:>8} ∩ {p[1]:>8}: J={p[2]:.2f} (both={p[3]}, either={p[4]})")
print("  最相关:")
for p in pairs[-5:]:
    print(f"    {p[0]:>8} ∩ {p[1]:>8}: J={p[2]:.2f} (both={p[3]}, either={p[4]})")

# ============================================================
# Part 2: 最优策略选择器（每期选最佳策略）
# ============================================================
print("\n" + "=" * 70)
print("Part 2: 动态策略选择器")
print("=" * 70)

# 基于最近N期表现选策略
for lookback in [5, 8, 10, 12, 15, 20]:
    hits = 0
    for pi in range(TEST_PERIODS):
        if pi < lookback:
            # 初始用最佳单一策略
            si = 3  # 冷号20
        else:
            # 找最近lookback期表现最好的策略
            perf = hit_matrix[pi-lookback:pi].sum(axis=0)
            si = np.argmax(perf)
            # 若有平局，按权重选
        
        if hit_matrix[pi, si] == 1:
            hits += 1
    print(f"  lookback={lookback:>2}: {hits}/{TEST_PERIODS} = {hits/TEST_PERIODS*100:.1f}%")

# ============================================================
# Part 3: 贪心集成 - 逐步添加策略
# ============================================================
print("\n" + "=" * 70)
print("Part 3: 贪心策略集成 (投票)")
print("=" * 70)

# 投票法: TOP4选入越多策略投票的生肖
for min_strats in [2, 3, 5, 7, 9, 11]:
    # 只用前min_strats个最好策略投票
    # 先按单独命中率排序
    sorted_strats = sorted(range(N), key=lambda s: -hit_matrix[:,s].sum())
    selected = sorted_strats[:min_strats]
    
    hits = 0
    for pi in range(TEST_PERIODS):
        i = START + pi
        votes = Counter()
        for si in selected:
            top4 = strat_funcs[si][1](i)
            for z in top4:
                votes[z] += 1
        top4_voted = [z for z, _ in votes.most_common(4)]
        if animals[i] in top4_voted:
            hits += 1
    print(f"  Top{min_strats}策略投票: {hits}/{TEST_PERIODS} = {hits/TEST_PERIODS*100:.1f}%")

# 互补策略投票
print("\n互补策略组合:")
# 选择互补性最好的策略
best_combo_rate = 0
best_combo = None
from itertools import combinations
for combo_size in [3, 4, 5]:
    for combo in combinations(range(N), combo_size):
        # 估算投票命中率
        hits = 0
        for pi in range(TEST_PERIODS):
            i = START + pi
            votes = Counter()
            for si in combo:
                top4 = strat_funcs[si][1](i)
                for z in top4:
                    votes[z] += 1
            top4_voted = [z for z, _ in votes.most_common(4)]
            if animals[i] in top4_voted:
                hits += 1
        rate = hits / TEST_PERIODS
        if rate > best_combo_rate:
            best_combo_rate = rate
            best_combo = combo
            names = [strat_funcs[s][0] for s in combo]
            if rate > 0.43:
                print(f"  {names}: {rate*100:.1f}% *best*")

print(f"\n最优组合: {[strat_funcs[s][0] for s in best_combo]} = {best_combo_rate*100:.1f}%")

# ============================================================
# Part 4: TOP5/TOP6 扩展分析 
# ============================================================
print("\n" + "=" * 70)
print("Part 4: TOP4/5/6 对比 (自适应扩展)")
print("=" * 70)

# 如果允许TOP5或TOP6，命中率如何？
for topk in [4, 5, 6]:
    hits = 0
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        freq = Counter(hist[-min(20,len(hist)):])
        mx = max(freq.values()) if freq else 1
        topN = sorted(ZODIAC_CYCLE, key=lambda z: freq.get(z,0))[:topk]
        if animals[i] in topN:
            hits += 1
    print(f"  冷号W20 TOP{topk}: {hits}/{TEST_PERIODS} = {hits/TEST_PERIODS*100:.1f}%")

# 自适应: 连续不中时扩展到TOP5/TOP6
for expand_after in [2, 3, 4]:
    hits = 0
    streak = 0
    bets_4 = 0; bets_5 = 0; bets_6 = 0
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        freq = Counter(hist[-min(20,len(hist)):])
        
        if streak >= expand_after * 2:
            topk = 6; bets_6 += 1
        elif streak >= expand_after:
            topk = 5; bets_5 += 1
        else:
            topk = 4; bets_4 += 1
        
        topN = sorted(ZODIAC_CYCLE, key=lambda z: freq.get(z,0))[:topk]
        if animals[i] in topN:
            hits += 1
            streak = 0
        else:
            streak += 1
    avg_bet = (bets_4*4 + bets_5*5 + bets_6*6) / TEST_PERIODS
    print(f"  自适应(expand@{expand_after}): {hits}/{TEST_PERIODS}={hits/TEST_PERIODS*100:.1f}% (avg bet:{avg_bet:.1f}肖, TOP4:{bets_4}|5:{bets_5}|6:{bets_6})")

# ============================================================
# Part 5: Markov理论天花板
# ============================================================
print("\n" + "=" * 70)
print("Part 5: 马尔可夫理论天花板")
print("=" * 70)

# 构建完整转移矩阵
trans = {}
for k in range(1, len(animals)):
    prev, curr = animals[k-1], animals[k]
    if prev not in trans:
        trans[prev] = Counter()
    trans[prev][curr] += 1

# 每个状态的TOP4概率和
print("每个状态的TOP4覆盖概率:")
top4_coverage = []
for z in ZODIAC_CYCLE:
    if z in trans:
        total = sum(trans[z].values())
        probs = [(tz, trans[z].get(tz,0)/total) for tz in ZODIAC_CYCLE]
        probs.sort(key=lambda x: -x[1])
        top4_prob = sum(p for _, p in probs[:4])
        top4_names = [n for n, _ in probs[:4]]
        top4_coverage.append(top4_prob)
        print(f"  {z} → TOP4={top4_names} 覆盖={top4_prob*100:.1f}%")

avg_coverage = np.mean(top4_coverage)
print(f"\n平均TOP4覆盖: {avg_coverage*100:.1f}% (这是完美马尔可夫的天花板)")

print(f"\n总耗时: {time.time()-t0:.1f}秒")
