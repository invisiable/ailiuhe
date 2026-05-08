"""
生肖TOP4优化器v3 - 高级策略探索
目标: 300期滚动验证命中率 >= 50%

策略方向:
1. 动态自适应权重（根据近期表现调整）
2. 投票集成（多策略独立选TOP4，统计投票）
3. 条件马尔可夫（根据上期生肖切换策略）
4. 数字级别特征→生肖概率
5. 排除法（排除最不可能的8个，剩余4个）
"""
import pandas as pd
import numpy as np
from collections import Counter
import time

t0 = time.time()

ZODIAC_CYCLE = ['马', '蛇', '龙', '兔', '虎', '牛', '鼠', '猪', '狗', '鸡', '猴', '羊']
Z_IDX = {z: i for i, z in enumerate(ZODIAC_CYCLE)}
NUM_TO_ZODIAC = {n: ZODIAC_CYCLE[(n - 1) % 12] for n in range(1, 50)}
ZODIAC_TO_NUMS = {}
for z in ZODIAC_CYCLE:
    ZODIAC_TO_NUMS[z] = [n for n in range(1, 50) if NUM_TO_ZODIAC[n] == z]

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].tolist()
animals = [NUM_TO_ZODIAC[n] for n in numbers]
TOTAL = len(animals)
TEST_PERIODS = 300
START = TOTAL - TEST_PERIODS

print(f"数据: {TOTAL}期, 测试{TEST_PERIODS}期")

# ============================================================
# 方法A: 动态自适应权重
# ============================================================
def method_adaptive_weights(lookback=15):
    """根据近N期各策略的表现动态调整权重"""
    strat_funcs = [
        lambda h: score_cold(h, 20),
        lambda h: score_gap(h),
        lambda h: score_markov1(h),
        lambda h: score_cold(h, 10),
        lambda h: score_periodicity(h),
        lambda h: score_markov2(h),
    ]
    n_strats = len(strat_funcs)
    # 初始等权
    weights = [1.0 / n_strats] * n_strats
    strat_history = [[] for _ in range(n_strats)]  # 各策略命中记录
    
    hits = 0
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        actual = animals[i]
        
        # 各策略评分
        all_scores = []
        for sf in strat_funcs:
            s = sf(hist)
            all_scores.append(s)
        
        # 加权融合
        final = {z: 0.0 for z in ZODIAC_CYCLE}
        for si in range(n_strats):
            for z in ZODIAC_CYCLE:
                final[z] += weights[si] * all_scores[si].get(z, 0)
        
        top4 = sorted(ZODIAC_CYCLE, key=lambda z: -final[z])[:4]
        hit = actual in top4
        if hit:
            hits += 1
        
        # 更新各策略的命中记录
        for si in range(n_strats):
            s_top4 = sorted(ZODIAC_CYCLE, key=lambda z: -all_scores[si].get(z, 0))[:4]
            strat_history[si].append(1 if actual in s_top4 else 0)
        
        # 动态调整权重（每期）
        if pi >= lookback:
            new_w = []
            for si in range(n_strats):
                recent = strat_history[si][-lookback:]
                rate = sum(recent) / len(recent)
                new_w.append(max(rate, 0.05))  # 最低保底
            ws = sum(new_w)
            weights = [w / ws for w in new_w]
    
    return hits / TEST_PERIODS

# ============================================================
# 方法B: 投票集成
# ============================================================
def method_voting_ensemble(top_k=4, min_votes=2):
    """多策略独立选TOP-K，按投票数排名"""
    hits = 0
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        actual = animals[i]
        
        # 各策略独立选TOP4
        selections = []
        s = score_cold(hist, 20)
        selections.append(sorted(ZODIAC_CYCLE, key=lambda z: -s[z])[:top_k])
        s = score_gap(hist)
        selections.append(sorted(ZODIAC_CYCLE, key=lambda z: -s[z])[:top_k])
        s = score_markov1(hist)
        selections.append(sorted(ZODIAC_CYCLE, key=lambda z: -s[z])[:top_k])
        s = score_cold(hist, 10)
        selections.append(sorted(ZODIAC_CYCLE, key=lambda z: -s[z])[:top_k])
        s = score_cold(hist, 50)
        selections.append(sorted(ZODIAC_CYCLE, key=lambda z: -s[z])[:top_k])
        s = score_periodicity(hist)
        selections.append(sorted(ZODIAC_CYCLE, key=lambda z: -s[z])[:top_k])
        s = score_anti_recent(hist)
        selections.append(sorted(ZODIAC_CYCLE, key=lambda z: -s[z])[:top_k])
        
        # 统计投票
        votes = Counter()
        for sel in selections:
            for z in sel:
                votes[z] += 1
        
        top4 = [z for z, _ in votes.most_common(4)]
        hit = actual in top4
        if hit:
            hits += 1
    
    return hits / TEST_PERIODS

# ============================================================
# 方法C: 条件策略（根据上期生肖选策略）
# ============================================================
def method_conditional():
    """基于马尔可夫转移矩阵 + 冷号的条件融合"""
    hits = 0
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        actual = animals[i]
        
        s_markov = score_markov1(hist)
        s_cold = score_cold(hist, 20)
        s_gap = score_gap(hist)
        
        # 判断马尔可夫信心
        markov_conf = max(s_markov.values()) if s_markov else 0
        
        if markov_conf > 0.15:  # 高信心用马尔可夫主导
            final = {z: 0.5*s_markov[z] + 0.3*s_cold[z] + 0.2*s_gap[z] for z in ZODIAC_CYCLE}
        else:  # 低信心用冷号+间隔
            final = {z: 0.5*s_cold[z] + 0.5*s_gap[z] for z in ZODIAC_CYCLE}
        
        top4 = sorted(ZODIAC_CYCLE, key=lambda z: -final[z])[:4]
        if actual in top4:
            hits += 1
    
    return hits / TEST_PERIODS

# ============================================================
# 方法D: 数字级分析→生肖概率
# ============================================================
def method_number_level():
    """从数字层面分析冷热，转化为生肖概率"""
    hits = 0
    for pi in range(TEST_PERIODS):
        i = START + pi
        num_hist = numbers[:i]
        actual = animals[i]
        
        # 49个数字的冷热度
        num_scores = {}
        w = min(30, len(num_hist))
        recent = num_hist[-w:]
        freq = Counter(recent)
        for n in range(1, 50):
            f = freq.get(n, 0)
            # 冷号加分
            num_scores[n] = 1.0 - f / max(max(freq.values()), 1)
            # 间隔加分
            last = -1
            for j in range(len(num_hist)-1, -1, -1):
                if num_hist[j] == n:
                    last = j
                    break
            gap = len(num_hist) - 1 - last if last >= 0 else len(num_hist)
            num_scores[n] += gap / 49.0 * 0.5
        
        # 数字得分汇总到生肖
        zodiac_scores = {z: 0.0 for z in ZODIAC_CYCLE}
        for n in range(1, 50):
            z = NUM_TO_ZODIAC[n]
            zodiac_scores[z] += num_scores[n]
        
        # 归一化
        mx = max(zodiac_scores.values())
        if mx > 0:
            zodiac_scores = {z: v/mx for z, v in zodiac_scores.items()}
        
        # 和生肖级间隔融合
        s_gap = score_gap(animals[:i])
        final = {z: 0.6*zodiac_scores[z] + 0.4*s_gap[z] for z in ZODIAC_CYCLE}
        
        top4 = sorted(ZODIAC_CYCLE, key=lambda z: -final[z])[:4]
        if actual in top4:
            hits += 1
    
    return hits / TEST_PERIODS

# ============================================================
# 方法E: 排除法（排最不可能的8个）
# ============================================================
def method_exclusion():
    """反向思维: 排除最可能不出的8个生肖"""
    hits = 0
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        actual = animals[i]
        
        # 综合热度 = 最不该选的
        s_hot = {}
        w = 15
        recent = hist[-min(w, len(hist)):]
        freq = Counter(recent)
        for z in ZODIAC_CYCLE:
            # 近期越热 + 刚出现过 → 排除分越高
            hotness = freq.get(z, 0) / max(max(freq.values()), 1)
            # 刚出现的额外惩罚
            recency = 0
            for j in range(min(3, len(hist))):
                if hist[-(j+1)] == z:
                    recency += (3 - j) / 3
            s_hot[z] = hotness * 0.6 + recency * 0.4
        
        # 排除得分最高的8个，剩下4个
        exclude8 = sorted(ZODIAC_CYCLE, key=lambda z: -s_hot[z])[:8]
        top4 = [z for z in ZODIAC_CYCLE if z not in exclude8]
        
        if actual in top4:
            hits += 1
    
    return hits / TEST_PERIODS

# ============================================================
# 方法F: 超级集成（多模型投票+动态权重+条件）
# ============================================================
def method_super_ensemble(adapt_window=20, each_topk=5):
    """
    超级集成策略:
    1. 7种基础策略各选TOP-K
    2. 按近N期命中率给策略加权
    3. 加权投票选TOP4
    """
    strat_funcs = [
        ('冷号W10', lambda h: score_cold(h, 10)),
        ('冷号W20', lambda h: score_cold(h, 20)),
        ('冷号W50', lambda h: score_cold(h, 50)),
        ('间隔', lambda h: score_gap(h)),
        ('马尔可夫1', lambda h: score_markov1(h)),
        ('马尔可夫2', lambda h: score_markov2(h)),
        ('反热门', lambda h: score_anti_recent(h)),
    ]
    
    n_strats = len(strat_funcs)
    perf_history = [[] for _ in range(n_strats)]
    strat_weights = [1.0] * n_strats
    
    hits = 0
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        actual = animals[i]
        
        # 加权投票
        vote_scores = {z: 0.0 for z in ZODIAC_CYCLE}
        strat_tops = []
        for si, (name, sf) in enumerate(strat_funcs):
            s = sf(hist)
            topk = sorted(ZODIAC_CYCLE, key=lambda z: -s[z])[:each_topk]
            strat_tops.append(topk)
            for rank, z in enumerate(topk):
                # 排名越高分越多, 加策略权重
                vote_scores[z] += strat_weights[si] * (each_topk - rank) / each_topk
        
        top4 = sorted(ZODIAC_CYCLE, key=lambda z: -vote_scores[z])[:4]
        hit = actual in top4
        if hit:
            hits += 1
        
        # 更新策略表现
        for si in range(n_strats):
            perf_history[si].append(1 if actual in strat_tops[si] else 0)
        
        # 动态调整权重
        if pi >= adapt_window:
            for si in range(n_strats):
                recent = perf_history[si][-adapt_window:]
                rate = sum(recent) / len(recent)
                strat_weights[si] = max(rate, 0.1)
    
    return hits / TEST_PERIODS

# ============================================================
# 方法G: 贝叶斯融合
# ============================================================
def method_bayesian():
    """贝叶斯后验概率融合"""
    hits = 0
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        actual = animals[i]
        
        # 先验: 整体频率
        freq_all = Counter(hist)
        total = len(hist)
        prior = {z: freq_all.get(z, 0.5) / total for z in ZODIAC_CYCLE}
        
        # 似然1: 马尔可夫转移
        likelihood_m = score_markov1(hist)
        
        # 似然2: 间隔概率（间隔越大越可能出现）
        likelihood_g = score_gap(hist)
        gap_sum = sum(likelihood_g.values())
        if gap_sum > 0:
            likelihood_g = {z: v / gap_sum for z, v in likelihood_g.items()}
        
        # 似然3: 冷号
        likelihood_c = score_cold(hist, 20)
        cold_sum = sum(likelihood_c.values())
        if cold_sum > 0:
            likelihood_c = {z: v / cold_sum for z, v in likelihood_c.items()}
        
        # 后验 ∝ 先验 × 似然1 × 似然2 × 似然3
        posterior = {}
        for z in ZODIAC_CYCLE:
            p = prior[z]
            l1 = likelihood_m.get(z, 1/12)
            l2 = likelihood_g.get(z, 1/12)
            l3 = likelihood_c.get(z, 1/12)
            posterior[z] = p * l1 * l2 * l3
        
        top4 = sorted(ZODIAC_CYCLE, key=lambda z: -posterior[z])[:4]
        if actual in top4:
            hits += 1
    
    return hits / TEST_PERIODS

# ============================================================
# 方法H: 滚动多窗口加权（去相关融合）
# ============================================================
def method_decorrelated_multi_window():
    """多窗口分析，根据窗口表现去相关融合"""
    windows = [5, 8, 12, 15, 20, 25, 30, 40, 50]
    hits = 0
    window_perfs = {w: [] for w in windows}
    window_weights = {w: 1.0 for w in windows}
    
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        actual = animals[i]
        
        vote_scores = {z: 0.0 for z in ZODIAC_CYCLE}
        window_tops = {}
        
        for w in windows:
            if len(hist) < w:
                continue
            recent = hist[-w:]
            freq = Counter(recent)
            # 冷号 + 间隔 混合
            for z in ZODIAC_CYCLE:
                f = freq.get(z, 0)
                mx = max(freq.values())
                cold = 1.0 - f / max(mx, 1)
                # 间隔
                last = -1
                for j in range(len(recent)-1, -1, -1):
                    if recent[j] == z:
                        last = j
                        break
                gap = (len(recent) - 1 - last) / 12 if last >= 0 else w / 12
                score = 0.6 * cold + 0.4 * gap
                vote_scores[z] += window_weights[w] * score
            
            topk = sorted(ZODIAC_CYCLE, key=lambda z: -vote_scores[z])[:4]
            window_tops[w] = topk
        
        top4 = sorted(ZODIAC_CYCLE, key=lambda z: -vote_scores[z])[:4]
        hit = actual in top4
        if hit:
            hits += 1
        
        # 更新窗口权重
        for w in windows:
            if w in window_tops:
                window_perfs[w].append(1 if actual in window_tops[w] else 0)
                if len(window_perfs[w]) >= 15:
                    rate = sum(window_perfs[w][-15:]) / 15
                    window_weights[w] = max(rate, 0.05)
    
    return hits / TEST_PERIODS

# ============================================================
# 辅助函数
# ============================================================
def score_cold(hist, window):
    w = min(window, len(hist))
    freq = Counter(hist[-w:])
    mx = max(freq.values()) if freq else 1
    return {z: 1.0 - freq.get(z, 0) / max(mx, 1) for z in ZODIAC_CYCLE}

def score_gap(hist):
    scores = {}
    for z in ZODIAC_CYCLE:
        last = -1
        for j in range(len(hist)-1, -1, -1):
            if hist[j] == z:
                last = j
                break
        scores[z] = (len(hist) - 1 - last) / 12 if last >= 0 else len(hist) / 12
    return scores

def score_markov1(hist):
    scores = {z: 1/12 for z in ZODIAC_CYCLE}
    if len(hist) < 2:
        return scores
    trans = {}
    for k in range(1, len(hist)):
        prev, curr = hist[k-1], hist[k]
        if prev not in trans:
            trans[prev] = Counter()
        trans[prev][curr] += 1
    state = hist[-1]
    if state in trans:
        total = sum(trans[state].values())
        for z in ZODIAC_CYCLE:
            scores[z] = trans[state].get(z, 0) / total
    return scores

def score_markov2(hist):
    scores = {z: 1/12 for z in ZODIAC_CYCLE}
    if len(hist) < 3:
        return scores
    trans = {}
    for k in range(2, len(hist)):
        prev = (hist[k-2], hist[k-1])
        curr = hist[k]
        if prev not in trans:
            trans[prev] = Counter()
        trans[prev][curr] += 1
    state = (hist[-2], hist[-1])
    if state in trans:
        total = sum(trans[state].values())
        for z in ZODIAC_CYCLE:
            scores[z] = trans[state].get(z, 0) / total
    return scores

def score_anti_recent(hist, decay=5):
    scores = {z: 1.0 for z in ZODIAC_CYCLE}
    dw = min(decay, len(hist))
    for j in range(dw):
        z = hist[-(j+1)]
        scores[z] -= (1.0 - j/dw) * 0.5
    return scores

def score_periodicity(hist):
    scores = {z: 0.0 for z in ZODIAC_CYCLE}
    if len(hist) < 20:
        return scores
    for z in ZODIAC_CYCLE:
        positions = [k for k, a in enumerate(hist) if a == z]
        if len(positions) >= 3:
            intervals = [positions[k+1]-positions[k] for k in range(len(positions)-1)]
            avg = np.mean(intervals)
            gap = len(hist) - 1 - positions[-1]
            if avg > 0:
                ratio = gap / avg
                if 0.8 <= ratio <= 1.5:
                    scores[z] = 0.5 + 0.5 * min(ratio, 1.0)
                elif ratio > 1.5:
                    scores[z] = 1.0
    return scores

# ============================================================
# 运行所有方法对比
# ============================================================
print("\n" + "="*70)
print("方法对比 (300期滚动验证)")
print("="*70)

results = {}

print("\n运行方法A: 动态自适应权重...")
for lb in [10, 15, 20, 25, 30]:
    r = method_adaptive_weights(lookback=lb)
    print(f"  lookback={lb}: {r*100:.1f}%")
    results[f'A_adapt_{lb}'] = r

print("\n运行方法B: 投票集成...")
for k in [4, 5, 6]:
    r = method_voting_ensemble(top_k=k)
    print(f"  each_topk={k}: {r*100:.1f}%")
    results[f'B_vote_k{k}'] = r

print("\n运行方法C: 条件策略...")
r = method_conditional()
print(f"  {r*100:.1f}%")
results['C_cond'] = r

print("\n运行方法D: 数字级分析...")
r = method_number_level()
print(f"  {r*100:.1f}%")
results['D_num'] = r

print("\n运行方法E: 排除法...")
r = method_exclusion()
print(f"  {r*100:.1f}%")
results['E_excl'] = r

print("\n运行方法F: 超级集成...")
for aw in [10, 15, 20, 25]:
    for k in [4, 5, 6]:
        r = method_super_ensemble(adapt_window=aw, each_topk=k)
        if r > 0.44:
            print(f"  adapt={aw}, k={k}: {r*100:.1f}% ⭐")
        results[f'F_super_{aw}_{k}'] = r
# 打印F类最优
f_best = max([(k,v) for k,v in results.items() if k.startswith('F_')], key=lambda x: x[1])
print(f"  F最优: {f_best[0]} = {f_best[1]*100:.1f}%")

print("\n运行方法G: 贝叶斯融合...")
r = method_bayesian()
print(f"  {r*100:.1f}%")
results['G_bayes'] = r

print("\n运行方法H: 去相关多窗口...")
r = method_decorrelated_multi_window()
print(f"  {r*100:.1f}%")
results['H_decorr'] = r

# ============================================================
# 排行榜
# ============================================================
print("\n" + "="*70)
print("排行榜 TOP10")
print("="*70)
sorted_results = sorted(results.items(), key=lambda x: -x[1])
for rank, (name, rate) in enumerate(sorted_results[:10], 1):
    mark = "⭐" if rate >= 0.50 else "★" if rate >= 0.45 else ""
    print(f"  #{rank:>2} {name:>20}: {rate*100:.1f}% {mark}")

print(f"\n总耗时: {time.time()-t0:.1f}秒")
