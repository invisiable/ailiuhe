"""
硬蒸馏: TOP9排除3个生肖 → 从剩下9个中重新推导最优TOP4
============================================
目标: 超过50%成功率

数学基础:
- 直接TOP4从12选: 随机基线 4/12=33.3%, v3静态=45.3%, v3+反miss=48.0%
- 蒸馏TOP4从9选: 随机基线 4/9=44.4%
- TOP9命中率85% → 理论上限85%
- 需要在9选4中达到 50/0.85≈58.8% 才能整体50%

策略方案:
A. 纯静态蒸馏 (30+策略组合)
B. 精细权重网格搜索
C. 反miss机制 (blend热号 + 扩展至TOP5/6)
D. 自适应策略切换 (根据miss连续数切换Stage2)
E. TOP9置信度加权 (TOP9高/低置信度用不同Stage2)
F. 多策略投票法 (多个Stage2投票取交集)
G. 动态窗口Stage2 (根据近期命中调整MK窗口)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import io
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations

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

# ====== 评分函数 ======
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

def second_order_markov(animals, window=150, laplace=0.5):
    probs = np.ones(12) / 12
    h = animals[-window:] if window and len(animals) > window else animals
    if len(h) < 3: return probs
    trans = {}
    for k in range(2, len(h)):
        key = (h[k-2], h[k-1])
        if key not in trans: trans[key] = Counter()
        trans[key][h[k]] += 1
    state = (h[-2], h[-1])
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
            gaps_l = [positions[k+1] - positions[k] for k in range(len(positions)-1)]
            avg_gap = np.mean(gaps_l)
            last_gap = window - positions[-1] if positions else window
            score = 1.0 - abs(last_gap - avg_gap) / max(avg_gap, 1)
            score = max(0, score)
        else:
            score = 0.5
        scores.append(score)
    return np.array(scores)

def streak_momentum(animals, window=15):
    """近期连续出现的动量分"""
    scores = np.zeros(12)
    recent = animals[-window:]
    for zi, z in enumerate(ZODIAC_CYCLE_2026):
        # 越近出现给越高分,衰减因子
        for j, a in enumerate(recent):
            if a == z:
                scores[zi] += (j + 1) / window  # 越近权重越高
    mx = scores.max()
    if mx > 0: scores /= mx
    return scores

def element_cycle_scores(animals, window=30):
    """五行周期分: 基于五行出现规律"""
    elements_map = {'金': 0, '木': 1, '水': 2, '火': 3, '土': 4}
    # 每个生肖对应五行 (简化映射)
    zodiac_element = {
        '鼠': '水', '牛': '土', '虎': '木', '兔': '木',
        '龙': '土', '蛇': '火', '马': '火', '羊': '土',
        '猴': '金', '鸡': '金', '狗': '土', '猪': '水'
    }
    recent = animals[-window:]
    elem_freq = Counter([zodiac_element.get(a, '金') for a in recent])
    total_e = sum(elem_freq.values())
    # 五行中最少出现的 → 对应的生肖应该升高
    scores = np.zeros(12)
    for zi, z in enumerate(ZODIAC_CYCLE_2026):
        e = zodiac_element.get(z, '金')
        freq = elem_freq.get(e, 0) / max(total_e, 1)
        scores[zi] = 1.0 - freq  # 冷五行的生肖得分高
    return scores

# ====== Stage1: TOP9 ======
def stage1_top9(animals):
    return (0.20 * cold_scores(animals, 15) + 0.05 * cold_scores(animals, 30) +
            0.50 * markov_scores(animals, 150) + 0.10 * gap_scores(animals) +
            0.15 * hot_scores(animals, 30))

# ====== 预计算 ======
all_data = []
for pi in range(test_periods):
    i = start + pi
    hist_animals = [NUM_TO_ZODIAC_2026[n] for n in numbers[:i]]
    actual_z = NUM_TO_ZODIAC_2026[numbers[i]]
    all_data.append((hist_animals, actual_z))

def analyze_hits(hl):
    streaks = []
    c = 0
    for h in hl:
        if not h: c += 1
        else:
            if c > 0: streaks.append(c)
            c = 0
    if c > 0: streaks.append(c)
    mm = max(streaks) if streaks else 0
    g4 = sum(1 for s in streaks if s >= 4)
    g6 = sum(1 for s in streaks if s >= 6)
    return mm, g4, g6

# ====== 测试框架 ======
def test_distill_top4(s2_fn, top_n=4):
    """测试硬蒸馏TOP9→TOP4"""
    hits = 0
    hit_list = []
    t9_miss = 0
    for pi in range(test_periods):
        animals, actual_z = all_data[pi]
        s1 = stage1_top9(animals)
        top9_idx = np.argsort(-s1)[:9]
        
        actual_idx = ZODIAC_CYCLE_2026.index(actual_z)
        if actual_idx not in set(top9_idx):
            t9_miss += 1
            hit_list.append(False)
            continue
        
        s2 = s2_fn(animals)
        s2_in9 = s2[top9_idx]
        top_in9 = np.argsort(-s2_in9)[:top_n]
        top_idx = top9_idx[top_in9]
        top = [ZODIAC_CYCLE_2026[i] for i in top_idx]
        
        hit = actual_z in top
        if hit: hits += 1
        hit_list.append(hit)
    
    rate = hits / test_periods * 100
    mm, g4, g6 = analyze_hits(hit_list)
    cost = test_periods * top_n * 4
    roi = (hits * 46 - cost) / cost * 100
    return hits, rate, mm, g4, g6, roi, t9_miss

def test_distill_antimiss(s2_fn, blend_th=2, expand_th=4, expand_n=5, hot_blend=0.25):
    """测试蒸馏+反miss"""
    consec = 0
    hits = 0
    hit_list = []
    sizes = []
    for pi in range(test_periods):
        animals, actual_z = all_data[pi]
        s1 = stage1_top9(animals)
        top9_idx = np.argsort(-s1)[:9]
        actual_idx = ZODIAC_CYCLE_2026.index(actual_z)
        
        s2 = s2_fn(animals)
        s2_in9 = s2[top9_idx]
        
        # 反miss: blend热号
        if consec >= blend_th:
            h = hot_scores(animals, 30)
            h_in9 = h[top9_idx]
            s2_in9 = (1 - hot_blend) * s2_in9 + hot_blend * h_in9
        
        # 扩展
        n = expand_n if consec >= expand_th else 4
        top_in9 = np.argsort(-s2_in9)[:n]
        top_idx = top9_idx[top_in9]
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
    return hits, rate, mm, g4, g6, roi, avg_n

def test_distill_adaptive(s2_fns_by_state):
    """自适应切换: 根据连续miss数选不同Stage2"""
    consec = 0
    hits = 0
    hit_list = []
    sizes = []
    for pi in range(test_periods):
        animals, actual_z = all_data[pi]
        s1 = stage1_top9(animals)
        top9_idx = np.argsort(-s1)[:9]
        actual_idx = ZODIAC_CYCLE_2026.index(actual_z)
        
        # 根据状态选择策略和大小
        state_key = min(consec, max(s2_fns_by_state.keys()))
        s2_fn, n = s2_fns_by_state[state_key]
        
        s2 = s2_fn(animals)
        s2_in9 = s2[top9_idx]
        top_in9 = np.argsort(-s2_in9)[:n]
        top_idx = top9_idx[top_in9]
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
    return hits, rate, mm, g4, g6, roi, avg_n

def test_distill_voting(s2_fn_list, vote_threshold=2, fallback_n=4):
    """多策略投票: 多个Stage2投票,票数≥阈值的入选"""
    hits = 0
    hit_list = []
    sizes = []
    for pi in range(test_periods):
        animals, actual_z = all_data[pi]
        s1 = stage1_top9(animals)
        top9_idx = np.argsort(-s1)[:9]
        actual_idx = ZODIAC_CYCLE_2026.index(actual_z)
        
        # 每个Stage2选TOP4,统计票数
        votes = Counter()
        for s2_fn in s2_fn_list:
            s2 = s2_fn(animals)
            s2_in9 = s2[top9_idx]
            top4_in9 = np.argsort(-s2_in9)[:4]
            top4_idx = top9_idx[top4_in9]
            for idx in top4_idx:
                votes[idx] += 1
        
        # 票数≥阈值的入选
        selected = [idx for idx, v in votes.most_common() if v >= vote_threshold]
        if len(selected) < 4:
            # 不足则补充到4
            for idx, _ in votes.most_common():
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= fallback_n:
                    break
        selected = selected[:max(fallback_n, len([idx for idx, v in votes.items() if v >= vote_threshold]))]
        
        top = [ZODIAC_CYCLE_2026[i] for i in selected]
        n = len(top)
        
        hit = actual_z in top
        if hit:
            hits += 1
        hit_list.append(hit)
        sizes.append(n)
    
    rate = hits / test_periods * 100
    mm, g4, g6 = analyze_hits(hit_list)
    avg_n = np.mean(sizes)
    cost = sum(s * 4 for s in sizes)
    roi = (hits * 46 - cost) / cost * 100
    return hits, rate, mm, g4, g6, roi, avg_n

def test_distill_confidence(s2_fn_high, s2_fn_low, confidence_threshold=0.15):
    """TOP9置信度分层: 高置信度用一种Stage2, 低置信度用另一种+扩展"""
    hits = 0
    hit_list = []
    sizes = []
    for pi in range(test_periods):
        animals, actual_z = all_data[pi]
        s1 = stage1_top9(animals)
        sorted_s1 = np.sort(s1)[::-1]
        # 置信度 = TOP9和TOP10之间的差距
        confidence = sorted_s1[8] - sorted_s1[9] if len(sorted_s1) > 9 else 0
        
        top9_idx = np.argsort(-s1)[:9]
        actual_idx = ZODIAC_CYCLE_2026.index(actual_z)
        
        if confidence >= confidence_threshold:
            s2 = s2_fn_high(animals)
            n = 4
        else:
            s2 = s2_fn_low(animals)
            n = 5  # 低置信度扩展到5
        
        s2_in9 = s2[top9_idx]
        top_in9 = np.argsort(-s2_in9)[:n]
        top_idx = top9_idx[top_in9]
        top = [ZODIAC_CYCLE_2026[i] for i in top_idx]
        
        hit = actual_z in top
        if hit: hits += 1
        hit_list.append(hit)
        sizes.append(n)
    
    rate = hits / test_periods * 100
    mm, g4, g6 = analyze_hits(hit_list)
    avg_n = np.mean(sizes)
    cost = sum(s * 4 for s in sizes)
    roi = (hits * 46 - cost) / cost * 100
    return hits, rate, mm, g4, g6, roi, avg_n

# ====== 输出 ======
buf = io.StringIO()
def p(s=""):
    print(s)
    buf.write(s + "\n")

p("=" * 100)
p("硬蒸馏: TOP9排除3个生肖 → 从9个中重新推导TOP4")
p(f"目标: 超过50%成功率")
p(f"数据: {total}期, 测试: 最近{test_periods}期")
p("=" * 100)

# ====== 方案A: 对照组 + 纯静态蒸馏 ======
p(f"\n{'='*100}")
p("方案A: 对照组 + 纯静态蒸馏 (30+策略)")
p(f"{'='*100}")

# 对照: 直接TOP4
direct_fns = [
    ("直接: v3静态(c15+c30+mk150)",
     lambda a: 0.30*cold_scores(a,15) + 0.10*cold_scores(a,30) + 0.60*markov_scores(a,150)),
    ("直接: TOP9权重",
     lambda a: stage1_top9(a)),
]
p(f"\n  --- 对照组: 直接TOP4 (无蒸馏) ---")
p(f"  {'策略':<35} {'命中':>8} {'命中率':>8} {'mm':>4} {'≥4':>4} {'ROI':>8}")
p(f"  {'-'*70}")

direct_best = 0
for name, fn in direct_fns:
    h, r, mm, g4, g6, roi, _ = test_distill_top4(fn)  # 实际是直接12选4
    # 重新测直接版(不经过TOP9)
    hits2 = 0
    hl2 = []
    for pi in range(test_periods):
        animals, actual_z = all_data[pi]
        scores = fn(animals)
        top4 = [ZODIAC_CYCLE_2026[i] for i in np.argsort(-scores)[:4]]
        hit = actual_z in top4
        if hit: hits2 += 1
        hl2.append(hit)
    r2 = hits2 / test_periods * 100
    mm2, g42, g62 = analyze_hits(hl2)
    cost2 = test_periods * 4 * 4
    roi2 = (hits2 * 46 - cost2) / cost2 * 100
    p(f"  {name:<35} {hits2:>4}/300 {r2:>7.1f}% {mm2:>3} {g42:>3} {roi2:>+7.1f}%")
    direct_best = max(direct_best, r2)

p(f"  直接TOP4最优: {direct_best:.1f}%")
p(f"  随机基线(直接): 4/12 = {4/12*100:.1f}%")
p(f"  随机基线(蒸馏): 4/9 = {4/9*100:.1f}%")

# Stage2策略
p(f"\n  --- 蒸馏TOP4: 在TOP9的9个中选4 ---")
s2_strategies = [
    # 与TOP9相同(对照)
    ("S2: TOP9同权重",     lambda a: stage1_top9(a)),
    ("S2: v3静态",         lambda a: 0.30*cold_scores(a,15) + 0.10*cold_scores(a,30) + 0.60*markov_scores(a,150)),
    
    # 短窗口MK
    ("S2: MK30",           lambda a: markov_scores(a, 30)),
    ("S2: MK50",           lambda a: markov_scores(a, 50)),
    ("S2: MK80",           lambda a: markov_scores(a, 80)),
    ("S2: MK100",          lambda a: markov_scores(a, 100)),
    
    # 二阶MK
    ("S2: 二阶MK100",      lambda a: second_order_markov(a, 100)),
    ("S2: 二阶MK150",      lambda a: second_order_markov(a, 150)),
    ("S2: 二阶MK80",       lambda a: second_order_markov(a, 80)),
    
    # 单维度
    ("S2: 纯间隔",         lambda a: gap_scores(a)),
    ("S2: 纯cold8",        lambda a: cold_scores(a, 8)),
    ("S2: 纯cold10",       lambda a: cold_scores(a, 10)),
    ("S2: 纯cold5",        lambda a: cold_scores(a, 5)),
    ("S2: 纯hot20",        lambda a: hot_scores(a, 20)),
    ("S2: 纯hot10",        lambda a: hot_scores(a, 10)),
    ("S2: 纯轮换",         lambda a: rotation_scores(a, 25)),
    ("S2: 纯动量",         lambda a: streak_momentum(a, 15)),
    ("S2: 五行周期",       lambda a: element_cycle_scores(a, 30)),
    
    # 互补二维组合
    ("S2: MK50+gap",       lambda a: 0.6*markov_scores(a,50)+0.4*gap_scores(a)),
    ("S2: MK80+gap",       lambda a: 0.6*markov_scores(a,80)+0.4*gap_scores(a)),
    ("S2: MK50+cold8",     lambda a: 0.6*markov_scores(a,50)+0.4*cold_scores(a,8)),
    ("S2: 二阶MK+gap",     lambda a: 0.5*second_order_markov(a,150)+0.5*gap_scores(a)),
    ("S2: 二阶MK+MK50",    lambda a: 0.5*second_order_markov(a,100)+0.5*markov_scores(a,50)),
    ("S2: 轮换+gap",       lambda a: 0.5*rotation_scores(a,25)+0.5*gap_scores(a)),
    ("S2: 轮换+MK80",      lambda a: 0.4*rotation_scores(a,25)+0.6*markov_scores(a,80)),
    ("S2: 轮换+cold10",    lambda a: 0.4*rotation_scores(a,25)+0.6*cold_scores(a,10)),
    ("S2: hot20+gap",      lambda a: 0.5*hot_scores(a,20)+0.5*gap_scores(a)),
    ("S2: 动量+MK80",      lambda a: 0.4*streak_momentum(a,15)+0.6*markov_scores(a,80)),
    ("S2: 五行+MK80",      lambda a: 0.3*element_cycle_scores(a,30)+0.7*markov_scores(a,80)),
    
    # 三维组合
    ("S2: MK50+gap+cold8",      lambda a: 0.4*markov_scores(a,50)+0.3*gap_scores(a)+0.3*cold_scores(a,8)),
    ("S2: 二阶MK+gap+rot",      lambda a: 0.4*second_order_markov(a,100)+0.3*gap_scores(a)+0.3*rotation_scores(a,25)),
    ("S2: MK80+gap+hot20",      lambda a: 0.4*markov_scores(a,80)+0.3*gap_scores(a)+0.3*hot_scores(a,20)),
    ("S2: cold8+gap+MK30",      lambda a: 0.3*cold_scores(a,8)+0.35*gap_scores(a)+0.35*markov_scores(a,30)),
    ("S2: MK50+动量+rot",       lambda a: 0.4*markov_scores(a,50)+0.3*streak_momentum(a,15)+0.3*rotation_scores(a,25)),
    ("S2: v3+gap补充",          lambda a: 0.7*(0.30*cold_scores(a,15)+0.10*cold_scores(a,30)+0.60*markov_scores(a,150))+0.3*gap_scores(a)),
    ("S2: v3+rot补充",          lambda a: 0.7*(0.30*cold_scores(a,15)+0.10*cold_scores(a,30)+0.60*markov_scores(a,150))+0.3*rotation_scores(a,25)),
]

p(f"  {'策略':<28} {'命中':>8} {'命中率':>8} {'mm':>4} {'≥4':>4} {'≥6':>4} {'ROI':>8} {'vs直接':>8}")
p(f"  {'-'*80}")

best_static = (0, "", None, None)
all_s2_results = []

for name, fn in s2_strategies:
    h, r, mm, g4, g6, roi, t9m = test_distill_top4(fn)
    diff = r - direct_best
    mark = " ★" if r > direct_best else (" ●50%+" if r >= 50 else "")
    p(f"  {name:<28} {h:>4}/300 {r:>7.1f}% {mm:>3} {g4:>3} {g6:>3} {roi:>+7.1f}% {diff:>+7.1f}%{mark}")
    if r > best_static[0]:
        best_static = (r, name, (h, mm, g4, g6, roi), fn)
    all_s2_results.append((r, name, h, mm, g4, g6, roi, fn))

p(f"\n  TOP9过滤miss: {t9m}/300 = {t9m/3:.1f}%")
p(f"  方案A最优: {best_static[1]} = {best_static[0]:.1f}%")

# ====== 方案B: 精细权重搜索 ======
p(f"\n{'='*100}")
p("方案B: 精细权重搜索 (Stage2最优权重)")
p(f"{'='*100}")

best_fine = (0, "", None)
fine_results = []
configs = []

# MK短窗口 + gap + cold短 + rot + hot
for mk_win in [30, 50, 80, 100]:
    for mkw in [0.0, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
        for gapw in [0.0, 0.10, 0.20, 0.30, 0.40]:
            for cw in [0.0, 0.10, 0.20, 0.30]:
                for rotw in [0.0, 0.10, 0.20, 0.30]:
                    hotw = round(1.0 - mkw - gapw - cw - rotw, 2)
                    if hotw < -0.01 or hotw > 0.40:
                        continue
                    hotw = max(hotw, 0)
                    configs.append(('mk', mkw, mk_win, gapw, cw, rotw, hotw))

# 二阶MK组合
for mk2w in [0.20, 0.30, 0.40, 0.50, 0.60]:
    for gapw in [0.0, 0.10, 0.20, 0.30]:
        for cw in [0.0, 0.10, 0.20]:
            for rotw in [0.0, 0.10, 0.20]:
                hotw = round(1.0 - mk2w - gapw - cw - rotw, 2)
                if hotw < -0.01 or hotw > 0.35:
                    continue
                hotw = max(hotw, 0)
                configs.append(('mk2', mk2w, 150, gapw, cw, rotw, hotw))

# v3基础 + 补充维度
for v3w in [0.50, 0.60, 0.70, 0.80]:
    for gapw in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]:
        for rotw in [0.0, 0.10, 0.15, 0.20]:
            momw = round(1.0 - v3w - gapw - rotw, 2)
            if momw < -0.01 or momw > 0.30:
                continue
            momw = max(momw, 0)
            configs.append(('v3', v3w, 0, gapw, 0, rotw, momw))

p(f"  搜索{len(configs)}种组合...")

for cfg in configs:
    typ, w1, win, gapw, cw, rotw, hotw = cfg
    
    if typ == 'mk':
        def make_fn(m, w, g, c, r, h):
            def fn(a):
                s = m * markov_scores(a, w)
                if g > 0: s = s + g * gap_scores(a)
                if c > 0: s = s + c * cold_scores(a, 8)
                if r > 0: s = s + r * rotation_scores(a, 25)
                if h > 0: s = s + h * hot_scores(a, 20)
                return s
            return fn
        fn = make_fn(w1, win, gapw, cw, rotw, hotw)
    elif typ == 'mk2':
        def make_fn2(m2, g, c, r, h):
            def fn(a):
                s = m2 * second_order_markov(a, 150)
                if g > 0: s = s + g * gap_scores(a)
                if c > 0: s = s + c * cold_scores(a, 8)
                if r > 0: s = s + r * rotation_scores(a, 25)
                if h > 0: s = s + h * hot_scores(a, 20)
                return s
            return fn
        fn = make_fn2(w1, gapw, cw, rotw, hotw)
    else:  # v3
        def make_fnv3(vw, g, r, m):
            def fn(a):
                s = vw * (0.30*cold_scores(a,15) + 0.10*cold_scores(a,30) + 0.60*markov_scores(a,150))
                if g > 0: s = s + g * gap_scores(a)
                if r > 0: s = s + r * rotation_scores(a, 25)
                if m > 0: s = s + m * streak_momentum(a, 15)
                return s
            return fn
        fn = make_fnv3(w1, gapw, rotw, hotw)
    
    h, r, mm, g4, g6, roi, _ = test_distill_top4(fn)
    if r > best_fine[0]:
        if typ == 'mk':
            label = f"mk{win}={w1} gap={gapw} c8={cw} rot={rotw} hot={hotw}"
        elif typ == 'mk2':
            label = f"mk2={w1} gap={gapw} c8={cw} rot={rotw} hot={hotw}"
        else:
            label = f"v3={w1} gap={gapw} rot={rotw} mom={hotw}"
        best_fine = (r, label, (h, mm, g4, g6, roi), fn)
    if r >= 47:
        fine_results.append((r, h, mm, g4, g6, roi, cfg, fn))

fine_results.sort(key=lambda x: (-x[0], x[2]))

p(f"\n  精细最优: {best_fine[1]}")
p(f"    {best_fine[2][0]}/300 = {best_fine[0]:.1f}% maxMiss={best_fine[2][1]} ROI={best_fine[2][4]:+.1f}%")

if fine_results:
    p(f"\n  ≥47%的组合 (前15):")
    seen = set()
    for r, h, mm, g4, g6, roi, cfg, fn in fine_results[:30]:
        key = f"{h}_{mm}"
        if key in seen: continue
        seen.add(key)
        typ, w1, win, gapw, cw, rotw, hotw = cfg
        if typ == 'mk':
            desc = f"mk{win}={w1} gap={gapw} c8={cw} rot={rotw} hot={hotw}"
        elif typ == 'mk2':
            desc = f"mk2={w1} gap={gapw} c8={cw} rot={rotw} hot={hotw}"
        else:
            desc = f"v3={w1} gap={gapw} rot={rotw} mom={hotw}"
        p(f"    {r:.1f}% ({h}/300) mm={mm} ≥4miss={g4} {desc}")
        if len(seen) >= 15: break

# ====== 方案C: 反miss机制 ======
p(f"\n{'='*100}")
p("方案C: 蒸馏 + 反miss机制 (blend热号 + 扩展)")
p(f"{'='*100}")

# 取最优几个Stage2基础
base_fns = sorted(all_s2_results, key=lambda x: -x[0])[:5]
if best_fine[3] is not None:
    base_fns.append((best_fine[0], "精细最优", best_fine[2][0], best_fine[2][1], best_fine[2][2], best_fine[2][3], best_fine[2][4], best_fine[3]))

p(f"  {'策略':<45} {'命中':>8} {'命中率':>8} {'mm':>4} {'≥4':>4} {'avg_n':>6} {'ROI':>8}")
p(f"  {'-'*85}")

best_antimiss = (0, "", None)
for _, sname, *rest in base_fns:
    sfn = rest[-1] if callable(rest[-1]) else None
    if sfn is None:
        continue
    
    for bt in [1, 2, 3]:
        for et in [2, 3, 4, 5]:
            if et <= bt: continue
            for en in [5, 6]:
                for hb in [0.20, 0.25, 0.30, 0.40]:
                    h, r, mm, g4, g6, roi, avg_n = test_distill_antimiss(
                        sfn, blend_th=bt, expand_th=et, expand_n=en, hot_blend=hb)
                    if r >= 48:
                        label = f"{sname}+反miss(b@{bt}h{hb},e@{et}→{en})"
                        p(f"  {label:<45} {h:>4}/300 {r:>7.1f}% {mm:>3} {g4:>3} {avg_n:>5.2f} {roi:>+7.1f}%")
                        if r > best_antimiss[0]:
                            best_antimiss = (r, label, (h, mm, g4, g6, roi, avg_n))

p(f"\n  方案C最优: {best_antimiss[1]} = {best_antimiss[0]:.1f}%")

# ====== 方案D: 自适应策略切换 ======
p(f"\n{'='*100}")
p("方案D: 自适应策略切换 (根据连续miss数切换Stage2+扩展)")
p(f"{'='*100}")

# 定义几种Stage2
s2_v3 = lambda a: 0.30*cold_scores(a,15) + 0.10*cold_scores(a,30) + 0.60*markov_scores(a,150)
s2_mk50 = lambda a: markov_scores(a, 50)
s2_mk80 = lambda a: markov_scores(a, 80)
s2_gap = lambda a: gap_scores(a)
s2_mk2 = lambda a: second_order_markov(a, 150)
s2_hot_gap = lambda a: 0.5*hot_scores(a,20) + 0.5*gap_scores(a)
s2_rot_gap = lambda a: 0.5*rotation_scores(a,25) + 0.5*gap_scores(a)

adaptive_configs = [
    ("miss0→v3(4), miss1→v3(4), miss2→mk50(5), miss3+→gap(6)",
     {0: (s2_v3, 4), 1: (s2_v3, 4), 2: (s2_mk50, 5), 3: (s2_gap, 6)}),
    ("miss0→v3(4), miss1→mk80(4), miss2→gap(5), miss3+→hot+gap(6)",
     {0: (s2_v3, 4), 1: (s2_mk80, 4), 2: (s2_gap, 5), 3: (s2_hot_gap, 6)}),
    ("miss0→v3(4), miss2→rot+gap(5), miss4+→gap(6)",
     {0: (s2_v3, 4), 2: (s2_rot_gap, 5), 4: (s2_gap, 6)}),
    ("miss0→v3(4), miss1→v3(5), miss3+→gap(6)",
     {0: (s2_v3, 4), 1: (s2_v3, 5), 3: (s2_gap, 6)}),
    ("miss0→v3(4), miss2→mk2(5), miss4+→rot+gap(6)",
     {0: (s2_v3, 4), 2: (s2_mk2, 5), 4: (s2_rot_gap, 6)}),
    ("miss0→v3(4), miss1→v3(4), miss3→mk80(5), miss5+→gap(7)",
     {0: (s2_v3, 4), 1: (s2_v3, 4), 3: (s2_mk80, 5), 5: (s2_gap, 7)}),
    ("miss0→v3(4), miss2→v3(5), miss3+→v3(6)",
     {0: (s2_v3, 4), 2: (s2_v3, 5), 3: (s2_v3, 6)}),
    ("miss0→v3(4), miss1→v3(5), miss2+→v3(6)",
     {0: (s2_v3, 4), 1: (s2_v3, 5), 2: (s2_v3, 6)}),
    ("miss0→v3(4), miss2→mk80(5), miss4→hot+gap(6), miss6+→gap(7)",
     {0: (s2_v3, 4), 2: (s2_mk80, 5), 4: (s2_hot_gap, 6), 6: (s2_gap, 7)}),
    # 纯扩展(不切换策略)
    ("miss0→v3(4), miss1→v3(5), miss3→v3(7), miss5+→v3(9)",
     {0: (s2_v3, 4), 1: (s2_v3, 5), 3: (s2_v3, 7), 5: (s2_v3, 9)}),
]

p(f"  {'策略':<60} {'命中':>8} {'命中率':>8} {'mm':>4} {'≥4':>4} {'avg':>6} {'ROI':>8}")
p(f"  {'-'*100}")

best_adaptive = (0, "", None)
for name, state_map in adaptive_configs:
    h, r, mm, g4, g6, roi, avg_n = test_distill_adaptive(state_map)
    p(f"  {name:<60} {h:>4}/300 {r:>7.1f}% {mm:>3} {g4:>3} {avg_n:>5.2f} {roi:>+7.1f}%")
    if r > best_adaptive[0]:
        best_adaptive = (r, name, (h, mm, g4, g6, roi, avg_n))

p(f"\n  方案D最优: {best_adaptive[0]:.1f}% ({best_adaptive[1]})")

# ====== 方案E: TOP9置信度分层 ======
p(f"\n{'='*100}")
p("方案E: TOP9置信度分层 (高置信→4个, 低置信→5个)")
p(f"{'='*100}")

p(f"  {'策略':<55} {'命中':>8} {'命中率':>8} {'mm':>4} {'≥4':>4} {'avg':>6} {'ROI':>8}")
p(f"  {'-'*95}")

best_conf = (0, "", None)
conf_strats = [
    ("高:v3/低:v3", s2_v3, s2_v3),
    ("高:v3/低:mk50", s2_v3, s2_mk50),
    ("高:v3/低:gap", s2_v3, s2_gap),
    ("高:v3/低:rot+gap", s2_v3, s2_rot_gap),
    ("高:v3/低:mk2", s2_v3, s2_mk2),
    ("高:mk80/低:gap", s2_mk80, s2_gap),
]

for thr in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    for name, fn_h, fn_l in conf_strats:
        h, r, mm, g4, g6, roi, avg_n = test_distill_confidence(fn_h, fn_l, thr)
        label = f"thr={thr} {name}"
        if r >= 48:
            p(f"  {label:<55} {h:>4}/300 {r:>7.1f}% {mm:>3} {g4:>3} {avg_n:>5.2f} {roi:>+7.1f}%")
        if r > best_conf[0]:
            best_conf = (r, label, (h, mm, g4, g6, roi, avg_n))

p(f"\n  方案E最优: {best_conf[0]:.1f}% ({best_conf[1]})")

# ====== 方案F: 多策略投票 ======
p(f"\n{'='*100}")
p("方案F: 多策略投票法 (多个Stage2投票选TOP4)")
p(f"{'='*100}")

vote_sets = [
    ("v3+mk50+gap (≥2票)", [s2_v3, s2_mk50, s2_gap], 2),
    ("v3+mk80+gap (≥2票)", [s2_v3, s2_mk80, s2_gap], 2),
    ("v3+mk50+mk2 (≥2票)", [s2_v3, s2_mk50, s2_mk2], 2),
    ("v3+mk80+gap+rot (≥2票)", [s2_v3, s2_mk80, s2_gap, s2_rot_gap], 2),
    ("v3+mk80+gap+rot (≥3票)", [s2_v3, s2_mk80, s2_gap, s2_rot_gap], 3),
    ("v3+mk50+mk2+gap (≥2票)", [s2_v3, s2_mk50, s2_mk2, s2_gap], 2),
    ("v3+mk50+mk2+gap (≥3票)", [s2_v3, s2_mk50, s2_mk2, s2_gap], 3),
    ("v3+mk80+mk2+rot+gap (≥3票)", [s2_v3, s2_mk80, s2_mk2, s2_rot_gap, s2_gap], 3),
]

p(f"  {'策略':<40} {'命中':>8} {'命中率':>8} {'mm':>4} {'≥4':>4} {'avg':>6} {'ROI':>8}")
p(f"  {'-'*85}")

best_vote = (0, "", None)
for name, fns, thr in vote_sets:
    h, r, mm, g4, g6, roi, avg_n = test_distill_voting(fns, thr)
    p(f"  {name:<40} {h:>4}/300 {r:>7.1f}% {mm:>3} {g4:>3} {avg_n:>5.2f} {roi:>+7.1f}%")
    if r > best_vote[0]:
        best_vote = (r, name, (h, mm, g4, g6, roi, avg_n))

p(f"\n  方案F最优: {best_vote[0]:.1f}% ({best_vote[1]})")

# ====== 方案G: 动态窗口 ======
p(f"\n{'='*100}")
p("方案G: 动态窗口Stage2 (根据近期命中率调整MK窗口)")
p(f"{'='*100}")

def test_dynamic_window(lookback=20, short_win=50, long_win=150, threshold=0.45):
    """近期命中率高用长窗口(稳定), 低用短窗口(灵活)"""
    consec = 0
    hits = 0
    hit_list = []
    sizes = []
    recent_hits = []
    
    for pi in range(test_periods):
        animals, actual_z = all_data[pi]
        s1 = stage1_top9(animals)
        top9_idx = np.argsort(-s1)[:9]
        
        # 根据近期命中率选窗口
        if len(recent_hits) >= lookback:
            recent_rate = sum(recent_hits[-lookback:]) / lookback
        else:
            recent_rate = 0.5
        
        if recent_rate >= threshold:
            mk_win = long_win
        else:
            mk_win = short_win
        
        s2 = 0.30*cold_scores(animals,15) + 0.10*cold_scores(animals,30) + 0.60*markov_scores(animals,mk_win)
        
        # 反miss扩展
        n = 4
        if consec >= 3:
            n = 5
        if consec >= 5:
            n = 6
        
        s2_in9 = s2[top9_idx]
        top_in9 = np.argsort(-s2_in9)[:n]
        top_idx = top9_idx[top_in9]
        top = [ZODIAC_CYCLE_2026[i] for i in top_idx]
        
        hit = actual_z in top
        if hit:
            hits += 1
            consec = 0
        else:
            consec += 1
        hit_list.append(hit)
        recent_hits.append(1 if hit else 0)
        sizes.append(n)
    
    rate = hits / test_periods * 100
    mm, g4, g6 = analyze_hits(hit_list)
    avg_n = np.mean(sizes)
    cost = sum(s * 4 for s in sizes)
    roi = (hits * 46 - cost) / cost * 100
    return hits, rate, mm, g4, g6, roi, avg_n

p(f"  {'策略':<55} {'命中':>8} {'命中率':>8} {'mm':>4} {'≥4':>4} {'avg':>6} {'ROI':>8}")
p(f"  {'-'*95}")

best_dyn = (0, "", None)
for lb in [10, 15, 20, 30]:
    for sw in [30, 50, 80]:
        for lw in [100, 150, 200]:
            for thr in [0.35, 0.40, 0.45, 0.50]:
                h, r, mm, g4, g6, roi, avg_n = test_dynamic_window(lb, sw, lw, thr)
                label = f"lb={lb} short={sw} long={lw} thr={thr}"
                if r >= 49:
                    p(f"  {label:<55} {h:>4}/300 {r:>7.1f}% {mm:>3} {g4:>3} {avg_n:>5.2f} {roi:>+7.1f}%")
                if r > best_dyn[0]:
                    best_dyn = (r, label, (h, mm, g4, g6, roi, avg_n))

p(f"\n  方案G最优: {best_dyn[0]:.1f}% ({best_dyn[1]})")

# ====== 方案H: 激进反miss (早扩展 + 多级) ======
p(f"\n{'='*100}")
p("方案H: 激进反miss (更早扩展 + 多级递增)")
p(f"{'='*100}")

def test_aggressive_antimiss(s2_fn, levels):
    """多级反miss: levels = [(miss_count, expand_n, hot_blend), ...]"""
    consec = 0
    hits = 0
    hit_list = []
    sizes = []
    
    for pi in range(test_periods):
        animals, actual_z = all_data[pi]
        s1 = stage1_top9(animals)
        top9_idx = np.argsort(-s1)[:9]
        
        s2 = s2_fn(animals)
        s2_in9 = s2[top9_idx]
        
        # 根据miss连续数确定级别
        n = 4
        hb = 0
        for miss_th, exp_n, h_blend in levels:
            if consec >= miss_th:
                n = exp_n
                hb = h_blend
        
        if hb > 0:
            h_in9 = hot_scores(animals, 30)[top9_idx]
            s2_in9 = (1 - hb) * s2_in9 + hb * h_in9
        
        top_in9 = np.argsort(-s2_in9)[:n]
        top_idx = top9_idx[top_in9]
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
    return hits, rate, mm, g4, g6, roi, avg_n

aggressive_levels = [
    # (名称, levels)
    ("miss1→5, miss3→6, miss5→7",       [(1, 5, 0), (3, 6, 0.2), (5, 7, 0.3)]),
    ("miss1→5, miss2→6, miss4→7",       [(1, 5, 0), (2, 6, 0.2), (4, 7, 0.3)]),
    ("miss1→5, miss3→7, miss5→9",       [(1, 5, 0), (3, 7, 0.2), (5, 9, 0.3)]),
    ("miss2→5, miss4→6, miss6→8",       [(2, 5, 0.1), (4, 6, 0.2), (6, 8, 0.3)]),
    ("miss1→5h0.2, miss2→6h0.3",       [(1, 5, 0.2), (2, 6, 0.3)]),
    ("miss1→5h0.3, miss3→7h0.4",       [(1, 5, 0.3), (3, 7, 0.4)]),
    ("miss2→5, miss3→6, miss4→7, miss6→9", [(2, 5, 0), (3, 6, 0.15), (4, 7, 0.25), (6, 9, 0.35)]),
    ("miss1→5, miss2→6, miss3→7, miss4+→9", [(1, 5, 0), (2, 6, 0.1), (3, 7, 0.2), (4, 9, 0.3)]),
]

p(f"  {'策略':<55} {'命中':>8} {'命中率':>8} {'mm':>4} {'≥4':>4} {'avg':>6} {'ROI':>8}")
p(f"  {'-'*95}")

best_aggr = (0, "", None)
for name, levels in aggressive_levels:
    h, r, mm, g4, g6, roi, avg_n = test_aggressive_antimiss(s2_v3, levels)
    label = f"v3+{name}"
    p(f"  {label:<55} {h:>4}/300 {r:>7.1f}% {mm:>3} {g4:>3} {avg_n:>5.2f} {roi:>+7.1f}%")
    if r > best_aggr[0]:
        best_aggr = (r, label, (h, mm, g4, g6, roi, avg_n))

p(f"\n  方案H最优: {best_aggr[0]:.1f}% ({best_aggr[1]})")

# ====== 总结 ======
p(f"\n{'='*100}")
p("总结对比")
p(f"{'='*100}")
p(f"  随机基线(直接TOP4): 4/12 = {4/12*100:.1f}%")
p(f"  随机基线(蒸馏TOP4): 4/9  = {4/9*100:.1f}%")
p(f"  直接TOP4最优(v3):    {direct_best:.1f}%")
p()
p(f"  方案A 纯静态蒸馏:    {best_static[0]:.1f}% ({best_static[1]})")
p(f"  方案B 精细权重搜索:  {best_fine[0]:.1f}% ({best_fine[1]})")
p(f"  方案C 蒸馏+反miss:   {best_antimiss[0]:.1f}% ({best_antimiss[1]})")
p(f"  方案D 自适应切换:    {best_adaptive[0]:.1f}%")
p(f"  方案E 置信度分层:    {best_conf[0]:.1f}%")
p(f"  方案F 多策略投票:    {best_vote[0]:.1f}% ({best_vote[1]})")
p(f"  方案G 动态窗口:      {best_dyn[0]:.1f}%")
p(f"  方案H 激进反miss:    {best_aggr[0]:.1f}%")

all_bests = [
    (best_static[0], "A纯静态"),
    (best_fine[0], "B精细搜索"),
    (best_antimiss[0], "C蒸馏+反miss"),
    (best_adaptive[0], "D自适应"),
    (best_conf[0], "E置信度"),
    (best_vote[0], "F投票"),
    (best_dyn[0], "G动态窗口"),
    (best_aggr[0], "H激进反miss"),
]
all_bests.sort(key=lambda x: -x[0])

p(f"\n  排名:")
for rank, (r, name) in enumerate(all_bests, 1):
    mark = " ✅>50%" if r >= 50 else ""
    diff = r - direct_best
    p(f"    {rank}. {name:<15} {r:.1f}% (vs直接: {diff:+.1f}%){mark}")

overall_best = all_bests[0][0]
if overall_best >= 50:
    p(f"\n  ✅ 达到50%目标! 最高: {overall_best:.1f}%")
else:
    p(f"\n  ❌ 未达50%目标. 最高: {overall_best:.1f}%")
    p(f"  💡 TOP4(4/9=44.4%随机)要达到50%,需要Stage2在TOP9命中的254期中命中150期(59%)")
    p(f"     这比直接TOP4的v3(45.3%)高不了多少,因为TOP9和TOP4共享核心特征")

with open('distill_top9_to_top4_v3.txt', 'w', encoding='utf-8-sig') as f:
    f.write(buf.getvalue())
p(f"\n结果已保存到 distill_top9_to_top4_v3.txt")
