"""
蒸馏方案精简版: TOP9过滤 → 从9个中选TOP4
减少穷举空间, 快速得到结论
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import io
import pandas as pd
import numpy as np
from collections import Counter

ZODIAC_CYCLE_2026 = ['马', '蛇', '龙', '兔', '虎', '牛', '鼠', '猪', '狗', '鸡', '猴', '羊']
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

# 预计算
all_data = []
for pi in range(test_periods):
    i = start + pi
    hist_animals = [NUM_TO_ZODIAC_2026[n] for n in numbers[:i]]
    actual_z = NUM_TO_ZODIAC_2026[numbers[i]]
    all_data.append((hist_animals, actual_z))

def stage1_top9(animals):
    return (0.20 * cold_scores(animals, 15) + 0.05 * cold_scores(animals, 30) +
            0.50 * markov_scores(animals, 150) + 0.10 * gap_scores(animals) +
            0.15 * hot_scores(animals, 30))

def analyze_hits(hl):
    streaks = []
    c = 0
    for h in hl:
        if not h: c += 1
        else:
            if c > 0: streaks.append(c)
            c = 0
    if c > 0: streaks.append(c)
    return max(streaks) if streaks else 0

def test_distill(s2_fn, top_n=4, label=""):
    """测试蒸馏策略"""
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
        
        s2 = s2_fn(animals)
        s2_in9 = s2[top9_idx]
        top_in9 = np.argsort(-s2_in9)[:top_n]
        top_idx = top9_idx[top_in9]
        top = [ZODIAC_CYCLE_2026[i] for i in top_idx]
        
        hit = actual_z in top
        if hit: hits += 1
        hit_list.append(hit)
    
    rate = hits / test_periods * 100
    mm = analyze_hits(hit_list)
    cost = test_periods * top_n * 4
    roi = (hits * 46 - cost) / cost * 100
    return hits, rate, mm, roi, t9_miss

def test_direct(score_fn, top_n=4):
    """测试直接策略(无过滤)"""
    hits = 0
    hit_list = []
    for pi in range(test_periods):
        animals, actual_z = all_data[pi]
        scores = score_fn(animals)
        top = [ZODIAC_CYCLE_2026[i] for i in np.argsort(-scores)[:top_n]]
        hit = actual_z in top
        if hit: hits += 1
        hit_list.append(hit)
    rate = hits / test_periods * 100
    mm = analyze_hits(hit_list)
    cost = test_periods * top_n * 4
    roi = (hits * 46 - cost) / cost * 100
    return hits, rate, mm, roi

def test_distill_antimiss(s2_fn, blend_t=2, expand_t=4):
    """蒸馏+反miss"""
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
        
        if consec >= blend_t:
            h = hot_scores(animals, 30)[top9_idx]
            s2_in9 = 0.75 * s2_in9 + 0.25 * h
        
        n = 5 if consec >= expand_t else 4
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
    mm = analyze_hits(hit_list)
    avg_n = np.mean(sizes)
    cost = sum(s * 4 for s in sizes)
    roi = (hits * 46 - cost) / cost * 100
    return hits, rate, mm, roi, avg_n

buf = io.StringIO()
def p(s=""):
    print(s)
    buf.write(s + "\n")

p("=" * 100)
p("蒸馏方案验证: TOP9(85%)过滤 → 从9个中选TOP4")
p(f"数据: {total}期, 测试: 最近{test_periods}期")
p("=" * 100)

# ===== 对照组 =====
p(f"\n--- 对照组: 直接TOP4 ---")
v3_static = lambda a: 0.30 * cold_scores(a, 15) + 0.10 * cold_scores(a, 30) + 0.60 * markov_scores(a, 150)
h1, r1, m1, roi1 = test_direct(v3_static, 4)
p(f"  直接v3静态TOP4:    {h1}/300 = {r1:.1f}% maxMiss={m1} ROI={roi1:+.1f}%")

top9w = lambda a: stage1_top9(a)
h2, r2, m2, roi2 = test_direct(top9w, 4)
p(f"  TOP9权重直选TOP4:  {h2}/300 = {r2:.1f}% maxMiss={m2} ROI={roi2:+.1f}%")

# ===== 蒸馏: 多种Stage2 =====
p(f"\n--- 蒸馏: TOP9过滤 → Stage2选TOP4 ---")

s2_strategies = [
    ("S2: v3静态",       lambda a: 0.30*cold_scores(a,15) + 0.10*cold_scores(a,30) + 0.60*markov_scores(a,150)),
    ("S2: MK50",         lambda a: markov_scores(a,50)),
    ("S2: MK80",         lambda a: markov_scores(a,80)),
    ("S2: MK100",        lambda a: markov_scores(a,100)),
    ("S2: MK150",        lambda a: markov_scores(a,150)),
    ("S2: cold10",       lambda a: cold_scores(a,10)),
    ("S2: cold15",       lambda a: cold_scores(a,15)),
    ("S2: cold8+MK80",   lambda a: 0.40*cold_scores(a,8) + 0.60*markov_scores(a,80)),
    ("S2: cold15+MK150", lambda a: 0.30*cold_scores(a,15) + 0.70*markov_scores(a,150)),
    ("S2: gap+MK150",    lambda a: 0.30*gap_scores(a) + 0.70*markov_scores(a,150)),
    ("S2: cold15+gap",   lambda a: 0.50*cold_scores(a,15) + 0.50*gap_scores(a)),
    ("S2: hot20+MK80",   lambda a: 0.30*hot_scores(a,20) + 0.70*markov_scores(a,80)),
    ("S2: hot30+MK150",  lambda a: 0.25*hot_scores(a,30) + 0.75*markov_scores(a,150)),
    ("S2: 综合",         lambda a: 0.20*cold_scores(a,15) + 0.05*cold_scores(a,30) + 0.50*markov_scores(a,150) + 0.10*gap_scores(a) + 0.15*hot_scores(a,30)),
]

p(f"  {'策略':<22} {'命中':>8} {'命中率':>8} {'mm':>4} {'ROI':>8} {'vs直接':>8}")
p(f"  {'-'*65}")

best_s2 = (0, None, "")
for name, fn in s2_strategies:
    h, r, m, roi, t9m = test_distill(fn, 4, name)
    diff = r - r1
    mark = " ★" if r > r1 else ""
    p(f"  {name:<22} {h:>4}/300 {r:>7.1f}% {m:>3} {roi:>+7.1f}% {diff:>+7.1f}%{mark}")
    if r > best_s2[0]:
        best_s2 = (r, fn, name)

p(f"\n  TOP9过滤miss: {t9m}/300 = {t9m/3:.1f}% (蒸馏理论上限: {(300-t9m)/300*100:.1f}%)")

# ===== 精细权重搜索 (减少维度) =====
p(f"\n--- 精细搜索: Stage2最优权重 ---")

best_fine = (0, "", None)
fine_results = []

for c15w in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    for mkw in [0.30, 0.40, 0.50, 0.60, 0.70]:
        for mk_win in [80, 100, 150]:
            remain = 1.0 - c15w - mkw
            if remain < -0.01: continue
            c30w = max(remain, 0)
            
            fn = lambda a, _c15=c15w, _c30=c30w, _mk=mkw, _w=mk_win: (
                _c15 * cold_scores(a, 15) + _c30 * cold_scores(a, 30) + _mk * markov_scores(a, _w)
            )
            h, r, m, roi, _ = test_distill(fn, 4)
            if r >= best_fine[0]:
                label = f"c15={c15w} c30={c30w:.2f} mk{mk_win}={mkw}"
                best_fine = (r, label, (h, m, roi))
            if r >= 46:
                fine_results.append((r, h, m, roi, c15w, c30w, mkw, mk_win))

# 加gap维度
for c15w in [0.20, 0.25, 0.30]:
    for mkw in [0.40, 0.50, 0.60]:
        for gapw in [0.10, 0.15, 0.20]:
            for mk_win in [80, 100, 150]:
                remain = 1.0 - c15w - mkw - gapw
                if remain < -0.01 or remain > 0.25: continue
                c30w = max(remain, 0)
                
                fn = lambda a, _c15=c15w, _c30=c30w, _mk=mkw, _g=gapw, _w=mk_win: (
                    _c15 * cold_scores(a, 15) + _c30 * cold_scores(a, 30) +
                    _mk * markov_scores(a, _w) + _g * gap_scores(a)
                )
                h, r, m, roi, _ = test_distill(fn, 4)
                if r >= best_fine[0]:
                    label = f"c15={c15w} c30={c30w:.2f} mk{mk_win}={mkw} gap={gapw}"
                    best_fine = (r, label, (h, m, roi))
                if r >= 46:
                    fine_results.append((r, h, m, roi, c15w, c30w, mkw, mk_win))

# 加hot维度
for c15w in [0.20, 0.25, 0.30]:
    for mkw in [0.40, 0.50, 0.60]:
        for hotw in [0.05, 0.10, 0.15, 0.20]:
            for mk_win in [80, 100, 150]:
                remain = 1.0 - c15w - mkw - hotw
                if remain < -0.01 or remain > 0.25: continue
                c30w = max(remain, 0)
                
                fn = lambda a, _c15=c15w, _c30=c30w, _mk=mkw, _h=hotw, _w=mk_win: (
                    _c15 * cold_scores(a, 15) + _c30 * cold_scores(a, 30) +
                    _mk * markov_scores(a, _w) + _h * hot_scores(a, 30)
                )
                h, r, m, roi, _ = test_distill(fn, 4)
                if r >= best_fine[0]:
                    label = f"c15={c15w} c30={c30w:.2f} mk{mk_win}={mkw} hot={hotw}"
                    best_fine = (r, label, (h, m, roi))
                if r >= 46:
                    fine_results.append((r, h, m, roi, c15w, c30w, mkw, mk_win))

fine_results.sort(key=lambda x: (-x[0], x[2]))
p(f"  Stage2最优: {best_fine[1]}")
p(f"    {best_fine[2][0]}/300 = {best_fine[0]:.1f}% maxMiss={best_fine[2][1]} ROI={best_fine[2][2]:+.1f}%")

if fine_results:
    p(f"\n  ≥46%的组合 (前5):")
    seen = set()
    for r, h, m, roi, *params in fine_results[:15]:
        key = f"{r:.1f}-{h}"
        if key in seen: continue
        seen.add(key)
        p(f"    {r:.1f}% ({h}/300) mm={m} ROI={roi:+.1f}%")
        if len(seen) >= 5: break

# ===== 蒸馏最优 + 反miss =====
p(f"\n--- 蒸馏最优 + 反miss ---")
# 用best_fine的参数或best_s2
for name, s2_fn in [(best_s2[2], best_s2[1])]:
    if s2_fn is None: continue
    for bt in [2, 3]:
        for et in [3, 4, 5]:
            if et <= bt: continue
            h, r, m, roi, avg = test_distill_antimiss(s2_fn, bt, et)
            diff = r - 48.0  # vs v3+antimiss的48%
            p(f"  {name} blend@{bt} exp@{et}: {h}/300={r:.1f}% mm={m} avg={avg:.2f} ROI={roi:+.1f}% vs基线{diff:+.1f}%")

# ===== 最终对比 =====
p(f"\n{'='*100}")
p("最终对比")
p(f"{'='*100}")
p(f"  直接TOP4 v3 (无反miss):     {r1:.1f}% ({h1}/300) maxMiss={m1}")
p(f"  直接TOP4 v3+反miss:         48.0% (144/300) maxMiss=7  [之前验证]")
p(f"  蒸馏最优Stage2 (无反miss):  {best_fine[0]:.1f}% ({best_fine[2][0]}/300) maxMiss={best_fine[2][1]}")
best_s2_rate = best_s2[0]
p(f"  蒸馏最优预设策略:            {best_s2_rate:.1f}%")

diff_vs_direct = best_fine[0] - r1
diff_vs_antimiss = best_fine[0] - 48.0

p(f"\n  蒸馏 vs 直接v3(无反miss): {diff_vs_direct:+.1f}%")
p(f"  蒸馏 vs 直接v3+反miss:    {diff_vs_antimiss:+.1f}%")

if diff_vs_antimiss > 0:
    p(f"\n  ✅ 蒸馏方案能提升TOP4! 最优提升 {diff_vs_antimiss:+.1f}%")
elif diff_vs_direct > 0:
    p(f"\n  ⚠️ 蒸馏比无反miss版有提升({diff_vs_direct:+.1f}%), 但不如v3+反miss版")
else:
    p(f"\n  ❌ 蒸馏方案未能提升TOP4命中率")
    p(f"  原因分析:")
    p(f"    1. TOP9过滤掉了{t9m}/300={t9m/3:.1f}%的正确答案, 这是不可恢复的损失")
    p(f"    2. 蒸馏上限={100-t9m/3:.1f}%, 而v3直接TOP4已达{r1:.1f}%+反miss=48.0%")
    p(f"    3. 在9个候选中选4个(4/9=44.4%随机率)的区分度有限")
    p(f"    4. Stage1和Stage2的评分维度相似, 过滤后信息增益不足")

with open('distill_top9_to_top4.txt', 'w', encoding='utf-8-sig') as f:
    f.write(buf.getvalue())
p(f"\n结果已保存到 distill_top9_to_top4.txt")
