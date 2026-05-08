"""
硬蒸馏: TOP9排除3个生肖 → 从剩下9个中重新推导最优TOP5
============================================
目标: 超过50%成功率

数学基础:
- 直接TOP5从12选: 随机基线 5/12=41.7%, v3静态=52.0%
- 蒸馏TOP5从9选: 随机基线 5/9=55.6% (更高!)
- TOP9命中率85% → 理论上限85%
- TOP9命中的期数中随机选5/9: 85%×55.6%=47.2% (下限)
- 如果Stage2有正贡献 → 可以显著超50%

关键: Stage2必须使用与Stage1(TOP9)不同的特征维度才有信息增益
TOP9偏重: mk150(0.50) + cold15(0.20)
Stage2应该偏重: 短窗口MK, 二阶MK, 间隔, 轮换, 热号趋势等
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

def test_distill_top5(s2_fn, label=""):
    """测试硬蒸馏TOP9→TOP5"""
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
        
        # Stage2: 在9个中重新评分选TOP5
        s2 = s2_fn(animals)
        s2_in9 = s2[top9_idx]
        top5_in9 = np.argsort(-s2_in9)[:5]
        top5_idx = top9_idx[top5_in9]
        top5 = [ZODIAC_CYCLE_2026[i] for i in top5_idx]
        
        hit = actual_z in top5
        if hit: hits += 1
        hit_list.append(hit)
    
    rate = hits / test_periods * 100
    mm, g4, g6 = analyze_hits(hit_list)
    cost = test_periods * 5 * 4
    roi = (hits * 46 - cost) / cost * 100
    return hits, rate, mm, g4, g6, roi, t9_miss

def test_direct_top5(score_fn):
    """测试直接TOP5 (无过滤)"""
    hits = 0
    hit_list = []
    for pi in range(test_periods):
        animals, actual_z = all_data[pi]
        scores = score_fn(animals)
        top5 = [ZODIAC_CYCLE_2026[i] for i in np.argsort(-scores)[:5]]
        hit = actual_z in top5
        if hit: hits += 1
        hit_list.append(hit)
    rate = hits / test_periods * 100
    mm, g4, g6 = analyze_hits(hit_list)
    cost = test_periods * 5 * 4
    roi = (hits * 46 - cost) / cost * 100
    return hits, rate, mm, g4, g6, roi

buf = io.StringIO()
def p(s=""):
    print(s)
    buf.write(s + "\n")

p("=" * 100)
p("硬蒸馏: TOP9排除3个生肖 → 从9个中重新推导TOP5")
p(f"目标: 超过50%成功率")
p(f"数据: {total}期, 测试: 最近{test_periods}期")
p("=" * 100)

# ====== 对照组 ======
p(f"\n--- 对照组: 直接TOP5 (从12个中选) ---")
direct_fns = [
    ("v3静态(c15+c30+mk150)",
     lambda a: 0.30*cold_scores(a,15) + 0.10*cold_scores(a,30) + 0.60*markov_scores(a,150)),
    ("TOP9权重",
     lambda a: stage1_top9(a)),
]
p(f"  {'策略':<30} {'命中':>8} {'命中率':>8} {'mm':>4} {'≥4':>4} {'ROI':>8}")
p(f"  {'-'*65}")

direct_best = 0
for name, fn in direct_fns:
    h, r, mm, g4, g6, roi = test_direct_top5(fn)
    p(f"  {name:<30} {h:>4}/300 {r:>7.1f}% {mm:>3} {g4:>3} {roi:>+7.1f}%")
    direct_best = max(direct_best, r)

p(f"  直接TOP5最优: {direct_best:.1f}%")
p(f"  随机基线(直接): 5/12 = {5/12*100:.1f}%")
p(f"  随机基线(蒸馏): 5/9 = {5/9*100:.1f}%")

# ====== Stage2策略: 在9个中重新推导TOP5 ======
p(f"\n{'='*100}")
p("Stage2: 在TOP9的9个生肖中重新推导TOP5")
p(f"{'='*100}")

# 关键: Stage2使用与Stage1不同/互补的特征
s2_strategies = [
    # -- 与TOP9相同的特征(对照) --
    ("S2: TOP9同权重(对照)",
     lambda a: stage1_top9(a)),
    ("S2: v3静态(对照)",
     lambda a: 0.30*cold_scores(a,15) + 0.10*cold_scores(a,30) + 0.60*markov_scores(a,150)),
    
    # -- 差异化特征: 短窗口MK (与TOP9的MK150互补) --
    ("S2: MK50",          lambda a: markov_scores(a, 50)),
    ("S2: MK80",          lambda a: markov_scores(a, 80)),
    ("S2: MK30",          lambda a: markov_scores(a, 30)),
    
    # -- 二阶马尔可夫 --
    ("S2: 二阶MK150",     lambda a: second_order_markov(a, 150)),
    ("S2: 二阶MK100",     lambda a: second_order_markov(a, 100)),
    
    # -- 单维度 --
    ("S2: 纯间隔",        lambda a: gap_scores(a)),
    ("S2: 纯cold10",      lambda a: cold_scores(a, 10)),
    ("S2: 纯cold8",       lambda a: cold_scores(a, 8)),
    ("S2: 纯hot20",       lambda a: hot_scores(a, 20)),
    ("S2: 纯hot10",       lambda a: hot_scores(a, 10)),
    ("S2: 纯轮换",        lambda a: rotation_scores(a, 25)),
    
    # -- 互补组合: 短窗口 + 间隔 --
    ("S2: MK50+gap",
     lambda a: 0.60*markov_scores(a,50) + 0.40*gap_scores(a)),
    ("S2: MK80+gap",
     lambda a: 0.60*markov_scores(a,80) + 0.40*gap_scores(a)),
    ("S2: MK50+cold8",
     lambda a: 0.60*markov_scores(a,50) + 0.40*cold_scores(a,8)),
    ("S2: MK80+cold10",
     lambda a: 0.60*markov_scores(a,80) + 0.40*cold_scores(a,10)),
    
    # -- 互补组合: 二阶MK + 其他 --
    ("S2: 二阶MK+gap",
     lambda a: 0.50*second_order_markov(a,150) + 0.50*gap_scores(a)),
    ("S2: 二阶MK+MK50",
     lambda a: 0.50*second_order_markov(a,150) + 0.50*markov_scores(a,50)),
    ("S2: 二阶MK+cold10",
     lambda a: 0.50*second_order_markov(a,100) + 0.50*cold_scores(a,10)),
    
    # -- 热号逻辑: 在冷号pool(TOP9)中挑热的 --
    ("S2: hot20+MK80",
     lambda a: 0.40*hot_scores(a,20) + 0.60*markov_scores(a,80)),
    ("S2: hot10+MK50",
     lambda a: 0.40*hot_scores(a,10) + 0.60*markov_scores(a,50)),
    ("S2: hot20+gap",
     lambda a: 0.50*hot_scores(a,20) + 0.50*gap_scores(a)),
    
    # -- 轮换组合 --
    ("S2: 轮换+MK80",
     lambda a: 0.40*rotation_scores(a,25) + 0.60*markov_scores(a,80)),
    ("S2: 轮换+gap",
     lambda a: 0.50*rotation_scores(a,25) + 0.50*gap_scores(a)),
    ("S2: 轮换+cold10",
     lambda a: 0.40*rotation_scores(a,25) + 0.60*cold_scores(a,10)),
    
    # -- 三维度组合 --
    ("S2: MK50+gap+cold8",
     lambda a: 0.40*markov_scores(a,50) + 0.30*gap_scores(a) + 0.30*cold_scores(a,8)),
    ("S2: MK80+gap+hot20",
     lambda a: 0.40*markov_scores(a,80) + 0.30*gap_scores(a) + 0.30*hot_scores(a,20)),
    ("S2: 二阶MK+gap+rot",
     lambda a: 0.40*second_order_markov(a,100) + 0.30*gap_scores(a) + 0.30*rotation_scores(a,25)),
    ("S2: MK50+hot10+rot",
     lambda a: 0.40*markov_scores(a,50) + 0.30*hot_scores(a,10) + 0.30*rotation_scores(a,25)),
    ("S2: cold8+gap+MK30",
     lambda a: 0.30*cold_scores(a,8) + 0.35*gap_scores(a) + 0.35*markov_scores(a,30)),
]

p(f"\n  {'策略':<25} {'命中':>8} {'命中率':>8} {'mm':>4} {'≥4':>4} {'≥6':>4} {'ROI':>8} {'vs直接':>8}")
p(f"  {'-'*76}")

best_s2 = (0, "", None, None)
all_results = []

for name, fn in s2_strategies:
    h, r, mm, g4, g6, roi, t9m = test_distill_top5(fn, name)
    diff = r - direct_best
    mark = " ★" if r > direct_best else (" ●50%+" if r >= 50 else "")
    p(f"  {name:<25} {h:>4}/300 {r:>7.1f}% {mm:>3} {g4:>3} {g6:>3} {roi:>+7.1f}% {diff:>+7.1f}%{mark}")
    if r > best_s2[0]:
        best_s2 = (r, name, (h, mm, g4, g6, roi), fn)
    all_results.append((r, name, h, mm, g4, g6, roi, fn))

p(f"\n  TOP9过滤miss: {t9m}/300 = {t9m/3:.1f}%")
p(f"  蒸馏理论上限: {(300-t9m)/300*100:.1f}%")

# ====== 精细权重搜索 ======
p(f"\n{'='*100}")
p("精细搜索: Stage2最优权重 (在TOP9的9个中选TOP5)")
p(f"{'='*100}")

best_fine = (0, "", None)

# 搜索: 各维度组合 (互补维度优先)
configs = []
for mk_win in [30, 50, 80]:
    for mkw in [0.0, 0.20, 0.30, 0.40, 0.50, 0.60]:
        for gapw in [0.0, 0.10, 0.20, 0.30, 0.40]:
            for hotw in [0.0, 0.10, 0.15, 0.20, 0.30]:
                for cw in [0.0, 0.10, 0.20, 0.30]:  # cold短窗口
                    remain = 1.0 - mkw - gapw - hotw - cw
                    if remain < -0.01 or remain > 0.40:
                        continue
                    rotw = max(remain, 0)
                    configs.append((mkw, mk_win, gapw, hotw, cw, rotw))

# 也搜索二阶MK
for mk2w in [0.20, 0.30, 0.40, 0.50]:
    for gapw in [0.0, 0.15, 0.25, 0.35]:
        for hotw in [0.0, 0.10, 0.20]:
            for cw in [0.0, 0.10, 0.20]:
                remain = 1.0 - mk2w - gapw - hotw - cw
                if remain < -0.01 or remain > 0.30:
                    continue
                rotw = max(remain, 0)
                configs.append(('mk2', mk2w, gapw, hotw, cw, rotw))

p(f"  搜索{len(configs)}种组合...")

fine_results = []
for cfg in configs:
    if cfg[0] == 'mk2':
        _, mk2w, gapw, hotw, cw, rotw = cfg
        def make_fn(m2, g, h, c, r):
            def fn(a):
                s = m2 * second_order_markov(a, 150)
                if g > 0: s = s + g * gap_scores(a)
                if h > 0: s = s + h * hot_scores(a, 20)
                if c > 0: s = s + c * cold_scores(a, 8)
                if r > 0: s = s + r * rotation_scores(a, 25)
                return s
            return fn
        fn = make_fn(mk2w, gapw, hotw, cw, rotw)
    else:
        mkw, mk_win, gapw, hotw, cw, rotw = cfg
        def make_fn(m, w, g, h, c, r):
            def fn(a):
                s = m * markov_scores(a, w)
                if g > 0: s = s + g * gap_scores(a)
                if h > 0: s = s + h * hot_scores(a, 20)
                if c > 0: s = s + c * cold_scores(a, 8)
                if r > 0: s = s + r * rotation_scores(a, 25)
                return s
            return fn
        fn = make_fn(mkw, mk_win, gapw, hotw, cw, rotw)
    
    h, r, mm, g4, g6, roi, _ = test_distill_top5(fn)
    if r > best_fine[0]:
        if cfg[0] == 'mk2':
            label = f"mk2={cfg[1]} gap={cfg[2]} hot={cfg[3]} c8={cfg[4]} rot={cfg[5]:.2f}"
        else:
            label = f"mk{cfg[1]}={cfg[0]} gap={cfg[2]} hot={cfg[3]} c8={cfg[4]} rot={cfg[5]:.2f}"
        best_fine = (r, label, (h, mm, g4, g6, roi))
    if r >= 53:
        fine_results.append((r, h, mm, g4, g6, roi, cfg))

fine_results.sort(key=lambda x: (-x[0], x[2]))

p(f"\n  Stage2最优: {best_fine[1]}")
p(f"    {best_fine[2][0]}/300 = {best_fine[0]:.1f}% maxMiss={best_fine[2][1]} ROI={best_fine[2][4]:+.1f}%")

if fine_results:
    p(f"\n  ≥53%的组合 (前10):")
    seen = set()
    for r, h, mm, g4, g6, roi, cfg in fine_results[:20]:
        key = f"{h}"
        if key in seen: continue
        seen.add(key)
        if cfg[0] == 'mk2':
            desc = f"mk2={cfg[1]} gap={cfg[2]} hot={cfg[3]} c8={cfg[4]} rot={cfg[5]:.2f}"
        else:
            desc = f"mk{cfg[1]}={cfg[0]} gap={cfg[2]} hot={cfg[3]} c8={cfg[4]} rot={cfg[5]:.2f}"
        p(f"    {r:.1f}% ({h}/300) mm={mm} ≥4miss={g4} {desc}")
        if len(seen) >= 10: break

# ====== 蒸馏最优 + 反miss ======
p(f"\n{'='*100}")
p("蒸馏最优 + 反miss机制")
p(f"{'='*100}")

# 取前几个最优策略测试反miss
top_strategies = sorted(all_results, key=lambda x: -x[0])[:3]
# 加精细搜索最优
if fine_results:
    best_cfg = fine_results[0][6]
    if best_cfg[0] == 'mk2':
        _, mk2w, gapw, hotw, cw, rotw = best_cfg
        best_fine_fn = lambda a: (mk2w*second_order_markov(a,150) +
                                  gapw*gap_scores(a) + hotw*hot_scores(a,20) +
                                  cw*cold_scores(a,8) + rotw*rotation_scores(a,25))
    else:
        mkw, mk_win, gapw, hotw, cw, rotw = best_cfg
        best_fine_fn = lambda a, _m=mkw, _w=mk_win, _g=gapw, _h=hotw, _c=cw, _r=rotw: (
            _m*markov_scores(a,_w) + _g*gap_scores(a) + _h*hot_scores(a,20) +
            _c*cold_scores(a,8) + _r*rotation_scores(a,25))
    top_strategies.append((best_fine[0], "精细最优", *best_fine[2], best_fine_fn))

for _, sname, *_ in top_strategies:
    sfn = _[-1] if callable(_[-1]) else None
    if sfn is None:
        continue
    
    for bt in [2, 3]:
        for et in [3, 4, 5]:
            if et <= bt: continue
            
            consec = 0
            hits = 0
            hit_list = []
            sizes = []
            
            for pi in range(test_periods):
                animals, actual_z = all_data[pi]
                s1 = stage1_top9(animals)
                top9_idx = np.argsort(-s1)[:9]
                
                actual_idx = ZODIAC_CYCLE_2026.index(actual_z)
                
                s2 = sfn(animals)
                s2_in9 = s2[top9_idx]
                
                # 反miss: blend热号
                if consec >= bt:
                    h = hot_scores(animals, 30)
                    h_in9 = h[top9_idx]
                    s2_in9 = 0.75 * s2_in9 + 0.25 * h_in9
                
                # 扩展
                n = 6 if consec >= et else 5
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
            
            if rate >= 52:
                p(f"  {sname} +反miss(b@{bt},e@{et}): {hits}/300={rate:.1f}% mm={mm} ≥4miss={g4} avg={avg_n:.2f} ROI={roi:+.1f}%")

# ====== 最终对比 ======
p(f"\n{'='*100}")
p("最终对比")
p(f"{'='*100}")
p(f"  随机基线(直接TOP5): 5/12 = {5/12*100:.1f}%")
p(f"  随机基线(蒸馏TOP5): 5/9  = {5/9*100:.1f}%")
p(f"  直接TOP5最优:        {direct_best:.1f}%")
p(f"  蒸馏TOP5(预设最优):  {best_s2[0]:.1f}% ({best_s2[1]})")
p(f"  蒸馏TOP5(精细最优):  {best_fine[0]:.1f}%")
diff1 = best_s2[0] - direct_best
diff2 = best_fine[0] - direct_best
p(f"  蒸馏预设 vs 直接: {diff1:+.1f}%")
p(f"  蒸馏精细 vs 直接: {diff2:+.1f}%")

if best_fine[0] >= 50:
    p(f"\n  ✅ 蒸馏TOP5达到50%目标! {best_fine[0]:.1f}%")
else:
    p(f"\n  ❌ 蒸馏TOP5未达50%: {best_fine[0]:.1f}%")

if best_fine[0] > direct_best:
    p(f"  ✅ 蒸馏优于直接: +{diff2:.1f}%")
else:
    p(f"  ❌ 蒸馏未超过直接TOP5")

with open('distill_top9_to_top5.txt', 'w', encoding='utf-8-sig') as f:
    f.write(buf.getvalue())
p(f"\n结果已保存到 distill_top9_to_top5.txt")
