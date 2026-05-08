"""
硬蒸馏: TOP9→TOP4 (优化版, 减少搜索量)
目标: 超过50%成功率
方案A~H, 8种方案全面测试
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import io
import pandas as pd
import numpy as np
from collections import Counter

ZODIAC_CYCLE_2026 = ['马','蛇','龙','兔','虎','牛','鼠','猪','狗','鸡','猴','羊']
NUM_TO_ZODIAC_2026 = {n: ZODIAC_CYCLE_2026[(n-1)%12] for n in range(1,50)}

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].tolist()
total = len(numbers)
test_periods = 300
start = total - test_periods

def cold_scores(animals, window):
    freq = Counter(animals[-min(window,len(animals)):])
    mx = max(freq.values()) if freq else 1
    return np.array([1.0 - freq.get(z,0)/max(mx,1) for z in ZODIAC_CYCLE_2026])

def hot_scores(animals, window):
    freq = Counter(animals[-min(window,len(animals)):])
    mx = max(freq.values()) if freq else 1
    return np.array([freq.get(z,0)/max(mx,1) for z in ZODIAC_CYCLE_2026])

def gap_scores(animals):
    scores = []
    for z in ZODIAC_CYCLE_2026:
        last = -1
        for j in range(len(animals)-1, -1, -1):
            if animals[j] == z: last = j; break
        gap = (len(animals)-1-last) if last >= 0 else len(animals)
        scores.append(gap / 12.0)
    return np.array(scores)

def markov_scores(animals, window=None, laplace=1.0):
    probs = np.ones(12)/12
    h = animals[-window:] if window and len(animals)>window else animals
    if len(h) < 2: return probs
    trans = {}
    for k in range(1, len(h)):
        p, c = h[k-1], h[k]
        if p not in trans: trans[p] = Counter()
        trans[p][c] += 1
    state = animals[-1]
    if state in trans:
        t = sum(trans[state].values()) + laplace*12
        for zi, z in enumerate(ZODIAC_CYCLE_2026):
            probs[zi] = (trans[state].get(z,0)+laplace)/t
    return probs

def second_order_markov(animals, window=150, laplace=0.5):
    probs = np.ones(12)/12
    h = animals[-window:] if window and len(animals)>window else animals
    if len(h) < 3: return probs
    trans = {}
    for k in range(2, len(h)):
        key = (h[k-2], h[k-1])
        if key not in trans: trans[key] = Counter()
        trans[key][h[k]] += 1
    state = (h[-2], h[-1])
    if state in trans:
        t = sum(trans[state].values()) + laplace*12
        for zi, z in enumerate(ZODIAC_CYCLE_2026):
            probs[zi] = (trans[state].get(z,0)+laplace)/t
    return probs

def rotation_scores(animals, window=25):
    scores = []
    for z in ZODIAC_CYCLE_2026:
        positions = [j for j, a in enumerate(animals[-window:]) if a == z]
        if len(positions) >= 2:
            gaps_l = [positions[k+1]-positions[k] for k in range(len(positions)-1)]
            avg_gap = np.mean(gaps_l)
            last_gap = window - positions[-1]
            score = max(0, 1.0 - abs(last_gap-avg_gap)/max(avg_gap,1))
        else:
            score = 0.5
        scores.append(score)
    return np.array(scores)

def stage1_top9(animals):
    return (0.20*cold_scores(animals,15) + 0.05*cold_scores(animals,30) +
            0.50*markov_scores(animals,150) + 0.10*gap_scores(animals) +
            0.15*hot_scores(animals,30))

# 预计算
print("预计算数据...")
all_data = []
for pi in range(test_periods):
    i = start + pi
    hist_animals = [NUM_TO_ZODIAC_2026[n] for n in numbers[:i]]
    actual_z = NUM_TO_ZODIAC_2026[numbers[i]]
    all_data.append((hist_animals, actual_z))

def analyze_hits(hl):
    streaks = []; c = 0
    for h in hl:
        if not h: c += 1
        else:
            if c > 0: streaks.append(c)
            c = 0
    if c > 0: streaks.append(c)
    mm = max(streaks) if streaks else 0
    g4 = sum(1 for s in streaks if s >= 4)
    return mm, g4

def test_distill(s2_fn, top_n=4):
    hits = 0; hit_list = []; t9m = 0
    for pi in range(test_periods):
        animals, actual_z = all_data[pi]
        s1 = stage1_top9(animals)
        top9_idx = np.argsort(-s1)[:9]
        actual_idx = ZODIAC_CYCLE_2026.index(actual_z)
        if actual_idx not in set(top9_idx):
            t9m += 1; hit_list.append(False); continue
        s2 = s2_fn(animals)
        s2_in9 = s2[top9_idx]
        top_in9 = np.argsort(-s2_in9)[:top_n]
        top_idx = top9_idx[top_in9]
        hit = actual_z in [ZODIAC_CYCLE_2026[i] for i in top_idx]
        if hit: hits += 1
        hit_list.append(hit)
    r = hits/test_periods*100
    mm, g4 = analyze_hits(hit_list)
    cost = test_periods*top_n*4
    roi = (hits*46-cost)/cost*100
    return hits, r, mm, g4, roi, t9m

def test_direct(score_fn, top_n=4):
    hits = 0; hl = []
    for pi in range(test_periods):
        animals, actual_z = all_data[pi]
        scores = score_fn(animals)
        top = [ZODIAC_CYCLE_2026[i] for i in np.argsort(-scores)[:top_n]]
        hit = actual_z in top
        if hit: hits += 1
        hl.append(hit)
    r = hits/test_periods*100
    mm, g4 = analyze_hits(hl)
    cost = test_periods*top_n*4
    roi = (hits*46-cost)/cost*100
    return hits, r, mm, g4, roi

def test_antimiss(s2_fn, bt, et, en, hb=0.25):
    consec = 0; hits = 0; hl = []; sizes = []
    for pi in range(test_periods):
        animals, actual_z = all_data[pi]
        s1 = stage1_top9(animals)
        top9_idx = np.argsort(-s1)[:9]
        s2 = s2_fn(animals)
        s2_in9 = s2[top9_idx]
        if consec >= bt:
            h_in9 = hot_scores(animals,30)[top9_idx]
            s2_in9 = (1-hb)*s2_in9 + hb*h_in9
        n = en if consec >= et else 4
        top_in9 = np.argsort(-s2_in9)[:n]
        top_idx = top9_idx[top_in9]
        hit = actual_z in [ZODIAC_CYCLE_2026[i] for i in top_idx]
        if hit: hits += 1; consec = 0
        else: consec += 1
        hl.append(hit); sizes.append(n)
    r = hits/test_periods*100
    mm, g4 = analyze_hits(hl)
    avg_n = np.mean(sizes)
    cost = sum(s*4 for s in sizes)
    roi = (hits*46-cost)/cost*100
    return hits, r, mm, g4, roi, avg_n

def test_multilevel(s2_fn, levels):
    """levels = [(miss_th, n, hot_blend), ...]"""
    consec = 0; hits = 0; hl = []; sizes = []
    for pi in range(test_periods):
        animals, actual_z = all_data[pi]
        s1 = stage1_top9(animals)
        top9_idx = np.argsort(-s1)[:9]
        s2 = s2_fn(animals)
        s2_in9 = s2[top9_idx]
        n = 4; hb = 0
        for miss_th, exp_n, h_blend in levels:
            if consec >= miss_th: n = exp_n; hb = h_blend
        if hb > 0:
            h_in9 = hot_scores(animals,30)[top9_idx]
            s2_in9 = (1-hb)*s2_in9 + hb*h_in9
        top_in9 = np.argsort(-s2_in9)[:n]
        top_idx = top9_idx[top_in9]
        hit = actual_z in [ZODIAC_CYCLE_2026[i] for i in top_idx]
        if hit: hits += 1; consec = 0
        else: consec += 1
        hl.append(hit); sizes.append(n)
    r = hits/test_periods*100
    mm, g4 = analyze_hits(hl)
    avg_n = np.mean(sizes)
    cost = sum(s*4 for s in sizes)
    roi = (hits*46-cost)/cost*100
    return hits, r, mm, g4, roi, avg_n

def test_adaptive(s2_fns_by_state):
    consec = 0; hits = 0; hl = []; sizes = []
    sorted_keys = sorted(s2_fns_by_state.keys())
    for pi in range(test_periods):
        animals, actual_z = all_data[pi]
        s1 = stage1_top9(animals)
        top9_idx = np.argsort(-s1)[:9]
        # 找最大的 key <= consec
        sk = sorted_keys[0]
        for k in sorted_keys:
            if k <= consec: sk = k
            else: break
        s2_fn, n = s2_fns_by_state[sk]
        s2 = s2_fn(animals)
        s2_in9 = s2[top9_idx]
        top_in9 = np.argsort(-s2_in9)[:n]
        top_idx = top9_idx[top_in9]
        hit = actual_z in [ZODIAC_CYCLE_2026[i] for i in top_idx]
        if hit: hits += 1; consec = 0
        else: consec += 1
        hl.append(hit); sizes.append(n)
    r = hits/test_periods*100
    mm, g4 = analyze_hits(hl)
    avg_n = np.mean(sizes)
    cost = sum(s*4 for s in sizes)
    roi = (hits*46-cost)/cost*100
    return hits, r, mm, g4, roi, avg_n

def test_voting(s2_fn_list, vote_thr=2):
    hits = 0; hl = []; sizes = []
    for pi in range(test_periods):
        animals, actual_z = all_data[pi]
        s1 = stage1_top9(animals)
        top9_idx = np.argsort(-s1)[:9]
        votes = Counter()
        for s2_fn in s2_fn_list:
            s2 = s2_fn(animals)
            s2_in9 = s2[top9_idx]
            for idx in top9_idx[np.argsort(-s2_in9)[:4]]:
                votes[idx] += 1
        sel = [idx for idx, v in votes.most_common() if v >= vote_thr]
        if len(sel) < 4:
            for idx, _ in votes.most_common():
                if idx not in sel: sel.append(idx)
                if len(sel) >= 4: break
        n = len(sel)
        hit = actual_z in [ZODIAC_CYCLE_2026[i] for i in sel]
        if hit: hits += 1
        hl.append(hit); sizes.append(n)
    r = hits/test_periods*100
    mm, g4 = analyze_hits(hl)
    avg_n = np.mean(sizes)
    cost = sum(s*4 for s in sizes)
    roi = (hits*46-cost)/cost*100
    return hits, r, mm, g4, roi, avg_n

def test_confidence_split(s2_fn_h, s2_fn_l, thr=0.10):
    hits = 0; hl = []; sizes = []
    for pi in range(test_periods):
        animals, actual_z = all_data[pi]
        s1 = stage1_top9(animals)
        sorted_s1 = np.sort(s1)[::-1]
        conf = sorted_s1[8] - sorted_s1[9] if len(sorted_s1)>9 else 0
        top9_idx = np.argsort(-s1)[:9]
        if conf >= thr:
            s2 = s2_fn_h(animals); n = 4
        else:
            s2 = s2_fn_l(animals); n = 5
        s2_in9 = s2[top9_idx]
        top_in9 = np.argsort(-s2_in9)[:n]
        top_idx = top9_idx[top_in9]
        hit = actual_z in [ZODIAC_CYCLE_2026[i] for i in top_idx]
        if hit: hits += 1
        hl.append(hit); sizes.append(n)
    r = hits/test_periods*100
    mm, g4 = analyze_hits(hl)
    avg_n = np.mean(sizes)
    cost = sum(s*4 for s in sizes)
    roi = (hits*46-cost)/cost*100
    return hits, r, mm, g4, roi, avg_n

# 常用Stage2
s2_v3 = lambda a: 0.30*cold_scores(a,15) + 0.10*cold_scores(a,30) + 0.60*markov_scores(a,150)
s2_mk50 = lambda a: markov_scores(a,50)
s2_mk80 = lambda a: markov_scores(a,80)
s2_gap = lambda a: gap_scores(a)
s2_mk2 = lambda a: second_order_markov(a,150)
s2_rot_gap = lambda a: 0.5*rotation_scores(a,25)+0.5*gap_scores(a)
s2_hot_gap = lambda a: 0.5*hot_scores(a,20)+0.5*gap_scores(a)

buf = io.StringIO()
def p(s=""):
    print(s); buf.write(s+"\n")

p("="*100)
p("硬蒸馏: TOP9→TOP4 (8种方案)")
p(f"数据: {total}期, 测试: {test_periods}期, 目标: >50%")
p("="*100)

# === 对照组 ===
p(f"\n--- 对照组 ---")
h1, r1, mm1, g41, roi1 = test_direct(s2_v3, 4)
h2, r2, mm2, g42, roi2 = test_direct(lambda a: stage1_top9(a), 4)
p(f"  直接TOP4 v3静态:  {h1}/300 = {r1:.1f}% mm={mm1} ≥4miss={g41} ROI={roi1:+.1f}%")
p(f"  直接TOP4 TOP9权重: {h2}/300 = {r2:.1f}% mm={mm2} ≥4miss={g42} ROI={roi2:+.1f}%")
direct_best = max(r1, r2)
p(f"  直接最优: {direct_best:.1f}%  |  随机: 直接4/12={4/12*100:.1f}%, 蒸馏4/9={4/9*100:.1f}%")

# === 方案A: 纯静态蒸馏 ===
p(f"\n{'='*100}")
p("方案A: 纯静态蒸馏 (在TOP9的9个中选4, 无反miss)")
p(f"{'='*100}")

s2_list = [
    ("TOP9同权重",   lambda a: stage1_top9(a)),
    ("v3静态",       s2_v3),
    ("MK30",         lambda a: markov_scores(a,30)),
    ("MK50",         s2_mk50),
    ("MK80",         s2_mk80),
    ("MK100",        lambda a: markov_scores(a,100)),
    ("二阶MK150",    s2_mk2),
    ("二阶MK80",     lambda a: second_order_markov(a,80)),
    ("纯间隔",       s2_gap),
    ("纯cold8",      lambda a: cold_scores(a,8)),
    ("纯cold10",     lambda a: cold_scores(a,10)),
    ("纯hot20",      lambda a: hot_scores(a,20)),
    ("纯轮换",       lambda a: rotation_scores(a,25)),
    ("MK50+gap",     lambda a: 0.6*markov_scores(a,50)+0.4*gap_scores(a)),
    ("MK80+gap",     lambda a: 0.6*markov_scores(a,80)+0.4*gap_scores(a)),
    ("轮换+gap",     s2_rot_gap),
    ("轮换+MK80",    lambda a: 0.4*rotation_scores(a,25)+0.6*markov_scores(a,80)),
    ("二阶MK+gap+rot", lambda a: 0.4*second_order_markov(a,100)+0.3*gap_scores(a)+0.3*rotation_scores(a,25)),
    ("v3+gap",       lambda a: 0.7*s2_v3(a)+0.3*gap_scores(a)),
    ("v3+rot",       lambda a: 0.7*s2_v3(a)+0.3*rotation_scores(a,25)),
]

p(f"  {'策略':<20} {'命中':>8} {'命中率':>8} {'mm':>4} {'≥4':>4} {'ROI':>8} {'vs直接':>8}")
p(f"  {'-'*65}")

best_A = (0, "", None, None)
for name, fn in s2_list:
    h, r, mm, g4, roi, t9m = test_distill(fn)
    diff = r - direct_best
    mark = " ★" if r > direct_best else ""
    p(f"  {name:<20} {h:>4}/300 {r:>7.1f}% {mm:>3} {g4:>3} {roi:>+7.1f}% {diff:>+7.1f}%{mark}")
    if r > best_A[0]: best_A = (r, name, (h,mm,g4,roi), fn)

p(f"\n  方案A最优: {best_A[1]} = {best_A[0]:.1f}% (TOP9 miss={t9m}/300)")

# === 方案B: 精细权重搜索 ===
p(f"\n{'='*100}")
p("方案B: 精细权重搜索")
p(f"{'='*100}")

best_B = (0, "", None, None)
count_B = 0
for mk_win in [50, 80, 100, 150]:
    for mkw in np.arange(0.2, 0.75, 0.10):
        for gapw in np.arange(0, 0.45, 0.10):
            for cw in np.arange(0, 0.35, 0.10):
                rotw = round(1.0 - mkw - gapw - cw, 2)
                if rotw < -0.01 or rotw > 0.40: continue
                rotw = max(rotw, 0)
                count_B += 1
                def make_fn(m, w, g, c, r):
                    return lambda a: m*markov_scores(a,w)+g*gap_scores(a)+c*cold_scores(a,8)+r*rotation_scores(a,25)
                fn = make_fn(mkw, mk_win, gapw, cw, rotw)
                h, r, mm, g4, roi, _ = test_distill(fn)
                if r > best_B[0]:
                    best_B = (r, f"mk{mk_win}={mkw:.1f} g={gapw:.1f} c8={cw:.1f} rot={rotw:.2f}", (h,mm,g4,roi), fn)

# 也搜索v3基础 + 补充
for v3w in np.arange(0.5, 0.9, 0.10):
    for gw in np.arange(0, 0.35, 0.10):
        rw = round(1.0 - v3w - gw, 2)
        if rw < -0.01 or rw > 0.35: continue
        rw = max(rw, 0)
        count_B += 1
        def make_fn_v3(vw, g, r):
            return lambda a: vw*s2_v3(a)+g*gap_scores(a)+r*rotation_scores(a,25)
        fn = make_fn_v3(v3w, gw, rw)
        h, r, mm, g4, roi, _ = test_distill(fn)
        if r > best_B[0]:
            best_B = (r, f"v3={v3w:.1f} gap={gw:.1f} rot={rw:.2f}", (h,mm,g4,roi), fn)

p(f"  搜索了{count_B}种组合")
p(f"  最优: {best_B[1]}")
p(f"    {best_B[2][0]}/300 = {best_B[0]:.1f}% mm={best_B[2][1]} ≥4miss={best_B[2][2]} ROI={best_B[2][3]:+.1f}%")

# === 方案C: 蒸馏+反miss ===
p(f"\n{'='*100}")
p("方案C: 蒸馏 + 反miss (blend热号 + 扩展)")
p(f"{'='*100}")
p(f"  {'策略':<45} {'命中':>8} {'率':>7} {'mm':>4} {'avg':>6} {'ROI':>8}")
p(f"  {'-'*80}")

best_C = (0, "", None)
base_C = [("v3", s2_v3), ("轮换+gap", s2_rot_gap)]
if best_B[3]: base_C.append(("精细B", best_B[3]))

for bname, bfn in base_C:
    for bt in [1, 2, 3]:
        for et in [2, 3, 4, 5]:
            if et <= bt: continue
            for en in [5, 6, 7]:
                for hb in [0.20, 0.30]:
                    h, r, mm, g4, roi, avg_n = test_antimiss(bfn, bt, et, en, hb)
                    label = f"{bname}+反miss(b@{bt}h{hb},e@{et}→{en})"
                    if r >= 48:
                        p(f"  {label:<45} {h:>4}/300 {r:>6.1f}% {mm:>3} {avg_n:>5.2f} {roi:>+7.1f}%")
                    if r > best_C[0]: best_C = (r, label, (h,mm,g4,roi,avg_n))

p(f"\n  方案C最优: {best_C[0]:.1f}% ({best_C[1]})")

# === 方案D: 自适应策略切换 ===
p(f"\n{'='*100}")
p("方案D: 自适应策略切换 (根据miss数切换Stage2+扩展)")
p(f"{'='*100}")
p(f"  {'策略':<60} {'命中':>8} {'率':>7} {'mm':>4} {'avg':>6} {'ROI':>8}")
p(f"  {'-'*95}")

best_D = (0, "", None)
adaptive_cfgs = [
    ("miss0→v3(4), miss1→v3(4), miss2→mk50(5), miss3+→gap(6)",
     {0:(s2_v3,4), 1:(s2_v3,4), 2:(s2_mk50,5), 3:(s2_gap,6)}),
    ("miss0→v3(4), miss1→mk80(4), miss2→gap(5), miss3+→hot+gap(6)",
     {0:(s2_v3,4), 1:(s2_mk80,4), 2:(s2_gap,5), 3:(s2_hot_gap,6)}),
    ("miss0→v3(4), miss2→rot+gap(5), miss4+→gap(6)",
     {0:(s2_v3,4), 2:(s2_rot_gap,5), 4:(s2_gap,6)}),
    ("miss0→v3(4), miss1→v3(5), miss3+→gap(6)",
     {0:(s2_v3,4), 1:(s2_v3,5), 3:(s2_gap,6)}),
    ("miss0→v3(4), miss2→v3(5), miss3+→v3(6)",
     {0:(s2_v3,4), 2:(s2_v3,5), 3:(s2_v3,6)}),
    ("miss0→v3(4), miss1→v3(5), miss2+→v3(6)",
     {0:(s2_v3,4), 1:(s2_v3,5), 2:(s2_v3,6)}),
    ("miss0→v3(4), miss2→mk80(5), miss4→hot+gap(6), miss6+→gap(7)",
     {0:(s2_v3,4), 2:(s2_mk80,5), 4:(s2_hot_gap,6), 6:(s2_gap,7)}),
    ("miss0→v3(4), miss1→v3(5), miss3→v3(7), miss5+→v3(9)",
     {0:(s2_v3,4), 1:(s2_v3,5), 3:(s2_v3,7), 5:(s2_v3,9)}),
]

for name, sm in adaptive_cfgs:
    h, r, mm, g4, roi, avg_n = test_adaptive(sm)
    p(f"  {name:<60} {h:>4}/300 {r:>6.1f}% {mm:>3} {avg_n:>5.2f} {roi:>+7.1f}%")
    if r > best_D[0]: best_D = (r, name, (h,mm,g4,roi,avg_n))

p(f"\n  方案D最优: {best_D[0]:.1f}%")

# === 方案E: TOP9置信度分层 ===
p(f"\n{'='*100}")
p("方案E: TOP9置信度分层 (高置信→4, 低置信→5)")
p(f"{'='*100}")
p(f"  {'策略':<45} {'命中':>8} {'率':>7} {'mm':>4} {'avg':>6} {'ROI':>8}")
p(f"  {'-'*80}")

best_E = (0, "", None)
for thr in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    for n_h, fn_h in [("v3",s2_v3), ("mk80",s2_mk80)]:
        for n_l, fn_l in [("v3",s2_v3), ("gap",s2_gap), ("rot+gap",s2_rot_gap), ("mk50",s2_mk50)]:
            h, r, mm, g4, roi, avg_n = test_confidence_split(fn_h, fn_l, thr)
            label = f"thr={thr} 高:{n_h}/低:{n_l}"
            if r >= 48:
                p(f"  {label:<45} {h:>4}/300 {r:>6.1f}% {mm:>3} {avg_n:>5.2f} {roi:>+7.1f}%")
            if r > best_E[0]: best_E = (r, label, (h,mm,g4,roi,avg_n))

p(f"\n  方案E最优: {best_E[0]:.1f}% ({best_E[1]})")

# === 方案F: 多策略投票 ===
p(f"\n{'='*100}")
p("方案F: 多策略投票法")
p(f"{'='*100}")
p(f"  {'策略':<45} {'命中':>8} {'率':>7} {'mm':>4} {'avg':>6} {'ROI':>8}")
p(f"  {'-'*80}")

best_F = (0, "", None)
vote_cfgs = [
    ("v3+mk50+gap (≥2票)", [s2_v3,s2_mk50,s2_gap], 2),
    ("v3+mk80+gap (≥2票)", [s2_v3,s2_mk80,s2_gap], 2),
    ("v3+mk50+mk2 (≥2票)", [s2_v3,s2_mk50,s2_mk2], 2),
    ("v3+mk80+gap+rot (≥2票)", [s2_v3,s2_mk80,s2_gap,s2_rot_gap], 2),
    ("v3+mk80+gap+rot (≥3票)", [s2_v3,s2_mk80,s2_gap,s2_rot_gap], 3),
    ("v3+mk50+mk2+gap (≥2票)", [s2_v3,s2_mk50,s2_mk2,s2_gap], 2),
    ("v3+mk50+mk2+gap (≥3票)", [s2_v3,s2_mk50,s2_mk2,s2_gap], 3),
    ("v3+mk80+mk2+rot+gap(≥3)", [s2_v3,s2_mk80,s2_mk2,s2_rot_gap,s2_gap], 3),
]
for name, fns, vt in vote_cfgs:
    h, r, mm, g4, roi, avg_n = test_voting(fns, vt)
    p(f"  {name:<45} {h:>4}/300 {r:>6.1f}% {mm:>3} {avg_n:>5.2f} {roi:>+7.1f}%")
    if r > best_F[0]: best_F = (r, name, (h,mm,g4,roi,avg_n))

p(f"\n  方案F最优: {best_F[0]:.1f}% ({best_F[1]})")

# === 方案G: 激进多级反miss ===
p(f"\n{'='*100}")
p("方案G: 激进多级反miss (快速扩展)")
p(f"{'='*100}")
p(f"  {'策略':<55} {'命中':>8} {'率':>7} {'mm':>4} {'avg':>6} {'ROI':>8}")
p(f"  {'-'*90}")

best_G = (0, "", None)
levels_cfgs = [
    ("miss1→5, miss3→6, miss5→7",          [(1,5,0), (3,6,0.2), (5,7,0.3)]),
    ("miss1→5, miss2→6, miss4→7",          [(1,5,0), (2,6,0.2), (4,7,0.3)]),
    ("miss1→5, miss3→7, miss5→9",          [(1,5,0), (3,7,0.2), (5,9,0.3)]),
    ("miss2→5, miss4→6, miss6→8",          [(2,5,0.1), (4,6,0.2), (6,8,0.3)]),
    ("miss1→5h0.2, miss2→6h0.3",          [(1,5,0.2), (2,6,0.3)]),
    ("miss1→5h0.3, miss3→7h0.4",          [(1,5,0.3), (3,7,0.4)]),
    ("miss2→5, miss3→6, miss4→7, miss6→9", [(2,5,0),(3,6,0.15),(4,7,0.25),(6,9,0.35)]),
    ("miss1→5, miss2→6, miss3→7, miss4+→9",[(1,5,0),(2,6,0.1),(3,7,0.2),(4,9,0.3)]),
    # 更激进
    ("miss1→6, miss2→7, miss3+→9",         [(1,6,0.1),(2,7,0.2),(3,9,0.3)]),
    ("miss1→5, miss2→7, miss3+→9",         [(1,5,0),(2,7,0.2),(3,9,0.3)]),
]

for name, levels in levels_cfgs:
    h, r, mm, g4, roi, avg_n = test_multilevel(s2_v3, levels)
    label = f"v3+{name}"
    p(f"  {label:<55} {h:>4}/300 {r:>6.1f}% {mm:>3} {avg_n:>5.2f} {roi:>+7.1f}%")
    if r > best_G[0]: best_G = (r, label, (h,mm,g4,roi,avg_n))

p(f"\n  方案G最优: {best_G[0]:.1f}%")

# === 方案H: 动态窗口 ===
p(f"\n{'='*100}")
p("方案H: 动态窗口 (近期命中率调整MK窗口) + 反miss扩展")
p(f"{'='*100}")

def test_dynwin(lb=20, sw=50, lw=150, thr=0.45):
    consec = 0; hits = 0; hl = []; sizes = []; recent_h = []
    for pi in range(test_periods):
        animals, actual_z = all_data[pi]
        s1 = stage1_top9(animals)
        top9_idx = np.argsort(-s1)[:9]
        rr = sum(recent_h[-lb:])/lb if len(recent_h)>=lb else 0.5
        mk_win = lw if rr >= thr else sw
        s2 = 0.30*cold_scores(animals,15)+0.10*cold_scores(animals,30)+0.60*markov_scores(animals,mk_win)
        n = 4
        if consec >= 3: n = 5
        if consec >= 5: n = 6
        s2_in9 = s2[top9_idx]
        top_in9 = np.argsort(-s2_in9)[:n]
        top_idx = top9_idx[top_in9]
        hit = actual_z in [ZODIAC_CYCLE_2026[i] for i in top_idx]
        if hit: hits += 1; consec = 0
        else: consec += 1
        hl.append(hit); recent_h.append(1 if hit else 0); sizes.append(n)
    r = hits/test_periods*100
    mm, g4 = analyze_hits(hl)
    avg_n = np.mean(sizes)
    cost = sum(s*4 for s in sizes)
    roi = (hits*46-cost)/cost*100
    return hits, r, mm, g4, roi, avg_n

p(f"  {'策略':<45} {'命中':>8} {'率':>7} {'mm':>4} {'avg':>6} {'ROI':>8}")
p(f"  {'-'*80}")

best_H = (0, "", None)
for lb in [10, 15, 20, 30]:
    for sw in [30, 50, 80]:
        for lw in [100, 150]:
            for thr in [0.35, 0.40, 0.45, 0.50]:
                h, r, mm, g4, roi, avg_n = test_dynwin(lb, sw, lw, thr)
                label = f"lb={lb} short={sw} long={lw} thr={thr}"
                if r >= 48:
                    p(f"  {label:<45} {h:>4}/300 {r:>6.1f}% {mm:>3} {avg_n:>5.2f} {roi:>+7.1f}%")
                if r > best_H[0]: best_H = (r, label, (h,mm,g4,roi,avg_n))

p(f"\n  方案H最优: {best_H[0]:.1f}% ({best_H[1]})")

# === 总结 ===
p(f"\n{'='*100}")
p("★ 总结对比 ★")
p(f"{'='*100}")
p(f"  直接TOP4最优(v3): {direct_best:.1f}%")
p(f"  随机基线: 直接={4/12*100:.1f}% 蒸馏={4/9*100:.1f}%")
p()

all_bests = [
    (best_A[0], "A纯静态蒸馏", best_A[1]),
    (best_B[0], "B精细搜索", best_B[1]),
    (best_C[0], "C蒸馏+反miss", best_C[1]),
    (best_D[0], "D自适应切换", best_D[1]),
    (best_E[0], "E置信度分层", best_E[1]),
    (best_F[0], "F多策略投票", best_F[1]),
    (best_G[0], "G激进反miss", best_G[1]),
    (best_H[0], "H动态窗口", best_H[1]),
]
all_bests.sort(key=lambda x: -x[0])

for rank, (r, label, detail) in enumerate(all_bests, 1):
    diff = r - direct_best
    mark = " ✅>50%" if r >= 50 else ""
    p(f"  {rank}. {label:<16} {r:.1f}% (vs直接: {diff:+.1f}%){mark}")
    p(f"     └─ {detail}")

overall = all_bests[0][0]
if overall >= 50:
    p(f"\n  ✅ 目标达成! 最高: {overall:.1f}%")
else:
    p(f"\n  ❌ 未达50%. 最高: {overall:.1f}%")
    gap_pct = 50.0 - overall
    p(f"     差距: {gap_pct:.1f}% ({int(gap_pct*3)}期)")
    p(f"     注: 4/9=44.4%随机→要50%需Stage2在TOP9命中的254期中选对150期(59.1%)")

with open('distill_top9_to_top4_v3.txt', 'w', encoding='utf-8-sig') as f:
    f.write(buf.getvalue())
p(f"\n结果已保存到 distill_top9_to_top4_v3.txt")
