"""
精细搜索: 能否用TOP8达80%? TOP9最优组合是什么?
测试更多策略权重组合
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

# 预计算
all_data = []
for pi in range(test_periods):
    i = start + pi
    hist_animals = [NUM_TO_ZODIAC_2026[n] for n in numbers[:i]]
    actual_z = NUM_TO_ZODIAC_2026[numbers[i]]
    all_data.append((hist_animals, actual_z))

def test_strategy(score_fn, top_n, label=""):
    hits = 0
    hit_list = []
    for pi in range(test_periods):
        hist_animals, actual_z = all_data[pi]
        scores = score_fn(hist_animals)
        top = [ZODIAC_CYCLE_2026[i] for i in np.argsort(-scores)[:top_n]]
        hit = actual_z in top
        if hit: hits += 1
        hit_list.append(hit)
    
    # max miss
    streaks = []
    c = 0
    for h in hit_list:
        if not h: c += 1
        else:
            if c > 0: streaks.append(c)
            c = 0
    if c > 0: streaks.append(c)
    max_miss = max(streaks) if streaks else 0
    
    rate = hits / test_periods * 100
    cost = test_periods * top_n * 4
    roi = (hits * 46 - cost) / cost * 100
    return hits, rate, max_miss, roi

buf = io.StringIO()
def p(s=""):
    print(s)
    buf.write(s + "\n")

p("=" * 90)
p("精细搜索: TOP8能否达80%? TOP9最优组合")
p("=" * 90)

# ===== TOP8精细搜索 =====
p(f"\n--- TOP8精细权重搜索 ---")
best8 = (0, "", None)

configs8 = []
# 穷举 cold_w / mk_w / gap_w / hot_w
for c15w in [0.15, 0.20, 0.25, 0.30]:
    for mkw in [0.30, 0.40, 0.50, 0.60]:
        for gapw in [0.0, 0.10, 0.15, 0.20, 0.25]:
            for hotw in [0.0, 0.05, 0.10, 0.15]:
                remain = 1.0 - c15w - mkw - gapw - hotw
                if abs(remain) < 0.001:
                    remain = 0
                if remain < -0.01 or remain > 0.35:
                    continue
                configs8.append((c15w, mkw, gapw, hotw, remain))

p(f"  测试{len(configs8)}种权重组合...")

top8_results = []
for c15w, mkw, gapw, hotw, c30w in configs8:
    def make_fn(c15, mk, gp, ht, c30):
        def fn(a):
            s = c15 * cold_scores(a, 15) + c30 * cold_scores(a, 30) + mk * markov_scores(a, 150)
            if gp > 0: s += gp * gap_scores(a)
            if ht > 0: s += ht * hot_scores(a, 30)
            return s
        return fn
    
    hits, rate, mm, roi = test_strategy(make_fn(c15w, mkw, gapw, hotw, c30w), 8)
    if rate >= best8[0]:
        label = f"c15={c15w:.2f} c30={c30w:.2f} mk={mkw:.2f} gap={gapw:.2f} hot={hotw:.2f}"
        best8 = (rate, label, (hits, mm, roi))
    if rate >= 78:
        top8_results.append((rate, hits, mm, roi, c15w, c30w, mkw, gapw, hotw))

top8_results.sort(key=lambda x: (-x[0], x[2]))
p(f"  TOP8最优: {best8[1]}")
p(f"    命中率={best8[0]:.1f}% ({best8[2][0]}/300) maxMiss={best8[2][1]} ROI={best8[2][2]:+.1f}%")
p(f"\n  TOP8 ≥78%的组合 (共{len(top8_results)}个):")
for rate, hits, mm, roi, c15w, c30w, mkw, gapw, hotw in top8_results[:10]:
    p(f"    {rate:.1f}% ({hits}/300) mm={mm} c15={c15w} c30={c30w} mk={mkw} gap={gapw} hot={hotw}")

# ===== TOP8 + 反miss =====
p(f"\n--- TOP8+反miss(能否到80%?) ---")
for c15w, mkw, gapw, hotw, c30w in [
    (0.30, 0.60, 0.0, 0.0, 0.10),   # v3静态
    (0.20, 0.50, 0.20, 0.0, 0.10),   # gap增强
    (0.25, 0.50, 0.15, 0.10, 0.0),   # 热号增强
]:
    for blend_t in [2, 3]:
        for expand_t in [3, 4, 5]:
            if expand_t <= blend_t: continue
            for expand_to in [9, 10]:
                consec = 0
                hits = 0
                hit_list = []
                sizes = []
                for pi in range(test_periods):
                    hist_animals, actual_z = all_data[pi]
                    s = c15w * cold_scores(hist_animals, 15) + c30w * cold_scores(hist_animals, 30) + mkw * markov_scores(hist_animals, 150)
                    if gapw > 0: s += gapw * gap_scores(hist_animals)
                    if hotw > 0: s += hotw * hot_scores(hist_animals, 30)
                    
                    if consec >= blend_t:
                        h = hot_scores(hist_animals, 30)
                        s = 0.75 * s + 0.25 * h
                    
                    n = expand_to if consec >= expand_t else 8
                    top = [ZODIAC_CYCLE_2026[i] for i in np.argsort(-s)[:n]]
                    hit = actual_z in top
                    if hit:
                        hits += 1
                        consec = 0
                    else:
                        consec += 1
                    hit_list.append(hit)
                    sizes.append(n)
                
                rate = hits / test_periods * 100
                streaks = []
                c = 0
                for h2 in hit_list:
                    if not h2: c += 1
                    else:
                        if c > 0: streaks.append(c)
                        c = 0
                if c > 0: streaks.append(c)
                max_miss = max(streaks) if streaks else 0
                avg_n = np.mean(sizes)
                cost = sum(s2 * 4 for s2 in sizes)
                roi = (hits * 46 - cost) / cost * 100
                
                if rate >= 79:
                    label = f"c15={c15w} mk={mkw} gap={gapw} hot={hotw}"
                    p(f"  {label} blend@{blend_t} expand→{expand_to}@{expand_t}: {hits}/300={rate:.1f}% mm={max_miss} avg={avg_n:.1f} ROI={roi:+.1f}%")

# ===== TOP9精细搜索 =====
p(f"\n--- TOP9精细权重搜索 ---")
best9 = (0, "", None)

for c15w in [0.15, 0.20, 0.25, 0.30]:
    for mkw in [0.30, 0.40, 0.50, 0.60]:
        for gapw in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]:
            for hotw in [0.0, 0.05, 0.10, 0.15]:
                remain = 1.0 - c15w - mkw - gapw - hotw
                if remain < -0.01 or remain > 0.35:
                    continue
                c30w = max(remain, 0)
                
                def make_fn9(c15, mk, gp, ht, c30):
                    def fn(a):
                        s = c15 * cold_scores(a, 15) + c30 * cold_scores(a, 30) + mk * markov_scores(a, 150)
                        if gp > 0: s += gp * gap_scores(a)
                        if ht > 0: s += ht * hot_scores(a, 30)
                        return s
                    return fn
                
                hits, rate, mm, roi = test_strategy(make_fn9(c15w, mkw, gapw, hotw, c30w), 9)
                if rate >= best9[0] or (rate == best9[0] and mm < best9[2][1]):
                    label = f"c15={c15w:.2f} c30={c30w:.2f} mk={mkw:.2f} gap={gapw:.2f} hot={hotw:.2f}"
                    best9 = (rate, label, (hits, mm, roi))

p(f"  TOP9最优: {best9[1]}")
p(f"    命中率={best9[0]:.1f}% ({best9[2][0]}/300) maxMiss={best9[2][1]} ROI={best9[2][2]:+.1f}%")

# ===== TOP9 + 反miss ===== 
p(f"\n--- TOP9+反miss (降低maxMiss, 不一定提升命中率) ---")
for blend_t in [2]:
    for expand_t in [3, 4]:
        for base_config_name, c15w, c30w, mkw, gapw, hotw in [
            ("v3静态", 0.30, 0.10, 0.60, 0.0, 0.0),
            ("增强gap", 0.20, 0.0, 0.50, 0.20, 0.10),
            ("best9", *[float(x.split('=')[1]) for x in best9[1].split()]),
        ]:
            consec = 0
            hits = 0
            hit_list = []
            sizes = []
            for pi in range(test_periods):
                hist_animals, actual_z = all_data[pi]
                s = c15w * cold_scores(hist_animals, 15) + c30w * cold_scores(hist_animals, 30) + mkw * markov_scores(hist_animals, 150)
                if gapw > 0: s += gapw * gap_scores(hist_animals)
                if hotw > 0: s += hotw * hot_scores(hist_animals, 30)
                
                if consec >= blend_t:
                    h = hot_scores(hist_animals, 30)
                    s = 0.75 * s + 0.25 * h
                
                n = 10 if consec >= expand_t else 9
                top = [ZODIAC_CYCLE_2026[i] for i in np.argsort(-s)[:n]]
                hit = actual_z in top
                if hit:
                    hits += 1
                    consec = 0
                else:
                    consec += 1
                hit_list.append(hit)
                sizes.append(n)
            
            rate = hits / test_periods * 100
            streaks2 = []
            c = 0
            for h2 in hit_list:
                if not h2: c += 1
                else:
                    if c > 0: streaks2.append(c)
                    c = 0
            if c > 0: streaks2.append(c)
            max_miss = max(streaks2) if streaks2 else 0
            avg_n = np.mean(sizes)
            cost = sum(s2 * 4 for s2 in sizes)
            roi = (hits * 46 - cost) / cost * 100
            
            p(f"  {base_config_name} blend@{blend_t} expand→10@{expand_t}: {hits}/300={rate:.1f}% mm={max_miss} avg={avg_n:.1f} ROI={roi:+.1f}%")

# ===== 最终结论 =====
p(f"\n{'='*90}")
p("最终结论")
p(f"{'='*90}")
p(f"  达到80%命中率最少需要: TOP9 (9个生肖)")
p(f"  TOP8最高只能到: {best8[0]:.1f}% (差距较大)")
p(f"  TOP9最优: {best9[0]:.1f}%, maxMiss={best9[2][1]}")
p(f"  推荐: TOP9 + 反miss机制 → 命中率>82%, 最大连miss≤2")

with open('search_80pct_fine.txt', 'w', encoding='utf-8-sig') as f:
    f.write(buf.getvalue())
p(f"\n结果已保存到 search_80pct_fine.txt")
