"""
v3设计: 多级反miss机制精细搜索
基于发现: hot30是最佳互补策略(静态miss时45.7%命中)
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

def hot_scores(animals, window):
    w = min(window, len(animals))
    freq = Counter(animals[-w:])
    mx = max(freq.values()) if freq else 1
    return np.array([freq.get(z, 0) / max(mx, 1) for z in ZODIAC_CYCLE_2026])

def static_predict(animals):
    cold15 = cold_scores(animals, 15)
    cold30 = cold_scores(animals, 30)
    mk150 = markov_scores(animals, window=150)
    return 0.30 * cold15 + 0.10 * cold30 + 0.60 * mk150

# 预计算所有期数据
all_data = []
for pi in range(test_periods):
    i = start + pi
    hist_animals = [NUM_TO_ZODIAC_2026[n] for n in numbers[:i]]
    actual_z = NUM_TO_ZODIAC_2026[numbers[i]]
    all_data.append((hist_animals, actual_z))

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
    ge4 = sum(1 for s in streaks if s >= 4)
    ge6 = sum(1 for s in streaks if s >= 6)
    return max_miss, ge4, ge6

def run_strategy(get_top_fn, label=""):
    """统一运行策略"""
    consec = 0
    hits = 0
    hit_list = []
    sizes = []
    for pi in range(test_periods):
        hist_animals, actual_z = all_data[pi]
        top = get_top_fn(hist_animals, consec)
        hit = actual_z in top
        if hit:
            hits += 1
            consec = 0
        else:
            consec += 1
        hit_list.append(hit)
        sizes.append(len(top))
    
    max_miss, ge4, ge6 = analyze_miss(hit_list)
    hit_rate = hits / test_periods * 100
    avg_size = np.mean(sizes)
    
    # 投注成本: 每个生肖4元
    total_bet = sum(s * 4 for s in sizes)
    total_win = sum(46 for h in hit_list if h)
    profit = total_win - total_bet
    roi = profit / total_bet * 100
    
    return {
        'label': label,
        'hits': hits,
        'rate': hit_rate,
        'max_miss': max_miss,
        'ge4': ge4,
        'ge6': ge6,
        'avg_size': avg_size,
        'profit': profit,
        'roi': roi,
        'hit_list': hit_list,
    }


buf = io.StringIO()
def p(s=""):
    print(s)
    buf.write(s + "\n")

p("=" * 90)
p("v3方案设计: 多级反miss机制搜索")
p("=" * 90)

results = []

# === 基准 ===
p("\n--- 基准方案 ---")
# 纯静态TOP4
r = run_strategy(lambda a, c: [ZODIAC_CYCLE_2026[i] for i in np.argsort(-static_predict(a))[:4]], "纯静态TOP4")
results.append(r)
p(f"  {r['label']:<40} {r['hits']}/300={r['rate']:.1f}% max_miss={r['max_miss']} ≥4miss={r['ge4']} ≥6miss={r['ge6']} ROI={r['roi']:+.1f}%")

# 当前v2
from zodiac_top4_v2_predictor import ZodiacTop4V2Predictor
pred_v2 = ZodiacTop4V2Predictor()
def v2_fn(a, c_unused):
    # v2 uses its own consecutive miss tracking
    return pred_v2.predict([numbers[start + all_data.index((a, None))] if False else 0], top_n=4)
# 重新运行v2
v2_hits = []
pred_v2 = ZodiacTop4V2Predictor()
for pi in range(test_periods):
    i = start + pi
    predicted = pred_v2.predict(numbers[:i], top_n=4)
    actual_z = NUM_TO_ZODIAC_2026[numbers[i]]
    hit = actual_z in predicted
    v2_hits.append(hit)
    pred_v2.record_result(hit)
v2_max, v2_ge4, v2_ge6 = analyze_miss(v2_hits)
v2_profit = sum(46 for h in v2_hits if h) - 300*16
p(f"  {'当前v2(静态+miss切换)':<40} {sum(v2_hits)}/300={sum(v2_hits)/300*100:.1f}% max_miss={v2_max} ≥4miss={v2_ge4} ≥6miss={v2_ge6} ROI={v2_profit/(300*16)*100:+.1f}%")

# === 方案A: 纯扩展(静态TOP5) ===
p("\n--- 方案A: 连miss时扩展(静态排名) ---")
for thresh in [2, 3]:
    for expand_to in [5, 6]:
        def make_fn(t, e):
            def fn(a, c):
                s = static_predict(a)
                n = e if c >= t else 4
                return [ZODIAC_CYCLE_2026[i] for i in np.argsort(-s)[:n]]
            return fn
        r = run_strategy(make_fn(thresh, expand_to), f"静态扩展TOP{expand_to}@miss>={thresh}")
        results.append(r)
        p(f"  {r['label']:<40} {r['hits']}/300={r['rate']:.1f}% max_miss={r['max_miss']} ≥4miss={r['ge4']} ≥6miss={r['ge6']} avg={r['avg_size']:.1f} ROI={r['roi']:+.1f}%")

# === 方案B: 扩展5th来自互补(hot30) ===
p("\n--- 方案B: TOP4+互补hot扩展 ---")
for thresh in [2, 3]:
    for hot_w in [20, 30, 40]:
        def make_fn(t, hw):
            def fn(a, c):
                s = static_predict(a)
                top4_idx = list(np.argsort(-s)[:4])
                if c >= t:
                    h = hot_scores(a, hw)
                    h_sorted = np.argsort(-h)
                    for idx in h_sorted:
                        if idx not in top4_idx:
                            top4_idx.append(idx)
                            break
                return [ZODIAC_CYCLE_2026[i] for i in top4_idx]
            return fn
        r = run_strategy(make_fn(thresh, hot_w), f"TOP4+hot{hot_w}扩展@miss>={thresh}")
        results.append(r)
        p(f"  {r['label']:<40} {r['hits']}/300={r['rate']:.1f}% max_miss={r['max_miss']} ≥4miss={r['ge4']} ≥6miss={r['ge6']} avg={r['avg_size']:.1f} ROI={r['roi']:+.1f}%")

# === 方案C: 多级梯度扩展 ===
p("\n--- 方案C: 多级梯度扩展 ---")
configs_c = [
    (2, 5, 4, 5, "miss>=2→5, >=4→5(hot)"),
    (2, 5, 5, 6, "miss>=2→5, >=5→6"),
    (2, 5, 4, 6, "miss>=2→5, >=4→6"),
    (3, 5, 5, 6, "miss>=3→5, >=5→6"),
    (3, 5, 6, 6, "miss>=3→5, >=6→6"),
]
for t1, n1, t2, n2, label in configs_c:
    def make_fn(t1_, n1_, t2_, n2_):
        def fn(a, c):
            s = static_predict(a)
            if c >= t2_:
                n = n2_
            elif c >= t1_:
                n = n1_
            else:
                n = 4
            return [ZODIAC_CYCLE_2026[i] for i in np.argsort(-s)[:n]]
        return fn
    r = run_strategy(make_fn(t1, n1, t2, n2), label)
    results.append(r)
    p(f"  {r['label']:<40} {r['hits']}/300={r['rate']:.1f}% max_miss={r['max_miss']} ≥4miss={r['ge4']} ≥6miss={r['ge6']} avg={r['avg_size']:.1f} ROI={r['roi']:+.1f}%")

# === 方案D: 多级梯度 + 互补 ===
p("\n--- 方案D: 多级扩展+互补替换 ---")
for t1 in [2, 3]:
    for t2 in [4, 5]:
        def make_fn(t1_, t2_):
            def fn(a, c):
                s = static_predict(a)
                top_idx = list(np.argsort(-s)[:4])
                if c >= t2_:
                    # 级别2: TOP4 + hot30的top1(不重复) + 静态5th
                    h = hot_scores(a, 30)
                    h_sorted = np.argsort(-h)
                    added = 0
                    for idx in h_sorted:
                        if idx not in top_idx:
                            top_idx.append(idx)
                            added += 1
                            if added >= 2:
                                break
                elif c >= t1_:
                    # 级别1: TOP4 + hot30的top1(不重复)
                    h = hot_scores(a, 30)
                    h_sorted = np.argsort(-h)
                    for idx in h_sorted:
                        if idx not in top_idx:
                            top_idx.append(idx)
                            break
                return [ZODIAC_CYCLE_2026[i] for i in top_idx]
            return fn
        r = run_strategy(make_fn(t1, t2), f"静态+hot30互补: L1@{t1},L2@{t2}")
        results.append(r)
        p(f"  {r['label']:<40} {r['hits']}/300={r['rate']:.1f}% max_miss={r['max_miss']} ≥4miss={r['ge4']} ≥6miss={r['ge6']} avg={r['avg_size']:.1f} ROI={r['roi']:+.1f}%")

# === 方案E: 静态+互补blend(不扩展,替换最弱的) ===
p("\n--- 方案E: miss时blend互补(不扩展,保持TOP4) ---")
for thresh in [2, 3]:
    for blend_w in [0.15, 0.25, 0.35]:
        def make_fn(t, w):
            def fn(a, c):
                s = static_predict(a)
                if c >= t:
                    h = hot_scores(a, 30)
                    combined = (1-w) * s + w * h
                    return [ZODIAC_CYCLE_2026[i] for i in np.argsort(-combined)[:4]]
                return [ZODIAC_CYCLE_2026[i] for i in np.argsort(-s)[:4]]
            return fn
        r = run_strategy(make_fn(thresh, blend_w), f"blend{blend_w:.0%}hot@miss>={thresh}")
        results.append(r)
        p(f"  {r['label']:<40} {r['hits']}/300={r['rate']:.1f}% max_miss={r['max_miss']} ≥4miss={r['ge4']} ≥6miss={r['ge6']} ROI={r['roi']:+.1f}%")

# === 方案F: 混合策略(静态base + miss时渐进blend + 扩展) ===
p("\n--- 方案F: 渐进blend + 扩展 ---")
for blend_base in [0.05, 0.10]:
    for expand_thresh in [3, 4]:
        def make_fn(bb, et):
            def fn(a, c):
                s = static_predict(a)
                if c >= 1:
                    h = hot_scores(a, 30)
                    w = min(bb + (c - 1) * 0.05, 0.4)
                    combined = (1-w) * s + w * h
                else:
                    combined = s
                n = 5 if c >= et else 4
                return [ZODIAC_CYCLE_2026[i] for i in np.argsort(-combined)[:n]]
            return fn
        r = run_strategy(make_fn(blend_base, expand_thresh), f"渐进blend{blend_base}+扩展@{expand_thresh}")
        results.append(r)
        p(f"  {r['label']:<40} {r['hits']}/300={r['rate']:.1f}% max_miss={r['max_miss']} ≥4miss={r['ge4']} ≥6miss={r['ge6']} avg={r['avg_size']:.1f} ROI={r['roi']:+.1f}%")

# === 方案G: 原始模型做互补(miss时用原始模型的推荐扩展5th) ===
p("\n--- 方案G: 原始模型互补扩展 ---")
from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy
for thresh in [2, 3]:
    orig = RecommendedZodiacTop4Strategy(use_emergency_backup=False)
    def make_fn_g(t, orig_):
        def fn(a, c):
            s = static_predict(a)
            top4_idx = list(np.argsort(-s)[:4])
            top4_z = [ZODIAC_CYCLE_2026[i] for i in top4_idx]
            if c >= t:
                result = orig_.predict_top4(a)
                for z in result['top4']:
                    if z not in top4_z:
                        top4_z.append(z)
                        break
            return top4_z
        return fn
    r = run_strategy(make_fn_g(thresh, orig), f"静态+原始互补@miss>={thresh}")
    results.append(r)
    p(f"  {r['label']:<40} {r['hits']}/300={r['rate']:.1f}% max_miss={r['max_miss']} ≥4miss={r['ge4']} ≥6miss={r['ge6']} avg={r['avg_size']:.1f} ROI={r['roi']:+.1f}%")

# ===== 排名总结 =====
p(f"\n{'='*90}")
p("综合排名 (按命中率排序, 相同命中率按max_miss排序)")
p(f"{'='*90}")
p(f"  {'排名':>4} {'方案':<42} {'命中':>6} {'命中率':>7} {'maxM':>5} {'≥4M':>4} {'≥6M':>4} {'avg':>5} {'ROI':>8}")
p(f"  {'-'*86}")

all_results = results
all_results.sort(key=lambda x: (-x['rate'], x['max_miss'], x['ge6']))

for rank, r in enumerate(all_results, 1):
    marker = " ★" if r['max_miss'] <= 8 and r['rate'] >= 46 else ""
    p(f"  {rank:>4} {r['label']:<42} {r['hits']:>4}/300 {r['rate']:>6.1f}% {r['max_miss']:>5} {r['ge4']:>4} {r['ge6']:>4} {r['avg_size']:>5.1f} {r['roi']:>+7.1f}%{marker}")

# 输出最佳方案的详细hit_list
best = all_results[0]
p(f"\n最佳方案: {best['label']}")
p(f"命中率={best['rate']:.1f}%, 最大连miss={best['max_miss']}, ≥6期miss={best['ge6']}次, ROI={best['roi']:+.1f}%")

# 保存
with open('v3_design_search.txt', 'w', encoding='utf-8-sig') as f:
    f.write(buf.getvalue())
print("\n结果已保存到 v3_design_search.txt")
