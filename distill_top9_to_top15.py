"""
蒸馏方案: TOP9生肖过滤 → 在剩余号码中选TOP15
============================================
流程:
1. Stage1: TOP9生肖预测器选出9个生肖 → 约37个候选号码
2. Stage2: 在这37个号码中, 用多种号码评分方法选出TOP15
3. 验证: 实际号码是否在TOP15中

对照组:
- 直接TOP15(从49个中选): 已有基线约50-58%
- 蒸馏TOP15: 从~37个中选15, 随机率≈15/37=40.5%
- 直接TOP15随机率: 15/49=30.6%
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import io
import pandas as pd
import numpy as np
from collections import Counter
from scipy import stats

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

# ====== 生肖评分 (TOP9 Stage1) ======
def cold_scores_z(animals, window):
    w = min(window, len(animals))
    freq = Counter(animals[-w:])
    mx = max(freq.values()) if freq else 1
    return np.array([1.0 - freq.get(z, 0) / max(mx, 1) for z in ZODIAC_CYCLE_2026])

def hot_scores_z(animals, window):
    w = min(window, len(animals))
    freq = Counter(animals[-w:])
    mx = max(freq.values()) if freq else 1
    return np.array([freq.get(z, 0) / max(mx, 1) for z in ZODIAC_CYCLE_2026])

def gap_scores_z(animals):
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

def markov_scores_z(animals, window=None, laplace=1.0):
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

def stage1_top9(animals):
    """TOP9最优评分"""
    return (0.20 * cold_scores_z(animals, 15) + 0.05 * cold_scores_z(animals, 30) +
            0.50 * markov_scores_z(animals, 150) + 0.10 * gap_scores_z(animals) +
            0.15 * hot_scores_z(animals, 30))

def get_top9_numbers(animals):
    """获取TOP9生肖对应的所有号码"""
    s1 = stage1_top9(animals)
    top9_idx = np.argsort(-s1)[:9]
    top9_zodiacs = [ZODIAC_CYCLE_2026[i] for i in top9_idx]
    candidate_nums = []
    for z in top9_zodiacs:
        candidate_nums.extend(ZODIAC_NUMS_2026[z])
    return sorted(candidate_nums), top9_zodiacs

# ====== 号码评分方法 (Stage2: 从候选号码中选TOP15) ======

def num_frequency_score(nums_hist, candidates, windows=[50, 30, 20, 10]):
    """多窗口频率评分: 出现少的号码高分(回补逻辑)"""
    scores = {}
    for n in candidates:
        s = 0
        w_weights = [0.2, 0.25, 0.25, 0.3]
        for w, ww in zip(windows, w_weights):
            recent = nums_hist[-w:] if len(nums_hist) >= w else nums_hist
            freq = recent.count(n) / len(recent)
            expected = 1 / 49
            deficit = max(0, expected - freq) / expected
            s += ww * deficit
        scores[n] = s
    return scores

def num_gap_score(nums_hist, candidates):
    """间隔评分: 距上次出现多远"""
    scores = {}
    for n in candidates:
        last = -1
        for j in range(len(nums_hist) - 1, -1, -1):
            if nums_hist[j] == n:
                last = j
                break
        if last < 0:
            gap = len(nums_hist)
        else:
            gap = len(nums_hist) - 1 - last
        
        # 间隔5-20期得分最高
        if 5 <= gap <= 20:
            s = 1.0
        elif gap < 5:
            s = 0.1  # 太近,惩罚
        elif gap <= 40:
            s = 0.8
        else:
            s = 0.6  # 太久可能是冷号
        scores[n] = s
    return scores

def num_poisson_score(nums_hist, candidates, window=100):
    """泊松分布评分: 欠债回补"""
    scores = {}
    recent = nums_hist[-window:] if len(nums_hist) >= window else nums_hist
    expected = len(recent) / 49
    freq = Counter(recent)
    for n in candidates:
        actual = freq.get(n, 0)
        if actual < expected:
            deficit_ratio = (expected - actual) / max(expected, 0.01)
            scores[n] = min(1.0, deficit_ratio)
        else:
            scores[n] = 0.1
    return scores

def num_markov_score(nums_hist, candidates, window=100):
    """号码级马尔可夫转移概率"""
    scores = {}
    h = nums_hist[-window:] if len(nums_hist) >= window else nums_hist
    if len(h) < 2:
        return {n: 0.5 for n in candidates}
    
    # 用区间转移 (号码太多直接用会稀疏)
    # 分5个区间: 1-10, 11-20, 21-30, 31-40, 41-49
    def zone(x):
        if x <= 10: return 0
        elif x <= 20: return 1
        elif x <= 30: return 2
        elif x <= 40: return 3
        else: return 4
    
    trans = np.ones((5, 5))  # laplace
    for k in range(1, len(h)):
        trans[zone(h[k-1])][zone(h[k])] += 1
    
    last_zone = zone(h[-1])
    probs = trans[last_zone] / trans[last_zone].sum()
    
    for n in candidates:
        z = zone(n)
        scores[n] = float(probs[z])
    return scores

def num_recent_penalty(nums_hist, candidates, penalty_window=5, penalty_factor=0.2):
    """最近N期出现的号码惩罚"""
    recent = set(nums_hist[-penalty_window:]) if len(nums_hist) >= penalty_window else set(nums_hist)
    return {n: penalty_factor if n in recent else 1.0 for n in candidates}

def num_cycle_score(nums_hist, candidates, cycles=[3, 5, 7, 10]):
    """周期回补: 每N期出现一次的模式"""
    scores = {}
    for n in candidates:
        s = 0
        positions = [j for j, x in enumerate(nums_hist[-60:]) if x == n]
        if len(positions) >= 2:
            gaps = [positions[k+1] - positions[k] for k in range(len(positions) - 1)]
            for c in cycles:
                cycle_match = sum(1 for g in gaps if abs(g - c) <= 1)
                if cycle_match > 0:
                    # 检查是否"到期"
                    dist_from_last = 60 - positions[-1] if positions else 60
                    for cyc in cycles:
                        if abs(dist_from_last - cyc) <= 1:
                            s += 0.3
                            break
        scores[n] = min(1.0, max(0.1, 0.3 + s))
    return scores

# ====== 组合策略 ======

def combine_scores(nums_hist, candidates, weights):
    """组合多个评分方法"""
    final = {n: 0.0 for n in candidates}
    
    methods = {
        'freq': (num_frequency_score, 0),
        'gap': (num_gap_score, 0),
        'poisson': (num_poisson_score, 0),
        'markov': (num_markov_score, 0),
        'cycle': (num_cycle_score, 0),
    }
    
    for method_name, w in weights.items():
        if w <= 0: continue
        fn = methods[method_name][0]
        sc = fn(nums_hist, candidates)
        # 归一化到0-1
        vals = list(sc.values())
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx > mn else 1.0
        for n in candidates:
            final[n] += w * (sc[n] - mn) / rng
    
    # 最近5期惩罚
    penalty = num_recent_penalty(nums_hist, candidates)
    for n in candidates:
        final[n] *= penalty[n]
    
    return final

def select_top15(scores_dict, k=15):
    """从评分字典中选TOP-K"""
    sorted_nums = sorted(scores_dict.items(), key=lambda x: -x[1])
    return [n for n, s in sorted_nums[:k]]

# ====== 预计算 ======
all_data = []
for pi in range(test_periods):
    i = start + pi
    hist = numbers[:i]
    actual = numbers[i]
    hist_animals = [NUM_TO_ZODIAC_2026[n] for n in hist]
    all_data.append((hist, hist_animals, actual))

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

buf = io.StringIO()
def p(s=""):
    print(s)
    buf.write(s + "\n")

p("=" * 100)
p("蒸馏方案: TOP9生肖过滤 → 号码TOP15推测")
p(f"数据: {total}期, 测试: 最近{test_periods}期")
p("=" * 100)

# ====== Step1: TOP9过滤的基础统计 ======
p(f"\n--- Step1: TOP9过滤基础统计 ---")
t9_hit = 0
candidate_sizes = []
for pi in range(test_periods):
    hist, hist_animals, actual = all_data[pi]
    cands, _ = get_top9_numbers(hist_animals)
    candidate_sizes.append(len(cands))
    if actual in cands:
        t9_hit += 1

p(f"  TOP9生肖命中: {t9_hit}/300 = {t9_hit/3:.1f}%")
p(f"  候选号码数: 平均{np.mean(candidate_sizes):.1f}个 (范围{min(candidate_sizes)}-{max(candidate_sizes)})")
p(f"  蒸馏TOP15随机基线: 15/{np.mean(candidate_sizes):.0f} = {15/np.mean(candidate_sizes)*100:.1f}%")
p(f"  直接TOP15随机基线: 15/49 = {15/49*100:.1f}%")

# ====== Step2: 对照组 - 直接TOP15 (从49个中选) ======
p(f"\n--- 对照组: 直接从49个号码中选TOP15 ---")

direct_configs = [
    ("直接-统计均衡",    {'freq': 0.30, 'gap': 0.25, 'poisson': 0.25, 'markov': 0.10, 'cycle': 0.10}),
    ("直接-重泊松",      {'freq': 0.20, 'gap': 0.20, 'poisson': 0.40, 'markov': 0.10, 'cycle': 0.10}),
    ("直接-重间隔",      {'freq': 0.20, 'gap': 0.40, 'poisson': 0.20, 'markov': 0.10, 'cycle': 0.10}),
    ("直接-重频率",      {'freq': 0.45, 'gap': 0.20, 'poisson': 0.15, 'markov': 0.10, 'cycle': 0.10}),
    ("直接-重MK",        {'freq': 0.15, 'gap': 0.15, 'poisson': 0.15, 'markov': 0.45, 'cycle': 0.10}),
]

p(f"  {'策略':<22} {'命中':>8} {'命中率':>8} {'maxMiss':>8}")
p(f"  {'-'*50}")

direct_baseline = 0
for name, weights in direct_configs:
    hits = 0
    hit_list = []
    for pi in range(test_periods):
        hist, _, actual = all_data[pi]
        all_nums = list(range(1, 50))
        scores = combine_scores(hist, all_nums, weights)
        top15 = select_top15(scores, 15)
        hit = actual in top15
        if hit: hits += 1
        hit_list.append(hit)
    rate = hits / test_periods * 100
    mm = analyze_hits(hit_list)
    p(f"  {name:<22} {hits:>4}/300 {rate:>7.1f}% {mm:>7}")
    direct_baseline = max(direct_baseline, rate)

p(f"\n  直接TOP15最优: {direct_baseline:.1f}%")

# ====== Step3: 蒸馏 - TOP9过滤后选TOP15 ======
p(f"\n--- 蒸馏: TOP9过滤 → 从候选中选TOP15 ---")

distill_configs = [
    ("蒸馏-统计均衡",    {'freq': 0.30, 'gap': 0.25, 'poisson': 0.25, 'markov': 0.10, 'cycle': 0.10}),
    ("蒸馏-重泊松",      {'freq': 0.20, 'gap': 0.20, 'poisson': 0.40, 'markov': 0.10, 'cycle': 0.10}),
    ("蒸馏-重间隔",      {'freq': 0.20, 'gap': 0.40, 'poisson': 0.20, 'markov': 0.10, 'cycle': 0.10}),
    ("蒸馏-重频率",      {'freq': 0.45, 'gap': 0.20, 'poisson': 0.15, 'markov': 0.10, 'cycle': 0.10}),
    ("蒸馏-重MK",        {'freq': 0.15, 'gap': 0.15, 'poisson': 0.15, 'markov': 0.45, 'cycle': 0.10}),
    ("蒸馏-纯泊松",      {'freq': 0.0,  'gap': 0.0,  'poisson': 1.0,  'markov': 0.0,  'cycle': 0.0}),
    ("蒸馏-纯间隔",      {'freq': 0.0,  'gap': 1.0,  'poisson': 0.0,  'markov': 0.0,  'cycle': 0.0}),
    ("蒸馏-纯频率",      {'freq': 1.0,  'gap': 0.0,  'poisson': 0.0,  'markov': 0.0,  'cycle': 0.0}),
    ("蒸馏-freq+gap",    {'freq': 0.50, 'gap': 0.50, 'poisson': 0.0,  'markov': 0.0,  'cycle': 0.0}),
    ("蒸馏-poisson+gap", {'freq': 0.0,  'gap': 0.50, 'poisson': 0.50, 'markov': 0.0,  'cycle': 0.0}),
]

p(f"  {'策略':<22} {'命中':>8} {'命中率':>8} {'maxMiss':>8} {'T9miss':>7} {'vs直接':>8}")
p(f"  {'-'*68}")

best_distill = (0, "")
for name, weights in distill_configs:
    hits = 0
    hit_list = []
    t9_miss_count = 0
    for pi in range(test_periods):
        hist, hist_animals, actual = all_data[pi]
        
        # Stage1: TOP9过滤
        cands, _ = get_top9_numbers(hist_animals)
        
        if actual not in cands:
            t9_miss_count += 1
            hit_list.append(False)
            continue
        
        # Stage2: 从候选号码中选TOP15
        scores = combine_scores(hist, cands, weights)
        top15 = select_top15(scores, 15)
        hit = actual in top15
        if hit: hits += 1
        hit_list.append(hit)
    
    rate = hits / test_periods * 100
    mm = analyze_hits(hit_list)
    diff = rate - direct_baseline
    mark = " ★" if diff > 0 else ""
    p(f"  {name:<22} {hits:>4}/300 {rate:>7.1f}% {mm:>7} {t9_miss_count:>6} {diff:>+7.1f}%{mark}")
    if rate > best_distill[0]:
        best_distill = (rate, name)

# ====== Step4: 使用现有TOP15预测器 + TOP9过滤 ======
p(f"\n--- 蒸馏: TOP9 + 现有TOP15预测器 ---")

try:
    from top15_statistical_predictor import Top15StatisticalPredictor
    has_stat = True
except:
    has_stat = False

try:
    from precise_top15_predictor import PreciseTop15Predictor
    has_precise = True
except:
    has_precise = False

existing_predictors = []
if has_stat:
    existing_predictors.append(("统计TOP15", Top15StatisticalPredictor()))
if has_precise:
    existing_predictors.append(("精准TOP15", PreciseTop15Predictor()))

for pred_name, predictor in existing_predictors:
    # 直接版 (无过滤)
    hits_direct = 0
    hl_direct = []
    # 蒸馏版 (TOP9过滤)
    hits_distill = 0
    hl_distill = []
    t9m = 0
    
    for pi in range(test_periods):
        hist, hist_animals, actual = all_data[pi]
        
        # 直接版
        try:
            direct_pred = predictor.predict(hist)
            if isinstance(direct_pred, list) and len(direct_pred) > 0:
                hit_d = actual in direct_pred[:15]
            else:
                hit_d = False
        except:
            hit_d = False
        if hit_d: hits_direct += 1
        hl_direct.append(hit_d)
        
        # 蒸馏版: 取TOP9候选号码与直接TOP15/TOP20的交集
        cands, _ = get_top9_numbers(hist_animals)
        if actual not in cands:
            t9m += 1
        
        try:
            # 取更多候选(TOP20+), 与TOP9号码交集后取TOP15
            if hasattr(predictor, 'predict_top20'):
                full_pred = predictor.predict_top20(hist)
            else:
                full_pred = predictor.predict(hist)
            
            # 交集: 在TOP9号码中且在预测器排名中
            filtered = [n for n in full_pred if n in set(cands)]
            # 如果交集不足15, 从候选中补充
            if len(filtered) < 15:
                for n in cands:
                    if n not in filtered:
                        filtered.append(n)
                    if len(filtered) >= 15:
                        break
            hit_dist = actual in filtered[:15]
        except:
            hit_dist = False
        if hit_dist: hits_distill += 1
        hl_distill.append(hit_dist)
    
    r_d = hits_direct / test_periods * 100
    r_dist = hits_distill / test_periods * 100
    mm_d = analyze_hits(hl_direct)
    mm_dist = analyze_hits(hl_distill)
    diff = r_dist - r_d
    
    p(f"  {pred_name} 直接:    {hits_direct}/300 = {r_d:.1f}% maxMiss={mm_d}")
    p(f"  {pred_name} 蒸馏:    {hits_distill}/300 = {r_dist:.1f}% maxMiss={mm_dist} (T9miss={t9m}) diff={diff:+.1f}%")

# ====== Step5: 精细权重搜索 ======
p(f"\n--- 精细搜索: 蒸馏Stage2最优权重 ---")

best_s2 = (0, "", None)
for fw in [0.0, 0.15, 0.25, 0.35, 0.50]:
    for gw in [0.0, 0.15, 0.25, 0.35, 0.50]:
        for pw in [0.0, 0.15, 0.25, 0.35, 0.50]:
            for mw in [0.0, 0.10, 0.20, 0.35]:
                cw = 1.0 - fw - gw - pw - mw
                if cw < -0.01 or cw > 0.30: continue
                cw = max(cw, 0)
                
                weights = {'freq': fw, 'gap': gw, 'poisson': pw, 'markov': mw, 'cycle': cw}
                
                hits = 0
                hit_list = []
                for pi in range(test_periods):
                    hist, hist_animals, actual = all_data[pi]
                    cands, _ = get_top9_numbers(hist_animals)
                    if actual not in cands:
                        hit_list.append(False)
                        continue
                    scores = combine_scores(hist, cands, weights)
                    top15 = select_top15(scores, 15)
                    hit = actual in top15
                    if hit: hits += 1
                    hit_list.append(hit)
                
                rate = hits / test_periods * 100
                if rate > best_s2[0]:
                    mm = analyze_hits(hit_list)
                    label = f"freq={fw} gap={gw} poisson={pw} mk={mw} cycle={cw:.2f}"
                    best_s2 = (rate, label, (hits, mm))

p(f"  最优: {best_s2[1]}")
p(f"    {best_s2[2][0]}/300 = {best_s2[0]:.1f}% maxMiss={best_s2[2][1]}")

# ====== 最终总结 ======
p(f"\n{'='*100}")
p("最终对比总结")
p(f"{'='*100}")
p(f"  随机基线:")
p(f"    直接TOP15: 15/49 = {15/49*100:.1f}%")
p(f"    蒸馏TOP15: 15/{np.mean(candidate_sizes):.0f} = {15/np.mean(candidate_sizes)*100:.1f}%")
p(f"")
p(f"  实测结果:")
p(f"    直接TOP15最优:     {direct_baseline:.1f}%")
p(f"    蒸馏TOP15最优(预设): {best_distill[0]:.1f}% ({best_distill[1]})")
p(f"    蒸馏TOP15最优(搜索): {best_s2[0]:.1f}%")
diff_final = best_s2[0] - direct_baseline
p(f"")
if diff_final > 0:
    p(f"  ✅ 蒸馏方案提升TOP15: +{diff_final:.1f}%")
elif diff_final > -2:
    p(f"  ⚠️ 蒸馏方案与直接TOP15接近: {diff_final:+.1f}%")
else:
    p(f"  ❌ 蒸馏方案未能提升TOP15: {diff_final:+.1f}%")
    
p(f"\n  关键限制: TOP9过滤miss={100-t9_hit/3:.1f}%, 这些期无论如何不可能命中")

with open('distill_top9_to_top15.txt', 'w', encoding='utf-8-sig') as f:
    f.write(buf.getvalue())
p(f"\n结果已保存到 distill_top9_to_top15.txt")
