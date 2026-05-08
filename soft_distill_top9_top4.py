"""
软蒸馏方案: TOP9作为先验信息辅助推导TOP4
============================================
与之前"硬屏蔽"的区别:
- 硬屏蔽: TOP9排除3个生肖 → 在9个中选4个 → 信息丢失
- 软蒸馏: TOP9为12个生肖提供"置信度评分" → 与独立TOP4评分融合 → 不丢弃任何生肖

核心思想: TOP9的排名本身就是有价值的信息(85%命中率)
将TOP9评分作为"先验概率", 乘以独立的"似然评分", 得到"后验TOP4"

多种软蒸馏策略:
A. 乘法融合: TOP4_score × TOP9_score^α (α控制先验强度)
B. 加法融合: β×TOP4_score + (1-β)×TOP9_score
C. 排名加权: TOP9排名转权重 × 独立TOP4评分
D. 动态置信: TOP9得分差距大时强过滤, 差距小时弱过滤
E. 残差修正: TOP4基础评分 + TOP9排名修正项
F. 软屏蔽梯度: TOP9排名1-9正常, 10-12降权但不清零
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

def second_order_markov(animals, window=150, laplace=0.5):
    """二阶马尔可夫: 用前两期状态预测"""
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

# ====== TOP9评分 (先验) ======
def top9_scores(animals):
    """TOP9多维度评分 - 作为先验信息"""
    return (0.20 * cold_scores(animals, 15) +
            0.05 * cold_scores(animals, 30) +
            0.50 * markov_scores(animals, 150) +
            0.10 * gap_scores(animals) +
            0.15 * hot_scores(animals, 30))

# ====== 独立TOP4评分 (似然) - 多种独立视角 ======
def top4_v3_scores(animals):
    """v3静态: cold15+cold30+mk150"""
    return (0.30 * cold_scores(animals, 15) +
            0.10 * cold_scores(animals, 30) +
            0.60 * markov_scores(animals, 150))

def top4_short_mk(animals):
    """短窗口MK: 与TOP9的MK150互补"""
    return (0.40 * markov_scores(animals, 50) +
            0.30 * markov_scores(animals, 80) +
            0.30 * cold_scores(animals, 10))

def top4_gap_heavy(animals):
    """重间隔: 回补逻辑"""
    return (0.45 * gap_scores(animals) +
            0.35 * markov_scores(animals, 100) +
            0.20 * cold_scores(animals, 15))

def top4_mk2_blend(animals):
    """二阶MK + 一阶MK混合"""
    return (0.40 * second_order_markov(animals, 150) +
            0.30 * markov_scores(animals, 150) +
            0.30 * cold_scores(animals, 15))

def top4_rotation(animals):
    """轮换+间隔"""
    return (0.35 * rotation_scores(animals, 25) +
            0.30 * gap_scores(animals) +
            0.35 * markov_scores(animals, 100))

def top4_hot_cold_mix(animals):
    """冷热混合: 在冷号基础上加入热号修正"""
    return (0.25 * cold_scores(animals, 15) +
            0.15 * hot_scores(animals, 20) +
            0.40 * markov_scores(animals, 150) +
            0.20 * gap_scores(animals))

# ====== 归一化工具 ======
def normalize(scores):
    """归一化到0-1"""
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-10:
        return np.ones_like(scores) / len(scores)
    return (scores - mn) / (mx - mn)

def softmax(scores, temperature=1.0):
    """Softmax概率化"""
    e = np.exp((scores - scores.max()) / temperature)
    return e / e.sum()

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

def test_strategy(score_fn, top_n=4, label=""):
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
    mm, g4, g6 = analyze_hits(hit_list)
    cost = test_periods * top_n * 4
    roi = (hits * 46 - cost) / cost * 100
    return hits, rate, mm, g4, g6, roi, hit_list

buf = io.StringIO()
def p(s=""):
    print(s)
    buf.write(s + "\n")

p("=" * 100)
p("软蒸馏方案: TOP9先验 × 独立TOP4似然 → 后验TOP4")
p("核心: 不硬屏蔽任何生肖, TOP9排名作为软权重辅助推导")
p(f"数据: {total}期, 测试: 最近{test_periods}期")
p("=" * 100)

# ====== 对照组 ======
p(f"\n{'='*100}")
p("对照组")
p(f"{'='*100}")

h, r, mm, g4, g6, roi, _ = test_strategy(top4_v3_scores, 4)
p(f"  直接v3 TOP4:         {h}/300 = {r:.1f}% maxMiss={mm} ≥4miss={g4} ≥6miss={g6} ROI={roi:+.1f}%")
baseline_rate = r

# 直接用TOP9权重选TOP4
h2, r2, mm2, g42, g62, roi2, _ = test_strategy(top9_scores, 4)
p(f"  TOP9权重直选TOP4:    {h2}/300 = {r2:.1f}% maxMiss={mm2} ≥4miss={g42} ≥6miss={g62} ROI={roi2:+.1f}%")

# 硬屏蔽基线(之前的蒸馏)
def hard_filter_top4(animals):
    s1 = top9_scores(animals)
    s2 = top4_v3_scores(animals)
    top9_idx = set(np.argsort(-s1)[:9].tolist())
    # 硬屏蔽: 不在TOP9的设为-inf
    result = np.full(12, -999.0)
    for i in top9_idx:
        result[i] = s2[i]
    return result

h3, r3, mm3, g43, g63, roi3, _ = test_strategy(hard_filter_top4, 4)
p(f"  硬屏蔽(TOP9→TOP4):  {h3}/300 = {r3:.1f}% maxMiss={mm3} ≥4miss={g43} ≥6miss={g63} ROI={roi3:+.1f}%")

# ====== 软蒸馏策略 ======
p(f"\n{'='*100}")
p("软蒸馏策略 (TOP9先验 × 独立似然)")
p(f"{'='*100}")

top4_methods = [
    ("v3静态",     top4_v3_scores),
    ("短窗口MK",   top4_short_mk),
    ("重间隔",     top4_gap_heavy),
    ("二阶MK",     top4_mk2_blend),
    ("轮换+间隔",  top4_rotation),
    ("冷热混合",   top4_hot_cold_mix),
]

# === 策略A: 乘法融合 P(z) ∝ prior^α × likelihood ===
p(f"\n--- A. 乘法融合: score = TOP9^α × TOP4_independent ---")
p(f"  {'似然方法':<12} {'α':>4} {'命中':>8} {'命中率':>8} {'mm':>4} {'≥4':>4} {'ROI':>8} {'vs基线':>8}")
p(f"  {'-'*62}")

best_A = (0, "", None)
for method_name, method_fn in top4_methods:
    for alpha in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        def make_multiply(fn, a):
            def score_fn(animals):
                prior = normalize(top9_scores(animals)) + 0.01  # 避免0
                likelihood = normalize(fn(animals)) + 0.01
                return (prior ** a) * likelihood
            return score_fn
        
        h, r, mm, g4, g6, roi, _ = test_strategy(make_multiply(method_fn, alpha), 4)
        diff = r - baseline_rate
        if r > best_A[0]:
            best_A = (r, f"{method_name} α={alpha}", (h, mm, g4, g6, roi))
        if r >= 46:
            p(f"  {method_name:<12} {alpha:>4.1f} {h:>4}/300 {r:>7.1f}% {mm:>3} {g4:>3} {roi:>+7.1f}% {diff:>+7.1f}%")

p(f"  >> A最优: {best_A[1]} → {best_A[0]:.1f}% mm={best_A[2][1]}")

# === 策略B: 加法融合 score = β×TOP4 + (1-β)×TOP9 ===
p(f"\n--- B. 加法融合: score = β×TOP4 + (1-β)×TOP9 ---")
p(f"  {'似然方法':<12} {'β':>4} {'命中':>8} {'命中率':>8} {'mm':>4} {'≥4':>4} {'ROI':>8} {'vs基线':>8}")
p(f"  {'-'*62}")

best_B = (0, "", None)
for method_name, method_fn in top4_methods:
    for beta in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        def make_add(fn, b):
            def score_fn(animals):
                prior = normalize(top9_scores(animals))
                likelihood = normalize(fn(animals))
                return b * likelihood + (1 - b) * prior
            return score_fn
        
        h, r, mm, g4, g6, roi, _ = test_strategy(make_add(method_fn, beta), 4)
        diff = r - baseline_rate
        if r > best_B[0]:
            best_B = (r, f"{method_name} β={beta}", (h, mm, g4, g6, roi))
        if r >= 46:
            p(f"  {method_name:<12} {beta:>4.1f} {h:>4}/300 {r:>7.1f}% {mm:>3} {g4:>3} {roi:>+7.1f}% {diff:>+7.1f}%")

p(f"  >> B最优: {best_B[1]} → {best_B[0]:.1f}% mm={best_B[2][1]}")

# === 策略C: 排名权重 TOP9排名→软权重 ===
p(f"\n--- C. 排名软权重: TOP9排名→权重乘子 × TOP4评分 ---")
p(f"  {'似然方法':<12} {'衰减':>6} {'命中':>8} {'命中率':>8} {'mm':>4} {'vs基线':>8}")
p(f"  {'-'*55}")

best_C = (0, "", None)
for method_name, method_fn in top4_methods:
    for decay_style in ['linear', 'exp', 'step']:
        def make_rank_weight(fn, decay):
            def score_fn(animals):
                s1 = top9_scores(animals)
                ranks = np.argsort(-s1)  # rank 0=best, 11=worst
                rank_of = np.zeros(12)
                for r_pos, idx in enumerate(ranks):
                    rank_of[idx] = r_pos
                
                if decay == 'linear':
                    # 排名1→权重1.0, 排名12→权重0.1
                    weights = 1.0 - 0.9 * rank_of / 11
                elif decay == 'exp':
                    # 指数衰减
                    weights = np.exp(-0.3 * rank_of)
                elif decay == 'step':
                    # 阶梯: TOP9=1.0, 排名10-12=0.15
                    weights = np.where(rank_of < 9, 1.0, 0.15)
                
                likelihood = fn(animals)
                return weights * likelihood
            return score_fn
        
        h, r, mm, g4, g6, roi, _ = test_strategy(make_rank_weight(method_fn, decay_style), 4)
        diff = r - baseline_rate
        if r > best_C[0]:
            best_C = (r, f"{method_name} {decay_style}", (h, mm, g4, g6, roi))
        if r >= 45:
            p(f"  {method_name:<12} {decay_style:>6} {h:>4}/300 {r:>7.1f}% {mm:>3} {diff:>+7.1f}%")

p(f"  >> C最优: {best_C[1]} → {best_C[0]:.1f}% mm={best_C[2][1]}")

# === 策略D: 动态置信度 ===
p(f"\n--- D. 动态置信: TOP9分差大→强先验, 分差小→弱先验 ---")

best_D = (0, "", None)
for method_name, method_fn in top4_methods:
    for base_alpha in [0.5, 1.0, 1.5]:
        def make_dynamic(fn, ba):
            def score_fn(animals):
                s1 = top9_scores(animals)
                sorted_s1 = np.sort(s1)[::-1]
                # 第9名和第10名的差距 → 置信度
                gap_9_10 = sorted_s1[8] - sorted_s1[9] if len(sorted_s1) > 9 else 0
                # 归一化差距
                confidence = min(1.0, gap_9_10 / 0.05) if gap_9_10 > 0 else 0
                
                # 置信度高 → 先验强(alpha大); 置信度低 → 先验弱(alpha小)
                alpha = ba * (0.3 + 0.7 * confidence)
                
                prior = normalize(s1) + 0.01
                likelihood = normalize(fn(animals)) + 0.01
                return (prior ** alpha) * likelihood
            return score_fn
        
        h, r, mm, g4, g6, roi, _ = test_strategy(make_dynamic(method_fn, base_alpha), 4)
        diff = r - baseline_rate
        if r > best_D[0]:
            best_D = (r, f"{method_name} base_α={base_alpha}", (h, mm, g4, g6, roi))

p(f"  >> D最优: {best_D[1]} → {best_D[0]:.1f}% mm={best_D[2][1]}")

# === 策略E: 残差修正 ===
p(f"\n--- E. 残差修正: TOP4 + λ×(TOP9排名修正) ---")

best_E = (0, "", None)
for method_name, method_fn in top4_methods:
    for lam in [0.1, 0.2, 0.3, 0.5]:
        def make_residual(fn, l):
            def score_fn(animals):
                base = normalize(fn(animals))
                s1 = top9_scores(animals)
                # 修正项: TOP9高排名的生肖加分
                correction = normalize(s1)
                return base + l * correction
            return score_fn
        
        h, r, mm, g4, g6, roi, _ = test_strategy(make_residual(method_fn, lam), 4)
        diff = r - baseline_rate
        if r > best_E[0]:
            best_E = (r, f"{method_name} λ={lam}", (h, mm, g4, g6, roi))

p(f"  >> E最优: {best_E[1]} → {best_E[0]:.1f}% mm={best_E[2][1]}")

# === 策略F: 梯度软屏蔽 ===
p(f"\n--- F. 梯度软屏蔽: 排名越低降权越多(但不清零) ---")

best_F = (0, "", None)
for method_name, method_fn in top4_methods:
    for floor_w in [0.05, 0.10, 0.20, 0.30]:
        def make_gradient(fn, fw):
            def score_fn(animals):
                s1 = top9_scores(animals)
                ranks = np.argsort(np.argsort(-s1))  # 0=best
                # 权重: TOP1=1.0 → TOP12=floor_w
                weights = 1.0 - (1.0 - fw) * ranks / 11
                likelihood = fn(animals)
                return weights * likelihood
            return score_fn
        
        h, r, mm, g4, g6, roi, _ = test_strategy(make_gradient(method_fn, floor_w), 4)
        diff = r - baseline_rate
        if r > best_F[0]:
            best_F = (r, f"{method_name} floor={floor_w}", (h, mm, g4, g6, roi))
        if r >= 46:
            p(f"  {method_name:<12} floor={floor_w} {h}/300={r:.1f}% mm={mm} {diff:>+.1f}%")

p(f"  >> F最优: {best_F[1]} → {best_F[0]:.1f}% mm={best_F[2][1]}")

# === 策略G: Softmax温度融合 ===
p(f"\n--- G. Softmax融合: softmax(TOP9/T) × softmax(TOP4/T) ---")

best_G = (0, "", None)
for method_name, method_fn in top4_methods:
    for t1 in [0.5, 1.0, 2.0]:
        for t2 in [0.5, 1.0, 2.0]:
            def make_softmax(fn, temp1, temp2):
                def score_fn(animals):
                    prior = softmax(top9_scores(animals), temp1)
                    likelihood = softmax(fn(animals), temp2)
                    return prior * likelihood
                return score_fn
            
            h, r, mm, g4, g6, roi, _ = test_strategy(make_softmax(method_fn, t1, t2), 4)
            if r > best_G[0]:
                best_G = (r, f"{method_name} T1={t1} T2={t2}", (h, mm, g4, g6, roi))

p(f"  >> G最优: {best_G[1]} → {best_G[0]:.1f}% mm={best_G[2][1]}")

# ====== 总榜: 所有策略最优对比 ======
p(f"\n{'='*100}")
p("总榜: 所有软蒸馏策略最优对比")
p(f"{'='*100}")
p(f"  {'策略':<35} {'命中率':>8} {'mm':>4} {'≥4miss':>6} {'≥6miss':>6} {'ROI':>8} {'vs基线':>8}")
p(f"  {'-'*82}")

all_bests = [
    ("A.乘法融合", best_A),
    ("B.加法融合", best_B),
    ("C.排名软权重", best_C),
    ("D.动态置信", best_D),
    ("E.残差修正", best_E),
    ("F.梯度软屏蔽", best_F),
    ("G.Softmax融合", best_G),
]

overall_best = (0, "", "", None)
for cat, (rate, name, detail) in all_bests:
    diff = rate - baseline_rate
    mark = " ★" if diff > 0 else ""
    h, mm, g4, g6, roi = detail
    label = f"{cat}: {name}"
    p(f"  {label:<35} {rate:>7.1f}% {mm:>3} {g4:>5} {g6:>5} {roi:>+7.1f}% {diff:>+7.1f}%{mark}")
    if rate > overall_best[0]:
        overall_best = (rate, label, cat, detail)

p(f"\n  基线(v3 TOP4):                       {baseline_rate:.1f}%")
p(f"  硬屏蔽(TOP9→TOP4):                   {r3:.1f}%")

# ====== 最优软蒸馏 + 反miss ======
p(f"\n{'='*100}")
p(f"最优软蒸馏 + 反miss机制")
p(f"{'='*100}")

# 根据最优策略重建score函数
# 我们需要找到最优的具体函数...
# 为了通用性, 遍历所有最优的方法+参数组合加反miss

# 找到总最优的策略类型和参数
best_overall_rate = overall_best[0]
p(f"  最优软蒸馏: {overall_best[1]} = {overall_best[0]:.1f}%")

# 对每个类别的最优, 加反miss测试
for cat, (rate, name, detail) in all_bests:
    if rate < baseline_rate - 1:  # 只测试接近或超过基线的
        continue
    
    # 重建该策略的score函数(简化: 用最优参数)
    # 这里我们用通用的方式: 基于最优策略信息来重建
    # 为简化代码, 测试A.乘法和B.加法中的最优
    
for method_name, method_fn in top4_methods:
    # 测试最佳A策略参数
    for alpha in [0.5, 0.8, 1.0]:
        def make_antimiss_multiply(fn, a, blend_t, expand_t):
            class State:
                consec = 0
            state = State()
            def predict(animals):
                prior = normalize(top9_scores(animals)) + 0.01
                likelihood = normalize(fn(animals)) + 0.01
                scores = (prior ** a) * likelihood
                
                if state.consec >= blend_t:
                    hot = hot_scores(animals, 30)
                    scores = 0.75 * scores + 0.25 * normalize(hot)
                return scores, state
            return predict
        
        for bt in [2, 3]:
            for et in [3, 4, 5]:
                if et <= bt: continue
                predict_fn = make_antimiss_multiply(method_fn, alpha, bt, et)
                
                hits = 0
                hit_list = []
                sizes = []
                consec = 0
                
                for pi in range(test_periods):
                    animals, actual_z = all_data[pi]
                    prior = normalize(top9_scores(animals)) + 0.01
                    likelihood = normalize(method_fn(animals)) + 0.01
                    scores = (prior ** alpha) * likelihood
                    
                    if consec >= bt:
                        hot = hot_scores(animals, 30)
                        scores = 0.75 * scores + 0.25 * normalize(hot)
                    
                    n = 5 if consec >= et else 4
                    top = [ZODIAC_CYCLE_2026[i] for i in np.argsort(-scores)[:n]]
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
                
                if rate >= 48:
                    p(f"  A.{method_name} α={alpha} +反miss(b@{bt},e@{et}): {hits}/300={rate:.1f}% mm={mm} avg={avg_n:.2f} ROI={roi:+.1f}%")

# ====== 最终结论 ======
p(f"\n{'='*100}")
p("最终结论")
p(f"{'='*100}")
diff = overall_best[0] - baseline_rate
diff_hard = overall_best[0] - r3
p(f"  v3 直接TOP4:      {baseline_rate:.1f}%")
p(f"  硬屏蔽TOP9→TOP4:  {r3:.1f}%")
p(f"  软蒸馏最优:        {overall_best[0]:.1f}% ({overall_best[1]})")
p(f"  vs 直接v3: {diff:+.1f}%")
p(f"  vs 硬屏蔽: {diff_hard:+.1f}%")

if diff > 1:
    p(f"\n  ✅ 软蒸馏显著提升TOP4! +{diff:.1f}%")
elif diff > 0:
    p(f"\n  ⚠️ 软蒸馏略有提升 +{diff:.1f}%, 但幅度有限")
else:
    p(f"\n  ❌ 软蒸馏未能提升TOP4")
    p(f"  原因: TOP9先验和TOP4似然使用相似特征(冷号+MK), 融合信息增益不足")

with open('soft_distill_top9_top4.txt', 'w', encoding='utf-8-sig') as f:
    f.write(buf.getvalue())
p(f"\n结果已保存到 soft_distill_top9_top4.txt")
