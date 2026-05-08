"""
生肖TOP4优化v5 - 聚焦突破方向
1. 从PreciseTop15预测器的数字预测转化为生肖概率
2. 优化冷号/间隔的评分曲线（非线性）
3. 冷号窗口滑动优化
4. 强化"过期回归"信号
"""
import pandas as pd
import numpy as np
from collections import Counter
import sys, os
sys.path.insert(0, os.getcwd())
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

print(f"数据: {TOTAL}期, 测试{TEST_PERIODS}期")

# ============================================================
# 策略1: 非线性冷号评分
# ============================================================
def score_cold_nonlinear(hist, window, power=1.5):
    """非线性冷号评分: (1 - freq/max_freq)^power"""
    w = min(window, len(hist))
    freq = Counter(hist[-w:])
    mx = max(freq.values()) if freq else 1
    scores = {}
    for z in ZODIAC_CYCLE:
        f = freq.get(z, 0)
        base = 1.0 - f / max(mx, 1)
        scores[z] = base ** power  # 非线性放大冷号差异
    return scores

# ============================================================
# 策略2: 对数间隔评分
# ============================================================
def score_gap_log(hist):
    """对数间隔: log(gap+1)/log(max_gap+1)"""
    gaps = {}
    for z in ZODIAC_CYCLE:
        last = -1
        for j in range(len(hist)-1, -1, -1):
            if hist[j] == z:
                last = j
                break
        gaps[z] = len(hist) - 1 - last if last >= 0 else len(hist)
    
    max_gap = max(gaps.values())
    scores = {}
    for z in ZODIAC_CYCLE:
        scores[z] = np.log1p(gaps[z]) / np.log1p(max(max_gap, 1))
    return scores

# ============================================================
# 策略3: 过期回归评分
# ============================================================
def score_overdue(hist, decay_rate=0.08):
    """
    基于各生肖的平均间隔，计算"过期"程度
    过期越多 = 越可能出现
    """
    scores = {}
    for z in ZODIAC_CYCLE:
        positions = [k for k, a in enumerate(hist) if a == z]
        if len(positions) >= 2:
            intervals = [positions[k+1]-positions[k] for k in range(len(positions)-1)]
            avg_interval = np.mean(intervals)
            gap_now = len(hist) - 1 - positions[-1]
            overdue = gap_now / max(avg_interval, 1)
            # sigmoid: 越过期越高分
            scores[z] = 1.0 / (1.0 + np.exp(-3 * (overdue - 1)))
        elif len(positions) == 1:
            gap = len(hist) - 1 - positions[0]
            scores[z] = min(gap / 12.0, 1.0)
        else:
            scores[z] = 0.8  # 从未出现
    return scores

# ============================================================
# 策略4: 数字级冷热→生肖概率
# ============================================================
def score_number_to_zodiac(numbers_hist, window=30):
    """从49个数字的冷热度，聚合为生肖概率"""
    w = min(window, len(numbers_hist))
    recent = numbers_hist[-w:]
    num_freq = Counter(recent)
    
    # 每个数字的"过期"得分
    num_scores = {}
    for n in range(1, 50):
        f = num_freq.get(n, 0)
        cold = 1.0 - f / max(max(num_freq.values()), 1)
        # 数字间隔
        last = -1
        for j in range(len(numbers_hist)-1, -1, -1):
            if numbers_hist[j] == n:
                last = j
                break
        gap = len(numbers_hist) - 1 - last if last >= 0 else len(numbers_hist)
        num_scores[n] = 0.5 * cold + 0.5 * (np.log1p(gap) / np.log1p(50))
    
    # 聚合到生肖
    zodiac_scores = {}
    for z in ZODIAC_CYCLE:
        nums = ZODIAC_NUMS[z]
        zodiac_scores[z] = sum(num_scores[n] for n in nums) / len(nums)
    
    return zodiac_scores

# ============================================================
# 策略5: 一阶马尔可夫（平滑版）
# ============================================================
def score_markov_smooth(hist, alpha=1.0):
    """Laplace平滑马尔可夫"""
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
        total = sum(trans[state].values()) + alpha * 12
        for z in ZODIAC_CYCLE:
            scores[z] = (trans[state].get(z, 0) + alpha) / total
    return scores

# ============================================================
# 策略6: 反热门（近期出现降权，指数衰减）
# ============================================================
def score_anti_hot_exp(hist, decay_len=6, decay_rate=0.5):
    scores = {z: 1.0 for z in ZODIAC_CYCLE}
    for j in range(min(decay_len, len(hist))):
        z = hist[-(j+1)]
        scores[z] *= (1.0 - decay_rate * np.exp(-j * 0.3))
    return scores

# ============================================================
# 评估框架
# ============================================================
def evaluate_strategy(predict_func, name=""):
    hits = 0
    streak_miss = 0
    max_miss = 0
    seg_hits = [0] * 6  # 6段各50期
    
    for pi in range(TEST_PERIODS):
        i = START + pi
        top4 = predict_func(i)
        actual = animals[i]
        hit = actual in top4
        if hit:
            hits += 1
            streak_miss = 0
        else:
            streak_miss += 1
            max_miss = max(max_miss, streak_miss)
        seg_hits[pi // 50] += 1 if hit else 0
    
    rate = hits / TEST_PERIODS
    seg_str = " | ".join([f"{h}/50={h*2}%" for h in seg_hits])
    return rate, max_miss, seg_str


# ============================================================
# 大规模参数搜索: 6策略权重 + 冷号参数
# ============================================================
print("\n" + "="*70)
print("阶段1: 预计算所有策略得分矩阵")
print("="*70)

# 预计算
N_STRATS = 10
score_mat = np.zeros((N_STRATS, TEST_PERIODS, 12))

for pi in range(TEST_PERIODS):
    i = START + pi
    hist = animals[:i]
    num_hist = numbers[:i]
    
    # 0: 冷号W10 非线性
    s = score_cold_nonlinear(hist, 10, power=1.5)
    for zi, z in enumerate(ZODIAC_CYCLE): score_mat[0, pi, zi] = s[z]
    
    # 1: 冷号W15 非线性
    s = score_cold_nonlinear(hist, 15, power=1.5)
    for zi, z in enumerate(ZODIAC_CYCLE): score_mat[1, pi, zi] = s[z]
    
    # 2: 冷号W20 非线性
    s = score_cold_nonlinear(hist, 20, power=1.5)
    for zi, z in enumerate(ZODIAC_CYCLE): score_mat[2, pi, zi] = s[z]
    
    # 3: 冷号W30 非线性
    s = score_cold_nonlinear(hist, 30, power=1.5)
    for zi, z in enumerate(ZODIAC_CYCLE): score_mat[3, pi, zi] = s[z]
    
    # 4: 对数间隔
    s = score_gap_log(hist)
    for zi, z in enumerate(ZODIAC_CYCLE): score_mat[4, pi, zi] = s[z]
    
    # 5: 过期回归
    s = score_overdue(hist)
    for zi, z in enumerate(ZODIAC_CYCLE): score_mat[5, pi, zi] = s[z]
    
    # 6: 数字→生肖
    s = score_number_to_zodiac(num_hist, window=30)
    for zi, z in enumerate(ZODIAC_CYCLE): score_mat[6, pi, zi] = s[z]
    
    # 7: 马尔可夫平滑
    s = score_markov_smooth(hist)
    for zi, z in enumerate(ZODIAC_CYCLE): score_mat[7, pi, zi] = s[z]
    
    # 8: 反热门指数
    s = score_anti_hot_exp(hist)
    for zi, z in enumerate(ZODIAC_CYCLE): score_mat[8, pi, zi] = s[z]
    
    # 9: 数字→生肖(窗口50)
    s = score_number_to_zodiac(num_hist, window=50)
    for zi, z in enumerate(ZODIAC_CYCLE): score_mat[9, pi, zi] = s[z]

actual_idx = np.array([Z_IDX[animals[START + pi]] for pi in range(TEST_PERIODS)])
print(f"预计算完成: {time.time()-t0:.1f}秒")

# 单策略基线
strat_names = ['冷号NL10', '冷号NL15', '冷号NL20', '冷号NL30', '对数间隔', 
               '过期回归', '数字→肖30', '马尔可夫S', '反热门E', '数字→肖50']

print("\n单策略基线:")
for si in range(N_STRATS):
    hits = 0
    for pi in range(TEST_PERIODS):
        top4 = np.argsort(-score_mat[si, pi])[:4]
        if actual_idx[pi] in top4:
            hits += 1
    print(f"  {strat_names[si]:>10}: {hits}/{TEST_PERIODS} = {hits/TEST_PERIODS*100:.1f}%")

# ============================================================
# 快速评估
# ============================================================
def fast_eval(weights):
    w = np.array(weights).reshape(-1, 1, 1)
    combined = (score_mat * w).sum(axis=0)
    hits = 0
    for pi in range(TEST_PERIODS):
        top4 = np.argsort(-combined[pi])[:4]
        if actual_idx[pi] in top4:
            hits += 1
    return hits / TEST_PERIODS

# ============================================================
# 阶段2: 大规模权重搜索
# ============================================================
print("\n" + "="*70)
print("阶段2: 10维权重搜索")
print("="*70)

t1 = time.time()
best_rate = 0
best_w = None
count = 0

# 分阶段搜索: 先搜Top6策略
vals = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
# 关键策略索引: 冷号NL15(1), 冷号NL20(2), 对数间隔(4), 过期回归(5), 数字→肖30(6), 马尔可夫S(7)
key_indices = [1, 2, 4, 5, 6, 7]

for w1 in vals:
    for w2 in vals:
        for w4 in vals:
            for w5 in [0, 0.1, 0.2, 0.3]:
                for w6 in [0, 0.1, 0.2, 0.3]:
                    for w7 in [0, 0.1, 0.2, 0.3]:
                        w = [0]*10
                        w[1] = w1; w[2] = w2; w[4] = w4
                        w[5] = w5; w[6] = w6; w[7] = w7
                        ws = sum(w)
                        if ws == 0: continue
                        w = [x/ws for x in w]
                        rate = fast_eval(w)
                        count += 1
                        if rate > best_rate:
                            best_rate = rate
                            best_w = w[:]
                            if rate > 0.43:
                                active = [(strat_names[i], f"{w[i]:.3f}") for i in range(10) if w[i] > 0.001]
                                print(f"  #{count} {rate*100:.1f}% {active}")

print(f"\n阶段2: {count}组合, {time.time()-t1:.1f}s, 最优{best_rate*100:.1f}%")

# 阶段2b: 加入其余策略微调
print("\n阶段2b: 加入冷号NL10/NL30, 反热门, 数字→肖50")
t2 = time.time()
base = best_w[:]
for w0 in [0, 0.05, 0.1]:
    for w3 in [0, 0.05, 0.1]:
        for w8 in [0, 0.05, 0.1]:
            for w9 in [0, 0.05, 0.1]:
                w = base[:]
                w[0] = w0; w[3] = w3; w[8] = w8; w[9] = w9
                ws = sum(w)
                if ws == 0: continue
                w = [x/ws for x in w]
                rate = fast_eval(w)
                count += 1
                if rate > best_rate:
                    best_rate = rate
                    best_w = w[:]
                    active = [(strat_names[i], f"{w[i]:.3f}") for i in range(10) if w[i] > 0.001]
                    print(f"  新最优 {rate*100:.1f}% {active}")

print(f"阶段2b: {time.time()-t2:.1f}s, 最优{best_rate*100:.1f}%")

# ============================================================
# 阶段3: 精细搜索
# ============================================================
print("\n" + "="*70)
print("阶段3: 精细搜索 (围绕最优±0.025)")
print("="*70)
t3 = time.time()
base = best_w[:]
active_idx = [i for i in range(10) if base[i] > 0.005]
print(f"活跃维度: {[strat_names[i] for i in active_idx]}")

from itertools import product as iprod
deltas = [-0.025, -0.01, 0, 0.01, 0.025]
fine_count = 0
for combo in iprod(deltas, repeat=min(len(active_idx), 6)):
    w = base[:]
    for j, idx in enumerate(active_idx[:6]):
        w[idx] = max(0, base[idx] + combo[j])
    ws = sum(w)
    if ws == 0: continue
    w = [x/ws for x in w]
    rate = fast_eval(w)
    fine_count += 1
    if rate > best_rate:
        best_rate = rate
        best_w = w[:]
        active = [(strat_names[i], f"{w[i]:.4f}") for i in range(10) if w[i] > 0.001]
        print(f"  #{fine_count} 新最优 {rate*100:.1f}% {active}")

print(f"精细搜索: {fine_count}组合, {time.time()-t3:.1f}s")

# ============================================================
# 最终结果
# ============================================================
print("\n" + "="*70)
print("最终结果")
print("="*70)
print(f"最优命中率: {best_rate*100:.1f}%  ({int(best_rate*TEST_PERIODS)}/{TEST_PERIODS})")
print(f"权重:")
for i in range(10):
    if best_w[i] > 0.001:
        print(f"  {strat_names[i]:>12}: {best_w[i]:.4f}")

# 分段统计
w_arr = np.array(best_w).reshape(-1, 1, 1)
combined = (score_mat * w_arr).sum(axis=0)
seg_hits = []
for seg in range(0, TEST_PERIODS, 50):
    end = min(seg + 50, TEST_PERIODS)
    h = 0
    for pi in range(seg, end):
        if actual_idx[pi] in np.argsort(-combined[pi])[:4]:
            h += 1
    seg_hits.append(h)
    print(f"  {seg+1:>3}-{end:>3}: {h}/50 = {h/50*100:.1f}%")

# 连续不中统计
streak = 0
max_streak = 0
for pi in range(TEST_PERIODS):
    if actual_idx[pi] in np.argsort(-combined[pi])[:4]:
        streak = 0
    else:
        streak += 1
        max_streak = max(max_streak, streak)
print(f"最长连续不中: {max_streak}期")

print(f"\n总耗时: {time.time()-t0:.1f}秒")
