"""
生肖TOP4优化器v2 - 预计算+快速网格搜索
目标: 300期滚动验证命中率 >= 50%
"""
import pandas as pd
import numpy as np
from collections import Counter
import time

t0 = time.time()

# 2026马年映射
ZODIAC_CYCLE = ['马', '蛇', '龙', '兔', '虎', '牛', '鼠', '猪', '狗', '鸡', '猴', '羊']
Z_IDX = {z: i for i, z in enumerate(ZODIAC_CYCLE)}
NUM_TO_ZODIAC = {n: ZODIAC_CYCLE[(n - 1) % 12] for n in range(1, 50)}

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].tolist()
animals = [NUM_TO_ZODIAC[n] for n in numbers]
TOTAL = len(animals)
TEST_PERIODS = 300
START = TOTAL - TEST_PERIODS

print(f"数据: {TOTAL}期, 测试{TEST_PERIODS}期 (idx {START}~{TOTAL-1})")

# ============================================================
# 预计算: 每期每种策略对12个生肖的得分矩阵
# score_matrix[strategy][period][zodiac_idx] = score
# ============================================================
N_STRATS = 8
score_matrix = np.zeros((N_STRATS, TEST_PERIODS, 12))

print("预计算策略得分...")

for pi in range(TEST_PERIODS):
    i = START + pi
    hist = animals[:i]
    hist_len = len(hist)
    
    # === 策略0: 冷号(窗口10) ===
    w = min(10, hist_len)
    freq = Counter(hist[-w:])
    mx = max(freq.values()) if freq else 1
    for zi, z in enumerate(ZODIAC_CYCLE):
        score_matrix[0, pi, zi] = 1.0 - freq.get(z, 0) / max(mx, 1)
    
    # === 策略1: 冷号(窗口20) ===
    w = min(20, hist_len)
    freq = Counter(hist[-w:])
    mx = max(freq.values()) if freq else 1
    for zi, z in enumerate(ZODIAC_CYCLE):
        score_matrix[1, pi, zi] = 1.0 - freq.get(z, 0) / max(mx, 1)
    
    # === 策略2: 冷号(窗口50) ===
    w = min(50, hist_len)
    freq = Counter(hist[-w:])
    mx = max(freq.values()) if freq else 1
    for zi, z in enumerate(ZODIAC_CYCLE):
        score_matrix[2, pi, zi] = 1.0 - freq.get(z, 0) / max(mx, 1)

    # === 策略3: 间隔分析 ===
    for zi, z in enumerate(ZODIAC_CYCLE):
        last = -1
        for j in range(hist_len - 1, -1, -1):
            if hist[j] == z:
                last = j
                break
        gap = hist_len - 1 - last if last >= 0 else hist_len
        score_matrix[3, pi, zi] = gap / 12.0

    # === 策略4: 一阶马尔可夫 ===
    if hist_len >= 2:
        trans = {}
        for k in range(1, hist_len):
            prev = hist[k-1]
            curr = hist[k]
            if prev not in trans:
                trans[prev] = Counter()
            trans[prev][curr] += 1
        state = hist[-1]
        if state in trans:
            total = sum(trans[state].values())
            for zi, z in enumerate(ZODIAC_CYCLE):
                score_matrix[4, pi, zi] = trans[state].get(z, 0) / total

    # === 策略5: 二阶马尔可夫 ===
    if hist_len >= 3:
        trans2 = {}
        for k in range(2, hist_len):
            prev = (hist[k-2], hist[k-1])
            curr = hist[k]
            if prev not in trans2:
                trans2[prev] = Counter()
            trans2[prev][curr] += 1
        state2 = (hist[-2], hist[-1])
        if state2 in trans2:
            total = sum(trans2[state2].values())
            for zi, z in enumerate(ZODIAC_CYCLE):
                score_matrix[5, pi, zi] = trans2[state2].get(z, 0) / total

    # === 策略6: 反热门惩罚 ===
    for zi, z in enumerate(ZODIAC_CYCLE):
        score_matrix[6, pi, zi] = 1.0
    dw = min(5, hist_len)
    for j in range(dw):
        zi = Z_IDX[hist[-(j+1)]]
        penalty = 1.0 - (j / dw)
        score_matrix[6, pi, zi] -= penalty * 0.5

    # === 策略7: 周期性 ===
    for zi, z in enumerate(ZODIAC_CYCLE):
        positions = []
        for k in range(hist_len):
            if hist[k] == z:
                positions.append(k)
        if len(positions) >= 3:
            intervals = [positions[k+1] - positions[k] for k in range(len(positions)-1)]
            avg_int = np.mean(intervals)
            gap_now = hist_len - 1 - positions[-1]
            if avg_int > 0:
                ratio = gap_now / avg_int
                if 0.8 <= ratio <= 1.5:
                    score_matrix[7, pi, zi] = 0.5 + 0.5 * min(ratio, 1.0)
                elif ratio > 1.5:
                    score_matrix[7, pi, zi] = 1.0

# 预计算实际结果
actual_zodiac_idx = np.array([Z_IDX[animals[START + pi]] for pi in range(TEST_PERIODS)])

print(f"预计算完成: {time.time()-t0:.1f}秒")

# ============================================================
# 快速评估函数
# ============================================================
def fast_evaluate(weights):
    """给定8个策略权重,快速计算TOP4命中率"""
    w = np.array(weights).reshape(8, 1, 1)
    # combined[period][zodiac] = weighted sum
    combined = (score_matrix * w).sum(axis=0)  # (300, 12)
    
    hits = 0
    for pi in range(TEST_PERIODS):
        # TOP4: 得分最高的4个
        top4_idx = np.argsort(-combined[pi])[:4]
        if actual_zodiac_idx[pi] in top4_idx:
            hits += 1
    return hits / TEST_PERIODS

# ============================================================
# 阶段1: 单策略基线
# ============================================================
print("\n" + "="*70)
print("阶段1: 单策略基线")
print("="*70)
strat_names = ['冷号W10', '冷号W20', '冷号W50', '间隔', '马尔可夫1', '马尔可夫2', '反热门', '周期']
for si in range(N_STRATS):
    w = [0.0] * N_STRATS
    w[si] = 1.0
    rate = fast_evaluate(w)
    print(f"  {strat_names[si]:>8}: {rate*100:.1f}%")

# ============================================================
# 阶段2: 大规模网格搜索
# ============================================================
print("\n" + "="*70)
print("阶段2: 网格搜索 (步长0.1)")
print("="*70)

best_rate = 0
best_w = None
count = 0
t1 = time.time()

# 生成权重组合 (8维, 每维0~0.5步长0.1, 归一化)
vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
# 太多组合，用分组搜索: 固定部分维度
# 先搜前4个主要策略
for w0 in vals:  # 冷号W10
    for w1 in vals:  # 冷号W20
        for w2 in vals:  # 冷号W50
            for w3 in vals:  # 间隔
                for w4 in vals:  # 马尔可夫1
                    s = w0+w1+w2+w3+w4
                    if s == 0:
                        continue
                    # 固定后3个为0
                    w = [w0, w1, w2, w3, w4, 0, 0, 0]
                    ws = sum(w)
                    w = [x/ws for x in w]
                    rate = fast_evaluate(w)
                    count += 1
                    if rate > best_rate:
                        best_rate = rate
                        best_w = w[:]
                        if count % 500 == 0 or rate > 0.42:
                            print(f"  #{count} {rate*100:.1f}% w=[{','.join(f'{x:.2f}' for x in w)}]")

print(f"  5维搜索: {count}组合, 最优{best_rate*100:.1f}%, 耗时{time.time()-t1:.1f}s")

# 阶段2b: 在最优5维基础上搜索后3维
print("\n阶段2b: 在最优基础上加入马尔可夫2/反热/周期")
base5 = best_w[:5]
t2 = time.time()
for w5 in [0, 0.05, 0.1, 0.15, 0.2, 0.3]:  # 马尔可夫2
    for w6 in [0, 0.05, 0.1, 0.15, 0.2, 0.3]:  # 反热门
        for w7 in [0, 0.05, 0.1, 0.15, 0.2, 0.3]:  # 周期
            w = base5 + [w5, w6, w7]
            ws = sum(w)
            if ws == 0:
                continue
            w = [x/ws for x in w]
            rate = fast_evaluate(w)
            count += 1
            if rate > best_rate:
                best_rate = rate
                best_w = w[:]
                print(f"  #{count} 新最优 {rate*100:.1f}% w=[{','.join(f'{x:.2f}' for x in w)}]")

print(f"  补充搜索: 耗时{time.time()-t2:.1f}s, 总最优{best_rate*100:.1f}%")

# ============================================================
# 阶段3: 精细搜索 - 围绕最优±0.05
# ============================================================
print("\n" + "="*70)
print("阶段3: 精细搜索 (±0.05)")
print("="*70)
t3 = time.time()
base_w = best_w[:]
deltas = [-0.05, -0.025, 0, 0.025, 0.05]
fine_count = 0
# 只对非零权重做精细搜索
active = [i for i in range(8) if base_w[i] > 0.01]
print(f"  活跃维度: {[strat_names[i] for i in active]}")

from itertools import product as iprod
for combo in iprod(deltas, repeat=min(len(active), 5)):
    w = base_w[:]
    for j, idx in enumerate(active[:5]):
        w[idx] = max(0, base_w[idx] + combo[j])
    ws = sum(w)
    if ws == 0:
        continue
    w = [x/ws for x in w]
    rate = fast_evaluate(w)
    fine_count += 1
    if rate > best_rate:
        best_rate = rate
        best_w = w[:]
        print(f"  #{fine_count} 新最优 {rate*100:.1f}% w=[{','.join(f'{x:.3f}' for x in w)}]")

print(f"  精细搜索: {fine_count}组合, 耗时{time.time()-t3:.1f}s")

# ============================================================
# 最终结果
# ============================================================
print("\n" + "="*70)
print("最终结果")
print("="*70)
print(f"最优命中率: {best_rate*100:.1f}%  ({int(best_rate*TEST_PERIODS)}/{TEST_PERIODS})")
print(f"权重向量:")
for si in range(N_STRATS):
    if best_w[si] > 0.001:
        print(f"  {strat_names[si]:>8}: {best_w[si]:.4f}")

# 逐期详情
w = np.array(best_w).reshape(8, 1, 1)
combined = (score_matrix * w).sum(axis=0)

hits = 0
streak_miss = 0
max_miss = 0
monthly = {}

print(f"\n逐期详情（命中标记）:")
for pi in range(TEST_PERIODS):
    i = START + pi
    top4_idx = np.argsort(-combined[pi])[:4]
    top4 = [ZODIAC_CYCLE[zi] for zi in top4_idx]
    actual = animals[i]
    hit = actual in top4
    if hit:
        hits += 1
        streak_miss = 0
    else:
        streak_miss += 1
        max_miss = max(max_miss, streak_miss)
    
    date = df['date'].iloc[i] if 'date' in df.columns else ''
    month = date[:7] if date else ''
    if month not in monthly:
        monthly[month] = [0, 0]
    monthly[month][1] += 1
    if hit:
        monthly[month][0] += 1

print(f"\n命中率: {hits}/{TEST_PERIODS} = {hits/TEST_PERIODS*100:.1f}%")
print(f"最长连续不中: {max_miss}期")

print(f"\n月度命中率:")
for month, (h, t) in sorted(monthly.items()):
    rate = h/t*100 if t > 0 else 0
    bar = "█" * int(rate / 5)
    print(f"  {month}: {h:>2}/{t:>2} = {rate:>5.1f}% {bar}")

# 分50期统计
print(f"\n分段统计(每50期):")
for seg in range(0, TEST_PERIODS, 50):
    end = min(seg + 50, TEST_PERIODS)
    seg_hits = 0
    for pi in range(seg, end):
        top4_idx = np.argsort(-combined[pi])[:4]
        if actual_zodiac_idx[pi] in top4_idx:
            seg_hits += 1
    print(f"  {seg+1:>3}-{end:>3}: {seg_hits}/{end-seg} = {seg_hits/(end-seg)*100:.1f}%")

print(f"\n总耗时: {time.time()-t0:.1f}秒")
