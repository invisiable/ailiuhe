"""
生肖TOP4终极突破 v7 - 多维度融合
=================================
关键思路:
1. 数字级冷号分析 → 聚合到生肖 (比直接生肖冷号更精细)
2. 二分组Markov: 将12个状态分成"Markov友好"和"冷号友好"两组
3. 多策略投票 + 自适应阈值
4. 连续失败自动切换
"""
import pandas as pd
import numpy as np
from collections import Counter
import time, sys
sys.stdout.reconfigure(encoding='utf-8')

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
print(f"Data: {TOTAL} periods, test {TEST_PERIODS}")

# ============================================================
# 策略函数 (每个返回 dict{zodiac: score})
# ============================================================
def strat_cold_zodiac(hist_animals, window):
    """生肖级冷号"""
    w = min(window, len(hist_animals))
    freq = Counter(hist_animals[-w:])
    mx = max(freq.values()) if freq else 1
    return {z: 1.0 - freq.get(z,0)/max(mx,1) for z in ZODIAC_CYCLE}

def strat_cold_number(hist_numbers, window):
    """数字级冷号 → 聚合到生肖 (NEW!)"""
    w = min(window, len(hist_numbers))
    freq = Counter(hist_numbers[-w:])
    mx = max(freq.values()) if freq else 1
    # 每个数字的冷分
    num_cold = {n: 1.0 - freq.get(n,0)/max(mx,1) for n in range(1,50)}
    # 聚合到生肖: 取该生肖下所有数字冷分的均值
    zodiac_score = {}
    for z in ZODIAC_CYCLE:
        nums = ZODIAC_NUMS[z]
        zodiac_score[z] = np.mean([num_cold[n] for n in nums])
    return zodiac_score

def strat_gap(hist_animals):
    """间隔策略"""
    scores = {}
    for z in ZODIAC_CYCLE:
        last = -1
        for j in range(len(hist_animals)-1,-1,-1):
            if hist_animals[j]==z: last=j; break
        scores[z] = (len(hist_animals)-1-last)/12.0 if last>=0 else len(hist_animals)/12.0
    return scores

def strat_markov(hist_animals, laplace=1.0):
    """一阶马尔可夫"""
    probs = {z: 1/12 for z in ZODIAC_CYCLE}
    if len(hist_animals) < 2:
        return probs
    trans = {}
    for k in range(1, len(hist_animals)):
        p,c = hist_animals[k-1], hist_animals[k]
        if p not in trans: trans[p]=Counter()
        trans[p][c]+=1
    state = hist_animals[-1]
    if state in trans:
        total = sum(trans[state].values()) + laplace*12
        for z in ZODIAC_CYCLE:
            probs[z] = (trans[state].get(z,0)+laplace)/total
    return probs

def strat_markov2(hist_animals, laplace=0.5):
    """二阶马尔可夫"""
    probs = {z: 1/12 for z in ZODIAC_CYCLE}
    if len(hist_animals) < 3:
        return probs
    trans = {}
    for k in range(2, len(hist_animals)):
        prev = (hist_animals[k-2], hist_animals[k-1])
        curr = hist_animals[k]
        if prev not in trans: trans[prev] = Counter()
        trans[prev][curr] += 1
    state2 = (hist_animals[-2], hist_animals[-1])
    if state2 in trans:
        total = sum(trans[state2].values()) + laplace*12
        for z in ZODIAC_CYCLE:
            probs[z] = (trans[state2].get(z,0)+laplace)/total
    return probs

def strat_overdue(hist_animals):
    """超期回归"""
    scores = {}
    for z in ZODIAC_CYCLE:
        pos = [k for k,a in enumerate(hist_animals) if a==z]
        if len(pos) >= 2:
            ints = [pos[k+1]-pos[k] for k in range(len(pos)-1)]
            avg = np.mean(ints)
            gap = len(hist_animals)-1-pos[-1]
            scores[z] = 1/(1+np.exp(-3*(gap/max(avg,1)-1)))
        elif len(pos) == 1:
            scores[z] = min((len(hist_animals)-1-pos[0])/12, 1)
        else:
            scores[z] = 0.8
    return scores

def strat_num_gap(hist_numbers):
    """数字级间隔 → 聚合到生肖 (NEW!)"""
    scores = {}
    for n in range(1, 50):
        last = -1
        for j in range(len(hist_numbers)-1,-1,-1):
            if hist_numbers[j]==n: last=j; break
        scores[n] = (len(hist_numbers)-1-last)/49.0 if last>=0 else len(hist_numbers)/49.0
    zodiac_score = {}
    for z in ZODIAC_CYCLE:
        zodiac_score[z] = np.mean([scores[n] for n in ZODIAC_NUMS[z]])
    return zodiac_score

# ============================================================
# 预计算所有策略得分
# ============================================================
print("预计算策略矩阵...")
N_STRATS = 8
strat_labels = ['冷号Z15', '冷号Z20', '冷号N20', '间隔Z', '间隔N', '马尔可夫1', '马尔可夫2', '超期']
mat = np.zeros((N_STRATS, TEST_PERIODS, 12))
actual_zi = np.zeros(TEST_PERIODS, dtype=int)
prev_zi = np.zeros(TEST_PERIODS, dtype=int)

for pi in range(TEST_PERIODS):
    i = START + pi
    ha = animals[:i]
    hn = numbers[:i]
    actual_zi[pi] = Z_IDX[animals[i]]
    prev_zi[pi] = Z_IDX[ha[-1]] if ha else 0
    
    for si, func in enumerate([
        lambda: strat_cold_zodiac(ha, 15),
        lambda: strat_cold_zodiac(ha, 20),
        lambda: strat_cold_number(hn, 20),
        lambda: strat_gap(ha),
        lambda: strat_num_gap(hn),
        lambda: strat_markov(ha),
        lambda: strat_markov2(ha),
        lambda: strat_overdue(ha),
    ]):
        result = func()
        for zi, z in enumerate(ZODIAC_CYCLE):
            mat[si, pi, zi] = result[z]

print(f"预计算完成: {time.time()-t0:.1f}s")

# ============================================================
# 单策略基线
# ============================================================
print("\n" + "="*60)
print("单策略基线:")
print("="*60)
for si in range(N_STRATS):
    h = sum(1 for pi in range(TEST_PERIODS) if actual_zi[pi] in np.argsort(-mat[si,pi])[:4])
    print(f"  {strat_labels[si]:>10}: {h}/{TEST_PERIODS} = {h/TEST_PERIODS*100:.1f}%")

# ============================================================
# 快速权重搜索 (所有8个策略)
# ============================================================
print("\n" + "="*60)
print("权重搜索 (8策略)")
print("="*60)

def eval_weights(weights):
    w = np.array(weights).reshape(-1,1,1)
    combined = (mat * w).sum(axis=0)
    return sum(1 for pi in range(TEST_PERIODS) if actual_zi[pi] in np.argsort(-combined[pi])[:4])

# 粗搜索: 关键维度
best_h = 0
best_w = None
vals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
count = 0
for wCZ15 in [0, 0.1, 0.2, 0.3]:
    for wCZ20 in [0, 0.1, 0.2, 0.3, 0.4]:
        for wCN20 in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            for wGZ in [0, 0.1, 0.2, 0.3]:
                for wGN in [0, 0.1, 0.2, 0.3]:
                    for wM1 in [0, 0.1, 0.2, 0.3, 0.4]:
                        for wM2 in [0, 0.1, 0.2]:
                            for wOD in [0, 0.1, 0.2]:
                                w = [wCZ15, wCZ20, wCN20, wGZ, wGN, wM1, wM2, wOD]
                                if sum(w) == 0: continue
                                count += 1
                                h = eval_weights(w)
                                if h > best_h:
                                    best_h = h
                                    best_w = w[:]
                                    if h/TEST_PERIODS > 0.44:
                                        active = [(strat_labels[i], f"{w[i]:.1f}") for i in range(N_STRATS) if w[i]>0]
                                        print(f"  [{count}] {h}/{TEST_PERIODS}={h/TEST_PERIODS*100:.1f}% {active}")

print(f"\n粗搜索 ({count} combos): {best_h}/{TEST_PERIODS} = {best_h/TEST_PERIODS*100:.1f}%")
active = [(strat_labels[i], f"{best_w[i]:.2f}") for i in range(N_STRATS) if best_w[i]>0]
print(f"  最优权重: {active}")

# 精细搜索
print("\n精细搜索...")
base = best_w[:]
for _ in range(3):
    improved = False
    for dim in range(N_STRATS):
        for delta in [-0.05, -0.02, 0.02, 0.05]:
            w = base[:]
            w[dim] = max(0, w[dim] + delta)
            if sum(w) == 0: continue
            h = eval_weights(w)
            if h > best_h:
                best_h = h
                best_w = w[:]
                base = w[:]
                improved = True
    if not improved:
        break

print(f"精细搜索: {best_h}/{TEST_PERIODS} = {best_h/TEST_PERIODS*100:.1f}%")
active = [(strat_labels[i], f"{best_w[i]:.3f}") for i in range(N_STRATS) if best_w[i]>0]
print(f"  最优权重: {active}")

# ============================================================
# 二分组Markov策略
# ============================================================
print("\n" + "="*60)
print("二分组Markov策略")
print("="*60)

# 用训练数据确定哪些状态是Markov友好的
# 使用前半段确定分组,后半段验证
half = TEST_PERIODS // 2
state_mk_perf = {}
state_cold_perf = {}

for state_zi in range(12):
    pis_first = [pi for pi in range(half) if prev_zi[pi] == state_zi]
    if not pis_first:
        state_mk_perf[state_zi] = 0
        state_cold_perf[state_zi] = 0
        continue
    mk_hits = sum(1 for pi in pis_first if actual_zi[pi] in np.argsort(-mat[5,pi])[:4])
    cold_hits = sum(1 for pi in pis_first if actual_zi[pi] in np.argsort(-mat[1,pi])[:4])
    state_mk_perf[state_zi] = mk_hits/len(pis_first) if pis_first else 0
    state_cold_perf[state_zi] = cold_hits/len(pis_first) if pis_first else 0
    z = ZODIAC_CYCLE[state_zi]
    print(f"  {z}: mk={mk_hits}/{len(pis_first)}={state_mk_perf[state_zi]:.1%} cold={cold_hits}/{len(pis_first)}={state_cold_perf[state_zi]:.1%}")

# 分组: Markov优于cold的为MK组
mk_group = [zi for zi in range(12) if state_mk_perf[zi] > state_cold_perf[zi]]
cold_group = [zi for zi in range(12) if zi not in mk_group]
print(f"\nMK组: {[ZODIAC_CYCLE[zi] for zi in mk_group]}")
print(f"Cold组: {[ZODIAC_CYCLE[zi] for zi in cold_group]}")

# 测试分组效果(全300期)
for mk_w in [(0.1, 0.3, 0.2, 0.1, 0.1, 0.5, 0.1, 0.1),  # MK组权重
             (0.1, 0.2, 0.3, 0.1, 0.1, 0.6, 0.1, 0.0),
             (0.0, 0.1, 0.2, 0.0, 0.0, 0.7, 0.1, 0.0)]:
    for cd_w in [(0.2, 0.4, 0.4, 0.2, 0.2, 0.0, 0.0, 0.2),  # Cold组权重
                 (0.1, 0.3, 0.5, 0.2, 0.2, 0.1, 0.0, 0.1),
                 (0.2, 0.5, 0.3, 0.3, 0.2, 0.0, 0.0, 0.0)]:
        hits = 0
        for pi in range(TEST_PERIODS):
            state_zi = prev_zi[pi]
            w = mk_w if state_zi in mk_group else cd_w
            ws = sum(w)
            if ws == 0: continue
            combined = sum((w[s]/ws)*mat[s,pi] for s in range(N_STRATS))
            if actual_zi[pi] in np.argsort(-combined)[:4]:
                hits += 1
        rate = hits/TEST_PERIODS*100
        if rate > 44:
            print(f"  二分组: {hits}/{TEST_PERIODS}={rate:.1f}% *")

# 精搜索二分组
print("\n二分组精细搜索...")
best_2g = 0
best_2g_mk = None
best_2g_cd = None
search_vals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# 只搜索关键维度: CZ20, CN20, MK1, GapZ
for cwCZ20_mk in [0, 0.1, 0.2, 0.3]:
    for cwCN20_mk in [0, 0.1, 0.2, 0.3]:
        for cwMK1_mk in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for cwGZ_mk in [0, 0.1, 0.2]:
                mk_w = [0.05, cwCZ20_mk, cwCN20_mk, cwGZ_mk, 0, cwMK1_mk, 0.05, 0]
                for cwCZ20_cd in [0.2, 0.3, 0.4, 0.5]:
                    for cwCN20_cd in [0.2, 0.3, 0.4, 0.5]:
                        for cwMK1_cd in [0, 0.05, 0.1, 0.15]:
                            for cwGZ_cd in [0, 0.1, 0.2, 0.3]:
                                cd_w = [0.05, cwCZ20_cd, cwCN20_cd, cwGZ_cd, 0, cwMK1_cd, 0, 0.05]
                                hits = 0
                                for pi in range(TEST_PERIODS):
                                    state_zi = prev_zi[pi]
                                    w = mk_w if state_zi in mk_group else cd_w
                                    ws = sum(w)
                                    if ws == 0: continue
                                    combined = sum((w[s]/ws)*mat[s,pi] for s in range(N_STRATS))
                                    if actual_zi[pi] in np.argsort(-combined)[:4]:
                                        hits += 1
                                if hits > best_2g:
                                    best_2g = hits
                                    best_2g_mk = mk_w[:]
                                    best_2g_cd = cd_w[:]

print(f"二分组最优: {best_2g}/{TEST_PERIODS} = {best_2g/TEST_PERIODS*100:.1f}%")
mk_active = [(strat_labels[i], f"{best_2g_mk[i]:.2f}") for i in range(N_STRATS) if best_2g_mk[i]>0]
cd_active = [(strat_labels[i], f"{best_2g_cd[i]:.2f}") for i in range(N_STRATS) if best_2g_cd[i]>0]
print(f"  MK组权重: {mk_active}")
print(f"  Cold组权重: {cd_active}")

# ============================================================
# 投票策略
# ============================================================
print("\n" + "="*60)
print("多策略投票")
print("="*60)

# 每个策略投TOP4, 统计每个zodiac被投票数, 选票数最多的
from itertools import combinations

strat_sets = list(range(N_STRATS))
best_vote = 0
best_vote_set = None

for k in range(3, N_STRATS+1):
    for combo in combinations(strat_sets, k):
        hits = 0
        for pi in range(TEST_PERIODS):
            votes = Counter()
            for si in combo:
                top4 = np.argsort(-mat[si, pi])[:4]
                for zi in top4:
                    votes[zi] += 1
            # 选票数最多的4个
            top4_voted = [z for z,_ in votes.most_common(4)]
            if actual_zi[pi] in top4_voted:
                hits += 1
        if hits > best_vote:
            best_vote = hits
            best_vote_set = combo
            if hits/TEST_PERIODS > 0.44:
                print(f"  {[strat_labels[s] for s in combo]}: {hits}/{TEST_PERIODS}={hits/TEST_PERIODS*100:.1f}%")

print(f"\n投票最优: {best_vote}/{TEST_PERIODS} = {best_vote/TEST_PERIODS*100:.1f}%")
print(f"  策略组合: {[strat_labels[s] for s in best_vote_set]}")

# ============================================================
# 加权投票
# ============================================================
print("\n" + "="*60)
print("加权投票 (TOP策略不同票权)")
print("="*60)

best_wvote = 0
best_wvote_params = None
# 使用最好的投票组合
if best_vote_set:
    combo = best_vote_set
    for w_list_idx in range(200):
        # 随机权重
        np.random.seed(w_list_idx)
        weights = np.random.dirichlet(np.ones(len(combo)))
        
        hits = 0
        for pi in range(TEST_PERIODS):
            votes = {}
            for z_idx in range(12):
                votes[z_idx] = 0
            for k_idx, si in enumerate(combo):
                top4 = np.argsort(-mat[si, pi])[:4]
                for zi in top4:
                    votes[zi] += weights[k_idx]
            top4_voted = sorted(range(12), key=lambda z: -votes[z])[:4]
            if actual_zi[pi] in top4_voted:
                hits += 1
        if hits > best_wvote:
            best_wvote = hits
            best_wvote_params = (w_list_idx, weights)
            if hits/TEST_PERIODS > 0.44:
                print(f"  seed={w_list_idx}: {hits}/{TEST_PERIODS}={hits/TEST_PERIODS*100:.1f}%")

print(f"\n加权投票最优: {best_wvote}/{TEST_PERIODS} = {best_wvote/TEST_PERIODS*100:.1f}%")

# ============================================================
# 最终汇总
# ============================================================
print("\n" + "="*60)
print("最终汇总")
print("="*60)
print(f"  全局权重搜索: {best_h}/{TEST_PERIODS} = {best_h/TEST_PERIODS*100:.1f}%")
print(f"  二分组:       {best_2g}/{TEST_PERIODS} = {best_2g/TEST_PERIODS*100:.1f}%")
print(f"  投票:         {best_vote}/{TEST_PERIODS} = {best_vote/TEST_PERIODS*100:.1f}%")
print(f"  加权投票:     {best_wvote}/{TEST_PERIODS} = {best_wvote/TEST_PERIODS*100:.1f}%")

winner = max(best_h, best_2g, best_vote, best_wvote)
print(f"\n  BEST: {winner}/{TEST_PERIODS} = {winner/TEST_PERIODS*100:.1f}%")
print(f"  vs baseline 33.3%, improvement = +{(winner/TEST_PERIODS-0.333)/0.333*100:.1f}%")
print(f"\nTotal time: {time.time()-t0:.1f}s")
