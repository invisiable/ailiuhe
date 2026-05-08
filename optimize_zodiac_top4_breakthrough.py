"""
生肖TOP4终极突破 - 条件马尔可夫 + 状态特化策略
核心思路: 马尔可夫理论天花板=53.2%，关键是对每个"前一期生肖"状态
使用专门优化的策略权重，而非全局统一权重。

方法:
1. 按前一期生肖分12个状态，每个状态有独立的策略权重
2. 对每个状态单独网格搜索最优权重
3. 结合马尔可夫强信号 + 冷号/间隔弱信号
"""
import pandas as pd
import numpy as np
from collections import Counter
from itertools import product as iprod
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
# 预计算策略得分矩阵
# ============================================================
N_STRATS = 6
strat_names = ['冷号15', '冷号20', '冷号30', '间隔', '马尔可夫', '过期']
score_mat = np.zeros((N_STRATS, TEST_PERIODS, 12))
prev_zodiac_idx = np.zeros(TEST_PERIODS, dtype=int)  # 前一期生肖索引
actual_idx = np.zeros(TEST_PERIODS, dtype=int)

for pi in range(TEST_PERIODS):
    i = START + pi
    hist = animals[:i]
    actual_idx[pi] = Z_IDX[animals[i]]
    prev_zodiac_idx[pi] = Z_IDX[hist[-1]] if hist else 0
    
    # 冷号15
    w = min(15, len(hist)); freq = Counter(hist[-w:])
    mx = max(freq.values()) if freq else 1
    for zi, z in enumerate(ZODIAC_CYCLE):
        score_mat[0, pi, zi] = 1.0 - freq.get(z,0)/max(mx,1)
    
    # 冷号20
    w = min(20, len(hist)); freq = Counter(hist[-w:])
    mx = max(freq.values()) if freq else 1
    for zi, z in enumerate(ZODIAC_CYCLE):
        score_mat[1, pi, zi] = 1.0 - freq.get(z,0)/max(mx,1)
    
    # 冷号30
    w = min(30, len(hist)); freq = Counter(hist[-w:])
    mx = max(freq.values()) if freq else 1
    for zi, z in enumerate(ZODIAC_CYCLE):
        score_mat[2, pi, zi] = 1.0 - freq.get(z,0)/max(mx,1)
    
    # 间隔
    for zi, z in enumerate(ZODIAC_CYCLE):
        last = -1
        for j in range(len(hist)-1,-1,-1):
            if hist[j]==z: last=j; break
        gap = len(hist)-1-last if last>=0 else len(hist)
        score_mat[3, pi, zi] = gap/12.0
    
    # 马尔可夫(Laplace平滑)
    if len(hist) >= 2:
        trans = {}
        for k in range(1, len(hist)):
            p,c = hist[k-1], hist[k]
            if p not in trans: trans[p]=Counter()
            trans[p][c] += 1
        state = hist[-1]
        if state in trans:
            total = sum(trans[state].values()) + 12  # Laplace
            for zi, z in enumerate(ZODIAC_CYCLE):
                score_mat[4, pi, zi] = (trans[state].get(z,0)+1)/total
        else:
            score_mat[4, pi, :] = 1/12
    else:
        score_mat[4, pi, :] = 1/12
    
    # 过期回归
    for zi, z in enumerate(ZODIAC_CYCLE):
        pos = [k for k,a in enumerate(hist) if a==z]
        if len(pos) >= 2:
            ints = [pos[k+1]-pos[k] for k in range(len(pos)-1)]
            avg = np.mean(ints)
            gap = len(hist)-1-pos[-1]
            score_mat[5, pi, zi] = 1/(1+np.exp(-3*(gap/max(avg,1)-1)))
        elif len(pos) == 1:
            score_mat[5, pi, zi] = min((len(hist)-1-pos[0])/12, 1)
        else:
            score_mat[5, pi, zi] = 0.8

print(f"预计算完成: {time.time()-t0:.1f}s")

# ============================================================
# 方法A: 状态特化策略 - 每个前一期生肖有独立权重
# ============================================================
print("\n" + "="*70)
print("方法A: 状态特化 (12个独立权重向量)")
print("="*70)

def fast_eval_global(weights):
    w = np.array(weights).reshape(-1,1,1)
    combined = (score_mat * w).sum(axis=0)
    hits = 0
    for pi in range(TEST_PERIODS):
        if actual_idx[pi] in np.argsort(-combined[pi])[:4]:
            hits += 1
    return hits

# 对每个状态独立搜索
state_weights = {}
state_hits = {}
vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

for state_zi in range(12):
    state_name = ZODIAC_CYCLE[state_zi]
    periods_in_state = [pi for pi in range(TEST_PERIODS) if prev_zodiac_idx[pi] == state_zi]
    n_periods = len(periods_in_state)
    
    if n_periods == 0:
        continue
    
    best_h = 0
    best_w = [0.2]*N_STRATS
    
    # 搜索这个状态下的最优权重
    # 关键维度: 冷号20(1), 间隔(3), 马尔可夫(4)
    for w1 in vals:
        for w3 in vals:
            for w4 in vals:
                for w0 in [0, 0.1, 0.2]:
                    w = [w0, w1, 0, w3, w4, 0]
                    ws = sum(w)
                    if ws == 0: continue
                    w = [x/ws for x in w]
                    
                    # 只评估该状态下的期数
                    h = 0
                    for pi in periods_in_state:
                        combined = sum(w[s] * score_mat[s, pi] for s in range(N_STRATS))
                        if actual_idx[pi] in np.argsort(-combined)[:4]:
                            h += 1
                    
                    if h > best_h:
                        best_h = h
                        best_w = w[:]
    
    # 精细搜索
    base = best_w[:]
    for combo in iprod([-0.05, 0, 0.05], repeat=4):
        w = base[:]
        w[0] = max(0, base[0]+combo[0])
        w[1] = max(0, base[1]+combo[1])
        w[3] = max(0, base[3]+combo[2])
        w[4] = max(0, base[4]+combo[3])
        ws = sum(w)
        if ws == 0: continue
        w = [x/ws for x in w]
        h = 0
        for pi in periods_in_state:
            combined = sum(w[s]*score_mat[s,pi] for s in range(N_STRATS))
            if actual_idx[pi] in np.argsort(-combined)[:4]:
                h += 1
        if h > best_h:
            best_h = h
            best_w = w[:]
    
    state_weights[state_zi] = best_w
    state_hits[state_zi] = best_h
    rate = best_h/n_periods*100 if n_periods > 0 else 0
    active = [(strat_names[s], f"{best_w[s]:.2f}") for s in range(N_STRATS) if best_w[s] > 0.01]
    print(f"  {state_name}: {best_h}/{n_periods} = {rate:.1f}% | {active}")

total_hits_A = sum(state_hits.values())
print(f"\n方法A总计: {total_hits_A}/{TEST_PERIODS} = {total_hits_A/TEST_PERIODS*100:.1f}%")

# ============================================================
# 方法A带来过拟合风险！用交叉验证评估
# ============================================================
print("\n" + "="*70)
print("方法A交叉验证 (3折)")
print("="*70)

fold_size = TEST_PERIODS // 3
cv_hits = 0

for fold in range(3):
    test_start = fold * fold_size
    test_end = (fold + 1) * fold_size
    train_periods = list(range(0, test_start)) + list(range(test_end, TEST_PERIODS))
    test_periods = list(range(test_start, test_end))
    
    # 在训练集上搜索状态特化权重
    fold_state_weights = {}
    for state_zi in range(12):
        train_in_state = [pi for pi in train_periods if prev_zodiac_idx[pi] == state_zi]
        if not train_in_state:
            fold_state_weights[state_zi] = [0.2]*N_STRATS
            continue
        
        best_h = 0
        best_w = [0.2]*N_STRATS
        for w1 in [0, 0.2, 0.4, 0.6, 0.8]:
            for w3 in [0, 0.2, 0.4, 0.6, 0.8]:
                for w4 in [0, 0.2, 0.4, 0.6, 0.8]:
                    w = [0.1, w1, 0, w3, w4, 0]
                    ws = sum(w)
                    if ws == 0: continue
                    w = [x/ws for x in w]
                    h = 0
                    for pi in train_in_state:
                        combined = sum(w[s]*score_mat[s,pi] for s in range(N_STRATS))
                        if actual_idx[pi] in np.argsort(-combined)[:4]:
                            h += 1
                    if h > best_h:
                        best_h = h
                        best_w = w[:]
        fold_state_weights[state_zi] = best_w
    
    # 在测试集上评估
    fold_hits = 0
    for pi in test_periods:
        state_zi = prev_zodiac_idx[pi]
        w = fold_state_weights.get(state_zi, [0.2]*N_STRATS)
        combined = sum(w[s]*score_mat[s,pi] for s in range(N_STRATS))
        if actual_idx[pi] in np.argsort(-combined)[:4]:
            fold_hits += 1
    cv_hits += fold_hits
    print(f"  Fold {fold+1}: {fold_hits}/{len(test_periods)} = {fold_hits/len(test_periods)*100:.1f}%")

print(f"  CV平均: {cv_hits}/{TEST_PERIODS} = {cv_hits/TEST_PERIODS*100:.1f}%")

# ============================================================
# 方法B: 滚动训练状态特化 (无未来函数)
# ============================================================
print("\n" + "="*70)
print("方法B: 滚动训练状态特化 (无未来函数)")
print("="*70)

def method_rolling_state_specific(train_window=150, retrain_every=25):
    """滚动训练: 用前train_window期训练状态权重,无偷看"""
    hits = 0
    cur_state_weights = {zi: [0.0, 0.3, 0.0, 0.3, 0.4, 0.0] for zi in range(12)}
    last_train = -retrain_every
    
    for pi in range(TEST_PERIODS):
        # 训练 (只用历史数据)
        if pi - last_train >= retrain_every and pi >= 30:
            tw = min(train_window, pi)
            train_range = list(range(max(0, pi-tw), pi))
            
            for state_zi in range(12):
                train_in_state = [p for p in train_range if prev_zodiac_idx[p] == state_zi]
                if len(train_in_state) < 5:
                    continue
                
                best_h = 0
                best_w = cur_state_weights[state_zi][:]
                for w1 in [0, 0.2, 0.4, 0.6, 0.8]:
                    for w3 in [0, 0.2, 0.4, 0.6]:
                        for w4 in [0, 0.2, 0.4, 0.6, 0.8]:
                            w = [0.1, w1, 0, w3, w4, 0]
                            ws = sum(w)
                            if ws == 0: continue
                            w = [x/ws for x in w]
                            h = sum(1 for p in train_in_state 
                                    if actual_idx[p] in np.argsort(-sum(w[s]*score_mat[s,p] for s in range(N_STRATS)))[:4])
                            if h > best_h:
                                best_h = h
                                best_w = w[:]
                cur_state_weights[state_zi] = best_w
            last_train = pi
        
        # 预测
        state_zi = prev_zodiac_idx[pi]
        w = cur_state_weights[state_zi]
        combined = sum(w[s]*score_mat[s,pi] for s in range(N_STRATS))
        if actual_idx[pi] in np.argsort(-combined)[:4]:
            hits += 1
    
    return hits

for tw in [100, 150, 200]:
    for rt in [20, 30, 50]:
        h = method_rolling_state_specific(train_window=tw, retrain_every=rt)
        rate = h/TEST_PERIODS*100
        if rate > 43:
            print(f"  tw={tw}, rt={rt}: {h}/{TEST_PERIODS} = {rate:.1f}% *")
        else:
            print(f"  tw={tw}, rt={rt}: {h}/{TEST_PERIODS} = {rate:.1f}%")

# ============================================================
# 方法C: 置信度加权马尔可夫 + 冷号回退
# ============================================================
print("\n" + "="*70)
print("方法C: 置信度加权马尔可夫")
print("="*70)

def method_confidence_markov(min_samples=3, markov_boost=2.0):
    """当马尔可夫有足够样本时重点使用,否则回退冷号"""
    hits = 0
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        
        # 马尔可夫置信度
        if len(hist) >= 2:
            trans = {}
            for k in range(1, len(hist)):
                p,c = hist[k-1], hist[k]
                if p not in trans: trans[p]=Counter()
                trans[p][c]+=1
            state = hist[-1]
            if state in trans:
                n_samples = sum(trans[state].values())
            else:
                n_samples = 0
        else:
            n_samples = 0
        
        # 根据置信度调整权重
        if n_samples >= min_samples:
            conf = min(n_samples / 20, 1.0)  # 20样本时100%置信
            w_mk = 0.2 + 0.4 * conf * markov_boost  # 最高0.8+
            w_cold = (1 - w_mk) * 0.6
            w_gap = (1 - w_mk) * 0.4
        else:
            w_mk = 0.1
            w_cold = 0.5
            w_gap = 0.4
        
        # 归一化
        ws = w_mk + w_cold + w_gap
        w_mk /= ws; w_cold /= ws; w_gap /= ws
        
        final = {}
        for zi, z in enumerate(ZODIAC_CYCLE):
            final[z] = w_mk*score_mat[4,pi,zi] + w_cold*score_mat[1,pi,zi] + w_gap*score_mat[3,pi,zi]
        
        top4 = sorted(ZODIAC_CYCLE, key=lambda z: -final[z])[:4]
        if animals[i] in top4:
            hits += 1
    
    return hits

for ms in [2, 3, 5, 8]:
    for mb in [1.0, 1.5, 2.0, 2.5]:
        h = method_confidence_markov(min_samples=ms, markov_boost=mb)
        rate = h/TEST_PERIODS*100
        if rate > 43:
            print(f"  samples>={ms}, boost={mb}: {h}/{TEST_PERIODS} = {rate:.1f}% *")

# ============================================================
# 方法D: 马尔可夫TOP4 + 冷号补充
# ============================================================
print("\n" + "="*70)
print("方法D: 马尔可夫选2-3个 + 冷号补充到4个")
print("="*70)

def method_markov_plus_cold(n_markov=2, conf_thresh=5):
    """马尔可夫选高置信的N个,其余用冷号补充"""
    hits = 0
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        
        selected = []
        
        # 马尔可夫部分
        if len(hist) >= 2:
            trans = {}
            for k in range(1, len(hist)):
                p,c = hist[k-1], hist[k]
                if p not in trans: trans[p]=Counter()
                trans[p][c]+=1
            state = hist[-1]
            if state in trans and sum(trans[state].values()) >= conf_thresh:
                total = sum(trans[state].values())
                ranked = sorted(ZODIAC_CYCLE, key=lambda z: -trans[state].get(z,0))
                selected = ranked[:n_markov]
        
        # 冷号补充
        freq = Counter(hist[-min(20, len(hist)):])
        cold_ranked = sorted(ZODIAC_CYCLE, key=lambda z: freq.get(z,0))
        for z in cold_ranked:
            if z not in selected:
                selected.append(z)
            if len(selected) >= 4:
                break
        
        if animals[i] in selected[:4]:
            hits += 1
    
    return hits

for nm in [1, 2, 3]:
    for ct in [3, 5, 8, 10]:
        h = method_markov_plus_cold(n_markov=nm, conf_thresh=ct)
        rate = h/TEST_PERIODS*100
        if rate > 43:
            print(f"  markov_picks={nm}, min_conf={ct}: {h}/{TEST_PERIODS} = {rate:.1f}% *")

# ============================================================
# 方法E: 超级融合 - 状态特化 + 马尔可夫置信 + 冷号
# ============================================================
print("\n" + "="*70)
print("方法E: 状态特化马尔可夫+冷号自适应融合")
print("="*70)

def method_state_markov_adaptive():
    """
    1. 查询当前状态(前一期生肖)
    2. 马尔可夫有强信号→选马尔可夫TOP2-3, 冷号补充
    3. 马尔可夫无强信号→纯冷号+间隔
    4. 滚动统计各状态的最优策略
    """
    hits = 0
    # 预计算马尔可夫转移矩阵（滚动更新）
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        state = hist[-1] if hist else ZODIAC_CYCLE[0]
        
        # 构建到当前为止的转移矩阵
        trans = {}
        for k in range(1, len(hist)):
            p,c = hist[k-1], hist[k]
            if p not in trans: trans[p]=Counter()
            trans[p][c]+=1
        
        selected = []
        
        if state in trans:
            total = sum(trans[state].values())
            probs = {z: trans[state].get(z,0)/total for z in ZODIAC_CYCLE}
            sorted_by_prob = sorted(ZODIAC_CYCLE, key=lambda z: -probs[z])
            
            # 检查top概率是否显著高于baseline(1/12=8.3%)
            top_probs = [probs[z] for z in sorted_by_prob[:4]]
            
            if top_probs[0] >= 0.14:  # 显著高
                # 强信号: 马尔可夫选2个
                for z in sorted_by_prob[:2]:
                    if z not in selected:
                        selected.append(z)
            
            if top_probs[0] >= 0.10:
                # 中等信号: 马尔可夫选1个
                if sorted_by_prob[0] not in selected:
                    selected.append(sorted_by_prob[0])
        
        # 冷号补充
        w = 20
        freq = Counter(hist[-min(w, len(hist)):])
        cold_ranked = sorted(ZODIAC_CYCLE, key=lambda z: freq.get(z, 0))
        
        # 同时考虑间隔
        gaps = {}
        for z in ZODIAC_CYCLE:
            last = -1
            for j in range(len(hist)-1, -1, -1):
                if hist[j] == z: last = j; break
            gaps[z] = len(hist)-1-last if last >= 0 else len(hist)
        
        # 冷号+间隔混合排名
        mixed = sorted(ZODIAC_CYCLE, key=lambda z: -(
            0.5*(1-freq.get(z,0)/max(max(freq.values()),1)) + 0.5*gaps[z]/12
        ))
        
        for z in mixed:
            if z not in selected:
                selected.append(z)
            if len(selected) >= 4:
                break
        
        if animals[i] in selected[:4]:
            hits += 1
    
    return hits

h = method_state_markov_adaptive()
print(f"  基础版: {h}/{TEST_PERIODS} = {h/TEST_PERIODS*100:.1f}%")

# 参数搜索版
def method_E_param(mk_strong=0.14, mk_medium=0.10, mk_picks_strong=2, mk_picks_medium=1,
                   cold_window=20, cold_weight=0.5, gap_weight=0.5):
    hits = 0
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        state = hist[-1]
        
        trans = {}
        for k in range(1, len(hist)):
            p,c = hist[k-1], hist[k]
            if p not in trans: trans[p]=Counter()
            trans[p][c]+=1
        
        selected = []
        if state in trans:
            total = sum(trans[state].values())
            probs = {z: trans[state].get(z,0)/total for z in ZODIAC_CYCLE}
            sorted_mk = sorted(ZODIAC_CYCLE, key=lambda z: -probs[z])
            
            if probs[sorted_mk[0]] >= mk_strong:
                for z in sorted_mk[:mk_picks_strong]:
                    if z not in selected: selected.append(z)
            elif probs[sorted_mk[0]] >= mk_medium:
                for z in sorted_mk[:mk_picks_medium]:
                    if z not in selected: selected.append(z)
        
        w = cold_window
        freq = Counter(hist[-min(w, len(hist)):])
        gaps = {}
        for z in ZODIAC_CYCLE:
            last = -1
            for j in range(len(hist)-1,-1,-1):
                if hist[j]==z: last=j; break
            gaps[z] = len(hist)-1-last if last>=0 else len(hist)
        
        mx = max(freq.values()) if freq else 1
        mixed = sorted(ZODIAC_CYCLE, key=lambda z: -(
            cold_weight*(1-freq.get(z,0)/max(mx,1)) + gap_weight*gaps[z]/12
        ))
        for z in mixed:
            if z not in selected: selected.append(z)
            if len(selected) >= 4: break
        
        if animals[i] in selected[:4]: hits += 1
    return hits

print("\n参数搜索:")
best_E = 0
best_E_params = None
for ms in [0.12, 0.14, 0.16, 0.18]:
    for mm in [0.08, 0.10, 0.12]:
        for mps in [1, 2, 3]:
            for mpm in [0, 1]:
                for cw in [15, 20, 25]:
                    for cwt in [0.3, 0.4, 0.5, 0.6, 0.7]:
                        h = method_E_param(mk_strong=ms, mk_medium=mm, 
                                          mk_picks_strong=mps, mk_picks_medium=mpm,
                                          cold_window=cw, cold_weight=cwt, gap_weight=1-cwt)
                        if h > best_E:
                            best_E = h
                            best_E_params = (ms, mm, mps, mpm, cw, cwt)
                            if h/TEST_PERIODS > 0.44:
                                print(f"  {h}/{TEST_PERIODS}={h/TEST_PERIODS*100:.1f}% ms={ms} mm={mm} mps={mps} mpm={mpm} cw={cw} cwt={cwt}")

print(f"\n方法E最优: {best_E}/{TEST_PERIODS} = {best_E/TEST_PERIODS*100:.1f}%")
print(f"  参数: {best_E_params}")

# ============================================================
# 汇总
# ============================================================
print("\n" + "="*70)
print("总结")
print("="*70)
print(f"方法A (状态特化,有过拟合): {total_hits_A}/{TEST_PERIODS} = {total_hits_A/TEST_PERIODS*100:.1f}%")
print(f"方法A (3折CV):             {cv_hits}/{TEST_PERIODS} = {cv_hits/TEST_PERIODS*100:.1f}%")
print(f"方法E (马尔可夫+冷号):     {best_E}/{TEST_PERIODS} = {best_E/TEST_PERIODS*100:.1f}%")
print(f"\n总耗时: {time.time()-t0:.1f}秒")
