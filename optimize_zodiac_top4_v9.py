"""
生肖TOP4突破 v9 - 高效预计算 + 全方位搜索
============================================
所有策略预计算到矩阵, 参数搜索只做矩阵运算
"""
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
import time, sys
sys.stdout.reconfigure(encoding='utf-8')

t0 = time.time()
ZODIAC_CYCLE = ['马', '蛇', '龙', '兔', '虎', '牛', '鼠', '猪', '狗', '鸡', '猴', '羊']
Z_IDX = {z: i for i, z in enumerate(ZODIAC_CYCLE)}
NUM_TO_ZODIAC = {n: ZODIAC_CYCLE[(n - 1) % 12] for n in range(1, 50)}

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].tolist()
animals = [NUM_TO_ZODIAC[n] for n in numbers]
TOTAL = len(animals)
T = 300  # test periods
S = TOTAL - T
print(f"Data: {TOTAL}, test {T}")

# ============================================================
# 预计算所有策略 (只算一次!)
# ============================================================
print("预计算中...")
actual = np.array([Z_IDX[animals[S+pi]] for pi in range(T)])
prev_z = np.array([Z_IDX[animals[S+pi-1]] for pi in range(T)])

strat_names = []
strat_mat = []  # list of (T, 12) arrays

def add_strat(name, scores_list):
    strat_names.append(name)
    strat_mat.append(np.array(scores_list))

# 冷号 (多个窗口)
for w in [10, 15, 20, 25, 30, 40, 50]:
    scores = []
    for pi in range(T):
        hist = animals[:S+pi]
        freq = Counter(hist[-min(w, len(hist)):])
        mx = max(freq.values()) if freq else 1
        scores.append([1.0 - freq.get(z,0)/max(mx,1) for z in ZODIAC_CYCLE])
    add_strat(f'冷号{w}', scores)

# 间隔
scores = []
for pi in range(T):
    hist = animals[:S+pi]
    row = []
    for z in ZODIAC_CYCLE:
        last = -1
        for j in range(len(hist)-1,-1,-1):
            if hist[j]==z: last=j; break
        row.append((len(hist)-1-last)/12 if last>=0 else len(hist)/12)
    scores.append(row)
add_strat('间隔', scores)

# 马尔可夫 (全历史 + 多滑窗)
for mkw in [None, 50, 100, 150, 200]:
    scores = []
    for pi in range(T):
        hist = animals[:S+pi]
        h = hist[-mkw:] if mkw and len(hist)>mkw else hist
        probs = [1/12]*12
        if len(h) >= 2:
            trans = {}
            for k in range(1, len(h)):
                p,c = h[k-1], h[k]
                if p not in trans: trans[p]=Counter()
                trans[p][c]+=1
            state = hist[-1]
            if state in trans:
                total = sum(trans[state].values()) + 12
                probs = [(trans[state].get(z,0)+1)/total for z in ZODIAC_CYCLE]
        scores.append(probs)
    label = f'MK全' if mkw is None else f'MK{mkw}'
    add_strat(label, scores)

# 超期回归
scores = []
for pi in range(T):
    hist = animals[:S+pi]
    row = []
    for z in ZODIAC_CYCLE:
        pos = [k for k,a in enumerate(hist) if a==z]
        if len(pos) >= 2:
            ints = [pos[k+1]-pos[k] for k in range(len(pos)-1)]
            avg = np.mean(ints)
            g = len(hist)-1-pos[-1]
            row.append(1/(1+np.exp(-3*(g/max(avg,1)-1))))
        elif len(pos) == 1:
            row.append(min((len(hist)-1-pos[0])/12, 1))
        else:
            row.append(0.8)
    scores.append(row)
add_strat('超期', scores)

# 热号 (与冷号相反)
for w in [10, 15, 20]:
    scores = []
    for pi in range(T):
        hist = animals[:S+pi]
        freq = Counter(hist[-min(w, len(hist)):])
        mx = max(freq.values()) if freq else 1
        scores.append([freq.get(z,0)/max(mx,1) for z in ZODIAC_CYCLE])
    add_strat(f'热号{w}', scores)

# 二阶马尔可夫
scores = []
for pi in range(T):
    hist = animals[:S+pi]
    probs = [1/12]*12
    if len(hist) >= 3:
        trans = {}
        for k in range(2, len(hist)):
            prev = (hist[k-2], hist[k-1])
            curr = hist[k]
            if prev not in trans: trans[prev]=Counter()
            trans[prev][curr]+=1
        state2 = (hist[-2], hist[-1])
        if state2 in trans:
            total = sum(trans[state2].values()) + 6
            probs = [(trans[state2].get(z,0)+0.5)/total for z in ZODIAC_CYCLE]
    scores.append(probs)
add_strat('MK2阶', scores)

N = len(strat_names)
mat = np.array(strat_mat)  # (N, T, 12)
print(f"预计算完成: {N}个策略, {time.time()-t0:.1f}s")

# ============================================================
# 单策略基线
# ============================================================
print(f"\n{'='*60}\n单策略基线:\n{'='*60}")
for si in range(N):
    h = sum(1 for pi in range(T) if actual[pi] in np.argsort(-mat[si,pi])[:4])
    print(f"  {strat_names[si]:>8}: {h}/{T} = {h/T*100:.1f}%")

# ============================================================
# 快速权重评估函数
# ============================================================
def eval_w(w_indices, w_vals):
    """只用选中的策略做加权"""
    combined = np.zeros((T, 12))
    for idx, w in zip(w_indices, w_vals):
        combined += w * mat[idx]
    hits = 0
    for pi in range(T):
        if actual[pi] in np.argsort(-combined[pi])[:4]:
            hits += 1
    return hits

# ============================================================
# 搜索: TOP策略组合 (2-5个策略)
# ============================================================
print(f"\n{'='*60}\n策略组合搜索\n{'='*60}")

best_global = 0
best_global_info = None

# 识别有价值的策略 (>35%)
useful = [si for si in range(N) if sum(1 for pi in range(T) if actual[pi] in np.argsort(-mat[si,pi])[:4]) > T*0.34]
print(f"有效策略({len(useful)}个): {[strat_names[s] for s in useful]}")

# 2-4策略组合搜索
w_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
for n_strats in [2, 3, 4]:
    for combo in combinations(useful, n_strats):
        # 权重搜索(简化版: 对每个策略试几个权重)
        if n_strats == 2:
            for w0 in w_grid:
                w1 = 1.0 - w0
                if w1 <= 0: continue
                h = eval_w(combo, [w0, w1])
                if h > best_global:
                    best_global = h
                    best_global_info = (combo, [w0, w1])
        elif n_strats == 3:
            for w0 in [0.1, 0.2, 0.3, 0.4, 0.5]:
                for w1 in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    w2 = 1.0 - w0 - w1
                    if w2 <= 0: continue
                    h = eval_w(combo, [w0, w1, w2])
                    if h > best_global:
                        best_global = h
                        best_global_info = (combo, [w0, w1, w2])
        elif n_strats == 4:
            for w0 in [0.1, 0.2, 0.3, 0.4]:
                for w1 in [0.1, 0.2, 0.3, 0.4]:
                    for w2 in [0.1, 0.2, 0.3, 0.4]:
                        w3 = 1.0 - w0 - w1 - w2
                        if w3 <= 0: continue
                        h = eval_w(combo, [w0, w1, w2, w3])
                        if h > best_global:
                            best_global = h
                            best_global_info = (combo, [w0, w1, w2, w3])

combo, weights = best_global_info
combo_names = [strat_names[s] for s in combo]
print(f"\n最优全局组合: {best_global}/{T} = {best_global/T*100:.1f}%")
print(f"  策略: {combo_names}")
print(f"  权重: {[f'{w:.2f}' for w in weights]}")

# 精搜索
base_w = list(weights)
for _ in range(5):
    imp = False
    for dim in range(len(base_w)):
        for d in [-0.05, -0.02, 0.02, 0.05, 0.08]:
            w = base_w[:]
            w[dim] = max(0.01, w[dim]+d)
            ws = sum(w)
            w = [x/ws for x in w]
            h = eval_w(combo, w)
            if h > best_global:
                best_global = h
                base_w = w[:]
                imp = True
    if not imp: break

print(f"精搜索: {best_global}/{T} = {best_global/T*100:.1f}%")
print(f"  权重: {[f'{w:.3f}' for w in base_w]}")

# ============================================================
# 投票 (不用权重, 每个策略投TOP4)
# ============================================================
print(f"\n{'='*60}\n投票策略\n{'='*60}")
best_vote = 0
best_vote_combo = None
for k in range(3, min(N+1, 8)):
    for combo in combinations(useful, k):
        hits = 0
        for pi in range(T):
            votes = np.zeros(12)
            for si in combo:
                top4 = np.argsort(-mat[si,pi])[:4]
                votes[top4] += 1
            top4_v = np.argsort(-votes)[:4]
            if actual[pi] in top4_v:
                hits += 1
        if hits > best_vote:
            best_vote = hits
            best_vote_combo = combo
            if hits/T > 0.44:
                print(f"  {[strat_names[s] for s in combo]}: {hits}/{T}={hits/T*100:.1f}%")

print(f"\n投票最优: {best_vote}/{T} = {best_vote/T*100:.1f}%")
print(f"  组合: {[strat_names[s] for s in best_vote_combo]}")

# ============================================================
# 优先级策略: 马尔可夫优先 + 冷号填充
# ============================================================
print(f"\n{'='*60}\n优先级策略: MK优先+冷号填充\n{'='*60}")

# 预计算MK概率矩阵
mk_all_idx = strat_names.index('MK全')
best_prior = 0
best_prior_p = None

for mk_idx_name in ['MK全', 'MK50', 'MK100', 'MK150', 'MK200']:
    if mk_idx_name not in strat_names: continue
    mk_idx = strat_names.index(mk_idx_name)
    
    for thresh in [0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18]:
        for n_mk in [1, 2, 3]:
            for cold_name in ['冷号10', '冷号15', '冷号20', '冷号25', '冷号30']:
                if cold_name not in strat_names: continue
                cold_idx = strat_names.index(cold_name)
                gap_idx = strat_names.index('间隔')
                
                for cgr in [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
                    hits = 0
                    for pi in range(T):
                        mk_probs = mat[mk_idx, pi]
                        sorted_mk = np.argsort(-mk_probs)
                        
                        selected = []
                        for k in range(min(n_mk, 12)):
                            if mk_probs[sorted_mk[k]] >= thresh:
                                selected.append(sorted_mk[k])
                        
                        # 冷号+间隔混合
                        mixed = cgr*mat[cold_idx,pi] + (1-cgr)*mat[gap_idx,pi]
                        for zi in np.argsort(-mixed):
                            if zi not in selected:
                                selected.append(zi)
                            if len(selected) >= 4:
                                break
                        
                        if actual[pi] in selected[:4]:
                            hits += 1
                    
                    if hits > best_prior:
                        best_prior = hits
                        best_prior_p = (mk_idx_name, thresh, n_mk, cold_name, cgr)
                        if hits/T > 0.44:
                            print(f"  {hits}/{T}={hits/T*100:.1f}% {mk_idx_name} t={thresh} n={n_mk} {cold_name} cgr={cgr}")

print(f"\n优先级最优: {best_prior}/{T} = {best_prior/T*100:.1f}%")
print(f"  参数: {best_prior_p}")

# ============================================================
# EWMA在线学习 (预计算版)
# ============================================================
print(f"\n{'='*60}\nEWMA在线学习\n{'='*60}")

# 用最好的3-4个独立策略做EWMA
top_strats = sorted(range(N), key=lambda si: -sum(1 for pi in range(T) if actual[pi] in np.argsort(-mat[si,pi])[:4]))[:6]

best_ewma = 0
best_ewma_p = None
for alpha in [0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]:
    for combo in combinations(top_strats, 3):
        # 初始等权
        ws = np.array([1.0/3]*3)
        hits = 0
        for pi in range(T):
            # 预测
            combined = sum(ws[k]*mat[combo[k],pi] for k in range(3))
            top4 = np.argsort(-combined)[:4]
            if actual[pi] in top4:
                hits += 1
            
            # 更新权重
            for k in range(3):
                s_top4 = set(np.argsort(-mat[combo[k],pi])[:4])
                hit_k = 1.0 if actual[pi] in s_top4 else 0.0
                ws[k] = max(0.05, (1-alpha)*ws[k] + alpha*hit_k)
            ws = ws / ws.sum()
        
        if hits > best_ewma:
            best_ewma = hits
            best_ewma_p = (alpha, [strat_names[s] for s in combo])

print(f"EWMA最优: {best_ewma}/{T} = {best_ewma/T*100:.1f}%")
print(f"  参数: {best_ewma_p}")

# ============================================================
# 连miss切换: 正常用最优静态, 连miss>=N时切换策略
# ============================================================
print(f"\n{'='*60}\n连miss策略切换\n{'='*60}")

# 最优静态组合
static_combo, static_w = best_global_info
static_combined = np.zeros((T, 12))
for idx, w in zip(static_combo, static_w):
    static_combined += w * mat[idx]

# 备选策略 (与最优不太相关的)
# 找和static命中模式最不相关的策略
static_hits_set = set(pi for pi in range(T) if actual[pi] in np.argsort(-static_combined[pi])[:4])

alt_strats = []
for si in range(N):
    si_hits = set(pi for pi in range(T) if actual[pi] in np.argsort(-mat[si,pi])[:4])
    overlap = len(static_hits_set & si_hits)
    union = len(static_hits_set | si_hits)
    jaccard = overlap / union if union > 0 else 0
    unique = len(si_hits - static_hits_set)
    alt_strats.append((si, jaccard, unique, len(si_hits)))

alt_strats.sort(key=lambda x: -x[2])  # 按unique hits排序
print("互补策略排名:")
for si, jac, uniq, total_h in alt_strats[:8]:
    print(f"  {strat_names[si]:>8}: jaccard={jac:.2f}, unique={uniq}, total={total_h}")

best_switch = 0
best_switch_p = None
for alt_si, _, _, _ in alt_strats[:5]:
    for switch_thresh in [2, 3, 4, 5]:
        for switch_w_static in [0.0, 0.2, 0.3, 0.4, 0.5]:
            switch_w_alt = 1.0 - switch_w_static
            
            hits = 0
            consec_miss = 0
            for pi in range(T):
                if consec_miss >= switch_thresh:
                    # 切换: 混合static和alt
                    combined = switch_w_static*static_combined[pi] + switch_w_alt*mat[alt_si,pi]
                else:
                    combined = static_combined[pi]
                
                top4 = np.argsort(-combined)[:4]
                if actual[pi] in top4:
                    hits += 1
                    consec_miss = 0
                else:
                    consec_miss += 1
            
            if hits > best_switch:
                best_switch = hits
                best_switch_p = (strat_names[alt_si], switch_thresh, switch_w_static)

print(f"\n连miss切换最优: {best_switch}/{T} = {best_switch/T*100:.1f}%")
print(f"  参数: {best_switch_p}")

# ============================================================
# 集中度自适应(预计算版)
# ============================================================
print(f"\n{'='*60}\n集中度自适应\n{'='*60}")

mk_full = mat[strat_names.index('MK全')]
hhi = np.sum(mk_full**2, axis=1)  # (T,) HHI per period
hhi_uniform = 1/12

best_conc = 0
best_conc_p = None
for cold_name in ['冷号15', '冷号20', '冷号25']:
    cold_idx = strat_names.index(cold_name)
    gap_idx = strat_names.index('间隔')
    mk_idx = strat_names.index('MK全')
    
    for base_wm in [0.1, 0.15, 0.2, 0.25, 0.3]:
        for base_wc in [0.3, 0.4, 0.5]:
            base_wg = max(0, 1-base_wm-base_wc)
            for boost in [1, 2, 3, 4, 5, 6, 8, 10, 15]:
                hits = 0
                for pi in range(T):
                    conc = max(0, hhi[pi] - hhi_uniform) / hhi_uniform
                    wm = base_wm * (1 + conc * boost)
                    wc = base_wc
                    wg = base_wg
                    ws = wm + wc + wg
                    combined = (wm*mat[mk_idx,pi] + wc*mat[cold_idx,pi] + wg*mat[gap_idx,pi]) / ws
                    if actual[pi] in np.argsort(-combined)[:4]:
                        hits += 1
                
                if hits > best_conc:
                    best_conc = hits
                    best_conc_p = (cold_name, base_wm, base_wc, boost)

print(f"集中度自适应最优: {best_conc}/{T} = {best_conc/T*100:.1f}%")
print(f"  参数: {best_conc_p}")

# ============================================================
# 非线性融合: 分数排名制
# ============================================================
print(f"\n{'='*60}\n排名融合 (非线性)\n{'='*60}")

# 每个策略对12个zodiac排名(1-12), 融合排名而非分数
rank_mat = np.zeros_like(mat)  # (N, T, 12)
for si in range(N):
    for pi in range(T):
        order = np.argsort(-mat[si, pi])
        for r, zi in enumerate(order):
            rank_mat[si, pi, zi] = 12 - r  # 最高=12分, 最低=1分

best_rank = 0
best_rank_combo = None
for combo in combinations(useful, 3):
    rank_sum = sum(rank_mat[si] for si in combo)
    h = sum(1 for pi in range(T) if actual[pi] in np.argsort(-rank_sum[pi])[:4])
    if h > best_rank:
        best_rank = h
        best_rank_combo = combo

for combo in combinations(useful, 4):
    rank_sum = sum(rank_mat[si] for si in combo)
    h = sum(1 for pi in range(T) if actual[pi] in np.argsort(-rank_sum[pi])[:4])
    if h > best_rank:
        best_rank = h
        best_rank_combo = combo

for combo in combinations(useful, 5):
    rank_sum = sum(rank_mat[si] for si in combo)
    h = sum(1 for pi in range(T) if actual[pi] in np.argsort(-rank_sum[pi])[:4])
    if h > best_rank:
        best_rank = h
        best_rank_combo = combo

print(f"排名融合最优: {best_rank}/{T} = {best_rank/T*100:.1f}%")
print(f"  策略: {[strat_names[s] for s in best_rank_combo]}")

# ============================================================
# 最终汇总
# ============================================================
print(f"\n{'='*60}")
print("最终汇总")
print(f"{'='*60}")
all_results = [
    ('全局权重搜索', best_global),
    ('投票', best_vote),
    ('优先级MK+冷号', best_prior),
    ('EWMA在线学习', best_ewma),
    ('连miss切换', best_switch),
    ('集中度自适应', best_conc),
    ('排名融合', best_rank),
]
for name, h in sorted(all_results, key=lambda x: -x[1]):
    star = ' ***' if h/T >= 0.46 else ' **' if h/T >= 0.45 else ' *' if h/T >= 0.44 else ''
    print(f"  {name:>14}: {h}/{T} = {h/T*100:.1f}%{star}")

winner = max(all_results, key=lambda x: x[1])
print(f"\nBEST: {winner[0]} = {winner[1]/T*100:.1f}%")
print(f"Random baseline: 33.3%, improvement: +{(winner[1]/T-0.333)/0.333*100:.1f}%")
print(f"Total time: {time.time()-t0:.1f}s")
