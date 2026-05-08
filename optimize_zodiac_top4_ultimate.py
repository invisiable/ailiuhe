"""
生肖TOP4终极优化器 - ML + 元学习 + 策略去相关
目标: 300期滚动验证命中率 >= 50%
"""
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
import time

t0 = time.time()

ZODIAC_CYCLE = ['马', '蛇', '龙', '兔', '虎', '牛', '鼠', '猪', '狗', '鸡', '猴', '羊']
Z_IDX = {z: i for i, z in enumerate(ZODIAC_CYCLE)}
NUM_TO_ZODIAC = {n: ZODIAC_CYCLE[(n - 1) % 12] for n in range(1, 50)}

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].tolist()
animals = [NUM_TO_ZODIAC[n] for n in numbers]
TOTAL = len(animals)
TEST_PERIODS = 300
START = TOTAL - TEST_PERIODS

print(f"数据: {TOTAL}期, 测试{TEST_PERIODS}期")

# ============================================================
# 特征工程：为每期生成丰富的特征向量
# ============================================================
def build_features(animals_hist, numbers_hist):
    """
    为每个时间点构建特征，返回(n_features,)向量
    """
    n = len(animals_hist)
    features = []
    
    # 特征组1: 各生肖在多窗口的频率 (12 * 5 = 60)
    for window in [5, 10, 15, 20, 30]:
        w = min(window, n)
        freq = Counter(animals_hist[-w:])
        for z in ZODIAC_CYCLE:
            features.append(freq.get(z, 0) / w)
    
    # 特征组2: 各生肖的间隔 (12)
    for z in ZODIAC_CYCLE:
        last = -1
        for j in range(n-1, -1, -1):
            if animals_hist[j] == z:
                last = j
                break
        gap = n - 1 - last if last >= 0 else n
        features.append(gap / 12.0)
    
    # 特征组3: 一阶马尔可夫转移概率 (12)
    if n >= 2:
        trans = {}
        for k in range(1, n):
            prev, curr = animals_hist[k-1], animals_hist[k]
            if prev not in trans:
                trans[prev] = Counter()
            trans[prev][curr] += 1
        state = animals_hist[-1]
        if state in trans:
            total = sum(trans[state].values())
            for z in ZODIAC_CYCLE:
                features.append(trans[state].get(z, 0) / total)
        else:
            features.extend([1/12] * 12)
    else:
        features.extend([1/12] * 12)
    
    # 特征组4: 二阶马尔可夫 (12)
    if n >= 3:
        trans2 = {}
        for k in range(2, n):
            prev = (animals_hist[k-2], animals_hist[k-1])
            curr = animals_hist[k]
            if prev not in trans2:
                trans2[prev] = Counter()
            trans2[prev][curr] += 1
        state2 = (animals_hist[-2], animals_hist[-1])
        if state2 in trans2:
            total = sum(trans2[state2].values())
            for z in ZODIAC_CYCLE:
                features.append(trans2[state2].get(z, 0) / total)
        else:
            features.extend([1/12] * 12)
    else:
        features.extend([1/12] * 12)
    
    # 特征组5: 最近5期one-hot (5 * 12 = 60)
    for j in range(5):
        if n > j:
            idx = Z_IDX[animals_hist[-(j+1)]]
            for zi in range(12):
                features.append(1.0 if zi == idx else 0.0)
        else:
            features.extend([0.0] * 12)
    
    # 特征组6: 趋势特征 - 近10期频率 vs 近30期频率 (12)
    if n >= 30:
        freq10 = Counter(animals_hist[-10:])
        freq30 = Counter(animals_hist[-30:])
        for z in ZODIAC_CYCLE:
            r10 = freq10.get(z, 0) / 10
            r30 = freq30.get(z, 0) / 30
            features.append(r10 - r30)  # 正=变热，负=变冷
    else:
        features.extend([0.0] * 12)
    
    # 特征组7: 数字级特征→生肖聚合 (12)
    if len(numbers_hist) >= 20:
        w = min(30, len(numbers_hist))
        num_freq = Counter(numbers_hist[-w:])
        for z in ZODIAC_CYCLE:
            z_nums = [n for n in range(1, 50) if NUM_TO_ZODIAC[n] == z]
            total_freq = sum(num_freq.get(n, 0) for n in z_nums)
            features.append(total_freq / w)
    else:
        features.extend([0.0] * 12)
    
    # 特征组8: 周期性特征 (12)
    for z in ZODIAC_CYCLE:
        positions = [k for k, a in enumerate(animals_hist) if a == z]
        if len(positions) >= 3:
            intervals = [positions[k+1]-positions[k] for k in range(len(positions)-1)]
            avg_interval = np.mean(intervals)
            gap = n - 1 - positions[-1]
            features.append(gap / max(avg_interval, 1))
        else:
            features.append(0.0)
    
    return np.array(features)


# ============================================================
# 方法1: ML多分类 (GradientBoosting)
# ============================================================
def method_ml_multiclass(model_cls, retrain_every=50, min_train=80):
    """ML多分类预测器，定期重训练"""
    hits = 0
    model = None
    last_train = -retrain_every
    
    for pi in range(TEST_PERIODS):
        i = START + pi
        
        # 定期训练
        if pi - last_train >= retrain_every or model is None:
            # 构建训练集
            train_start = max(30, i - 200)  # 最近200期训练
            X_train = []
            y_train = []
            for t in range(train_start, i):
                feat = build_features(animals[:t], numbers[:t])
                label = Z_IDX[animals[t]]
                X_train.append(feat)
                y_train.append(label)
            
            if len(X_train) >= min_train:
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                model = model_cls()
                model.fit(X_train, y_train)
                last_train = pi
        
        if model is None:
            continue
        
        # 预测
        feat = build_features(animals[:i], numbers[:i])
        proba = model.predict_proba(feat.reshape(1, -1))[0]
        
        # 映射到生肖（model.classes_可能不是0-11连续）
        zodiac_proba = {z: 0.0 for z in ZODIAC_CYCLE}
        for ci, cls in enumerate(model.classes_):
            zodiac_proba[ZODIAC_CYCLE[cls]] = proba[ci]
        
        top4 = sorted(ZODIAC_CYCLE, key=lambda z: -zodiac_proba[z])[:4]
        if animals[i] in top4:
            hits += 1
    
    return hits / TEST_PERIODS


# ============================================================
# 方法2: ML二分类集成 (每个生肖独立建模)
# ============================================================
def method_ml_binary_ensemble(retrain_every=50, min_train=60):
    """12个独立二分类器，分别预测每个生肖"""
    hits = 0
    models = {}
    last_train = -retrain_every
    
    for pi in range(TEST_PERIODS):
        i = START + pi
        
        if pi - last_train >= retrain_every or not models:
            train_start = max(30, i - 200)
            X_train = []
            y_trains = {z: [] for z in ZODIAC_CYCLE}
            for t in range(train_start, i):
                feat = build_features(animals[:t], numbers[:t])
                X_train.append(feat)
                actual_z = animals[t]
                for z in ZODIAC_CYCLE:
                    y_trains[z].append(1 if actual_z == z else 0)
            
            if len(X_train) >= min_train:
                X_train = np.array(X_train)
                models = {}
                for z in ZODIAC_CYCLE:
                    y = np.array(y_trains[z])
                    if y.sum() >= 3:
                        clf = GradientBoostingClassifier(
                            n_estimators=50, max_depth=3, 
                            learning_rate=0.1, random_state=42
                        )
                        clf.fit(X_train, y)
                        models[z] = clf
                last_train = pi
        
        if not models:
            continue
        
        feat = build_features(animals[:i], numbers[:i]).reshape(1, -1)
        zodiac_proba = {}
        for z in ZODIAC_CYCLE:
            if z in models:
                zodiac_proba[z] = models[z].predict_proba(feat)[0][1]
            else:
                zodiac_proba[z] = 1/12
        
        top4 = sorted(ZODIAC_CYCLE, key=lambda z: -zodiac_proba[z])[:4]
        if animals[i] in top4:
            hits += 1
    
    return hits / TEST_PERIODS


# ============================================================
# 方法3: 元策略切换器
# ============================================================
def method_meta_switcher(lookback=10):
    """根据近N期表现选择最佳单一策略"""
    strat_funcs = [
        lambda h: sorted(ZODIAC_CYCLE, key=lambda z: -score_cold(h, 10)[z])[:4],
        lambda h: sorted(ZODIAC_CYCLE, key=lambda z: -score_cold(h, 15)[z])[:4],
        lambda h: sorted(ZODIAC_CYCLE, key=lambda z: -score_cold(h, 20)[z])[:4],
        lambda h: sorted(ZODIAC_CYCLE, key=lambda z: -score_cold(h, 30)[z])[:4],
        lambda h: sorted(ZODIAC_CYCLE, key=lambda z: -score_gap(h)[z])[:4],
        lambda h: sorted(ZODIAC_CYCLE, key=lambda z: -score_markov1(h)[z])[:4],
        lambda h: sorted(ZODIAC_CYCLE, key=lambda z: -(0.5*score_cold(h,20)[z]+0.5*score_gap(h)[z]))[: 4],
        lambda h: sorted(ZODIAC_CYCLE, key=lambda z: -(0.4*score_cold(h,20)[z]+0.3*score_gap(h)[z]+0.3*score_markov1(h)[z]))[:4],
    ]
    n_strats = len(strat_funcs)
    strat_history = [[] for _ in range(n_strats)]
    active = 0
    
    hits = 0
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        actual = animals[i]
        
        # 用当前active策略预测
        top4 = strat_funcs[active](hist)
        hit = actual in top4
        if hit:
            hits += 1
        
        # 记录所有策略的表现
        for si in range(n_strats):
            s_top4 = strat_funcs[si](hist)
            strat_history[si].append(1 if actual in s_top4 else 0)
        
        # 每期检查是否切换
        if pi >= lookback:
            rates = []
            for si in range(n_strats):
                r = sum(strat_history[si][-lookback:]) / lookback
                rates.append(r)
            best_si = np.argmax(rates)
            if best_si != active:
                active = best_si
    
    return hits / TEST_PERIODS


# ============================================================
# 方法4: 去相关投票 + 动态权重
# ============================================================
def method_decorr_voting():
    """策略去相关分析 + 动态权重投票"""
    strat_funcs = [
        ('冷号10', lambda h: sorted(ZODIAC_CYCLE, key=lambda z: -score_cold(h, 10)[z])[:5]),
        ('冷号20', lambda h: sorted(ZODIAC_CYCLE, key=lambda z: -score_cold(h, 20)[z])[:5]),
        ('间隔', lambda h: sorted(ZODIAC_CYCLE, key=lambda z: -score_gap(h)[z])[:5]),
        ('马尔可夫', lambda h: sorted(ZODIAC_CYCLE, key=lambda z: -score_markov1(h)[z])[:5]),
        ('混合', lambda h: sorted(ZODIAC_CYCLE, key=lambda z: -(0.5*score_cold(h,20)[z]+0.5*score_gap(h)[z]))[:5]),
    ]
    n_strats = len(strat_funcs)
    strat_history = [[] for _ in range(n_strats)]
    weights = [1.0] * n_strats
    
    hits = 0
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        actual = animals[i]
        
        # 加权投票
        votes = {z: 0.0 for z in ZODIAC_CYCLE}
        strat_preds = []
        for si, (name, sf) in enumerate(strat_funcs):
            topk = sf(hist)
            strat_preds.append(topk)
            for rank, z in enumerate(topk):
                votes[z] += weights[si] * (5 - rank)
        
        top4 = sorted(ZODIAC_CYCLE, key=lambda z: -votes[z])[:4]
        hit = actual in top4
        if hit:
            hits += 1
        
        # 更新
        for si in range(n_strats):
            strat_history[si].append(1 if actual in strat_preds[si] else 0)
            if len(strat_history[si]) >= 15:
                r = sum(strat_history[si][-15:]) / 15
                weights[si] = max(r, 0.05)
    
    return hits / TEST_PERIODS


# ============================================================
# 方法5: 混合ML+规则（ML概率 + 冷号/间隔启发式融合）
# ============================================================
def method_hybrid_ml_rules(retrain_every=30):
    """ML概率与规则评分融合"""
    hits = 0
    model = None
    last_train = -retrain_every
    
    for pi in range(TEST_PERIODS):
        i = START + pi
        
        if pi - last_train >= retrain_every or model is None:
            train_start = max(30, i - 200)
            X_train, y_train = [], []
            for t in range(train_start, i):
                feat = build_features(animals[:t], numbers[:t])
                X_train.append(feat)
                y_train.append(Z_IDX[animals[t]])
            
            if len(X_train) >= 50:
                model = GradientBoostingClassifier(
                    n_estimators=80, max_depth=4, learning_rate=0.1, random_state=42
                )
                model.fit(np.array(X_train), np.array(y_train))
                last_train = pi
        
        hist = animals[:i]
        
        # ML概率
        ml_proba = {z: 1/12 for z in ZODIAC_CYCLE}
        if model is not None:
            feat = build_features(hist, numbers[:i]).reshape(1, -1)
            proba = model.predict_proba(feat)[0]
            for ci, cls in enumerate(model.classes_):
                ml_proba[ZODIAC_CYCLE[cls]] = proba[ci]
        
        # 规则评分
        s_cold = score_cold(hist, 20)
        s_gap = score_gap(hist)
        s_mk = score_markov1(hist)
        
        # 归一化规则评分
        for scores in [s_cold, s_gap, s_mk]:
            s = sum(scores.values())
            if s > 0:
                for z in scores:
                    scores[z] /= s
        
        # 融合: 40%ML + 25%冷号 + 20%间隔 + 15%马尔可夫
        final = {}
        for z in ZODIAC_CYCLE:
            final[z] = 0.40 * ml_proba[z] + 0.25 * s_cold[z] + 0.20 * s_gap[z] + 0.15 * s_mk[z]
        
        top4 = sorted(ZODIAC_CYCLE, key=lambda z: -final[z])[:4]
        if animals[i] in top4:
            hits += 1
    
    return hits / TEST_PERIODS


# ============================================================
# 方法6: 自适应混合 - 根据连续miss/hit调策略
# ============================================================
def method_adaptive_regime():
    """根据连续不中/命中切换攻守策略"""
    hits = 0
    streak_miss = 0
    streak_hit = 0
    
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        actual = animals[i]
        
        # 防守模式(连续不中>=3): 用最稳的策略
        if streak_miss >= 3:
            s_cold = score_cold(hist, 25)
            s_gap = score_gap(hist)
            final = {z: 0.5*s_cold[z] + 0.5*s_gap[z] for z in ZODIAC_CYCLE}
        # 进攻模式(连续命中>=2): 用马尔可夫追热
        elif streak_hit >= 2:
            s_mk = score_markov1(hist)
            s_cold = score_cold(hist, 15)
            final = {z: 0.6*s_mk[z] + 0.4*s_cold[z] for z in ZODIAC_CYCLE}
        # 平衡模式
        else:
            s_cold = score_cold(hist, 20)
            s_gap = score_gap(hist)
            s_mk = score_markov1(hist)
            final = {z: 0.45*s_cold[z] + 0.30*s_gap[z] + 0.25*s_mk[z] for z in ZODIAC_CYCLE}
        
        top4 = sorted(ZODIAC_CYCLE, key=lambda z: -final[z])[:4]
        hit = actual in top4
        if hit:
            hits += 1
            streak_hit += 1
            streak_miss = 0
        else:
            streak_miss += 1
            streak_hit = 0
    
    return hits / TEST_PERIODS


# ============================================================
# 方法7: 超级融合 - 结合ML + 规则 + 动态 + 投票
# ============================================================
def method_ultimate_fusion(retrain_every=30, ml_weight=0.3):
    """终极融合: ML概率 + 多规则动态权重投票"""
    hits = 0
    model = None
    last_train = -retrain_every
    
    rule_funcs = [
        ('冷号15', lambda h: score_cold(h, 15)),
        ('冷号25', lambda h: score_cold(h, 25)),
        ('间隔', lambda h: score_gap(h)),
        ('马尔可夫', lambda h: score_markov1(h)),
    ]
    rule_history = [[] for _ in range(len(rule_funcs))]
    rule_weights = [0.25] * len(rule_funcs)
    
    for pi in range(TEST_PERIODS):
        i = START + pi
        hist = animals[:i]
        actual = animals[i]
        
        # ML部分
        if pi - last_train >= retrain_every or model is None:
            train_start = max(30, i - 200)
            X_train, y_train = [], []
            for t in range(train_start, i):
                feat = build_features(animals[:t], numbers[:t])
                X_train.append(feat)
                y_train.append(Z_IDX[animals[t]])
            if len(X_train) >= 50:
                model = GradientBoostingClassifier(
                    n_estimators=60, max_depth=3, learning_rate=0.1, random_state=42
                )
                model.fit(np.array(X_train), np.array(y_train))
                last_train = pi
        
        ml_proba = {z: 1/12 for z in ZODIAC_CYCLE}
        if model is not None:
            feat = build_features(hist, numbers[:i]).reshape(1, -1)
            proba = model.predict_proba(feat)[0]
            for ci, cls in enumerate(model.classes_):
                ml_proba[ZODIAC_CYCLE[cls]] = proba[ci]
        
        # 规则部分 - 各策略评分
        all_scores = []
        for ri, (name, sf) in enumerate(rule_funcs):
            s = sf(hist)
            # 归一化到概率
            total = sum(s.values())
            if total > 0:
                s = {z: v/total for z, v in s.items()}
            all_scores.append(s)
        
        # 融合
        final = {z: 0.0 for z in ZODIAC_CYCLE}
        for z in ZODIAC_CYCLE:
            # ML部分
            final[z] += ml_weight * ml_proba[z]
            # 规则部分
            for ri in range(len(rule_funcs)):
                final[z] += (1 - ml_weight) * rule_weights[ri] * all_scores[ri].get(z, 0)
        
        top4 = sorted(ZODIAC_CYCLE, key=lambda z: -final[z])[:4]
        hit = actual in top4
        if hit:
            hits += 1
        
        # 更新规则权重
        for ri in range(len(rule_funcs)):
            s_top4 = sorted(ZODIAC_CYCLE, key=lambda z: -all_scores[ri].get(z, 0))[:4]
            rule_history[ri].append(1 if actual in s_top4 else 0)
            if len(rule_history[ri]) >= 15:
                r = sum(rule_history[ri][-15:]) / 15
                rule_weights[ri] = max(r, 0.05)
        # 归一化
        ws = sum(rule_weights)
        rule_weights = [w/ws for w in rule_weights]
    
    return hits / TEST_PERIODS


# ============================================================
# 辅助函数
# ============================================================
def score_cold(hist, window):
    w = min(window, len(hist))
    freq = Counter(hist[-w:])
    mx = max(freq.values()) if freq else 1
    return {z: 1.0 - freq.get(z, 0) / max(mx, 1) for z in ZODIAC_CYCLE}

def score_gap(hist):
    scores = {}
    for z in ZODIAC_CYCLE:
        last = -1
        for j in range(len(hist)-1, -1, -1):
            if hist[j] == z:
                last = j
                break
        scores[z] = (len(hist) - 1 - last) / 12 if last >= 0 else len(hist) / 12
    return scores

def score_markov1(hist):
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
        total = sum(trans[state].values())
        for z in ZODIAC_CYCLE:
            scores[z] = trans[state].get(z, 0) / total
    return scores


# ============================================================
# 运行对比
# ============================================================
print("\n" + "="*70)
print("方法对比 (300期滚动验证)")
print("="*70)

results = {}

print("\n1️⃣  ML多分类 (GradientBoosting)...")
for retrain in [30, 50]:
    r = method_ml_multiclass(
        lambda: GradientBoostingClassifier(n_estimators=80, max_depth=4, learning_rate=0.1, random_state=42),
        retrain_every=retrain
    )
    print(f"   GB retrain={retrain}: {r*100:.1f}%")
    results[f'ML_GB_r{retrain}'] = r

print("\n2️⃣  ML多分类 (RandomForest)...")
for retrain in [30, 50]:
    r = method_ml_multiclass(
        lambda: RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        retrain_every=retrain
    )
    print(f"   RF retrain={retrain}: {r*100:.1f}%")
    results[f'ML_RF_r{retrain}'] = r

print("\n3️⃣  ML二分类集成...")
r = method_ml_binary_ensemble(retrain_every=30)
print(f"   Binary Ensemble: {r*100:.1f}%")
results['ML_Binary'] = r

print("\n4️⃣  元策略切换...")
for lb in [8, 10, 15, 20]:
    r = method_meta_switcher(lookback=lb)
    print(f"   lookback={lb}: {r*100:.1f}%")
    results[f'Meta_{lb}'] = r

print("\n5️⃣  去相关投票...")
r = method_decorr_voting()
print(f"   {r*100:.1f}%")
results['Decorr'] = r

print("\n6️⃣  混合ML+规则...")
r = method_hybrid_ml_rules(retrain_every=30)
print(f"   {r*100:.1f}%")
results['Hybrid_ML'] = r

print("\n7️⃣  自适应攻守...")
r = method_adaptive_regime()
print(f"   {r*100:.1f}%")
results['Adaptive'] = r

print("\n8️⃣  终极融合...")
for mw in [0.2, 0.3, 0.4, 0.5]:
    for rt in [20, 30, 50]:
        r = method_ultimate_fusion(retrain_every=rt, ml_weight=mw)
        results[f'Ultimate_mw{mw}_r{rt}'] = r
        if r > 0.44:
            print(f"   mw={mw},rt={rt}: {r*100:.1f}% ⭐")
u_best = max([(k,v) for k,v in results.items() if k.startswith('Ultimate')], key=lambda x: x[1])
print(f"   最优: {u_best[0]} = {u_best[1]*100:.1f}%")

# ============================================================
# 排行榜
# ============================================================
print("\n" + "="*70)
print("排行榜 TOP15")
print("="*70)
sorted_results = sorted(results.items(), key=lambda x: -x[1])
for rank, (name, rate) in enumerate(sorted_results[:15], 1):
    mark = "⭐⭐" if rate >= 0.50 else "⭐" if rate >= 0.45 else "★" if rate >= 0.43 else ""
    print(f"  #{rank:>2} {name:>25}: {rate*100:.1f}% {mark}")

print(f"\n总耗时: {time.time()-t0:.1f}秒")
