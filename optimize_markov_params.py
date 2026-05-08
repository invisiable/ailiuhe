"""
v6.0 马尔可夫倍投深度优化 - 在max=10约束下最大化ROI

优化维度:
1. 马尔可夫模式窗口长度 (2/3/4期)
2. 高概率加倍系数 (1.3/1.5/1.8/2.0)
3. 低概率减倍系数 (0.5/0.6/0.7/0.8)
4. 高概率阈值 (40%/45%/50%/55%)
5. 低概率阈值 (20%/25%/30%)
6. 最小样本数 (2/3/4/5)
7. 连败加速因子 (连败>=5期时额外加权)
8. lookback窗口 (8/10/12/15)
"""
import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor
from itertools import product

# === 加载数据生成命中序列 ===
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values
TEST_PERIODS = 300
start_idx = len(df) - TEST_PERIODS
PREDICT_K = 15
TRAIN_WINDOW = 25
BASE_COST = 15
WIN_REWARD = 47
MAX_MUL = 10

predictor = PreciseTop15Predictor()
hit_sequence = []
for i in range(start_idx, len(df)):
    lo = max(0, i - TRAIN_WINDOW)
    train = numbers[lo:i]
    preds = predictor.predict(train, k=PREDICT_K)
    actual = int(numbers[i])
    hit = actual in preds
    predictor.update_performance(preds, actual)
    hit_sequence.append(hit)

hits_total = sum(hit_sequence)
print(f"命中序列: {hits_total}/{TEST_PERIODS} = {hits_total/TEST_PERIODS*100:.1f}%")

FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

# === 参数化马尔可夫策略 ===
def simulate_markov(hit_seq, pause_mode=1, 
                    pattern_len=3, boost_factor=1.5, reduce_factor=0.7,
                    high_thresh=0.5, low_thresh=0.25, min_samples=3,
                    streak_boost=False, streak_threshold=5, streak_extra=1.2,
                    lookback=12):
    """参数化马尔可夫策略模拟"""
    fib_idx = 0
    balance = 0
    total_bet = 0
    total_win = 0
    min_balance = 0
    streak_loss = 0
    max_streak_loss = 0
    consecutive_miss = 0
    max_consecutive_miss = 0
    hit_max_count = 0
    recent = []
    pattern_stats = {}
    
    pause_rem = 0
    bet_periods = 0
    
    for h in hit_seq:
        if pause_rem > 0:
            pause_rem -= 1
            continue
        
        bet_periods += 1
        
        # 更新模式统计(处理前)
        if len(recent) >= pattern_len:
            pat = tuple(recent[-pattern_len:])
            if pat not in pattern_stats:
                pattern_stats[pat] = [0, 0]
            pattern_stats[pat][1] += 1
            if h:
                pattern_stats[pat][0] += 1
        
        # 获取Fib基础倍数
        base = min(FIB[min(fib_idx, len(FIB)-1)], MAX_MUL)
        
        # 马尔可夫调整
        mul = base
        if len(recent) >= pattern_len:
            pat = tuple(recent[-pattern_len:])
            stats = pattern_stats.get(pat, [0, 0])
            if stats[1] >= min_samples:
                rate = stats[0] / stats[1]
                if rate >= high_thresh:
                    mul = base * boost_factor
                elif rate <= low_thresh:
                    mul = base * reduce_factor
        
        # 连败加速
        if streak_boost and consecutive_miss >= streak_threshold:
            mul = mul * streak_extra
        
        mul = min(max(1, round(mul)), MAX_MUL)
        
        bet = BASE_COST * mul
        total_bet += bet
        if mul >= MAX_MUL:
            hit_max_count += 1
        
        if h:
            win = WIN_REWARD * mul
            total_win += win
            balance += (win - bet)
            fib_idx = 0
            streak_loss = 0
            consecutive_miss = 0
            if pause_mode >= 1:
                pause_rem = pause_mode
        else:
            balance -= bet
            fib_idx += 1
            streak_loss += bet
            consecutive_miss += 1
            if streak_loss > max_streak_loss:
                max_streak_loss = streak_loss
            if consecutive_miss > max_consecutive_miss:
                max_consecutive_miss = consecutive_miss
        
        if balance < min_balance:
            min_balance = balance
        
        recent.append(1 if h else 0)
        if len(recent) > lookback:
            recent.pop(0)
    
    dd = abs(min_balance)
    roi = (total_win - total_bet) / total_bet * 100 if total_bet > 0 else 0
    rr = balance / dd if dd > 0 else 0
    
    return {
        'profit': balance,
        'roi': roi,
        'drawdown': dd,
        'max_streak_loss': max_streak_loss,
        'hit_max_count': hit_max_count,
        'max_consecutive_miss': max_consecutive_miss,
        'risk_reward': rr,
        'bet_periods': bet_periods,
        'total_bet': total_bet,
        'total_win': total_win,
    }

# === 基线: 当前v6.0 ===
baseline = simulate_markov(hit_sequence, pause_mode=1,
    pattern_len=3, boost_factor=1.5, reduce_factor=0.7,
    high_thresh=0.5, low_thresh=0.25, min_samples=3,
    streak_boost=False, lookback=12)
print(f"\n当前v6.0基线: ROI={baseline['roi']:.1f}%, 利润={baseline['profit']:.0f}元, "
      f"回撤={baseline['drawdown']:.0f}元, 风险比={baseline['risk_reward']:.2f}")

# === 第一轮: 逐维度粗扫 ===
print(f"\n{'='*100}")
print("第一轮: 逐维度粗扫")
print(f"{'='*100}")

# 1. 模式窗口
print(f"\n--- 模式窗口长度 ---")
for pl in [2, 3, 4]:
    r = simulate_markov(hit_sequence, pattern_len=pl)
    print(f"  窗口={pl}: ROI={r['roi']:.1f}%, 利润={r['profit']:.0f}, 回撤={r['drawdown']:.0f}, 风险比={r['risk_reward']:.2f}")

# 2. 加倍系数
print(f"\n--- 高概率加倍系数 ---")
for bf in [1.0, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5]:
    r = simulate_markov(hit_sequence, boost_factor=bf)
    print(f"  boost={bf}: ROI={r['roi']:.1f}%, 利润={r['profit']:.0f}, 回撤={r['drawdown']:.0f}, 风险比={r['risk_reward']:.2f}")

# 3. 减倍系数
print(f"\n--- 低概率减倍系数 ---")
for rf in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    r = simulate_markov(hit_sequence, reduce_factor=rf)
    print(f"  reduce={rf}: ROI={r['roi']:.1f}%, 利润={r['profit']:.0f}, 回撤={r['drawdown']:.0f}, 风险比={r['risk_reward']:.2f}")

# 4. 高概率阈值
print(f"\n--- 高概率阈值 ---")
for ht in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    r = simulate_markov(hit_sequence, high_thresh=ht)
    print(f"  高阈值={ht:.0%}: ROI={r['roi']:.1f}%, 利润={r['profit']:.0f}, 回撤={r['drawdown']:.0f}, 风险比={r['risk_reward']:.2f}")

# 5. 低概率阈值
print(f"\n--- 低概率阈值 ---")
for lt in [0.15, 0.20, 0.25, 0.30, 0.35]:
    r = simulate_markov(hit_sequence, low_thresh=lt)
    print(f"  低阈值={lt:.0%}: ROI={r['roi']:.1f}%, 利润={r['profit']:.0f}, 回撤={r['drawdown']:.0f}, 风险比={r['risk_reward']:.2f}")

# 6. 最小样本数
print(f"\n--- 最小样本数 ---")
for ms in [1, 2, 3, 4, 5, 6]:
    r = simulate_markov(hit_sequence, min_samples=ms)
    print(f"  样本>={ms}: ROI={r['roi']:.1f}%, 利润={r['profit']:.0f}, 回撤={r['drawdown']:.0f}, 风险比={r['risk_reward']:.2f}")

# 7. 连败加速
print(f"\n--- 连败加速 ---")
for st, se in [(4, 1.2), (4, 1.3), (5, 1.2), (5, 1.3), (5, 1.5), (6, 1.3), (6, 1.5)]:
    r = simulate_markov(hit_sequence, streak_boost=True, streak_threshold=st, streak_extra=se)
    print(f"  连败>={st}+{se}x: ROI={r['roi']:.1f}%, 利润={r['profit']:.0f}, 回撤={r['drawdown']:.0f}, 风险比={r['risk_reward']:.2f}")

# 8. lookback窗口
print(f"\n--- lookback窗口 ---")
for lb in [6, 8, 10, 12, 15, 20]:
    r = simulate_markov(hit_sequence, lookback=lb)
    print(f"  lookback={lb}: ROI={r['roi']:.1f}%, 利润={r['profit']:.0f}, 回撤={r['drawdown']:.0f}, 风险比={r['risk_reward']:.2f}")

# === 第二轮: 网格搜索最优组合 ===
print(f"\n{'='*100}")
print("第二轮: 网格搜索最优组合 (TOP30)")
print(f"{'='*100}")

results = []
param_grid = list(product(
    [2, 3],              # pattern_len
    [1.3, 1.5, 1.8, 2.0, 2.5],  # boost
    [0.4, 0.5, 0.6, 0.7],       # reduce
    [0.40, 0.45, 0.50],         # high_thresh
    [0.20, 0.25, 0.30],         # low_thresh
    [2, 3],                      # min_samples
    [False, True],               # streak_boost
))

print(f"总参数组合: {len(param_grid)}")

for pl, bf, rf, ht, lt, ms, sb in param_grid:
    r = simulate_markov(hit_sequence, pause_mode=1,
        pattern_len=pl, boost_factor=bf, reduce_factor=rf,
        high_thresh=ht, low_thresh=lt, min_samples=ms,
        streak_boost=sb, streak_threshold=5, streak_extra=1.3,
        lookback=12)
    results.append({
        'params': f"窗口{pl} boost{bf} reduce{rf} 高{ht:.0%} 低{lt:.0%} 样本{ms} 连败{'是' if sb else '否'}",
        'pl': pl, 'bf': bf, 'rf': rf, 'ht': ht, 'lt': lt, 'ms': ms, 'sb': sb,
        **r
    })

# 按ROI排序
by_roi = sorted(results, key=lambda x: x['roi'], reverse=True)
print(f"\n--- TOP30 按ROI排序 ---")
print(f"{'排名':<4} {'参数':<55} {'ROI':>7} {'利润':>8} {'回撤':>7} {'连挂':>7} {'风险比':>7}")
print(f"{'-'*100}")
for i, r in enumerate(by_roi[:30], 1):
    marker = " ⭐" if i <= 5 else ""
    print(f"  {i:<3} {r['params']:<53} {r['roi']:>6.1f}% {r['profit']:>+7.0f} {r['drawdown']:>6.0f} {r['max_streak_loss']:>6.0f} {r['risk_reward']:>6.2f}{marker}")

# 按风险比排序
by_rr = sorted(results, key=lambda x: x['risk_reward'], reverse=True)
print(f"\n--- TOP15 按风险收益比排序 ---")
print(f"{'排名':<4} {'参数':<55} {'ROI':>7} {'利润':>8} {'回撤':>7} {'连挂':>7} {'风险比':>7}")
print(f"{'-'*100}")
for i, r in enumerate(by_rr[:15], 1):
    marker = " ⭐" if i <= 5 else ""
    print(f"  {i:<3} {r['params']:<53} {r['roi']:>6.1f}% {r['profit']:>+7.0f} {r['drawdown']:>6.0f} {r['max_streak_loss']:>6.0f} {r['risk_reward']:>6.2f}{marker}")

# === 综合评分TOP10 ===
print(f"\n{'='*100}")
print("综合评分 TOP10 (ROI权重40% + 风险比权重30% + 低回撤权重20% + 低连挂权重10%)")
print(f"{'='*100}")

max_roi = max(r['roi'] for r in results)
min_roi = min(r['roi'] for r in results)
max_rr = max(r['risk_reward'] for r in results)
min_rr = min(r['risk_reward'] for r in results)
max_dd = max(r['drawdown'] for r in results)
min_dd = min(r['drawdown'] for r in results)
max_sl = max(r['max_streak_loss'] for r in results)
min_sl = min(r['max_streak_loss'] for r in results)

for r in results:
    roi_score = (r['roi'] - min_roi) / (max_roi - min_roi) if max_roi > min_roi else 0
    rr_score = (r['risk_reward'] - min_rr) / (max_rr - min_rr) if max_rr > min_rr else 0
    dd_score = 1 - (r['drawdown'] - min_dd) / (max_dd - min_dd) if max_dd > min_dd else 0
    sl_score = 1 - (r['max_streak_loss'] - min_sl) / (max_sl - min_sl) if max_sl > min_sl else 0
    r['composite'] = roi_score * 0.4 + rr_score * 0.3 + dd_score * 0.2 + sl_score * 0.1

by_composite = sorted(results, key=lambda x: x['composite'], reverse=True)
print(f"{'排名':<4} {'综合分':>6} {'参数':<55} {'ROI':>7} {'利润':>8} {'回撤':>7} {'连挂':>7} {'风险比':>7}")
print(f"{'-'*110}")
for i, r in enumerate(by_composite[:10], 1):
    marker = " 🏆" if i <= 3 else ""
    print(f"  {i:<3} {r['composite']:>5.3f} {r['params']:<53} {r['roi']:>6.1f}% {r['profit']:>+7.0f} {r['drawdown']:>6.0f} {r['max_streak_loss']:>6.0f} {r['risk_reward']:>6.2f}{marker}")

# === 最优方案详情 ===
best = by_composite[0]
print(f"\n{'='*100}")
print(f"🏆 最优方案 vs 当前v6.0")
print(f"{'='*100}")
print(f"{'指标':<15} {'v6.0当前':>12} {'最优方案':>12} {'变化':>12}")
print(f"{'-'*55}")
print(f"  {'ROI':<13} {baseline['roi']:>11.1f}% {best['roi']:>11.1f}% {best['roi']-baseline['roi']:>+11.1f}%")
print(f"  {'净利润':<13} {baseline['profit']:>10.0f}元 {best['profit']:>10.0f}元 {best['profit']-baseline['profit']:>+10.0f}元")
print(f"  {'回撤':<13} {baseline['drawdown']:>10.0f}元 {best['drawdown']:>10.0f}元 {best['drawdown']-baseline['drawdown']:>+10.0f}元")
print(f"  {'连挂总额':<13} {baseline['max_streak_loss']:>10.0f}元 {best['max_streak_loss']:>10.0f}元 {best['max_streak_loss']-baseline['max_streak_loss']:>+10.0f}元")
print(f"  {'风险比':<13} {baseline['risk_reward']:>11.2f} {best['risk_reward']:>11.2f} {best['risk_reward']-baseline['risk_reward']:>+11.2f}")
print(f"  {'触顶次数':<13} {baseline['hit_max_count']:>10}次 {best['hit_max_count']:>10}次")
print(f"\n最优参数: 窗口={best['pl']}, boost={best['bf']}, reduce={best['rf']}, "
      f"高阈值={best['ht']:.0%}, 低阈值={best['lt']:.0%}, 样本>={best['ms']}, 连败加速={'是' if best['sb'] else '否'}")

# 同时输出ROI最高和风险比最高的方案
best_roi = by_roi[0]
best_rr = by_rr[0]
print(f"\nROI最高: 窗口={best_roi['pl']} boost={best_roi['bf']} reduce={best_roi['rf']} "
      f"高{best_roi['ht']:.0%} 低{best_roi['lt']:.0%} 样本{best_roi['ms']} → ROI={best_roi['roi']:.1f}% 利润={best_roi['profit']:.0f} 回撤={best_roi['drawdown']:.0f} 风险比={best_roi['risk_reward']:.2f}")
print(f"风险比最高: 窗口={best_rr['pl']} boost={best_rr['bf']} reduce={best_rr['rf']} "
      f"高{best_rr['ht']:.0%} 低{best_rr['lt']:.0%} 样本{best_rr['ms']} → ROI={best_rr['roi']:.1f}% 利润={best_rr['profit']:.0f} 回撤={best_rr['drawdown']:.0f} 风险比={best_rr['risk_reward']:.2f}")
