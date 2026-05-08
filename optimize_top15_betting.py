"""TOP15最优投注方案搜索 - 基于35.7%命中率，寻找最高ROI"""
import pandas as pd
import numpy as np
from collections import Counter
from precise_top15_predictor import PreciseTop15Predictor

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values
TEST_PERIODS = 300
start_idx = len(df) - TEST_PERIODS
PREDICT_K = 15
TRAIN_WINDOW = 25

# 先生成300期命中序列
predictor = PreciseTop15Predictor()
hit_sequence = []
for i in range(start_idx, len(df)):
    lo = max(0, i - TRAIN_WINDOW)
    train = numbers[lo:i]
    preds = predictor.predict(train, k=PREDICT_K)
    actual = int(numbers[i])
    hit_sequence.append(actual in preds)

hits = sum(hit_sequence)
print(f"TOP15 命中率: {hits}/{TEST_PERIODS} = {hits/TEST_PERIODS*100:.1f}%")
print(f"赔付规则: 买15个数, 成本15元/倍, 命中奖励47元/倍, 净利=32元/倍")

# === 模拟函数 ===
def simulate_strategy(hit_seq, base_cost, win_reward, fib_seq, max_mul, pause_after_hit=0, pause_after_N_miss=0, boost_after_miss=0):
    """通用策略模拟"""
    fib_idx = 0
    balance = 0
    min_balance = 0
    total_bet = 0
    total_win = 0
    streak_loss = 0
    max_streak_loss = 0
    pause_count = 0
    bet_periods = 0
    max_consec_loss = 0
    consec_loss = 0
    hit_10x = 0
    
    for i, hit in enumerate(hit_seq):
        # 暂停逻辑
        if pause_count > 0:
            pause_count -= 1
            continue
        
        bet_periods += 1
        mul = min(fib_seq[min(fib_idx, len(fib_seq)-1)], max_mul)
        bet = base_cost * mul
        total_bet += bet
        
        if mul >= max_mul:
            hit_10x += 1
        
        if hit:
            win = win_reward * mul
            total_win += win
            profit = win - bet
            balance += profit
            fib_idx = 0
            streak_loss = 0
            consec_loss = 0
            if pause_after_hit > 0:
                pause_count = pause_after_hit
        else:
            balance -= bet
            fib_idx += 1
            streak_loss += bet
            consec_loss += 1
            if streak_loss > max_streak_loss:
                max_streak_loss = streak_loss
            if consec_loss > max_consec_loss:
                max_consec_loss = consec_loss
        
        if balance < min_balance:
            min_balance = balance
    
    roi = (total_win - total_bet) / total_bet * 100 if total_bet > 0 else 0
    return {
        'profit': balance,
        'total_bet': total_bet,
        'total_win': total_win,
        'roi': roi,
        'max_drawdown': abs(min_balance),
        'max_streak_loss': max_streak_loss,
        'max_consec_loss': max_consec_loss,
        'bet_periods': bet_periods,
        'hit_10x': hit_10x,
    }

# 斐波那契数列
fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
# 线性数列 
linear = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# Martingale（翻倍）
martingale = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
# 温和递增
mild = [1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8]
# 平注
flat = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# DAlembert
dalembert = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# === 核心参数 ===
# TOP15: 买15个数, 每倍成本15元, 命中47元奖励
BASE_COST = 15  # 15个数 × 1元/数
WIN_REWARD = 47  # 命中赔付47元

print(f"\n{'='*80}")
print(f"测试不同投注策略组合（成本{BASE_COST}元/倍，奖励{WIN_REWARD}元/倍，净利{WIN_REWARD-BASE_COST}元/倍）")
print(f"{'='*80}\n")

# === 1. 测试不同数列 + 不同最大倍数 ===
sequences = {
    '斐波那契': fib,
    '线性递增': linear,
    '温和递增': mild,
    '平注(固定)': flat,
}

print(f"{'策略':<20} {'最大倍':<8} {'净利润':>8} {'ROI':>8} {'回撤':>8} {'连不中额':>10} {'最长连败':>8} {'触顶':>6} {'投注期':>8}")
print(f"{'-'*96}")

best_roi = -999
best_config = ""
all_results = []

for seq_name, seq in sequences.items():
    for max_mul in [5, 8, 10, 15, 20]:
        for pause in [0, 1, 2]:
            r = simulate_strategy(hit_sequence, BASE_COST, WIN_REWARD, seq, max_mul, pause_after_hit=pause)
            label = f"{seq_name}+停{pause}" if pause > 0 else seq_name
            all_results.append((label, max_mul, r))
            
            if r['roi'] > best_roi:
                best_roi = r['roi']
                best_config = f"{label} max={max_mul}"

# Sort by ROI
all_results.sort(key=lambda x: x[2]['roi'], reverse=True)

# Show top 20
for label, max_mul, r in all_results[:20]:
    marker = " ⭐" if r['roi'] >= 20 else ""
    print(f"{label:<20} {max_mul:<8} {r['profit']:>8.0f} {r['roi']:>7.1f}% {r['max_drawdown']:>8.0f} {r['max_streak_loss']:>10.0f} {r['max_consec_loss']:>8} {r['hit_10x']:>6} {r['bet_periods']:>8}{marker}")

print(f"\n最优ROI配置: {best_config} = {best_roi:.1f}%")

# === 2. 深入测试最优区间（斐波那契+暂停变体） ===
print(f"\n{'='*80}")
print(f"斐波那契详细变体测试")
print(f"{'='*80}\n")

print(f"{'配置':<35} {'净利润':>8} {'ROI':>8} {'回撤':>8} {'连不中额':>10} {'最长败':>6} {'风险收益比':>10}")
print(f"{'-'*100}")

for max_mul in [8, 10, 13, 15, 20]:
    for pause in [0, 1, 2]:
        r = simulate_strategy(hit_sequence, BASE_COST, WIN_REWARD, fib, max_mul, pause_after_hit=pause)
        risk_reward = r['profit'] / r['max_drawdown'] if r['max_drawdown'] > 0 else 0
        label = f"Fib max={max_mul} 停{pause}期"
        marker = ""
        if r['roi'] >= 15 and risk_reward >= 1.0:
            marker = " ⭐推荐"
        elif r['roi'] >= 20:
            marker = " 💰高ROI"
        print(f"{label:<35} {r['profit']:>8.0f} {r['roi']:>7.1f}% {r['max_drawdown']:>8.0f} {r['max_streak_loss']:>10.0f} {r['max_consec_loss']:>6} {risk_reward:>10.2f}{marker}")

# === 3. 对比 TOP15 vs TOP23 的经济性 ===
print(f"\n{'='*80}")
print(f"TOP15(成本15元,赔47元) vs TOP23(成本23元,赔47元) 经济性对比")
print(f"{'='*80}\n")

# TOP23 hit sequence
predictor23 = PreciseTop15Predictor()
hit_seq_23 = []
for i in range(start_idx, len(df)):
    lo = max(0, i - TRAIN_WINDOW)
    train = numbers[lo:i]
    preds = predictor23.predict(train, k=23)
    actual = int(numbers[i])
    hit_seq_23.append(actual in preds)

hits_23 = sum(hit_seq_23)

configs = [
    ("TOP15 平注 max=10", hit_sequence, 15, 47, flat, 10, 0),
    ("TOP15 平注+停1 max=10", hit_sequence, 15, 47, flat, 10, 1),
    ("TOP15 Fib max=10", hit_sequence, 15, 47, fib, 10, 0),
    ("TOP15 Fib+停1 max=10", hit_sequence, 15, 47, fib, 10, 1),
    ("TOP15 Fib max=15", hit_sequence, 15, 47, fib, 15, 0),
    ("TOP15 Fib+停1 max=15", hit_sequence, 15, 47, fib, 15, 1),
    ("TOP15 Fib max=20", hit_sequence, 15, 47, fib, 20, 0),
    ("TOP15 Fib+停1 max=20", hit_sequence, 15, 47, fib, 20, 1),
    ("TOP23 平注 max=10", hit_seq_23, 23, 47, flat, 10, 0),
    ("TOP23 平注+停1 max=10", hit_seq_23, 23, 47, flat, 10, 1),
    ("TOP23 Fib max=10", hit_seq_23, 23, 47, fib, 10, 0),
    ("TOP23 Fib+停1 max=10", hit_seq_23, 23, 47, fib, 10, 1),
]

print(f"TOP15 命中率: {sum(hit_sequence)}/{len(hit_sequence)} = {sum(hit_sequence)/len(hit_sequence)*100:.1f}%")
print(f"TOP23 命中率: {hits_23}/{len(hit_seq_23)} = {hits_23/len(hit_seq_23)*100:.1f}%\n")

print(f"{'配置':<30} {'净利润':>8} {'ROI':>8} {'回撤':>8} {'连不中额':>10} {'风险比':>8}")
print(f"{'-'*80}")

for label, hseq, cost, reward, seq, mm, pause in configs:
    r = simulate_strategy(hseq, cost, reward, seq, mm, pause_after_hit=pause)
    rr = r['profit'] / r['max_drawdown'] if r['max_drawdown'] > 0 else 0
    print(f"{label:<30} {r['profit']:>8.0f} {r['roi']:>7.1f}% {r['max_drawdown']:>8.0f} {r['max_streak_loss']:>10.0f} {rr:>8.2f}")

# === 4. 最终推荐 ===
print(f"\n{'='*80}")
print("最终推荐")
print(f"{'='*80}")

# Find best ROI for TOP15
best_top15 = None
best_top15_roi = -999
for label, max_mul, r in all_results:
    if r['roi'] > best_top15_roi:
        best_top15_roi = r['roi']
        best_top15 = (label, max_mul, r)

if best_top15:
    label, max_mul, r = best_top15
    print(f"\nTOP15最优ROI方案:")
    print(f"  策略: {label}, 最大倍数={max_mul}")
    print(f"  净利润: {r['profit']:.0f}元")
    print(f"  ROI: {r['roi']:.1f}%")
    print(f"  最大回撤: {r['max_drawdown']:.0f}元")
    print(f"  连续不中总额: {r['max_streak_loss']:.0f}元")
    print(f"  最长连败: {r['max_consec_loss']}期")
    print(f"  触顶次数: {r['hit_10x']}次")
