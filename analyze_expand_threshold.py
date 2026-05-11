"""蒸馏TOP15 - 扩展阈值对比: 连miss≥N期时扩展到TOP20"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from distill_top15_predictor import DistillTop15Predictor
from zodiac_top9_predictor import NUM_TO_ZODIAC_2026

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values.tolist()
total = len(df)
test_periods = min(300, total - 30)
start_idx = total - test_periods

FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
sep = '=' * 100

print(f'蒸馏TOP15 × Fibonacci - 扩展阈值对比分析')
print(sep)
print(f'数据: {total}期, 回测最近{test_periods}期')
print(f'基础: TOP15(成本15元/倍), 扩展: TOP20(成本20元/倍), 命中赔47元/倍')
print(f'Fibonacci倍投: 不中递进, 命中重置')
print(sep)

# 测试不同阈值
for threshold in [2, 3, 4, 5, 999]:  # 999=永不扩展(纯TOP15)
    predictor = DistillTop15Predictor()
    predictor.expand_threshold = threshold
    
    hit_records = []
    fib_index = 0
    total_cost = 0
    total_reward = 0
    balance = 0
    peak = 0
    max_drawdown = 0
    expand_count = 0
    
    for i in range(start_idx, total):
        hist = numbers[:i]
        actual = numbers[i]
        k = predictor._get_current_k()
        if k > 15:
            expand_count += 1
        final_nums, _, _ = predictor.predict_with_details(hist, top_n=k)
        hit = actual in final_nums
        hit_records.append(hit)
        
        fib_mul = min(FIB[min(fib_index, len(FIB) - 1)], 10)  # 最高10倍
        bet = k * fib_mul
        total_cost += bet
        
        if hit:
            reward = 47 * fib_mul
            total_reward += reward
            balance += reward - bet
            fib_index = 0
        else:
            balance -= bet
            fib_index = min(fib_index + 1, len(FIB) - 1)
        
        peak = max(peak, balance)
        max_drawdown = max(max_drawdown, peak - balance)
        predictor.record_result(hit)
    
    hits = sum(hit_records)
    hit_rate = hits / test_periods * 100
    net_profit = total_reward - total_cost
    roi = net_profit / total_cost * 100
    
    # 连续miss统计
    max_miss = 0
    cur_miss = 0
    streaks = []
    c = 0
    for h in hit_records:
        if not h:
            cur_miss += 1
            max_miss = max(max_miss, cur_miss)
            c += 1
        else:
            cur_miss = 0
            if c > 0:
                streaks.append(c)
            c = 0
    if c > 0:
        streaks.append(c)
    
    ge2 = sum(1 for s in streaks if s >= 2)
    ge3 = sum(1 for s in streaks if s >= 3)
    ge4 = sum(1 for s in streaks if s >= 4)
    ge5 = sum(1 for s in streaks if s >= 5)
    ge6 = sum(1 for s in streaks if s >= 6)
    ge7 = sum(1 for s in streaks if s >= 7)
    ge8 = sum(1 for s in streaks if s >= 8)
    
    label = f"≥{threshold}期→TOP20" if threshold < 999 else "永不扩展(纯TOP15)"
    print(f'\n{"─"*100}')
    print(f'方案: {label}')
    print(f'{"─"*100}')
    print(f'  命中: {hits}/{test_periods} = {hit_rate:.1f}%')
    print(f'  扩展使用: {expand_count}期 ({expand_count/test_periods*100:.1f}%)')
    print(f'  总投入: {total_cost}元, 总回报: {total_reward}元')
    print(f'  净利润: {net_profit:+d}元, ROI: {roi:+.1f}%')
    print(f'  最大回撤: {max_drawdown}元')
    print(f'  最大连续miss: {max_miss}期')
    print(f'  连续miss分布:')
    print(f'    ≥2期: {ge2}次 | ≥3期: {ge3}次 | ≥4期: {ge4}次 | ≥5期: {ge5}次 | ≥6期: {ge6}次 | ≥7期: {ge7}次 | ≥8期: {ge8}次')
    
    # 连续miss详情(≥4期的)
    streak_details = []
    c = 0
    s_start = 0
    for idx, h in enumerate(hit_records):
        if not h:
            if c == 0:
                s_start = idx + 1
            c += 1
        else:
            if c >= 4:
                streak_details.append((s_start, s_start + c - 1, c))
            c = 0
    if c >= 4:
        streak_details.append((s_start, s_start + c - 1, c))
    
    if streak_details:
        print(f'  ≥4期连miss详情:')
        for s, e, l in streak_details:
            nums = [numbers[start_idx + s - 1 + j] for j in range(l)]
            # 计算这段miss的Fib投入
            print(f'    第{s}-{e}期 ({l}期) 号码:{nums}')

# ====== 汇总对比表 ======
print(f'\n{sep}')
print(f'汇总对比表')
print(sep)
print(f'{"方案":<22} {"命中率":>8} {"净利润":>8} {"ROI":>8} {"maxMiss":>8} {"回撤":>8} {"≥3miss":>8} {"≥5miss":>8} {"扩展期":>8}')
print(f'{"-"*98}')

for threshold in [2, 3, 4, 5, 999]:
    predictor = DistillTop15Predictor()
    predictor.expand_threshold = threshold
    
    hit_records = []
    fib_index = 0
    total_cost = 0
    total_reward = 0
    balance = 0
    peak = 0
    max_drawdown = 0
    expand_count = 0
    
    for i in range(start_idx, total):
        hist = numbers[:i]
        actual = numbers[i]
        k = predictor._get_current_k()
        if k > 15:
            expand_count += 1
        final_nums, _, _ = predictor.predict_with_details(hist, top_n=k)
        hit = actual in final_nums
        hit_records.append(hit)
        
        fib_mul = min(FIB[min(fib_index, len(FIB) - 1)], 10)  # 最高10倍
        bet = k * fib_mul
        total_cost += bet
        if hit:
            total_reward += 47 * fib_mul
            fib_index = 0
        else:
            fib_index = min(fib_index + 1, len(FIB) - 1)
        
        balance = total_reward - total_cost
        peak = max(peak, balance)
        max_drawdown = max(max_drawdown, peak - balance)
        predictor.record_result(hit)
    
    hits = sum(hit_records)
    net_profit = total_reward - total_cost
    roi = net_profit / total_cost * 100
    
    max_miss = 0
    cur_miss = 0
    streaks = []
    c = 0
    for h in hit_records:
        if not h:
            cur_miss += 1
            max_miss = max(max_miss, cur_miss)
            c += 1
        else:
            cur_miss = 0
            if c > 0: streaks.append(c)
            c = 0
    if c > 0: streaks.append(c)
    ge3 = sum(1 for s in streaks if s >= 3)
    ge5 = sum(1 for s in streaks if s >= 5)
    
    label = f"≥{threshold}期→TOP20" if threshold < 999 else "纯TOP15"
    print(f'{label:<22} {hits/3:.1f}%{"":<3} {net_profit:>+7}元 {roi:>+6.1f}% {max_miss:>7}期 {max_drawdown:>7}元 {ge3:>7}次 {ge5:>7}次 {expand_count:>7}期')
