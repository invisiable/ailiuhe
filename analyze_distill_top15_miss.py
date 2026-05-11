"""蒸馏TOP15 深度分析 - 连续miss模式与改进方案探索"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from collections import Counter
from distill_top15_predictor import DistillTop15Predictor
from zodiac_top9_predictor import NUM_TO_ZODIAC_2026, ZODIAC_NUMS_2026

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values.tolist()
total = len(df)
test_periods = min(300, total - 30)
start_idx = total - test_periods

predictor = DistillTop15Predictor()

sep = '=' * 90

# ====== 1. 收集每期详细数据 ======
records = []
for i in range(start_idx, total):
    hist = numbers[:i]
    actual = numbers[i]
    final_nums, details, top9_z = predictor.predict_with_details(hist, top_n=15)
    hit = actual in final_nums
    
    # 实际号码在TOP9生肖池中吗?
    top9_pool = set()
    for z in top9_z:
        top9_pool.update(ZODIAC_NUMS_2026[z])
    in_top9_pool = actual in top9_pool
    
    # 实际号码在原始Top15中吗?
    in_original = actual in details['original_top15']
    
    # 实际号码的排名(在full_ranked中)
    actual_zodiac = NUM_TO_ZODIAC_2026[actual]
    
    records.append({
        'period': i - start_idx + 1,
        'actual': actual,
        'actual_zodiac': actual_zodiac,
        'hit': hit,
        'in_top9_pool': in_top9_pool,
        'in_original_top15': in_original,
        'kept_count': details['kept_count'],
        'supplement_count': details['supplement_count'],
        'excluded_count': len(details['excluded']),
        'top9_pool_size': details['top9_pool_size'],
        'final_nums': final_nums,
        'original_top15': details['original_top15'],
        'excluded': details['excluded'],
    })

hit_records = [r['hit'] for r in records]
hits = sum(hit_records)
print(f'{sep}')
print(f'蒸馏TOP15 深度分析 - 300期数据')
print(f'{sep}')
print(f'命中: {hits}/300 = {hits/3:.1f}%\n')

# ====== 2. miss原因分类 ======
print(f'{sep}')
print(f'一、MISS原因分类分析')
print(f'{sep}')

miss_records = [r for r in records if not r['hit']]
miss_count = len(miss_records)

# 分类: 实际号码不在TOP9池(Stage2过滤错误) vs 在池中但未被选中(Stage1排名问题)
miss_not_in_pool = [r for r in miss_records if not r['in_top9_pool']]
miss_in_pool_not_selected = [r for r in miss_records if r['in_top9_pool']]

print(f'总miss: {miss_count}期')
print(f'  A类: 实际号码不在TOP9生肖池(TOP9预测失败): {len(miss_not_in_pool)}期 ({len(miss_not_in_pool)/miss_count*100:.1f}%)')
print(f'  B类: 在TOP9池中但未被蒸馏TOP15选中: {len(miss_in_pool_not_selected)}期 ({len(miss_in_pool_not_selected)/miss_count*100:.1f}%)')

# B类进一步: 在原始Top15但被过滤 vs 两者都没选中
miss_b_was_in_original = [r for r in miss_in_pool_not_selected if r['in_original_top15']]
miss_b_neither = [r for r in miss_in_pool_not_selected if not r['in_original_top15']]
print(f'    B1: 在TOP9池且在原始Top15中(不应发生): {len(miss_b_was_in_original)}期')
print(f'    B2: 在TOP9池但不在原始Top15中(Precise排名低): {len(miss_b_neither)}期')

# ====== 3. 连续miss分析 ======
print(f'\n{sep}')
print(f'二、连续MISS详情')
print(f'{sep}')

streaks = []
cur_streak = []
for r in records:
    if not r['hit']:
        cur_streak.append(r)
    else:
        if cur_streak:
            streaks.append(cur_streak)
        cur_streak = []
if cur_streak:
    streaks.append(cur_streak)

# 按长度排序
long_streaks = sorted(streaks, key=len, reverse=True)[:10]
print(f'最长的10段连续miss:')
for rank, streak in enumerate(long_streaks, 1):
    start_p = streak[0]['period']
    end_p = streak[-1]['period']
    length = len(streak)
    # miss原因分布
    a_count = sum(1 for r in streak if not r['in_top9_pool'])
    b_count = sum(1 for r in streak if r['in_top9_pool'])
    nums = [r['actual'] for r in streak]
    print(f'  #{rank}: 第{start_p}-{end_p}期 ({length}期连miss) | A类(池外):{a_count} B类(池内):{b_count} | 号码:{nums}')

# ====== 4. TOP9生肖命中率 ======
print(f'\n{sep}')
print(f'三、TOP9生肖过滤效果')
print(f'{sep}')
top9_hits = sum(1 for r in records if r['in_top9_pool'])
print(f'TOP9生肖命中: {top9_hits}/300 = {top9_hits/3:.1f}%')
print(f'TOP9池平均大小: {np.mean([r["top9_pool_size"] for r in records]):.1f}个号码')
print(f'蒸馏后从{np.mean([r["top9_pool_size"] for r in records]):.0f}个中选15个')

# ====== 5. 扩展到不同K值的命中率 ======
print(f'\n{sep}')
print(f'四、不同号码数量的命中率对比')
print(f'{sep}')

# 重新跑不同K值
for k in [15, 18, 20, 23, 25]:
    k_hits = 0
    for i in range(start_idx, total):
        hist = numbers[:i]
        actual = numbers[i]
        final_nums, _, _ = predictor.predict_with_details(hist, top_n=k)
        if actual in final_nums:
            k_hits += 1
    cost = k
    breakeven = k / 47 * 100
    rate = k_hits / 300 * 100
    profit = k_hits * 47 - 300 * k
    print(f'  TOP{k}: {k_hits}/300={rate:.1f}% | 成本{k}元/倍 | 盈亏线{breakeven:.1f}% | 净利{profit:+d}元 | ROI{profit/(300*k)*100:+.1f}%')

# ====== 6. 反miss策略模拟 ======
print(f'\n{sep}')
print(f'五、反miss扩展策略模拟')
print(f'{sep}')
print(f'策略: 连续miss>=N期时, 扩展号码数从15→M个')

# 收集每期的full_ranked用于模拟
for expand_threshold in [2, 3]:
    for expand_to in [18, 20, 23]:
        sim_hits = 0
        sim_cost = 0
        sim_reward = 0
        sim_consec_miss = 0
        sim_max_miss = 0
        sim_hit_list = []
        
        for idx, r in enumerate(records):
            actual = r['actual']
            if sim_consec_miss >= expand_threshold:
                # 扩展: 用更大的K
                hist = numbers[:start_idx + idx]
                final_nums, _, _ = predictor.predict_with_details(hist, top_n=expand_to)
                bet_cost = expand_to
            else:
                final_nums = r['final_nums']
                bet_cost = 15
            
            hit = actual in final_nums
            sim_hit_list.append(hit)
            sim_cost += bet_cost
            if hit:
                sim_reward += 47
                sim_consec_miss = 0
            else:
                sim_consec_miss += 1
                sim_max_miss = max(sim_max_miss, sim_consec_miss)
        
        sim_hits = sum(sim_hit_list)
        sim_profit = sim_reward - sim_cost
        sim_roi = sim_profit / sim_cost * 100
        
        # 计算新的最大连续miss
        max_cm = 0
        cm = 0
        for h in sim_hit_list:
            if not h:
                cm += 1
                max_cm = max(max_cm, cm)
            else:
                cm = 0
        
        print(f'  连miss≥{expand_threshold}→扩展到TOP{expand_to}: 命中{sim_hits}/300={sim_hits/3:.1f}% | 成本{sim_cost}元 | 利润{sim_profit:+d}元 | ROI{sim_roi:+.1f}% | maxMiss={max_cm}')

# ====== 7. 混合补充策略模拟 ======
print(f'\n{sep}')
print(f'六、热号补充策略模拟')
print(f'{sep}')
print(f'策略: 连续miss>=N期时, 替换蒸馏TOP15中排名最低的M个为近期热号')

for miss_thresh in [2, 3]:
    for replace_count in [3, 5]:
        sim_hits = 0
        sim_cost = 0
        sim_reward = 0
        sim_consec_miss = 0
        sim_hit_list = []
        
        for idx, r in enumerate(records):
            actual = r['actual']
            final_nums = list(r['final_nums'])
            
            if sim_consec_miss >= miss_thresh:
                # 用近期热号替换末尾
                hist = numbers[:start_idx + idx]
                recent = hist[-30:]
                freq = Counter(recent)
                hot_nums = [n for n, _ in freq.most_common(49)]
                
                # 移除已在final_nums中的
                hot_candidates = [n for n in hot_nums if n not in final_nums]
                
                # 替换末尾replace_count个
                replaced = final_nums[:-replace_count] + hot_candidates[:replace_count]
                hit = actual in replaced
            else:
                hit = actual in final_nums
            
            sim_hit_list.append(hit)
            sim_cost += 15  # 成本不变，始终15个号
            if hit:
                sim_reward += 47
                sim_consec_miss = 0
            else:
                sim_consec_miss += 1
        
        sim_hits = sum(sim_hit_list)
        sim_profit = sim_reward - sim_cost
        sim_roi = sim_profit / sim_cost * 100
        max_cm = 0
        cm = 0
        for h in sim_hit_list:
            if not h:
                cm += 1
                max_cm = max(max_cm, cm)
            else:
                cm = 0
        print(f'  连miss≥{miss_thresh}→替换末{replace_count}为热号: 命中{sim_hits}/300={sim_hits/3:.1f}% | 利润{sim_profit:+d}元 | ROI{sim_roi:+.1f}% | maxMiss={max_cm}')

# ====== 8. 全域号码补充策略 ======
print(f'\n{sep}')
print(f'七、突破TOP9池限制策略模拟')
print(f'{sep}')
print(f'策略: 连续miss>=N期时, 补充非TOP9池的热号(打破生肖过滤限制)')

for miss_thresh in [2, 3, 4]:
    for replace_count in [3, 5]:
        sim_hits = 0
        sim_cost = 0
        sim_reward = 0
        sim_consec_miss = 0
        sim_hit_list = []
        
        for idx, r in enumerate(records):
            actual = r['actual']
            final_nums = list(r['final_nums'])
            
            if sim_consec_miss >= miss_thresh:
                hist = numbers[:start_idx + idx]
                # 用Precise模型的排名中, 取不在当前final_nums中的
                precise_ranked = predictor.precise.predict(hist, k=49)
                # 取排名最高但不在final_nums中的(不限TOP9池)
                outside_candidates = [n for n in precise_ranked if n not in final_nums]
                
                replaced = final_nums[:-replace_count] + outside_candidates[:replace_count]
                hit = actual in replaced
            else:
                hit = actual in final_nums
            
            sim_hit_list.append(hit)
            sim_cost += 15
            if hit:
                sim_reward += 47
                sim_consec_miss = 0
            else:
                sim_consec_miss += 1
        
        sim_hits = sum(sim_hit_list)
        sim_profit = sim_reward - sim_cost
        sim_roi = sim_profit / sim_cost * 100
        max_cm = 0
        cm = 0
        for h in sim_hit_list:
            if not h:
                cm += 1
                max_cm = max(max_cm, cm)
            else:
                cm = 0
        print(f'  连miss≥{miss_thresh}→Precise池外补{replace_count}: 命中{sim_hits}/300={sim_hits/3:.1f}% | 利润{sim_profit:+d}元 | ROI{sim_roi:+.1f}% | maxMiss={max_cm}')

print(f'\n{sep}')
print(f'分析完成')
print(sep)
