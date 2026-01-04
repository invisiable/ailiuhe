"""
深度分析v10.0最近10期失败率高的原因
"""

import pandas as pd
from collections import Counter
from zodiac_simple_smart import ZodiacSimpleSmart

def analyze_recent10_failure():
    predictor = ZodiacSimpleSmart()
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    
    print("="*110)
    print("v10.0 最近10期失败率分析")
    print("="*110)
    
    total = len(df)
    recent_10_data = df.iloc[-10:]  # 最近10期
    
    # 分析每一期的详细情况
    print("\n【详细分析】最近10期逐期检查:")
    print("-"*110)
    
    for idx in range(10):
        print(f"\n第{idx+1}期 ({recent_10_data.iloc[idx]['date']}):")
        
        # 获取历史数据
        animals = [str(a).strip() for a in df['animal'].values[:total - 10 + idx]]
        
        # 实际结果
        actual = str(recent_10_data.iloc[idx]['animal']).strip()
        
        # 分析最近20期数据特征
        recent_20 = animals[-20:] if len(animals) >= 20 else animals
        counter = Counter(recent_20)
        
        diversity = len(counter) / 12
        hot_zodiacs = sum(1 for count in counter.values() if count >= 3)
        concentration = sum(count for zodiac, count in counter.items() if count >= 3) / len(recent_20) if counter else 0
        
        # 获取预测
        scenario = predictor._detect_scenario(animals)
        prediction = predictor.predict_from_history(animals, top_n=5, debug=False)
        
        predicted_top5 = prediction['top5']
        model_used = prediction['selected_model']
        
        # 判断命中
        hit = actual in predicted_top5
        rank = predicted_top5.index(actual) + 1 if hit else 0
        
        # 打印详情
        print(f"  实际生肖: {actual}")
        print(f"  预测TOP5: {', '.join(predicted_top5)}")
        print(f"  命中情况: {'✓ TOP' + str(rank) if hit else '✗ 未命中'}")
        print(f"  选择模型: {model_used}")
        print(f"  场景判断: {scenario}")
        print(f"  数据特征: diversity={diversity:.2f}, concentration={concentration:.2f}, hot_count={hot_zodiacs}")
        
        # 分析实际生肖在最近的出现频率
        actual_count_20 = recent_20.count(actual)
        actual_count_30 = animals[-30:].count(actual) if len(animals) >= 30 else animals.count(actual)
        
        print(f"  '{actual}'在最近20期出现: {actual_count_20}次")
        print(f"  '{actual}'在最近30期出现: {actual_count_30}次")
        
        # 检查是否应该预测到
        if actual_count_20 >= 3:
            print(f"  ⚠️  '{actual}'是高频生肖（20期内{actual_count_20}次），但{'命中' if hit else '未被预测'}")
        elif actual_count_20 == 0:
            print(f"  ⚠️  '{actual}'在20期内未出现，冷门生肖突然出现")
        
        # 分析预测的TOP5在最近的频率
        print(f"  预测TOP5频率分析:")
        for z in predicted_top5:
            z_count = recent_20.count(z)
            print(f"    {z}: 最近20期{z_count}次", end="")
            if z_count >= 3:
                print(" (高频)", end="")
            elif z_count == 0:
                print(" (冷门)", end="")
            print()
    
    # 汇总分析
    print(f"\n{'='*110}")
    print("【汇总分析】最近10期整体特征:")
    print("-"*110)
    
    # 统计最近10期实际出现的生肖
    recent_10_animals = [str(a).strip() for a in recent_10_data['animal'].values]
    recent_10_counter = Counter(recent_10_animals)
    
    print(f"\n最近10期实际出现分布:")
    for zodiac, count in sorted(recent_10_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"  {zodiac}: {count}次", end="")
        if count >= 3:
            print(" ← 高频（热门）")
        else:
            print()
    
    # 检查这些高频生肖是否在之前也是高频
    print(f"\n高频生肖的历史趋势分析:")
    for zodiac, count in recent_10_counter.items():
        if count >= 2:  # 最近10期出现2次以上
            # 查看前20期（10-30期前）的频率
            prev_20_animals = df.iloc[-30:-10]['animal'].values
            prev_count = sum(1 for a in prev_20_animals if str(a).strip() == zodiac)
            
            print(f"  {zodiac}:")
            print(f"    最近10期: {count}次")
            print(f"    前20期: {prev_count}次")
            
            if count >= 2 and prev_count >= 3:
                print(f"    → 持续热门（应该被预测到）")
            elif count >= 2 and prev_count <= 1:
                print(f"    → 突然爆发（冷门转热门，难以预测）⚠️")
            elif count >= 2 and 1 < prev_count < 3:
                print(f"    → 逐渐升温")
    
    # 对比模型选择是否合理
    print(f"\n模型选择合理性分析:")
    
    # 重新分析最近10期开始前的20期数据
    animals_before_recent10 = [str(a).strip() for a in df['animal'].values[:-10]]
    recent_20_before = animals_before_recent10[-20:] if len(animals_before_recent10) >= 20 else animals_before_recent10
    counter_before = Counter(recent_20_before)
    
    diversity_before = len(counter_before) / 12
    hot_before = sum(1 for count in counter_before.values() if count >= 3)
    concentration_before = sum(count for zodiac, count in counter_before.items() if count >= 3) / len(recent_20_before) if counter_before else 0
    
    print(f"  最近10期开始前的20期特征:")
    print(f"    diversity={diversity_before:.2f}, concentration={concentration_before:.2f}, hot_count={hot_before}")
    
    scenario_before = predictor._detect_scenario(animals_before_recent10)
    print(f"    场景判断: {scenario_before}")
    
    # 检查实际最近10期的特征
    counter_actual_10 = Counter(recent_10_animals)
    diversity_actual = len(counter_actual_10) / 12
    hot_actual = sum(1 for count in counter_actual_10.values() if count >= 2)  # 10期内>=2就算高频
    concentration_actual = sum(count for zodiac, count in counter_actual_10.items() if count >= 2) / 10
    
    print(f"\n  最近10期的实际特征:")
    print(f"    diversity={diversity_actual:.2f}, concentration={concentration_actual:.2f}, 高频生肖数={hot_actual}")
    print(f"    高频生肖: {[z for z, c in counter_actual_10.items() if c >= 2]}")
    
    # 判断问题
    print(f"\n{'='*110}")
    print("【结论】失败原因诊断:")
    print("-"*110)
    
    # 统计命中和未命中的情况
    hit_count = 0
    for idx in range(10):
        animals = [str(a).strip() for a in df['animal'].values[:total - 10 + idx]]
        prediction = predictor.predict_from_history(animals, top_n=5, debug=False)
        actual = str(recent_10_data.iloc[idx]['animal']).strip()
        if actual in prediction['top5']:
            hit_count += 1
    
    print(f"\n1. 命中率: {hit_count}/10 = {hit_count*10}%")
    
    # 找出高频未命中的生肖
    high_freq_missed = []
    for zodiac, count in counter_actual_10.items():
        if count >= 2:
            # 检查这个生肖在10期中被命中了几次
            zodiac_hit = 0
            for idx in range(10):
                animals = [str(a).strip() for a in df['animal'].values[:total - 10 + idx]]
                prediction = predictor.predict_from_history(animals, top_n=5, debug=False)
                actual = str(recent_10_data.iloc[idx]['animal']).strip()
                if actual == zodiac and zodiac in prediction['top5']:
                    zodiac_hit += 1
            
            if zodiac_hit < count:
                high_freq_missed.append((zodiac, count, zodiac_hit))
    
    if high_freq_missed:
        print(f"\n2. 高频生肖预测失败:")
        for zodiac, total_count, hit_count in sorted(high_freq_missed, key=lambda x: x[1], reverse=True):
            miss_count = total_count - hit_count
            print(f"   '{zodiac}': 出现{total_count}次, 仅预测中{hit_count}次, 漏掉{miss_count}次 ⚠️")
    
    # 检查是否存在"热门突然爆发"的情况
    print(f"\n3. 数据模式变化:")
    sudden_changes = []
    for zodiac, count in counter_actual_10.items():
        if count >= 2:
            prev_20_animals = df.iloc[-30:-10]['animal'].values
            prev_count = sum(1 for a in prev_20_animals if str(a).strip() == zodiac)
            if prev_count <= 1 and count >= 2:
                sudden_changes.append((zodiac, prev_count, count))
    
    if sudden_changes:
        print(f"   发现突然爆发的生肖:")
        for zodiac, prev, now in sudden_changes:
            print(f"   '{zodiac}': 前20期{prev}次 → 最近10期{now}次 (冷门突然转热门) ⚠️")
        print(f"   → 这种突然变化是模型难以预测的")
    else:
        print(f"   未发现明显的突然爆发模式")
    
    print(f"\n4. 模型选择评估:")
    model_usage = {}
    for idx in range(10):
        animals = [str(a).strip() for a in df['animal'].values[:total - 10 + idx]]
        prediction = predictor.predict_from_history(animals, top_n=5, debug=False)
        model_usage[prediction['selected_model']] = model_usage.get(prediction['selected_model'], 0) + 1
    
    for model, count in sorted(model_usage.items(), key=lambda x: x[1], reverse=True):
        print(f"   {model}: {count}次 ({count*10}%)")
    
    # 给出建议
    print(f"\n{'='*110}")
    print("【建议】可能的优化方向:")
    print("-"*110)
    
    if sudden_changes:
        print(f"1. 最近10期存在'冷门突然爆发'现象，这是统计模型的固有局限")
        print(f"2. 可以考虑增加'冷门反转检测'机制")
    
    if high_freq_missed:
        print(f"3. 高频生肖预测失败，可能需要调整热门感知模型的触发条件")
    
    print(f"4. 20%的命中率可能是随机波动，建议观察更长时间")
    
    print(f"\n{'='*110}")

if __name__ == '__main__':
    analyze_recent10_failure()
