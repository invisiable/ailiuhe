# -*- coding: utf-8 -*-
"""
比较不同预测器的命中率，寻找能达到40%+命中率的模型
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

# 加载数据
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].tolist()
dates = df['date'].tolist()

test_periods = 300
test_start = len(numbers) - test_periods

print("=" * 70)
print("比较不同预测器命中率 - 寻找40%+命中率的模型")
print("=" * 70)
print(f"数据: {len(numbers)}期, 测试: {test_periods}期")
print(f"时间范围: {dates[test_start]} ~ {dates[-1]}")
print()

# 测试不同预测器
predictors_to_test = []

# 1. Top15 Statistical
try:
    from top15_statistical_predictor import Top15StatisticalPredictor
    predictors_to_test.append(('Top15统计', Top15StatisticalPredictor()))
except Exception as e:
    print(f"加载Top15统计失败: {e}")

# 2. Ensemble Top15
try:
    from ensemble_top15_predictor import EnsembleTop15Predictor
    predictors_to_test.append(('集成Top15', EnsembleTop15Predictor()))
except Exception as e:
    print(f"加载集成Top15失败: {e}")

# 3. Advanced Top15
try:
    from advanced_top15_predictor import AdvancedTop15Predictor
    predictors_to_test.append(('高级Top15', AdvancedTop15Predictor()))
except Exception as e:
    print(f"加载高级Top15失败: {e}")

# 4. Zodiac Enhanced (买4个生肖 = 16-17个号码)
try:
    from zodiac_enhanced_60_predictor import ZodiacEnhanced60Predictor
    predictors_to_test.append(('生肖增强60', ZodiacEnhanced60Predictor()))
except Exception as e:
    print(f"加载生肖增强60失败: {e}")

# 5. Zodiac v11
try:
    from zodiac_v11_predictor import ZodiacV11Predictor
    predictors_to_test.append(('生肖v11', ZodiacV11Predictor()))
except Exception as e:
    print(f"加载生肖v11失败: {e}")

print(f"成功加载 {len(predictors_to_test)} 个预测器")
print()

# 测试每个预测器
results = []

for name, predictor in predictors_to_test:
    print(f"测试 {name}...")
    hits = 0
    total_numbers = 0  # 预测的号码总数
    
    for i in range(test_start, len(numbers)):
        history = numbers[:i]
        actual = numbers[i]
        
        try:
            pred = predictor.predict(history)
            if isinstance(pred, (list, tuple, set)):
                pred_list = list(pred)
            else:
                pred_list = [pred]
            
            total_numbers += len(pred_list)
            if actual in pred_list:
                hits += 1
        except Exception as e:
            continue
    
    hit_rate = hits / test_periods * 100
    avg_numbers = total_numbers / test_periods
    
    # 计算理论ROI (假设每个号码投注15元，命中得45元)
    # 如果是Top15，投注15*15=225元，中奖得45元
    # 实际情况：投15元买一个生肖组（4-5个号码），中奖得45元
    
    # 简化计算：假设统一投注
    theoretical_roi = (hit_rate * 30 - (100 - hit_rate) * 15) / 15
    
    results.append({
        'name': name,
        'hit_rate': hit_rate,
        'hits': hits,
        'avg_numbers': avg_numbers,
        'theoretical_roi': theoretical_roi
    })

# 排序并显示
results.sort(key=lambda x: x['hit_rate'], reverse=True)

print()
print("=" * 70)
print("【预测器命中率排名】")
print("=" * 70)
print(f"{'排名':<4} {'预测器':<15} {'命中率':<10} {'命中次数':<10} {'平均预测数':<12} {'理论ROI':<10}")
print("-" * 70)

for i, r in enumerate(results, 1):
    status = "✅ 达标" if r['hit_rate'] >= 40 else ("⚠️ 接近" if r['hit_rate'] >= 35 else "❌ 不足")
    print(f"{i:<4} {r['name']:<15} {r['hit_rate']:.1f}%      {r['hits']:<10} {r['avg_numbers']:.1f}          {r['theoretical_roi']:.1f}%     {status}")

print()
print("=" * 70)
print("【结论】")
print("=" * 70)

best = results[0] if results else None
if best and best['hit_rate'] >= 40:
    print(f"✅ 找到40%+命中率的模型: {best['name']} ({best['hit_rate']:.1f}%)")
    print(f"   理论ROI可达: {best['theoretical_roi']:.1f}%")
elif best and best['hit_rate'] >= 35:
    print(f"⚠️ 最佳模型: {best['name']} ({best['hit_rate']:.1f}%)")
    print(f"   理论ROI: {best['theoretical_roi']:.1f}%")
    print(f"   距离20% ROI目标还需提升命中率到40%")
else:
    print(f"❌ 当前最佳模型: {best['name']} ({best['hit_rate']:.1f}%)")
    print(f"   理论ROI: {best['theoretical_roi']:.1f}%")
    print(f"   需要大幅提升预测准确率才能达到20% ROI")

print()
print("【提升ROI的可能方向】")
print("1. 提升预测模型准确率到40%+")
print("2. 减少预测数量（如Top10），提高命中时的盈利比")
print("3. 使用条件投注：只在高胜率情况下投注")
print("4. 组合多个模型投票，提高准确率")
