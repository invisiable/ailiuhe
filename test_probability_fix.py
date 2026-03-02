#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试修复后的概率预测策略
"""

import pandas as pd
from zodiac_simple_smart import ZodiacSimpleSmart
from zodiac_top5_probability_betting import ZodiacTop5ProbabilityBetting

print("=" * 80)
print("测试概率预测策略修复")
print("=" * 80)
print()

# 加载数据
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
animals = df['animal'].values

# 配置
prob_config = {
    'base_bet': 20,
    'win_reward': 47,
    'max_multiplier': 10,
    'lookback': 20,
    'prob_high_thresh': 0.45,
    'prob_low_thresh': 0.30,
    'high_mult': 1.5,
    'low_mult': 0.5
}

# 创建预测器和策略
predictor = ZodiacSimpleSmart()
prob_betting = ZodiacTop5ProbabilityBetting(prob_config)

print("测试前10期...")
test_periods = 10
start_idx = len(animals) - test_periods

for i in range(start_idx, start_idx + test_periods):
    # 预测
    train_data = animals[:i].tolist()
    result = predictor.predict_from_history(train_data, top_n=5, debug=False)
    top5 = result['top5']
    
    # 实际结果
    actual = animals[i]
    hit = actual in top5
    
    # 处理这一期
    period_result = prob_betting.process_period(hit)
    
    # 输出
    print(f"期{i-start_idx+1}: 预测={','.join(top5)}, 实际={actual}, "
          f"命中={'✅' if hit else '❌'}, "
          f"倍数={period_result['multiplier']:.1f}, "
          f"概率={period_result['predicted_prob']:.1%}, "
          f"余额={prob_betting.balance:+.0f}元")

print()
print("=" * 80)
print("✅ 测试成功！所有字段都能正确访问")
print("=" * 80)
print(f"总投注: {prob_betting.total_bet:.0f}元")
print(f"总收益: {prob_betting.total_win:.0f}元")
print(f"净利润: {prob_betting.balance:+.0f}元")
print(f"最大回撤: {prob_betting.max_drawdown:.0f}元")
print()
