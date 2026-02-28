"""
测试TOP15 vs TOP4对比功能已更新到300期
"""

import pandas as pd
from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy
from top15_predictor import Top15Predictor

# 读取数据
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')

print("="*120)
print("测试 TOP15 vs 生肖TOP4 对比分析 - 最近300期")
print("="*120)
print()

# 检查数据量
print(f"✓ 总数据量: {len(df)}期")

if len(df) < 300:
    print(f"❌ 数据不足300期，无法进行完整测试（当前仅{len(df)}期）")
    test_periods = len(df)
else:
    print(f"✓ 数据充足，可以测试完整300期")
    test_periods = 300

print(f"实际测试期数: {test_periods}期")
print()

# 模拟对比分析
start_idx = len(df) - test_periods

print("正在生成预测数据...")
top15_predictor = Top15Predictor()
top4_strategy = RecommendedZodiacTop4Strategy(use_emergency_backup=True)

comparison_data = []

for i in range(start_idx, len(df)):
    current_row = df.iloc[i]
    date = current_row['date']
    actual_number = current_row['number']
    actual_animal = current_row['animal']
    
    # TOP15预测
    train_numbers = df.iloc[:i]['number'].values
    top15_analysis = top15_predictor.get_analysis(train_numbers)
    top15_pred = top15_analysis['top15']
    top15_hit = actual_number in top15_pred
    
    # 生肖TOP4预测
    train_animals = [str(a).strip() for a in df['animal'].iloc[:i].tolist()]
    top4_prediction = top4_strategy.predict_top4(train_animals)
    top4_pred = top4_prediction['top4']
    top4_hit = actual_animal in top4_pred
    
    # 更新策略性能
    top4_strategy.update_performance(top4_hit)
    if (i - start_idx + 1) % 10 == 0:
        top4_strategy.check_and_switch_model()
    
    comparison_data.append({
        'date': date,
        'actual_number': actual_number,
        'actual_animal': actual_animal,
        'top15_hit': top15_hit,
        'top4_hit': top4_hit,
        'both_hit': top15_hit and top4_hit
    })
    
    if (i - start_idx + 1) % 50 == 0:
        print(f"  已处理 {i - start_idx + 1}/{test_periods} 期...")

print(f"\n✓ 预测生成完成！共 {len(comparison_data)} 期\n")

# 统计数据
top15_hits = sum(1 for d in comparison_data if d['top15_hit'])
top4_hits = sum(1 for d in comparison_data if d['top4_hit'])
both_hits = sum(1 for d in comparison_data if d['both_hit'])
both_miss = sum(1 for d in comparison_data if not d['top15_hit'] and not d['top4_hit'])

top15_hit_rate = top15_hits / len(comparison_data) * 100
top4_hit_rate = top4_hits / len(comparison_data) * 100
both_hit_rate = both_hits / len(comparison_data) * 100

print("="*120)
print("统计汇总")
print("="*120)
print()

print(f"【TOP15预测统计】")
print(f"  命中次数: {top15_hits}/{len(comparison_data)}")
print(f"  命中率: {top15_hit_rate:.2f}%")
print()

print(f"【生肖TOP4预测统计】（推荐策略v2.0）")
print(f"  使用模型: {top4_strategy.get_current_model_name()}")
print(f"  模型切换次数: {len(top4_strategy.switch_history)}次")
print(f"  命中次数: {top4_hits}/{len(comparison_data)}")
print(f"  命中率: {top4_hit_rate:.2f}%")
print()

print(f"【组合命中情况】")
print(f"  两种都中: {both_hits}期 ({both_hit_rate:.2f}%) 🌟")
print(f"  两种都未中: {both_miss}期 ({both_miss / len(comparison_data) * 100:.2f}%) 💔")
print()

# 收益对比
top15_cost = len(comparison_data) * 15
top15_reward = top15_hits * 47
top15_profit = top15_reward - top15_cost
top15_roi = (top15_profit / top15_cost * 100) if top15_cost > 0 else 0

top4_cost = len(comparison_data) * 16
top4_reward = top4_hits * 47
top4_profit = top4_reward - top4_cost
top4_roi = (top4_profit / top4_cost * 100) if top4_cost > 0 else 0

print("="*120)
print("收益对比")
print("="*120)
print()

print(f"【TOP15收益】")
print(f"  总投入: {top15_cost}元")
print(f"  总收益: {top15_profit:+.0f}元")
print(f"  ROI: {top15_roi:+.2f}%")
print()

print(f"【生肖TOP4收益】")
print(f"  总投入: {top4_cost}元")
print(f"  总收益: {top4_profit:+.0f}元")
print(f"  ROI: {top4_roi:+.2f}%")
print()

# 显示最后10期详细数据作为样本
print("="*120)
print(f"详细对比表（最后10期样本）")
print("="*120)
print()

print(f"{'日期':<12} {'中奖号':<6} {'生肖':<6} {'TOP15':<8} {'TOP4':<8} {'都中':<6}")
print("-" * 60)

for data in comparison_data[-10:]:
    top15_mark = "✓中" if data['top15_hit'] else "✗失"
    top4_mark = "✓中" if data['top4_hit'] else "✗失"
    both_hit_mark = "✓✓" if data['both_hit'] else ""
    
    if data['both_hit']:
        prefix = "🌟 "
    elif not data['top15_hit'] and not data['top4_hit']:
        prefix = "💔 "
    else:
        prefix = "   "
    
    print(f"{prefix}{str(data['date']):<12} "
          f"{data['actual_number']:<6} "
          f"{data['actual_animal']:<6} "
          f"{top15_mark:<8} "
          f"{top4_mark:<8} "
          f"{both_hit_mark:<6}")

print()
print("="*120)
print("✅ 测试完成！")
print("="*120)
print()
print("总结:")
print(f"  • 成功测试了{test_periods}期数据的对比分析")
print(f"  • TOP15命中率: {top15_hit_rate:.2f}%，收益: {top15_profit:+.0f}元")
print(f"  • TOP4命中率: {top4_hit_rate:.2f}%，收益: {top4_profit:+.0f}元")
print(f"  • 双重命中率: {both_hit_rate:.2f}%")
print(f"  • TOP4模型切换: {len(top4_strategy.switch_history)}次")
print(f"  • GUI中的对比表将显示全部{test_periods}期的详细数据")
