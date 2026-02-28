"""
测试生肖TOP4投注的最近300期回测功能
"""

import pandas as pd
from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy

print("="*120)
print("测试 生肖TOP4投注 - 最近300期回测")
print("="*120)
print()

# 读取数据
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
print(f"✓ 总数据量: {len(df)}期")

# 检查是否有足够的数据
if len(df) < 300:
    print(f"⚠️ 数据不足300期，实际可用: {len(df)}期")
    test_periods = len(df)
else:
    print(f"✓ 数据充足，测试300期")
    test_periods = 300

print()

# 创建推荐策略实例
strategy = RecommendedZodiacTop4Strategy(use_emergency_backup=True)

# 回测数据
start_idx = len(df) - test_periods
predictions_top4 = []
actuals = []
hit_records = []
predictor_records = []

print("正在生成每期的TOP4生肖预测（重训练v2.0模型）...")

for i in range(start_idx, len(df)):
    # 使用i之前的数据进行预测
    train_animals = [str(a).strip() for a in df['animal'].iloc[:i].tolist()]
    
    # 使用推荐策略进行预测
    prediction = strategy.predict_top4(train_animals)
    top4 = prediction['top4']
    predictor_name = prediction['predictor']
    
    predictions_top4.append(top4)
    predictor_records.append(predictor_name)
    
    # 实际结果
    actual = str(df.iloc[i]['animal']).strip()
    actuals.append(actual)
    
    # 判断命中
    hit = actual in top4
    hit_records.append(hit)
    
    # 更新策略性能监控
    strategy.update_performance(hit)
    
    # 每10期检查是否需要切换模型
    if (i - start_idx + 1) % 10 == 0:
        strategy.check_and_switch_model()
    
    if (i - start_idx + 1) % 50 == 0:
        stats = strategy.get_performance_stats()
        print(f"  已处理 {i - start_idx + 1}/{test_periods} 期... "
              f"最近{stats['recent_total']}期命中率: {stats['recent_rate']:.1f}%")

print(f"\n✓ 预测生成完成！共 {len(predictions_top4)} 期\n")

# 计算基础命中率
hits = sum(hit_records)
hit_rate = hits / len(hit_records)

# 获取最终性能统计
final_stats = strategy.get_performance_stats()

print("="*120)
print(f"回测结果分析（固定1倍投注，测试{test_periods}期）")
print("="*120)
print()

print(f"当前使用模型: {final_stats['current_model']}")
print(f"最近{final_stats['recent_total']}期命中率: {final_stats['recent_rate']:.1f}%")
print()

# 基础策略：固定投注（1倍）
base_profit = 0
for hit in hit_records:
    if hit:
        period_profit = 30  # 净利润 (46-16)
    else:
        period_profit = -16  # 亏损
    base_profit += period_profit

base_roi = (base_profit / (16 * len(hit_records))) * 100

print(f"命中次数: {hits}/{len(hit_records)} = {hit_rate*100:.2f}%")
print(f"理论命中率: 33.3%")
print(f"vs理论值: {hit_rate*100 - 33.3:+.1f}%")
print()
print(f"总投入: {16 * len(hit_records)}元")
print(f"总中奖: {hits * 46}元")
print(f"净收益: {base_profit:+.0f}元")
print(f"投资回报率: {base_roi:+.2f}%")
print()

# 最近300期分析
if len(hit_records) >= 300:
    recent_300_hits = sum(hit_records[-300:])
    recent_300_rate = recent_300_hits / 300
    recent_300_profit = recent_300_hits * 30 - (300 - recent_300_hits) * 16
    recent_300_investment = 16 * 300
    recent_300_roi = (recent_300_profit / recent_300_investment) * 100
    
    print("="*120)
    print("📈 最近300期表现分析")
    print("="*120)
    print(f"命中次数: {recent_300_hits}/300 = {recent_300_rate*100:.2f}%")
    print(f"总投入: {recent_300_investment}元")
    print(f"总中奖: {recent_300_hits * 46}元")
    print(f"净收益: {recent_300_profit:+.0f}元")
    print(f"投资回报率: {recent_300_roi:+.2f}%")
    
    # 计算连续命中和连续未中
    recent_300 = hit_records[-300:]
    max_consecutive_hits = 0
    max_consecutive_misses = 0
    current_hits = 0
    current_misses = 0
    for h in recent_300:
        if h:
            current_hits += 1
            current_misses = 0
            max_consecutive_hits = max(max_consecutive_hits, current_hits)
        else:
            current_misses += 1
            current_hits = 0
            max_consecutive_misses = max(max_consecutive_misses, current_misses)
    
    print(f"最大连续命中: {max_consecutive_hits}期")
    print(f"最大连续未中: {max_consecutive_misses}期")
    print()
    
    # 统计模型使用情况
    recent_300_predictors = predictor_records[-300:]
    primary_count = sum(1 for p in recent_300_predictors if '重训练' in p or '主力' in p)
    backup_count = 300 - primary_count
    
    print(f"模型使用统计: 主力模型 {primary_count}期  |  备份模型 {backup_count}期")
    print()
    
    # 显示最近10期详细记录（样本）
    print("="*120)
    print("📋 最近300期详细命中记录（显示最后10期样本）")
    print("="*120)
    print(f"{'期号':<6} {'日期':<12} {'实际':<8} {'预测TOP4':<35} {'模型':<20} {'结果':<6}")
    print("-"*120)
    
    recent_300_start_idx = len(hit_records) - 300
    for i in range(290, 300):  # 显示最后10期
        record_idx = recent_300_start_idx + i
        period_idx = start_idx + record_idx
        
        date_str = df.iloc[period_idx]['date']
        actual_animal = actuals[record_idx]
        predicted_top4 = predictions_top4[record_idx]
        predictor_used = predictor_records[record_idx]
        hit = hit_records[record_idx]
        
        top4_str = ', '.join(predicted_top4)
        result_mark = "✓" if hit else "✗"
        period_number = i + 1
        
        print(f"{period_number:<6} {date_str:<12} {actual_animal:<8} {top4_str:<35} {predictor_used:<20} {result_mark:<6}")
    
    print("="*120)
    print()

elif len(hit_records) >= 100:
    print(f"⚠️ 数据不足300期，显示最近{len(hit_records)}期的分析")
else:
    print(f"⚠️ 数据不足100期，无法进行详细分析")

print("="*120)
print("✅ 测试完成！")
print("="*120)
print()
print("总结:")
print(f"  • 测试期数: {test_periods}期")
print(f"  • 命中率: {hit_rate*100:.2f}%")
print(f"  • 净收益: {base_profit:+.0f}元")
print(f"  • ROI: {base_roi:+.2f}%")
if len(hit_records) >= 300:
    print(f"  • ✓ 最近300期分析功能正常")
else:
    print(f"  • ⚠️ 数据不足300期，显示了{len(hit_records)}期的结果")
