"""
生肖TOP4策略 - 最近300期回测验证
使用RecommendedZodiacTop4Strategy v2.0
"""
import sys
import io
import pandas as pd
from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy

# 设置UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def backtest_top4_300periods():
    print("="*80)
    print("生肖TOP4策略 - 最近300期回测分析")
    print("="*80)
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"\n✅ 数据加载完成: {len(df)}期")
    print(f"数据范围: {df.iloc[0]['date']} ~ {df.iloc[-1]['date']}")
    
    # 确定测试期数
    test_periods = min(300, len(df) - 50)  # 保留至少50期作为训练数据
    start_idx = len(df) - test_periods
    
    print(f"\n📊 回测配置:")
    print(f"  测试期数: {test_periods}期")
    print(f"  起始日期: {df.iloc[start_idx]['date']}")
    print(f"  结束日期: {df.iloc[-1]['date']}")
    
    # 创建策略
    strategy = RecommendedZodiacTop4Strategy(use_emergency_backup=True)
    print(f"\n✅ 策略创建完成:")
    print(f"  主力模型: {strategy.primary_model.__class__.__name__}")
    print(f"  备份模型: {strategy.backup_model.__class__.__name__}")
    
    # 回测
    print(f"\n{'='*80}")
    print("开始回测...")
    print("="*80)
    
    hit_records = []
    predictions = []
    actuals = []
    dates = []
    
    for i in range(start_idx, len(df)):
        # 训练数据
        train_animals = df['animal'].iloc[:i].tolist()
        
        # 预测
        prediction = strategy.predict_top4(train_animals)
        top4 = prediction['top4']
        
        # 实际结果
        actual = df.iloc[i]['animal']
        date = df.iloc[i]['date']
        
        # 判断命中
        hit = actual in top4
        
        hit_records.append(hit)
        predictions.append(top4)
        actuals.append(actual)
        dates.append(date)
        
        # 更新策略性能
        strategy.update_performance(hit)
        
        # 每10期检查模型切换
        if (i - start_idx + 1) % 10 == 0:
            strategy.check_and_switch_model()
        
        # 进度显示
        if (i - start_idx + 1) % 50 == 0:
            current_hits = sum(hit_records)
            current_rate = current_hits / len(hit_records) * 100
            print(f"  已处理 {i - start_idx + 1}/{test_periods} 期... 当前命中率: {current_rate:.2f}%")
    
    print(f"\n✅ 回测完成！\n")
    
    # 计算统计数据
    total_hits = sum(hit_records)
    hit_rate = total_hits / len(hit_records) * 100
    
    # 计算收益
    total_investment = 16 * len(hit_records)  # 每期16元
    total_win = total_hits * 46  # 命中奖励46元
    total_profit = total_hits * 30 - (len(hit_records) - total_hits) * 16  # 净利润
    roi = (total_profit / total_investment) * 100
    
    # 计算连续命中和连亏
    max_consecutive_hits = 0
    max_consecutive_misses = 0
    current_hits = 0
    current_misses = 0
    
    for hit in hit_records:
        if hit:
            current_hits += 1
            current_misses = 0
            max_consecutive_hits = max(max_consecutive_hits, current_hits)
        else:
            current_misses += 1
            current_hits = 0
            max_consecutive_misses = max(max_consecutive_misses, current_misses)
    
    # 分段统计（每50期）
    segment_stats = []
    segment_size = 50
    for i in range(0, len(hit_records), segment_size):
        segment = hit_records[i:i+segment_size]
        if len(segment) > 0:
            seg_hits = sum(segment)
            seg_rate = seg_hits / len(segment) * 100
            seg_start_date = dates[i]
            seg_end_date = dates[min(i+segment_size-1, len(dates)-1)]
            segment_stats.append({
                'period': f"{i+1}-{i+len(segment)}",
                'dates': f"{seg_start_date} ~ {seg_end_date}",
                'hits': seg_hits,
                'total': len(segment),
                'rate': seg_rate
            })
    
    # 输出结果
    print("="*80)
    print("回测结果汇总")
    print("="*80)
    
    print(f"\n【基础统计】")
    print(f"  测试期数: {len(hit_records)}期")
    print(f"  命中次数: {total_hits}次")
    print(f"  命中率: {hit_rate:.2f}%")
    print(f"  理论命中率: 33.33% (4/12生肖)")
    print(f"  vs理论值: {hit_rate - 33.33:+.2f}%")
    
    print(f"\n【投注收益】")
    print(f"  总投入: {total_investment}元 (每期16元)")
    print(f"  总中奖: {total_win}元 (每次46元)")
    print(f"  净收益: {total_profit:+.0f}元")
    print(f"  投资回报率: {roi:+.2f}%")
    
    print(f"\n【连续记录】")
    print(f"  最大连续命中: {max_consecutive_hits}期")
    print(f"  最大连续未中: {max_consecutive_misses}期")
    
    print(f"\n【模型切换】")
    final_stats = strategy.get_performance_stats()
    print(f"  当前使用模型: {strategy.get_current_model_name()}")
    print(f"  模型切换次数: {len(strategy.switch_history)}次")
    if len(strategy.switch_history) > 0:
        print(f"\n  切换记录:")
        for i, switch in enumerate(strategy.switch_history, 1):
            print(f"    {i}. {switch['from']} → {switch['to']}")
            print(f"       原因: {switch['reason']}")
            print(f"       当时命中率: {switch['recent_rate']:.1f}%")
    
    # 分段统计
    print(f"\n{'='*80}")
    print("分段统计分析（每50期）")
    print("="*80)
    print(f"\n{'期数范围':<15} {'日期范围':<30} {'命中':<10} {'命中率':<10}")
    print("-"*80)
    
    for seg in segment_stats:
        print(f"{seg['period']:<15} {seg['dates']:<30} {seg['hits']}/{seg['total']:<8} {seg['rate']:.2f}%")
    
    # 月度统计
    print(f"\n{'='*80}")
    print("月度收益统计")
    print("="*80)
    
    monthly_profits = {}
    for idx, hit in enumerate(hit_records):
        date_str = dates[idx]
        try:
            date_obj = pd.to_datetime(date_str)
            month_key = date_obj.strftime('%Y/%m')
        except:
            month_key = date_str[:7] if len(date_str) >= 7 else '未知'
        
        if month_key not in monthly_profits:
            monthly_profits[month_key] = {'hits': 0, 'total': 0, 'profit': 0}
        
        monthly_profits[month_key]['total'] += 1
        if hit:
            monthly_profits[month_key]['hits'] += 1
            monthly_profits[month_key]['profit'] += 30  # 净利润
        else:
            monthly_profits[month_key]['profit'] -= 16  # 亏损
    
    print(f"\n{'月份':<10} {'期数':<8} {'命中':<8} {'命中率':<10} {'收益':<12}")
    print("-"*80)
    
    for month in sorted(monthly_profits.keys()):
        data = monthly_profits[month]
        rate = data['hits'] / data['total'] * 100 if data['total'] > 0 else 0
        print(f"{month:<10} {data['total']:<8} {data['hits']:<8} {rate:.2f}%{'':<3} {data['profit']:>+8}元")
    
    # 评估和建议
    print(f"\n{'='*80}")
    print("评估和建议")
    print("="*80)
    
    if hit_rate >= 50:
        level = "优秀 🌟"
        advice = "表现优异，建议继续使用固定1倍投注策略"
    elif hit_rate >= 45:
        level = "良好 ✓"
        advice = "表现良好，可以稳健投注"
    elif hit_rate >= 40:
        level = "一般 ○"
        advice = "表现一般，建议降低投注额度"
    else:
        level = "需改进 ✗"
        advice = "表现不佳，建议暂停使用或优化模型"
    
    print(f"\n综合评级: {level}")
    print(f"命中率: {hit_rate:.2f}% (理论值33.33%，提升{hit_rate-33.33:.2f}%)")
    print(f"ROI: {roi:+.2f}%")
    print(f"\n建议: {advice}")
    
    print(f"\n💡 风险提示:")
    print(f"  - 历史表现不代表未来收益")
    print(f"  - 最大连亏{max_consecutive_misses}期需准备{max_consecutive_misses*16}元应对")
    print(f"  - 建议保留至少{max_consecutive_misses*32}元的风险准备金")
    print(f"  - 投资有风险，量力而行")
    
    print(f"\n{'='*80}")
    print("✅ 回测分析完成！")
    print("="*80)
    
    return {
        'test_periods': len(hit_records),
        'hits': total_hits,
        'hit_rate': hit_rate,
        'profit': total_profit,
        'roi': roi,
        'max_consecutive_hits': max_consecutive_hits,
        'max_consecutive_misses': max_consecutive_misses,
        'segment_stats': segment_stats
    }

if __name__ == '__main__':
    backtest_top4_300periods()
