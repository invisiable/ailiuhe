"""
测试v12.0平衡智能选择器
在10/20/30/50/100期数据上验证性能
"""

import pandas as pd
from zodiac_balanced_smart import ZodiacBalancedSmart

def test_periods(periods, label):
    """测试指定期数的预测准确率"""
    predictor = ZodiacBalancedSmart()
    df = pd.read_csv('data/lucky_numbers.csv')
    
    test_data = df.tail(periods)
    total = len(test_data)
    top5_hits = 0
    top3_hits = 0
    top2_hits = 0
    top1_hits = 0
    
    # 统计场景使用情况
    scenario_stats = {}
    
    for i in range(total):
        # 使用前面的数据作为训练集
        train_end = len(df) - total + i
        predictor_data = df.iloc[:train_end]
        
        if len(predictor_data) < 50:
            continue
        
        # 临时保存原始数据
        original_df = pd.read_csv('data/lucky_numbers.csv')
        predictor_data.to_csv('data/lucky_numbers.csv', index=False)
        
        # 获取策略信息
        info = predictor.get_strategy_info()
        scenario = info['scenario']
        scenario_stats[scenario] = scenario_stats.get(scenario, 0) + 1
        
        # 生成预测
        top5 = predictor.predict_top5()
        
        # 恢复原始数据
        original_df.to_csv('data/lucky_numbers.csv', index=False)
        
        # 获取实际结果
        actual = test_data.iloc[i]
        actual_zodiac = actual['animal']
        
        # 检查命中
        if actual_zodiac in top5:
            top5_hits += 1
            if actual_zodiac in top5[:3]:
                top3_hits += 1
                if actual_zodiac in top5[:2]:
                    top2_hits += 1
                    if top5[0] == actual_zodiac:
                        top1_hits += 1
    
    print(f"\n{'='*80}")
    print(f"{label} - {total}期测试结果")
    print(f"{'='*80}")
    print(f"TOP5命中: {top5_hits}/{total} = {top5_hits/total*100:.1f}%")
    print(f"TOP3命中: {top3_hits}/{total} = {top3_hits/total*100:.1f}%")
    print(f"TOP2命中: {top2_hits}/{total} = {top2_hits/total*100:.1f}%")
    print(f"TOP1命中: {top1_hits}/{total} = {top1_hits/total*100:.1f}%")
    
    print(f"\n场景使用统计:")
    for scenario, count in sorted(scenario_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {scenario:20s}: {count:3d}次 ({count/total*100:5.1f}%)")
    
    return {
        'total': total,
        'top5': top5_hits,
        'top3': top3_hits,
        'top2': top2_hits,
        'top1': top1_hits,
        'top5_rate': top5_hits/total,
        'scenarios': scenario_stats
    }

if __name__ == '__main__':
    print("="*80)
    print("v12.0 平衡智能选择器 - 综合性能测试")
    print("="*80)
    
    # 测试不同期数
    results = {}
    
    print("\n【关键测试】最近50期 - 目标>50%")
    results['50'] = test_periods(50, "最近50期")
    
    print("\n【短期测试】最近10期 - 验证爆发捕捉能力")
    results['10'] = test_periods(10, "最近10期")
    
    print("\n【中期测试】最近20期")
    results['20'] = test_periods(20, "最近20期")
    
    print("\n【中长期测试】最近30期")
    results['30'] = test_periods(30, "最近30期")
    
    print("\n【长期测试】100期 - 验证稳定性")
    results['100'] = test_periods(100, "100期")
    
    # 性能对比总结
    print(f"\n{'='*80}")
    print("v12.0 vs v10.0 vs v11.0 性能对比")
    print(f"{'='*80}")
    print(f"{'期数':8s} {'v12.0':>10s} {'v10.0':>10s} {'v11.0':>10s} {'vs v10.0':>10s} {'vs v11.0':>10s}")
    print(f"{'-'*80}")
    
    comparison = {
        '10': {'v10': 0.20, 'v11': 0.50},
        '20': {'v10': None, 'v11': 0.30},
        '30': {'v10': 0.433, 'v11': 0.30},
        '50': {'v10': None, 'v11': None},
        '100': {'v10': 0.52, 'v11': 0.38}
    }
    
    for period in ['10', '20', '30', '50', '100']:
        if period in results:
            v12_rate = results[period]['top5_rate']
            v10_rate = comparison[period]['v10']
            v11_rate = comparison[period]['v11']
            
            v10_diff = f"{(v12_rate - v10_rate)*100:+.1f}%" if v10_rate else "N/A"
            v11_diff = f"{(v12_rate - v11_rate)*100:+.1f}%" if v11_rate else "N/A"
            
            v10_str = f"{v10_rate*100:.1f}%" if v10_rate else "N/A"
            v11_str = f"{v11_rate*100:.1f}%" if v11_rate else "N/A"
            
            print(f"{period+'期':8s} {v12_rate*100:9.1f}% {v10_str:>10s} {v11_str:>10s} {v10_diff:>10s} {v11_diff:>10s}")
    
    print(f"{'='*80}")
    
    # 目标达成检查
    print(f"\n目标达成情况:")
    if results['50']['top5_rate'] >= 0.50:
        print(f"  ✓ 最近50期成功率 {results['50']['top5_rate']*100:.1f}% >= 50% - 目标达成!")
    else:
        print(f"  ✗ 最近50期成功率 {results['50']['top5_rate']*100:.1f}% < 50% - 需要继续优化")
    
    print(f"\n关键指标:")
    print(f"  最近10期: {results['10']['top5_rate']*100:.1f}% (爆发捕捉能力)")
    print(f"  最近50期: {results['50']['top5_rate']*100:.1f}% (目标指标)")
    print(f"  100期: {results['100']['top5_rate']*100:.1f}% (长期稳定性)")
