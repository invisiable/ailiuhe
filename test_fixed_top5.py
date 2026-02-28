"""
固定TOP5生肖投注策略测试
简单策略：始终投注预测的TOP5生肖
目标：通过增加覆盖率来降低连续不中
"""

import pandas as pd
import numpy as np
from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy
from collections import Counter


def test_fixed_top5():
    """测试固定TOP5策略"""
    print("="*80)
    print("固定TOP5生肖投注策略测试")
    print("策略：始终投注TOP5，简单稳定")
    print("="*80 + "\n")
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"数据加载完成：{len(df)}期\n")
    
    # 使用原始预测器
    predictor = RecommendedZodiacTop4Strategy(use_emergency_backup=False)
    
    # 测试300期
    test_periods = 300
    start_idx = len(df) - test_periods
    
    results = []
    total_profit = 0
    max_consecutive_misses = 0
    current_consecutive_misses = 0
    consecutive_miss_sequences = []
    hits = 0
    
    cost_per_period = 20  # TOP5: 5个生肖 × 4元 = 20元
    win_amount = 46
    
    print(f"测试期数：最近{test_periods}期")
    print(f"投注配置：固定TOP5，每期20元\n")
    print("开始回测...\n")
    
    for i in range(start_idx, len(df)):
        period = i - start_idx + 1
        
        # 获取历史数据
        history_animals = df['animal'].iloc[:i].tolist()
        
        # 预测TOP4，然后扩展到TOP5
        prediction = predictor.predict_top4(history_animals)
        base_top4 = prediction['top4']
        
        # 扩展到TOP5：从最近30期高频生肖中选择
        recent = history_animals[-30:] if len(history_animals) >= 30 else history_animals
        freq = Counter(recent)
        all_zodiacs = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
        sorted_by_freq = sorted(all_zodiacs, key=lambda z: freq.get(z, 0), reverse=True)
        
        top5 = base_top4.copy()
        for zodiac in sorted_by_freq:
            if zodiac not in top5:
                top5.append(zodiac)
                break
        
        # 实际结果
        actual = df.iloc[i]['animal']
        actual_date = df.iloc[i]['date']
        is_hit = actual in top5
        
        # 统计
        if is_hit:
            hits += 1
            profit = win_amount - cost_per_period
            current_consecutive_misses = 0
        else:
            profit = -cost_per_period
            current_consecutive_misses += 1
            
            if current_consecutive_misses == 1:
                consecutive_miss_sequences.append({
                    'start': period,
                    'length': 1
                })
            else:
                consecutive_miss_sequences[-1]['length'] = current_consecutive_misses
        
        max_consecutive_misses = max(max_consecutive_misses, current_consecutive_misses)
        total_profit += profit
        
        # 更新预测器
        predictor.update_performance(is_hit)
        
        results.append({
            'period': period,
            'date': actual_date,
            'top5': top5,
            'actual': actual,
            'is_hit': is_hit,
            'profit': profit,
            'cumulative_profit': total_profit,
            'consecutive_misses': current_consecutive_misses
        })
        
        if period % 50 == 0:
            print(f"  已测试 {period}/{test_periods} 期...")
    
    print("\n测试完成！\n")
    
    # 统计连续不中
    long_misses_5 = len([s for s in consecutive_miss_sequences if s['length'] >= 5])
    long_misses_4 = len([s for s in consecutive_miss_sequences if s['length'] >= 4])
    if consecutive_miss_sequences:
        avg_miss_length = np.mean([s['length'] for s in consecutive_miss_sequences])
    else:
        avg_miss_length = 0
    
    # 结果
    total_cost = test_periods * cost_per_period
    hit_rate = hits / test_periods
    roi = (total_profit / total_cost) * 100
    
    print("="*80)
    print("测试结果")
    print("="*80 + "\n")
    
    print("【基础统计】")
    print(f"  测试期数: {test_periods}")
    print(f"  命中次数: {hits}")
    print(f"  命中率: {hit_rate*100:.2f}%\n")
    
    print("【连续不中统计】⭐ 核心指标")
    print(f"  最大连续不中: {max_consecutive_misses}期")
    print(f"  平均连续不中: {avg_miss_length:.2f}期")
    print(f"  连续不中>=4期: {long_misses_4}次")
    print(f"  连续不中>=5期: {long_misses_5}次\n")
    
    print("【财务统计】")
    print(f"  总投注: {total_cost:.2f}元")
    print(f"  总收益: {total_profit:+.2f}元")
    print(f"  ROI: {roi:+.2f}%\n")
    
    # 目标达成
    print("="*80)
    print("目标达成情况")
    print("="*80 + "\n")
    
    target = 4
    print(f"目标: 最大连续不中 ≤ {target}期")
    print(f"实际: 最大连续不中 = {max_consecutive_misses}期\n")
    
    if max_consecutive_misses <= target:
        print(f"✅ 目标达成！")
        print(f"✅ 固定TOP5策略成功将最大连续不中控制在{max_consecutive_misses}期")
        improvement = ((12 - max_consecutive_misses) / 12) * 100
        print(f"✅ 相比原始TOP4的12期，降低了{improvement:.1f}%")
    else:
        gap = max_consecutive_misses - target
        improvement = ((12 - max_consecutive_misses) / 12) * 100
        print(f"⚠ 未达成目标，差{gap}期")
        print(f"   但相比原始TOP4的12期，已降低{improvement:.1f}%")
    
    # 详细序列
    print("\n【所有>=4期的连续不中】")
    long_misses = [s for s in consecutive_miss_sequences if s['length'] >= 4]
    long_misses.sort(key=lambda x: x['length'], reverse=True)
    if long_misses:
        for seq in long_misses:
            end = seq['start'] + seq['length'] - 1
            print(f"  {seq['length']}期不中: 第{seq['start']}期 到 第{end}期")
    else:
        print("  无4期以上连续不中情况 ✅")
    
    # 保存
    print("\n保存详细结果...")
    df_results = pd.DataFrame(results)
    df_results.to_csv('zodiac_top5_fixed_300periods.csv', index=False, encoding='utf-8-sig')
    print("  ✓ zodiac_top5_fixed_300periods.csv")
    
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)
    
    return {
        'max_consecutive_misses': max_consecutive_misses,
        'hit_rate': hit_rate,
        'roi': roi
    }


if __name__ == '__main__':
    test_fixed_top5()
