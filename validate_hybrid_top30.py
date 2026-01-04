"""
验证混合策略Top30预测成功率
扩展Top15策略到Top30
"""

import pandas as pd
import numpy as np
from collections import Counter
from final_hybrid_predictor import FinalHybridPredictor


def validate_hybrid_top30(csv_file='data/lucky_numbers.csv', periods=50):
    """验证混合模型Top30预测成功率"""
    
    predictor = FinalHybridPredictor()
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    
    # 确保至少有足够的数据
    if len(df) < periods + 1:
        print(f"数据不足，需要至少 {periods+1} 期数据")
        return
    
    print("=" * 80)
    print(f"混合策略Top30验证 - 最近{periods}期")
    print("=" * 80)
    print(f"验证规则：使用当期数据预测下一期，与实际结果比对")
    print(f"策略说明：")
    print(f"  TOP 1-5:   最近10期数据策略（精准预测）")
    print(f"  TOP 6-15:  全部历史数据策略（稳定覆盖）")
    print(f"  TOP 16-30: 扩展策略A+策略B组合")
    print("=" * 80)
    print()
    
    results = {
        'top5': [],
        'top10': [],
        'top15': [],
        'top20': [],
        'top30': [],
        'details': []
    }
    
    # 从后往前验证最近N期
    for i in range(periods):
        # 当期数据索引
        next_index = len(df) - periods + i
        
        # 用当期之前的所有数据预测下一期
        train_data = df.iloc[:next_index]
        current_date = train_data.iloc[-1]['date']
        current_number = train_data.iloc[-1]['number']
        
        # 下一期的实际数据
        next_actual = df.iloc[next_index]['number']
        next_date = df.iloc[next_index]['date']
        
        numbers = train_data['number'].values
        elements = train_data['element'].values
        
        # 策略A：全部历史数据（稳定覆盖）
        strategy_a = predictor._predict_strategy_a(numbers)
        
        # 策略B：最近10期数据（精准预测）
        strategy_b = predictor._predict_strategy_b(numbers, elements)
        
        # 扩展到Top30
        # TOP 1-5: 策略B
        top30_predictions = []
        for num in strategy_b[:5]:
            if num not in top30_predictions:
                top30_predictions.append(num)
        
        # TOP 6-15: 策略A
        for num in strategy_a:
            if num not in top30_predictions:
                top30_predictions.append(num)
            if len(top30_predictions) >= 15:
                break
        
        # TOP 16-30: 继续从策略B和策略A交替添加
        remaining_b = [n for n in strategy_b if n not in top30_predictions]
        remaining_a = [n for n in strategy_a if n not in top30_predictions]
        
        # 交替添加，优先策略B
        j = 0
        while len(top30_predictions) < 30:
            if j < len(remaining_b):
                top30_predictions.append(remaining_b[j])
            if len(top30_predictions) >= 30:
                break
            if j < len(remaining_a):
                top30_predictions.append(remaining_a[j])
            if len(top30_predictions) >= 30:
                break
            j += 1
        
        top5 = top30_predictions[:5]
        top10 = top30_predictions[:10]
        top15 = top30_predictions[:15]
        top20 = top30_predictions[:20]
        top30 = top30_predictions[:30]
        
        # 检查是否命中
        hit_top5 = next_actual in top5
        hit_top10 = next_actual in top10
        hit_top15 = next_actual in top15
        hit_top20 = next_actual in top20
        hit_top30 = next_actual in top30
        
        results['top5'].append(hit_top5)
        results['top10'].append(hit_top10)
        results['top15'].append(hit_top15)
        results['top20'].append(hit_top20)
        results['top30'].append(hit_top30)
        
        # 记录详细信息
        rank = None
        status = "❌ 未命中"
        if hit_top5:
            rank = top5.index(next_actual) + 1
            status = f"✅ TOP5命中 (#{rank})"
        elif hit_top10:
            rank = top10.index(next_actual) + 1
            status = f"✅ TOP10命中 (#{rank})"
        elif hit_top15:
            rank = top15.index(next_actual) + 1
            status = f"✅ TOP15命中 (#{rank})"
        elif hit_top20:
            rank = top20.index(next_actual) + 1
            status = f"○ TOP20命中 (#{rank})"
        elif hit_top30:
            rank = top30.index(next_actual) + 1
            status = f"○ TOP30命中 (#{rank})"
        
        results['details'].append({
            'period': i + 1,
            'current_date': current_date,
            'next_date': next_date,
            'actual': next_actual,
            'rank': rank,
            'status': status,
            'top30': top30
        })
        
        print(f"期数 {i+1:>2}/{periods} | {current_date} 预测 {next_date} | 实际: {next_actual:>2} | {status}")
    
    # 计算成功率
    print(f"\n{'='*80}")
    print("验证结果统计")
    print(f"{'='*80}\n")
    
    top5_success = sum(results['top5'])
    top10_success = sum(results['top10'])
    top15_success = sum(results['top15'])
    top20_success = sum(results['top20'])
    top30_success = sum(results['top30'])
    
    total = len(results['top5'])
    
    top5_rate = top5_success / total * 100
    top10_rate = top10_success / total * 100
    top15_rate = top15_success / total * 100
    top20_rate = top20_success / total * 100
    top30_rate = top30_success / total * 100
    
    print(f"验证期数: {total} 期")
    print(f"\n成功率统计:")
    print(f"  TOP 5  命中: {top5_success:>2}/{total} = {top5_rate:>5.1f}%")
    print(f"  TOP 10 命中: {top10_success:>2}/{total} = {top10_rate:>5.1f}%")
    print(f"  TOP 15 命中: {top15_success:>2}/{total} = {top15_rate:>5.1f}%")
    print(f"  TOP 20 命中: {top20_success:>2}/{total} = {top20_rate:>5.1f}%")
    print(f"  TOP 30 命中: {top30_success:>2}/{total} = {top30_rate:>5.1f}%")
    
    print(f"\n{'='*80}")
    print("结论")
    print(f"{'='*80}")
    print(f"\n✅ Top30预测成功率: {top30_rate:.1f}%")
    
    # 详细命中记录
    print(f"\n{'='*80}")
    print("详细命中记录")
    print(f"{'='*80}\n")
    
    for detail in results['details']:
        if detail['rank']:
            print(f"第{detail['period']:>2}期 | {detail['next_date']} | 实际: {detail['actual']:>2} | 排名: #{detail['rank']:>2}")
    
    return {
        'periods': total,
        'top5_rate': top5_rate,
        'top10_rate': top10_rate,
        'top15_rate': top15_rate,
        'top20_rate': top20_rate,
        'top30_rate': top30_rate,
        'top5_success': top5_success,
        'top10_success': top10_success,
        'top15_success': top15_success,
        'top20_success': top20_success,
        'top30_success': top30_success,
        'details': results['details']
    }


if __name__ == '__main__':
    results = validate_hybrid_top30(periods=50)
