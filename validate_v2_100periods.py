"""
V2版本100期回测
"""

from enhanced_top15_predictor_v2 import EnhancedTop15PredictorV2
import pandas as pd
import numpy as np


def validate_v2_100_periods():
    """V2版本100期回测"""
    
    print("=" * 80)
    print("Enhanced Top 15 Predictor V2 - 100期回测")
    print("=" * 80)
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    
    total_periods = len(numbers)
    test_periods = min(100, total_periods - 50)
    
    print(f"\n数据: {total_periods}期, 回测: {test_periods}期")
    
    # 创建预测器
    predictor = EnhancedTop15PredictorV2()
    
    # 统计
    results = {'top5': 0, 'top10': 0, 'top15': 0, 'details': []}
    
    print("\n期数\t实际\tTop5\tTop10\tTop15\t排名")
    print("-" * 60)
    
    # 回测
    for i in range(total_periods - test_periods, total_periods):
        period_num = i + 1
        actual = numbers[i]
        history = numbers[:i]
        
        if len(history) < 30:
            continue
        
        try:
            top15_pred = predictor.predict(history)
            
            top5_hit = actual in top15_pred[:5]
            top10_hit = actual in top15_pred[:10]
            top15_hit = actual in top15_pred
            
            if top5_hit:
                results['top5'] += 1
            if top10_hit:
                results['top10'] += 1
            if top15_hit:
                results['top15'] += 1
            
            rank = top15_pred.index(actual) + 1 if actual in top15_pred else '-'
            
            print(f"{period_num}\t{actual}\t{'V' if top5_hit else ''}\t{'V' if top10_hit else ''}\t{'V' if top15_hit else ''}\t{rank}")
            
            results['details'].append({
                'period': period_num,
                'actual': actual,
                'top15_hit': top15_hit
            })
            
        except Exception as e:
            print(f"{period_num}\t{actual}\tERR\tERR\tERR\t{e}")
    
    # 统计
    total = len(results['details'])
    if total == 0:
        print("\n无有效数据")
        return
    
    top5_rate = results['top5'] / total * 100
    top10_rate = results['top10'] / total * 100
    top15_rate = results['top15'] / total * 100
    
    print("\n" + "=" * 80)
    print("结果统计")
    print("=" * 80)
    
    print(f"\n总期数: {total}")
    print(f"Top 5:  {results['top5']}/{total} = {top5_rate:.1f}%")
    print(f"Top 10: {results['top10']}/{total} = {top10_rate:.1f}%")
    print(f"Top 15: {results['top15']}/{total} = {top15_rate:.1f}%")
    
    print(f"\n目标: 60%")
    if top15_rate >= 60:
        print(f"[成功] 已达成目标! {top15_rate:.1f}% >= 60%")
    else:
        print(f"[待提升] 还差 {60 - top15_rate:.1f}%")
    
    # 最近20期
    recent_20 = results['details'][-20:]
    recent_20_hits = sum(1 for d in recent_20 if d['top15_hit'])
    recent_20_rate = recent_20_hits / len(recent_20) * 100 if recent_20 else 0
    print(f"\n最近20期: {recent_20_hits}/{len(recent_20)} = {recent_20_rate:.1f}%")
    
    # 保存
    df_results = pd.DataFrame(results['details'])
    df_results.to_csv('enhanced_top15_v2_validation_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n结果已保存: enhanced_top15_v2_validation_results.csv")
    
    return top15_rate


if __name__ == '__main__':
    rate = validate_v2_100_periods()
    print(f"\n最终成功率: {rate:.1f}%")
