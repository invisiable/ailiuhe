"""
验证优化预测模型 - TOP5命中率测试
"""

import pandas as pd
from zodiac_optimized_predictor import ZodiacOptimizedPredictor


def validate_recent_periods(n_periods=30):
    """验证最近N期的TOP5命中率"""
    
    print(f"\n{'='*80}")
    print(f"验证最近{n_periods}期 - TOP5命中率测试")
    print(f"{'='*80}\n")
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total = len(df)
    
    if n_periods > total - 20:
        n_periods = total - 20
        print(f"调整验证期数为: {n_periods}\n")
    
    correct_top3 = 0
    correct_top5 = 0
    
    predictor = ZodiacOptimizedPredictor()
    
    print(f"{'期数':<6} {'实际':<4} {'预测TOP5':<40} {'TOP3':<4} {'TOP5':<4}")
    print("-" * 80)
    
    for i in range(n_periods):
        # 使用前面的数据预测
        train_df = df.iloc[:total-n_periods+i]
        actual = df.iloc[total-n_periods+i]['animal']
        period_num = total-n_periods+i+1
        
        # 保存临时数据
        train_df.to_csv('data/temp_validate.csv', index=False, encoding='utf-8-sig')
        
        # 预测
        result = predictor.predict(csv_file='data/temp_validate.csv', top_n=5)
        
        top5_zodiacs = [z for z, s in result['top5_zodiacs']]
        top3_zodiacs = top5_zodiacs[:3]
        
        # 统计命中
        hit_top3 = actual in top3_zodiacs
        hit_top5 = actual in top5_zodiacs
        
        if hit_top3:
            correct_top3 += 1
        if hit_top5:
            correct_top5 += 1
        
        # 显示结果
        status_3 = "✓" if hit_top3 else " "
        status_5 = "✓" if hit_top5 else "✗"
        
        print(f"{period_num:<6} {actual:<4} {str(top5_zodiacs):<40} {status_3:^4} {status_5:^4}")
    
    # 统计结果
    print("\n" + "="*80)
    print("验证结果统计")
    print("="*80)
    
    top3_rate = correct_top3 / n_periods * 100
    top5_rate = correct_top5 / n_periods * 100
    
    print(f"\nTOP3命中: {correct_top3}/{n_periods} = {top3_rate:.1f}%")
    print(f"TOP5命中: {correct_top5}/{n_periods} = {top5_rate:.1f}%")
    
    print(f"\n理论值对比:")
    print(f"  TOP3理论命中率: 25.0%")
    print(f"  TOP5理论命中率: 41.7%")
    
    print(f"\n提升幅度:")
    if top3_rate > 25.0:
        print(f"  TOP3提升: +{top3_rate-25.0:.1f}% ⬆️")
    else:
        print(f"  TOP3提升: {top3_rate-25.0:.1f}% ⬇️")
    
    if top5_rate > 41.7:
        print(f"  TOP5提升: +{top5_rate-41.7:.1f}% ⬆️")
    else:
        print(f"  TOP5提升: {top5_rate-41.7:.1f}% ⬇️")
    
    # 评级
    print(f"\n模型评级:")
    if top5_rate >= 55:
        grade = "A+ (优秀)"
    elif top5_rate >= 50:
        grade = "A (良好)"
    elif top5_rate >= 45:
        grade = "B (及格)"
    else:
        grade = "C (需改进)"
    
    print(f"  {grade}")
    
    print("\n" + "="*80 + "\n")
    
    return {
        'n_periods': n_periods,
        'top3_correct': correct_top3,
        'top5_correct': correct_top5,
        'top3_rate': top3_rate,
        'top5_rate': top5_rate
    }


def validate_multiple_ranges():
    """验证多个时间范围"""
    
    print("\n" + "="*80)
    print("多范围验证测试")
    print("="*80)
    
    ranges = [10, 20, 30, 50]
    results = []
    
    for n in ranges:
        print(f"\n测试最近{n}期...")
        result = validate_recent_periods(n)
        results.append((n, result))
    
    # 汇总
    print("\n" + "="*80)
    print("汇总报告")
    print("="*80)
    
    print(f"\n{'验证期数':<10} {'TOP3命中率':<12} {'TOP5命中率':<12} {'评级':<10}")
    print("-" * 50)
    
    for n, res in results:
        grade = "A+" if res['top5_rate'] >= 55 else "A" if res['top5_rate'] >= 50 else "B" if res['top5_rate'] >= 45 else "C"
        print(f"{n:<10} {res['top3_rate']:>6.1f}%      {res['top5_rate']:>6.1f}%      {grade:<10}")
    
    # 计算平均
    avg_top3 = sum(r['top3_rate'] for _, r in results) / len(results)
    avg_top5 = sum(r['top5_rate'] for _, r in results) / len(results)
    
    print(f"\n{'平均':<10} {avg_top3:>6.1f}%      {avg_top5:>6.1f}%")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("生肖优化预测模型 - 验证测试")
    print("="*80)
    
    # 运行验证
    print("\n选择验证模式:")
    print("1. 快速验证（最近10期）")
    print("2. 标准验证（最近30期）")
    print("3. 完整验证（最近50期）")
    print("4. 多范围验证（10/20/30/50期）")
    
    import sys
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = '2'  # 默认标准验证
    
    if mode == '1':
        validate_recent_periods(10)
    elif mode == '3':
        validate_recent_periods(50)
    elif mode == '4':
        validate_multiple_ranges()
    else:
        validate_recent_periods(30)
