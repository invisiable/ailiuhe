"""
验证超级预测器 - TOP5命中率专项测试
"""

import pandas as pd
from zodiac_super_predictor import ZodiacSuperPredictor
import sys


def validate_super_predictor(n_periods=30):
    """验证超级预测器"""
    
    print(f"\n{'='*80}")
    print(f"超级预测器验证 - 最近{n_periods}期")
    print(f"{'='*80}\n")
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total = len(df)
    
    if n_periods > total - 20:
        n_periods = total - 20
    
    correct_top1 = 0
    correct_top2 = 0
    correct_top3 = 0
    correct_top5 = 0
    
    predictor = ZodiacSuperPredictor()
    
    print(f"{'期数':<6} {'实际':<4} {'预测TOP5':<45} {'状态':<6}")
    print("-" * 80)
    
    details = []
    
    for i in range(n_periods):
        train_df = df.iloc[:total-n_periods+i]
        actual = df.iloc[total-n_periods+i]['animal']
        period_num = total-n_periods+i+1
        
        train_df.to_csv('data/temp_super_validate.csv', index=False, encoding='utf-8-sig')
        
        result = predictor.predict(csv_file='data/temp_super_validate.csv', top_n=5)
        
        top5_zodiacs = [z for z, s in result['top5_zodiacs']]
        
        # 统计命中位置
        if actual == top5_zodiacs[0]:
            correct_top1 += 1
            correct_top2 += 1
            correct_top3 += 1
            correct_top5 += 1
            status = "TOP1 ⭐⭐⭐"
        elif actual in top5_zodiacs[:2]:
            correct_top2 += 1
            correct_top3 += 1
            correct_top5 += 1
            status = "TOP2 ⭐⭐"
        elif actual in top5_zodiacs[:3]:
            correct_top3 += 1
            correct_top5 += 1
            status = "TOP3 ⭐"
        elif actual in top5_zodiacs:
            correct_top5 += 1
            status = "TOP5 ✓"
        else:
            status = "未中 ✗"
        
        print(f"{period_num:<6} {actual:<4} {str(top5_zodiacs):<45} {status:<6}")
        
        details.append({
            'period': period_num,
            'actual': actual,
            'predicted': top5_zodiacs,
            'hit': actual in top5_zodiacs
        })
    
    # 统计报告
    print("\n" + "="*80)
    print("详细统计报告")
    print("="*80)
    
    top1_rate = correct_top1 / n_periods * 100
    top2_rate = correct_top2 / n_periods * 100
    top3_rate = correct_top3 / n_periods * 100
    top5_rate = correct_top5 / n_periods * 100
    
    print(f"\n命中统计:")
    print(f"  TOP1命中: {correct_top1}/{n_periods} = {top1_rate:5.1f}%  (理论 8.3%)")
    print(f"  TOP2命中: {correct_top2}/{n_periods} = {top2_rate:5.1f}%  (理论16.7%)")
    print(f"  TOP3命中: {correct_top3}/{n_periods} = {top3_rate:5.1f}%  (理论25.0%)")
    print(f"  TOP5命中: {correct_top5}/{n_periods} = {top5_rate:5.1f}%  (理论41.7%) ⭐")
    
    print(f"\n提升幅度:")
    print(f"  TOP1: {'+' if top1_rate > 8.3 else ''}{top1_rate - 8.3:+.1f}%")
    print(f"  TOP2: {'+' if top2_rate > 16.7 else ''}{top2_rate - 16.7:+.1f}%")
    print(f"  TOP3: {'+' if top3_rate > 25.0 else ''}{top3_rate - 25.0:+.1f}%")
    print(f"  TOP5: {'+' if top5_rate > 41.7 else ''}{top5_rate - 41.7:+.1f}% ⭐")
    
    # 评级
    print(f"\n模型评级 (基于TOP5命中率):")
    if top5_rate >= 55:
        grade = "S级 - 卓越"
        emoji = "🏆"
    elif top5_rate >= 50:
        grade = "A级 - 优秀"
        emoji = "⭐"
    elif top5_rate >= 45:
        grade = "B级 - 良好"
        emoji = "✓"
    elif top5_rate >= 42:
        grade = "C级 - 及格"
        emoji = "○"
    else:
        grade = "D级 - 需改进"
        emoji = "✗"
    
    print(f"  {emoji} {grade} ({top5_rate:.1f}%)")
    
    # 连续命中分析
    print(f"\n连续命中分析:")
    max_streak = 0
    current_streak = 0
    for d in details:
        if d['hit']:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    print(f"  最长连续命中: {max_streak}期")
    
    # 最近10期表现
    recent_10_hits = sum(1 for d in details[-10:] if d['hit'])
    print(f"  最近10期命中: {recent_10_hits}/10 = {recent_10_hits*10:.0f}%")
    
    print("\n" + "="*80 + "\n")
    
    return {
        'n_periods': n_periods,
        'top1_rate': top1_rate,
        'top2_rate': top2_rate,
        'top3_rate': top3_rate,
        'top5_rate': top5_rate,
        'max_streak': max_streak
    }


def compare_predictors():
    """对比不同预测器"""
    
    print("\n" + "="*80)
    print("预测器性能对比")
    print("="*80 + "\n")
    
    from zodiac_optimized_predictor import ZodiacOptimizedPredictor
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total = len(df)
    n_periods = 20
    
    predictors = {
        '优化预测器': ZodiacOptimizedPredictor(),
        '超级预测器': ZodiacSuperPredictor()
    }
    
    results = {}
    
    for name, predictor in predictors.items():
        print(f"测试 {name}...")
        correct = 0
        
        for i in range(n_periods):
            train_df = df.iloc[:total-n_periods+i]
            actual = df.iloc[total-n_periods+i]['animal']
            
            train_df.to_csv('data/temp_compare.csv', index=False, encoding='utf-8-sig')
            
            result = predictor.predict(csv_file='data/temp_compare.csv', top_n=5)
            
            top5 = [z for z, s in result['top5_zodiacs']]
            
            if actual in top5:
                correct += 1
        
        rate = correct / n_periods * 100
        results[name] = rate
        print(f"  TOP5命中率: {correct}/{n_periods} = {rate:.1f}%\n")
    
    # 显示对比
    print("="*80)
    print(f"{'模型':<15} {'TOP5命中率':<15} {'评级':<10}")
    print("-" * 50)
    
    for name, rate in sorted(results.items(), key=lambda x: x[1], reverse=True):
        if rate >= 50:
            grade = "A级 ⭐"
        elif rate >= 45:
            grade = "B级 ✓"
        else:
            grade = "C级"
        print(f"{name:<15} {rate:>6.1f}%         {grade:<10}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("生肖超级预测器 - 验证系统")
    print("="*80)
    
    mode = sys.argv[1] if len(sys.argv) > 1 else '30'
    
    if mode == 'compare':
        compare_predictors()
    else:
        n = int(mode) if mode.isdigit() else 30
        result = validate_super_predictor(n)
        
        # 总结
        if result['top5_rate'] >= 50:
            print("🎉 恭喜！达到预期目标（TOP5 ≥ 50%）")
        else:
            print(f"⚠️  距离目标还差 {50 - result['top5_rate']:.1f}%")
