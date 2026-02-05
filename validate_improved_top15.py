"""
验证改进版 Top15 预测器
对比原版和改进版的性能
"""

import pandas as pd
import numpy as np
from improved_top15_predictor import ImprovedTop15Predictor
from top15_predictor import Top15Predictor


def validate_predictor(predictor, numbers, test_periods=100):
    """验证预测器性能"""
    results = []
    
    # 至少需要50期数据才开始预测
    start_idx = max(50, len(numbers) - test_periods - 1)
    
    for i in range(start_idx, len(numbers) - 1):
        # 使用前i期数据预测第i+1期
        train_data = numbers[:i+1]
        actual = numbers[i+1]
        
        # 预测
        predicted = predictor.predict(train_data)
        
        # 检查命中
        top5_hit = actual in predicted[:5]
        top10_hit = actual in predicted[:10]
        top15_hit = actual in predicted[:15]
        
        rank = predicted.index(actual) + 1 if actual in predicted else -1
        
        results.append({
            'period': i + 2,  # 期号从1开始
            'actual': actual,
            'predicted': predicted,
            'top5_hit': top5_hit,
            'top10_hit': top10_hit,
            'top15_hit': top15_hit,
            'rank': rank if rank > 0 else '-'
        })
    
    return pd.DataFrame(results)


def analyze_results(df_results, model_name):
    """分析验证结果"""
    print(f"\n{'='*80}")
    print(f"{model_name} - 性能分析报告")
    print(f"{'='*80}")
    
    total = len(df_results)
    top15_hits = df_results['top15_hit'].sum()
    top10_hits = df_results['top10_hit'].sum()
    top5_hits = df_results['top5_hit'].sum()
    
    print(f"\n【整体性能】")
    print(f"验证期数: {total}")
    print(f"Top 15 命中率: {top15_hits}/{total} = {top15_hits/total*100:.2f}%")
    print(f"Top 10 命中率: {top10_hits}/{total} = {top10_hits/total*100:.2f}%")
    print(f"Top 5 命中率:  {top5_hits}/{total} = {top5_hits/total*100:.2f}%")
    
    # 分段分析
    segment_size = total // 3
    segments = [
        ('前1/3', df_results.iloc[:segment_size]),
        ('中1/3', df_results.iloc[segment_size:segment_size*2]),
        ('后1/3', df_results.iloc[segment_size*2:])
    ]
    
    print(f"\n【稳定性分析】")
    for seg_name, seg_df in segments:
        hit_rate = seg_df['top15_hit'].sum() / len(seg_df) * 100
        print(f"{seg_name} Top15 命中率: {hit_rate:.2f}%")
    
    # 连续失败分析
    consecutive_failures = []
    current_failure_streak = 0
    
    for hit in df_results['top15_hit']:
        if not hit:
            current_failure_streak += 1
        else:
            if current_failure_streak > 0:
                consecutive_failures.append(current_failure_streak)
            current_failure_streak = 0
    
    if current_failure_streak > 0:
        consecutive_failures.append(current_failure_streak)
    
    print(f"\n【连续失败分析】")
    if consecutive_failures:
        print(f"最长连续失败: {max(consecutive_failures)}期")
        print(f"平均连续失败: {sum(consecutive_failures)/len(consecutive_failures):.2f}期")
        print(f"连续失败次数: {len(consecutive_failures)}次")
        
        from collections import Counter
        failure_counts = Counter(consecutive_failures)
        print(f"连续失败分布:")
        for length in sorted(failure_counts.keys()):
            print(f"  连续失败{length}期: {failure_counts[length]}次")
    else:
        print("无连续失败")
    
    # 命中排名分析
    ranked_hits = df_results[df_results['rank'] != '-']
    if len(ranked_hits) > 0:
        ranks = ranked_hits['rank'].astype(int)
        print(f"\n【命中排名分析】")
        print(f"平均命中排名: {ranks.mean():.2f}")
        print(f"Top 5 内命中: {(ranks <= 5).sum()}次")
        print(f"Top 6-10 命中: {((ranks > 5) & (ranks <= 10)).sum()}次")
        print(f"Top 11-15 命中: {(ranks > 10).sum()}次")


def main():
    """主函数"""
    print("=" * 80)
    print("Top15 预测器对比验证")
    print("=" * 80)
    
    # 读取数据
    print("\n读取历史数据...")
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    
    print(f"数据加载完成: {len(numbers)}期")
    
    # 决定验证期数
    test_periods = min(100, len(numbers) - 51)
    print(f"验证期数: {test_periods}期")
    
    # 验证原版
    print("\n" + "="*80)
    print("【1/2】验证原版 Top15Predictor...")
    print("="*80)
    original_predictor = Top15Predictor()
    df_original = validate_predictor(original_predictor, numbers, test_periods)
    analyze_results(df_original, "原版 Top15Predictor")
    
    # 验证改进版
    print("\n" + "="*80)
    print("【2/2】验证改进版 ImprovedTop15Predictor...")
    print("="*80)
    improved_predictor = ImprovedTop15Predictor()
    df_improved = validate_predictor(improved_predictor, numbers, test_periods)
    analyze_results(df_improved, "改进版 ImprovedTop15Predictor")
    
    # 对比总结
    print("\n" + "="*80)
    print("对比总结")
    print("="*80)
    
    original_rate = df_original['top15_hit'].sum() / len(df_original) * 100
    improved_rate = df_improved['top15_hit'].sum() / len(df_improved) * 100
    improvement = improved_rate - original_rate
    
    print(f"\nTop15 命中率对比:")
    print(f"  原版: {original_rate:.2f}%")
    print(f"  改进版: {improved_rate:.2f}%")
    print(f"  提升: {improvement:+.2f}%")
    
    # 分析改进版是否达到目标
    print(f"\n改进版性能评估:")
    if improved_rate >= 50:
        print(f"  ✅ Top15 命中率达标 ({improved_rate:.2f}% >= 50%)")
    else:
        print(f"  ❌ Top15 命中率未达标 ({improved_rate:.2f}% < 50%)")
    
    # 保存结果
    output_file = 'improved_top15_validation_results.csv'
    df_improved.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n改进版验证结果已保存到: {output_file}")
    
    # 保存对比报告
    comparison_file = 'top15_comparison_report.txt'
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write("Top15 预测器对比报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"验证期数: {test_periods}期\n")
        f.write(f"数据范围: 第{len(numbers)-test_periods}期 - 第{len(numbers)}期\n\n")
        f.write(f"原版 Top15 命中率: {original_rate:.2f}%\n")
        f.write(f"改进版 Top15 命中率: {improved_rate:.2f}%\n")
        f.write(f"提升幅度: {improvement:+.2f}%\n\n")
        
        # 详细对比
        f.write("详细对比:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'指标':<20} {'原版':<15} {'改进版':<15} {'差异':<15}\n")
        f.write("-"*80 + "\n")
        
        for metric in ['top15_hit', 'top10_hit', 'top5_hit']:
            orig_val = df_original[metric].sum() / len(df_original) * 100
            impr_val = df_improved[metric].sum() / len(df_improved) * 100
            diff = impr_val - orig_val
            
            metric_name = metric.replace('_hit', '').upper()
            f.write(f"{metric_name + ' 命中率':<20} {orig_val:>6.2f}%{'':<7} {impr_val:>6.2f}%{'':<7} {diff:>+6.2f}%\n")
    
    print(f"对比报告已保存到: {comparison_file}")
    
    print("\n" + "="*80)
    print("验证完成！")
    print("="*80)


if __name__ == '__main__':
    main()
