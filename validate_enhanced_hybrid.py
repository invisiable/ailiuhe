"""
验证增强版混合策略（结合奇偶预测）在最近50期的表现
"""

import pandas as pd
import os
from enhanced_hybrid_predictor import EnhancedHybridPredictor
from final_hybrid_predictor import FinalHybridPredictor


def validate_enhanced_strategy(num_periods=50):
    """验证增强策略"""
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total_records = len(df)
    
    if num_periods > total_records:
        num_periods = total_records
        print(f"警告: 请求的期数超过数据总量，调整为 {num_periods} 期")
    
    start_idx = total_records - num_periods
    
    print(f"\n{'='*80}")
    print(f"{'增强版混合策略验证（结合奇偶预测）':^74}")
    print(f"{'='*80}\n")
    print(f"验证期数: {num_periods} 期")
    print(f"验证范围: 第{start_idx + 1}期 - 第{total_records}期")
    print(f"日期范围: {df.iloc[start_idx]['date']} - {df.iloc[-1]['date']}")
    print(f"\n{'='*80}\n")
    
    # 存储结果
    results_base = []  # 基础混合策略
    results_enhanced = []  # 增强策略
    
    # 滚动验证
    for i in range(num_periods):
        current_idx = start_idx + i
        
        # 使用到当前期之前的所有数据
        train_df = df.iloc[:current_idx].copy()
        
        # 保存临时训练数据
        temp_csv = 'data/temp_train_enhanced.csv'
        train_df.to_csv(temp_csv, index=False, encoding='utf-8-sig')
        
        # 实际结果
        actual_number = df.iloc[current_idx]['number']
        period = df.iloc[current_idx]['期数'] if '期数' in df.columns else current_idx + 1
        date = df.iloc[current_idx]['date']
        
        try:
            # 创建预测器
            predictor = EnhancedHybridPredictor()
            
            # 1. 基础混合策略预测（不使用奇偶）
            base_result = predictor.predict(temp_csv, use_odd_even=False)
            base_top15 = base_result['top15']
            base_top10 = base_top15[:10]
            base_top5 = base_top15[:5]
            
            # 2. 增强策略预测（使用奇偶）
            enhanced_result = predictor.predict(temp_csv, use_odd_even=True)
            enhanced_top15 = enhanced_result['top15']
            enhanced_top10 = enhanced_top15[:10]
            enhanced_top5 = enhanced_top15[:5]
            odd_even_pred = enhanced_result['odd_even_prediction']
            
            # 记录基础策略结果
            results_base.append({
                'period': period,
                'date': date,
                'actual': actual_number,
                'in_top15': actual_number in base_top15,
                'in_top10': actual_number in base_top10,
                'in_top5': actual_number in base_top5,
            })
            
            # 记录增强策略结果
            results_enhanced.append({
                'period': period,
                'date': date,
                'actual': actual_number,
                'in_top15': actual_number in enhanced_top15,
                'in_top10': actual_number in enhanced_top10,
                'in_top5': actual_number in enhanced_top5,
                'odd_even_pred': odd_even_pred['predicted'],
                'confidence': odd_even_pred['confidence']
            })
            
            # 实时显示进度
            base_acc15 = sum([r['in_top15'] for r in results_base]) / len(results_base) * 100
            enhanced_acc15 = sum([r['in_top15'] for r in results_enhanced]) / len(results_enhanced) * 100
            
            base_status = '✅' if actual_number in base_top15 else '❌'
            enhanced_status = '✅' if actual_number in enhanced_top15 else '❌'
            
            print(f"{i+1:2d}. 第{period}期 ({date}) 实际:{actual_number:2d} | " +
                  f"基础:{base_status} 增强:{enhanced_status} | " +
                  f"奇偶预测:{odd_even_pred['predicted']:4s}({odd_even_pred['confidence']*100:4.1f}%) | " +
                  f"TOP15准确率 基础:{base_acc15:5.1f}% 增强:{enhanced_acc15:5.1f}%")
            
        except Exception as e:
            print(f"{i+1:2d}. 第{period}期 预测失败: {e}")
        
        # 清理临时文件
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
    
    # 计算统计数据
    base_df = pd.DataFrame(results_base)
    enhanced_df = pd.DataFrame(results_enhanced)
    
    base_acc15 = base_df['in_top15'].sum() / len(base_df) * 100
    base_acc10 = base_df['in_top10'].sum() / len(base_df) * 100
    base_acc5 = base_df['in_top5'].sum() / len(base_df) * 100
    
    enhanced_acc15 = enhanced_df['in_top15'].sum() / len(enhanced_df) * 100
    enhanced_acc10 = enhanced_df['in_top10'].sum() / len(enhanced_df) * 100
    enhanced_acc5 = enhanced_df['in_top5'].sum() / len(enhanced_df) * 100
    
    # 计算提升
    improvement_15 = enhanced_acc15 - base_acc15
    improvement_10 = enhanced_acc10 - base_acc10
    improvement_5 = enhanced_acc5 - base_acc5
    
    # 打印总结
    print("\n" + "="*80)
    print(f"{'验证结果对比':^74}")
    print("="*80 + "\n")
    
    print(f"{'指标':<20} {'基础混合策略':<20} {'增强策略(+奇偶)':<20} {'提升':<15}")
    print("-"*80)
    print(f"{'TOP15准确率':<20} {base_acc15:>6.2f}% ({base_df['in_top15'].sum()}/{len(base_df):<10}) " +
          f"{enhanced_acc15:>6.2f}% ({enhanced_df['in_top15'].sum()}/{len(enhanced_df):<10}) " +
          f"{improvement_15:>+6.2f}pp")
    print(f"{'TOP10准确率':<20} {base_acc10:>6.2f}% ({base_df['in_top10'].sum()}/{len(base_df):<10}) " +
          f"{enhanced_acc10:>6.2f}% ({enhanced_df['in_top10'].sum()}/{len(enhanced_df):<10}) " +
          f"{improvement_10:>+6.2f}pp")
    print(f"{'TOP5准确率':<20} {base_acc5:>6.2f}% ({base_df['in_top5'].sum()}/{len(base_df):<10}) " +
          f"{enhanced_acc5:>6.2f}% ({enhanced_df['in_top5'].sum()}/{len(enhanced_df):<10}) " +
          f"{improvement_5:>+6.2f}pp")
    
    print("\n" + "="*80)
    
    # 分析奇偶预测的影响
    print(f"\n奇偶预测统计:")
    print("-"*80)
    
    # 计算奇偶预测准确率
    odd_even_correct = 0
    for r in results_enhanced:
        actual_parity = '奇数' if r['actual'] % 2 == 1 else '偶数'
        if r['odd_even_pred'] == actual_parity:
            odd_even_correct += 1
    
    odd_even_acc = odd_even_correct / len(results_enhanced) * 100
    print(f"奇偶预测准确率: {odd_even_acc:.2f}% ({odd_even_correct}/{len(results_enhanced)})")
    
    # 按置信度分组分析
    high_conf = [r for r in results_enhanced if r['confidence'] >= 0.65]
    mid_conf = [r for r in results_enhanced if 0.55 <= r['confidence'] < 0.65]
    low_conf = [r for r in results_enhanced if r['confidence'] < 0.55]
    
    print(f"\n按奇偶预测置信度分组:")
    print(f"  高置信度(≥65%): {len(high_conf)}期, " +
          f"TOP15准确率 {sum([r['in_top15'] for r in high_conf])/len(high_conf)*100:.1f}%" if high_conf else "  高置信度(≥65%): 0期")
    print(f"  中等置信度(55-65%): {len(mid_conf)}期, " +
          f"TOP15准确率 {sum([r['in_top15'] for r in mid_conf])/len(mid_conf)*100:.1f}%" if mid_conf else "  中等置信度(55-65%): 0期")
    print(f"  低置信度(<55%): {len(low_conf)}期, " +
          f"TOP15准确率 {sum([r['in_top15'] for r in low_conf])/len(low_conf)*100:.1f}%" if low_conf else "  低置信度(<55%): 0期")
    
    # 保存详细结果
    result_summary = {
        'base_strategy': {
            'top15': f"{base_acc15:.2f}%",
            'top10': f"{base_acc10:.2f}%",
            'top5': f"{base_acc5:.2f}%"
        },
        'enhanced_strategy': {
            'top15': f"{enhanced_acc15:.2f}%",
            'top10': f"{enhanced_acc10:.2f}%",
            'top5': f"{enhanced_acc5:.2f}%"
        },
        'improvement': {
            'top15': f"{improvement_15:+.2f}pp",
            'top10': f"{improvement_10:+.2f}pp",
            'top5': f"{improvement_5:+.2f}pp"
        },
        'odd_even_accuracy': f"{odd_even_acc:.2f}%"
    }
    
    # 保存到CSV
    base_df.to_csv('base_hybrid_validation_50periods.csv', index=False, encoding='utf-8-sig')
    enhanced_df.to_csv('enhanced_hybrid_validation_50periods.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n详细结果已保存:")
    print(f"  基础策略: base_hybrid_validation_50periods.csv")
    print(f"  增强策略: enhanced_hybrid_validation_50periods.csv")
    
    # 生成报告
    generate_report(num_periods, result_summary, base_df, enhanced_df)
    
    print("\n" + "="*80)
    print("验证完成！")
    print("="*80 + "\n")
    
    return result_summary


def generate_report(num_periods, summary, base_df, enhanced_df):
    """生成验证报告"""
    from datetime import datetime
    
    report_filename = f'增强混合策略验证报告_{num_periods}期.md'
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(f"# 增强混合策略验证报告 - {num_periods}期\n\n")
        f.write(f"## 验证概况\n\n")
        f.write(f"- **验证期数**: {num_periods} 期\n")
        f.write(f"- **基础策略**: 固化混合组合策略\n")
        f.write(f"- **增强方法**: 结合奇偶预测模型（72特征+集成学习）\n")
        f.write(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## 策略对比\n\n")
        f.write(f"### 基础混合策略\n")
        f.write(f"- TOP 1-5: 使用最近10期数据策略（精准预测）\n")
        f.write(f"- TOP 6-15: 使用全部历史数据策略（稳定覆盖）\n\n")
        
        f.write(f"### 增强策略（基础+奇偶预测）\n")
        f.write(f"- 在基础策略的TOP15候选上应用奇偶预测\n")
        f.write(f"- 根据奇偶预测置信度调整候选数字的排序\n")
        f.write(f"- 高置信度(≥65%): 强调奇偶性（10:5比例）\n")
        f.write(f"- 中等置信度(55-65%): 适度调整（9:6比例）\n")
        f.write(f"- 低置信度(<55%): 轻微调整（8:7比例）\n\n")
        
        f.write(f"## 验证结果\n\n")
        f.write(f"| 指标 | 基础策略 | 增强策略 | 提升 |\n")
        f.write(f"|------|----------|----------|------|\n")
        f.write(f"| TOP15准确率 | {summary['base_strategy']['top15']} | " +
                f"{summary['enhanced_strategy']['top15']} | {summary['improvement']['top15']} |\n")
        f.write(f"| TOP10准确率 | {summary['base_strategy']['top10']} | " +
                f"{summary['enhanced_strategy']['top10']} | {summary['improvement']['top10']} |\n")
        f.write(f"| TOP5准确率 | {summary['base_strategy']['top5']} | " +
                f"{summary['enhanced_strategy']['top5']} | {summary['improvement']['top5']} |\n\n")
        
        f.write(f"### 奇偶预测贡献\n\n")
        f.write(f"- **奇偶预测准确率**: {summary['odd_even_accuracy']}\n\n")
        
        f.write(f"## 详细分析\n\n")
        f.write(f"### 逐期对比\n\n")
        f.write(f"| 期数 | 日期 | 实际 | 基础策略 | 增强策略 | 奇偶预测 |\n")
        f.write(f"|------|------|------|----------|----------|----------|\n")
        
        for i in range(len(base_df)):
            base_row = base_df.iloc[i]
            enhanced_row = enhanced_df.iloc[i]
            base_status = '✅ TOP15' if base_row['in_top15'] else ('🟡 TOP10' if base_row['in_top10'] else ('🟠 TOP5' if base_row['in_top5'] else '❌'))
            enhanced_status = '✅ TOP15' if enhanced_row['in_top15'] else ('🟡 TOP10' if enhanced_row['in_top10'] else ('🟠 TOP5' if enhanced_row['in_top5'] else '❌'))
            
            f.write(f"| {base_row['period']} | {base_row['date']} | {base_row['actual']} | " +
                   f"{base_status} | {enhanced_status} | {enhanced_row['odd_even_pred']} |\n")
        
        f.write(f"\n## 结论\n\n")
        
        improvement_15 = float(summary['improvement']['top15'].replace('pp', '').replace('+', ''))
        
        if improvement_15 > 2:
            f.write(f"✅ **增强策略显著优于基础策略**\n\n")
            f.write(f"结合奇偶预测后，TOP15准确率提升了{summary['improvement']['top15']}，说明奇偶预测能有效改善混合策略的预测质量。\n\n")
        elif improvement_15 > 0:
            f.write(f"🟡 **增强策略略优于基础策略**\n\n")
            f.write(f"结合奇偶预测后，TOP15准确率提升了{summary['improvement']['top15']}，有小幅改善。\n\n")
        elif improvement_15 == 0:
            f.write(f"⚠️ **增强策略与基础策略持平**\n\n")
            f.write(f"结合奇偶预测后，TOP15准确率没有变化，奇偶预测未能带来额外收益。\n\n")
        else:
            f.write(f"❌ **增强策略不如基础策略**\n\n")
            f.write(f"结合奇偶预测后，TOP15准确率下降了{abs(improvement_15):.2f}个百分点，说明奇偶预测可能干扰了原有策略。\n\n")
        
        f.write(f"### 建议\n\n")
        if improvement_15 > 0:
            f.write(f"- ✅ 建议使用增强策略，可以获得更好的预测效果\n")
            f.write(f"- 📊 奇偶预测模型有效，可以继续优化\n")
        else:
            f.write(f"- ⚠️ 建议继续使用基础混合策略\n")
            f.write(f"- 🔧 需要改进奇偶预测模型或调整结合方式\n")
    
    print(f"  报告: {report_filename}")


if __name__ == "__main__":
    validate_enhanced_strategy(num_periods=50)
