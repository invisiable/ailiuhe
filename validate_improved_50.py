"""
使用最佳改进模型验证50期
基于10期测试，选择表现最好的模型进行50期验证
"""

from improved_odd_even_predictor import ImprovedOddEvenPredictor
import pandas as pd
import os


def validate_50_periods_best_model():
    """使用最佳模型验证50期"""
    # 根据10期测试，ensemble_voting、ensemble_stacking、gradient_boosting、xgboost都是60%
    # 选择ensemble_voting（集成投票），理论上更稳定
    model_type = 'ensemble_voting'
    num_periods = 50
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total_records = len(df)
    start_idx = total_records - num_periods
    
    predictions = []
    correct_count = 0
    
    print(f"\n{'='*80}")
    print(f"使用改进模型验证最近50期: {model_type}")
    print(f"特征数量: 72 (相比原模型的28个特征，增加了44个)")
    print(f"{'='*80}\n")
    
    for i in range(num_periods):
        current_idx = start_idx + i
        train_df = df.iloc[:current_idx].copy()
        
        temp_csv = 'data/temp_train_improved_50.csv'
        train_df.to_csv(temp_csv, index=False, encoding='utf-8-sig')
        
        predictor = ImprovedOddEvenPredictor()
        
        try:
            # 静默训练
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            predictor.train_model(temp_csv, model_type=model_type, test_size=0.2)
            result = predictor.predict(temp_csv)
            
            sys.stdout = old_stdout
            
            # 获取实际结果
            actual_number = df.iloc[current_idx]['number']
            actual_parity = '奇数' if actual_number % 2 == 1 else '偶数'
            is_correct = (result['prediction'] == actual_parity)
            
            if is_correct:
                correct_count += 1
            
            predictions.append({
                'period': current_idx + 1,
                'date': df.iloc[current_idx]['date'],
                'predicted': result['prediction'],
                'actual': actual_parity,
                'actual_number': actual_number,
                'correct': is_correct,
                'confidence': result['probability']
            })
            
            status = '✅' if is_correct else '❌'
            print(f"{i+1:2d}. 第{current_idx+1}期 ({df.iloc[current_idx]['date']}) "
                  f"预测:{result['prediction']:4s} 实际:{actual_parity:4s} "
                  f"数字:{actual_number:2d} {status} "
                  f"置信度:{result['probability']*100:5.1f}% "
                  f"累计:{correct_count}/{i+1}={correct_count/(i+1)*100:.1f}%")
            
        except Exception as e:
            print(f"{i+1:2d}. 第{current_idx+1}期 预测失败: {e}")
            predictions.append({
                'period': current_idx + 1,
                'date': df.iloc[current_idx]['date'],
                'predicted': 'ERROR',
                'actual': '奇数' if df.iloc[current_idx]['number'] % 2 == 1 else '偶数',
                'actual_number': df.iloc[current_idx]['number'],
                'correct': False,
                'confidence': 0
            })
        
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
    
    # 计算统计数据
    results_df = pd.DataFrame(predictions)
    accuracy = correct_count / num_periods * 100
    
    # 按预测类型分析
    odd_predictions = results_df[results_df['predicted'] == '奇数']
    even_predictions = results_df[results_df['predicted'] == '偶数']
    
    odd_correct = odd_predictions['correct'].sum() if len(odd_predictions) > 0 else 0
    even_correct = even_predictions['correct'].sum() if len(even_predictions) > 0 else 0
    
    odd_acc = odd_correct / len(odd_predictions) * 100 if len(odd_predictions) > 0 else 0
    even_acc = even_correct / len(even_predictions) * 100 if len(even_predictions) > 0 else 0
    
    # 打印总结
    print("\n" + "="*80)
    print(f"{'改进模型 50期验证结果':^80}")
    print("="*80 + "\n")
    
    print(f"🎯 模型类型: {model_type}")
    print(f"📊 特征数量: 72 (vs 原模型 28)")
    print(f"✨ 总体准确率: {correct_count}/{num_periods} = {accuracy:.2f}%\n")
    
    print(f"📈 分类准确率:")
    print(f"  奇数预测: {odd_correct}/{len(odd_predictions)} = {odd_acc:.2f}%")
    print(f"  偶数预测: {even_correct}/{len(even_predictions)} = {even_acc:.2f}%\n")
    
    # 与原模型对比
    print(f"🔄 与原模型对比:")
    print(f"  原模型 (gradient_boosting, 28特征): 50.00% (25/50)")
    print(f"  改进模型 ({model_type}, 72特征): {accuracy:.2f}% ({correct_count}/50)")
    improvement = accuracy - 50.0
    if improvement > 0:
        print(f"  ✅ 提升: +{improvement:.2f}个百分点")
    elif improvement < 0:
        print(f"  ❌ 下降: {improvement:.2f}个百分点")
    else:
        print(f"  ⚠️ 持平")
    
    # 保存详细结果
    csv_filename = f'improved_odd_even_validation_50periods.csv'
    results_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到: {csv_filename}")
    
    # 生成报告
    from datetime import datetime
    report_filename = f'改进奇偶预测验证报告_50期.md'
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(f"# 改进奇偶预测验证报告 - 50期\n\n")
        f.write(f"## 验证概况\n\n")
        f.write(f"- **模型类型**: {model_type} (集成投票)\n")
        f.write(f"- **特征数量**: 72 (原模型: 28)\n")
        f.write(f"- **验证期数**: {num_periods} 期\n")
        f.write(f"- **验证范围**: 第 {start_idx + 1} 期 - 第 {total_records} 期\n")
        f.write(f"- **日期范围**: {df.iloc[start_idx]['date']} - {df.iloc[-1]['date']}\n")
        f.write(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## 模型改进点\n\n")
        f.write(f"### 1. 特征工程优化\n\n")
        f.write(f"原模型28个特征 → 改进模型72个特征，新增:\n\n")
        f.write(f"- **奇偶连续性特征**: 增加偶数连续性、最长连续记录\n")
        f.write(f"- **更多统计窗口**: 从4个窗口(3,5,7,10)扩展到6个(3,5,7,10,15,20)\n")
        f.write(f"- **倾向性特征**: 生肖奇偶倾向、五行奇偶倾向、区间奇偶倾向\n")
        f.write(f"- **细粒度分箱**: 数字区间从5档扩展到10档\n")
        f.write(f"- **周期性特征**: 增加7天、14天、30天多周期\n")
        f.write(f"- **交替模式**: 奇偶交替模式识别\n")
        f.write(f"- **波动性特征**: 标准差、偏度、峰度等统计特征\n")
        f.write(f"- **间隔特征**: 距离上次奇数/偶数的间隔\n")
        f.write(f"- **组合特征**: 生肖五行组合编码\n\n")
        
        f.write(f"### 2. 模型算法优化\n\n")
        f.write(f"使用 **{model_type}** 集成学习方法，结合:\n")
        f.write(f"- Gradient Boosting Classifier\n")
        f.write(f"- Random Forest Classifier\n")
        f.write(f"- XGBoost Classifier\n")
        f.write(f"- LightGBM Classifier\n")
        f.write(f"- Logistic Regression\n\n")
        f.write(f"采用软投票(soft voting)机制，综合多个模型的概率输出。\n\n")
        
        f.write(f"## 验证结果\n\n")
        f.write(f"### 总体性能\n\n")
        f.write(f"- **总体准确率**: {accuracy:.2f}% ({correct_count}/{num_periods})\n")
        f.write(f"- **奇数预测准确率**: {odd_acc:.2f}% ({odd_correct}/{len(odd_predictions)})\n")
        f.write(f"- **偶数预测准确率**: {even_acc:.2f}% ({even_correct}/{len(even_predictions)})\n\n")
        
        f.write(f"### 与原模型对比\n\n")
        f.write(f"| 指标 | 原模型 | 改进模型 | 变化 |\n")
        f.write(f"|------|--------|----------|------|\n")
        f.write(f"| 特征数量 | 28 | 72 | +44 |\n")
        f.write(f"| 算法 | Gradient Boosting | Ensemble Voting | 升级 |\n")
        f.write(f"| 准确率 | 50.00% | {accuracy:.2f}% | {improvement:+.2f}pp |\n\n")
        
        f.write(f"### 预测详情\n\n")
        f.write(f"| 期数 | 日期 | 预测 | 实际 | 数字 | 准确 | 置信度 |\n")
        f.write(f"|------|------|------|------|------|------|--------|\n")
        
        for _, row in results_df.iterrows():
            status = '✅' if row['correct'] else '❌'
            f.write(f"| {row['period']} | {row['date']} | {row['predicted']} | {row['actual']} | "
                   f"{row['actual_number']} | {status} | {row['confidence']*100:.1f}% |\n")
        
        f.write(f"\n## 结论\n\n")
        
        if accuracy > 55:
            f.write(f"✅ **改进模型表现优异**\n\n")
            f.write(f"改进后的模型在50期验证中达到 **{accuracy:.2f}%** 的准确率，")
            f.write(f"相比原模型的50%提升了 **{improvement:.2f}个百分点**，")
            f.write(f"明显优于随机猜测，证明：\n\n")
            f.write(f"1. **特征工程有效**: 72个精心设计的特征捕捉到了更多奇偶性规律\n")
            f.write(f"2. **集成学习优势**: 多模型投票提升了预测稳定性\n")
            f.write(f"3. **实用价值**: 模型具有实际预测能力\n\n")
        elif accuracy >= 50:
            f.write(f"⚠️ **改进模型效果有限**\n\n")
            f.write(f"改进后的模型准确率为 **{accuracy:.2f}%**，")
            f.write(f"仅比原模型提升 **{improvement:.2f}个百分点**，")
            f.write(f"接近随机猜测水平，说明：\n\n")
            f.write(f"1. 幸运数字奇偶性具有较强随机性\n")
            f.write(f"2. 虽然增加了特征但规律有限\n")
            f.write(f"3. 可能需要更长期的数据或其他方法\n\n")
        else:
            f.write(f"❌ **改进未达预期**\n\n")
            f.write(f"改进后的模型准确率为 **{accuracy:.2f}%**，")
            f.write(f"反而比原模型下降了 **{abs(improvement):.2f}个百分点**，可能原因：\n\n")
            f.write(f"1. 过多特征导致过拟合\n")
            f.write(f"2. 模型复杂度过高\n")
            f.write(f"3. 需要调整特征选择策略\n\n")
    
    print(f"报告已保存到: {report_filename}\n")
    print("="*80)
    
    return accuracy


if __name__ == "__main__":
    validate_50_periods_best_model()
