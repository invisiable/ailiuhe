"""
测试改进的奇偶预测模型在最近50期的表现
"""

import pandas as pd
import numpy as np
from improved_odd_even_predictor import ImprovedOddEvenPredictor
import os
from datetime import datetime


def validate_improved_model(model_type='ensemble_voting', num_periods=50):
    """
    验证改进模型的预测准确率
    
    参数:
        model_type: 模型类型
        num_periods: 验证的期数
    """
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total_records = len(df)
    
    if num_periods > total_records:
        num_periods = total_records
        print(f"警告: 请求的期数超过数据总量，调整为 {num_periods} 期")
    
    # 存储预测结果
    predictions = []
    
    print(f"\n{'='*80}")
    print(f"开始验证改进的奇偶预测模型: {model_type}")
    print(f"验证期数: 最近 {num_periods} 期")
    print(f"{'='*80}\n")
    
    # 滚动预测
    start_idx = total_records - num_periods
    
    for i in range(num_periods):
        current_idx = start_idx + i
        
        # 使用到当前期之前的所有数据进行训练
        train_df = df.iloc[:current_idx].copy()
        
        # 保存临时训练数据
        temp_csv = 'data/temp_train_improved.csv'
        train_df.to_csv(temp_csv, index=False, encoding='utf-8-sig')
        
        # 训练模型
        predictor = ImprovedOddEvenPredictor()
        
        try:
            predictor.train_model(temp_csv, model_type=model_type, test_size=0.2)
            
            # 预测
            result = predictor.predict(temp_csv)
            
            # 获取实际结果
            actual_number = df.iloc[current_idx]['number']
            actual_parity = '奇数' if actual_number % 2 == 1 else '偶数'
            
            # 记录结果
            is_correct = (result['prediction'] == actual_parity)
            
            predictions.append({
                'period': current_idx + 1,
                'date': df.iloc[current_idx]['date'],
                'predicted': result['prediction'],
                'actual': actual_parity,
                'actual_number': actual_number,
                'correct': is_correct,
                'confidence': result['probability']
            })
            
            # 打印进度
            current_acc = sum([p['correct'] for p in predictions]) / len(predictions) * 100
            status = '✅' if is_correct else '❌'
            print(f"预测第 {current_idx + 1} 期 ({df.iloc[current_idx]['date']}) - 进度: {i+1}/{num_periods}")
            print(f"预测结果: {result['prediction']} (置信度: {result['probability']*100:.2f}%)")
            print(f"实际结果: {actual_parity} (数字: {actual_number})")
            print(f"{status} {'预测正确！' if is_correct else '预测错误！'}\n")
            print(f"当前累计准确率: {sum([p['correct'] for p in predictions])}/{len(predictions)} = {current_acc:.2f}%\n")
            
        except Exception as e:
            print(f"预测第 {current_idx + 1} 期时出错: {e}\n")
            predictions.append({
                'period': current_idx + 1,
                'date': df.iloc[current_idx]['date'],
                'predicted': 'ERROR',
                'actual': '奇数' if df.iloc[current_idx]['number'] % 2 == 1 else '偶数',
                'actual_number': df.iloc[current_idx]['number'],
                'correct': False,
                'confidence': 0
            })
        
        # 清理临时文件
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
    
    # 计算统计数据
    results_df = pd.DataFrame(predictions)
    total_correct = results_df['correct'].sum()
    accuracy = total_correct / num_periods * 100
    
    # 按预测类型分析
    odd_predictions = results_df[results_df['predicted'] == '奇数']
    even_predictions = results_df[results_df['predicted'] == '偶数']
    
    odd_correct = odd_predictions['correct'].sum() if len(odd_predictions) > 0 else 0
    even_correct = even_predictions['correct'].sum() if len(even_predictions) > 0 else 0
    
    odd_acc = odd_correct / len(odd_predictions) * 100 if len(odd_predictions) > 0 else 0
    even_acc = even_correct / len(even_predictions) * 100 if len(even_predictions) > 0 else 0
    
    # 打印总结
    print("\n" + "="*80)
    print(f"{'改进模型验证结果':^80}")
    print(f"{'模型类型: ' + model_type:^80}")
    print("="*80 + "\n")
    
    print(f"✨ 总体准确率: {total_correct}/{num_periods} = {accuracy:.2f}%\n")
    
    print(f"📊 分类准确率:")
    print(f"  奇数预测: {odd_correct}/{len(odd_predictions)} = {odd_acc:.2f}%")
    print(f"  偶数预测: {even_correct}/{len(even_predictions)} = {even_acc:.2f}%\n")
    
    # 保存详细结果
    csv_filename = f'improved_odd_even_validation_{model_type}_{num_periods}periods.csv'
    results_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"详细结果已保存到: {csv_filename}")
    
    # 生成报告
    report_filename = f'改进奇偶预测验证报告_{model_type}_{num_periods}期.md'
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(f"# 改进奇偶预测验证报告 - {model_type}\n\n")
        f.write(f"## 验证概况\n\n")
        f.write(f"- **模型类型**: {model_type}\n")
        f.write(f"- **验证期数**: {num_periods} 期\n")
        f.write(f"- **验证范围**: 第 {start_idx + 1} 期 - 第 {total_records} 期\n")
        f.write(f"- **日期范围**: {df.iloc[start_idx]['date']} - {df.iloc[-1]['date']}\n")
        f.write(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## 验证结果\n\n")
        f.write(f"### 总体性能\n\n")
        f.write(f"- **总体准确率**: {accuracy:.2f}% ({total_correct}/{num_periods})\n")
        f.write(f"- **奇数预测准确率**: {odd_acc:.2f}% ({odd_correct}/{len(odd_predictions)})\n")
        f.write(f"- **偶数预测准确率**: {even_acc:.2f}% ({even_correct}/{len(even_predictions)})\n\n")
        
        f.write(f"### 预测详情\n\n")
        f.write(f"| 期数 | 日期 | 预测 | 实际 | 数字 | 准确 | 置信度 |\n")
        f.write(f"|------|------|------|------|------|------|--------|\n")
        
        for _, row in results_df.iterrows():
            status = '✅' if row['correct'] else '❌'
            f.write(f"| {row['period']} | {row['date']} | {row['predicted']} | {row['actual']} | "
                   f"{row['actual_number']} | {status} | {row['confidence']*100:.1f}% |\n")
        
        f.write(f"\n## 分析总结\n\n")
        f.write(f"该改进模型使用 **{model_type}** 算法，在最近 {num_periods} 期的验证中：\n\n")
        f.write(f"- 总体准确率为 **{accuracy:.2f}%**\n")
        
        if accuracy > 55:
            f.write(f"- ✅ 模型表现**优于随机猜测**（50%），显示出一定的预测能力\n")
        elif accuracy >= 50:
            f.write(f"- ⚠️ 模型表现**接近随机猜测**（50%），预测能力有限\n")
        else:
            f.write(f"- ❌ 模型表现**低于随机猜测**（50%），可能需要进一步优化\n")
        
        if abs(odd_acc - even_acc) > 10:
            f.write(f"- ⚠️ 奇数和偶数的预测准确率差异较大（相差{abs(odd_acc - even_acc):.1f}%），存在预测偏向\n")
    
    print(f"报告已保存到: {report_filename}\n")
    
    return accuracy, results_df


if __name__ == "__main__":
    # 测试多种改进模型
    models_to_test = [
        'ensemble_voting',      # 投票集成
        'ensemble_stacking',    # 堆叠集成
        'catboost',            # CatBoost
        'neural_network',      # 神经网络
        'gradient_boosting',   # 梯度提升（改进参数）
    ]
    
    results_summary = {}
    
    for model_type in models_to_test:
        try:
            print(f"\n\n{'#'*80}")
            print(f"{'#'*80}")
            print(f"测试模型: {model_type}")
            print(f"{'#'*80}")
            print(f"{'#'*80}\n")
            
            accuracy, _ = validate_improved_model(model_type=model_type, num_periods=50)
            results_summary[model_type] = accuracy
            
        except Exception as e:
            print(f"\n❌ 模型 {model_type} 验证失败: {e}\n")
            results_summary[model_type] = 0
    
    # 打印最终总结
    print("\n" + "="*80)
    print("所有模型验证结果总结")
    print("="*80)
    
    sorted_results = sorted(results_summary.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'排名':<6} {'模型类型':<30} {'准确率':<15}")
    print("-" * 80)
    
    for rank, (model, acc) in enumerate(sorted_results, 1):
        status = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{status} {rank:<4} {model:<30} {acc:.2f}%")
    
    print("\n" + "="*80)
    print(f"最佳模型: {sorted_results[0][0]} - 准确率: {sorted_results[0][1]:.2f}%")
    print("="*80)
