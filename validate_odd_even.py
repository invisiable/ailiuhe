"""
验证奇偶性预测模型的准确率
使用滚动预测方法验证最近N期的预测成功率
"""

import pandas as pd
from odd_even_predictor import OddEvenPredictor
import os


def validate_odd_even_prediction(num_periods=30):
    """
    验证奇偶性预测的准确率
    
    参数:
        num_periods: 验证的期数
    """
    print("=" * 80)
    print(f"奇偶性预测模型 - 最近{num_periods}期验证")
    print("=" * 80)
    
    # 读取完整数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total_records = len(df)
    
    print(f"\n数据集信息:")
    print(f"  总记录数: {total_records}")
    print(f"  最早日期: {df['date'].iloc[0]}")
    print(f"  最新日期: {df['date'].iloc[-1]}")
    print(f"  验证期数: 最近{num_periods}期 (第{total_records-num_periods+1}期 至 第{total_records}期)")
    
    print("\n" + "=" * 80)
    print("开始滚动预测...")
    print("=" * 80)
    
    results = []
    correct_predictions = 0
    
    # 对最近N期进行滚动预测
    for i in range(num_periods):
        test_index = total_records - num_periods + i
        train_size = test_index
        
        period_num = test_index + 1
        actual_row = df.iloc[test_index]
        actual_number = actual_row['number']
        actual_odd_even = '奇数' if actual_number % 2 == 1 else '偶数'
        actual_date = actual_row['date']
        
        print(f"\n{'='*80}")
        print(f"预测第 {period_num} 期 ({actual_date}) - 进度: {i+1}/{num_periods}")
        print(f"{'='*80}")
        
        # 准备训练数据
        train_df = df.iloc[:train_size].copy()
        temp_file = f'data/temp_train_odd_even_{i}.csv'
        train_df.to_csv(temp_file, index=False, encoding='utf-8-sig')
        
        try:
            # 训练模型（使用梯度提升）
            predictor = OddEvenPredictor()
            predictor.train_model(temp_file, model_type='gradient_boosting', test_size=0.2)
            
            # 进行预测
            prediction = predictor.predict()
            
            predicted_odd_even = prediction['prediction']
            confidence = prediction['confidence']
            
            print(f"\n预测结果: {predicted_odd_even} (置信度: {confidence*100:.2f}%)")
            print(f"实际结果: {actual_odd_even} (数字: {actual_number})")
            
            # 检查是否正确
            is_correct = (predicted_odd_even == actual_odd_even)
            
            if is_correct:
                print("✅ 预测正确！")
                correct_predictions += 1
            else:
                print("❌ 预测错误！")
            
            results.append({
                '期数': period_num,
                '日期': actual_date,
                '实际数字': actual_number,
                '实际奇偶': actual_odd_even,
                '预测奇偶': predicted_odd_even,
                '置信度': f"{confidence*100:.2f}%",
                '是否正确': '✅' if is_correct else '❌'
            })
            
            # 显示当前累计准确率
            current_count = i + 1
            current_acc = correct_predictions / current_count
            print(f"\n当前累计准确率: {correct_predictions}/{current_count} = {current_acc*100:.2f}%")
            
        except Exception as e:
            print(f"❌ 预测失败: {str(e)}")
            results.append({
                '期数': period_num,
                '日期': actual_date,
                '实际数字': actual_number,
                '实际奇偶': actual_odd_even,
                '预测奇偶': '预测失败',
                '置信度': 'N/A',
                '是否正确': '❌'
            })
        
        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    # 生成最终报告
    print("\n" + "=" * 80)
    print(f"最近{num_periods}期奇偶性预测准确率统计")
    print("=" * 80)
    
    accuracy = correct_predictions / num_periods
    print(f"\n✨ 总体准确率: {correct_predictions}/{num_periods} = {accuracy*100:.2f}%")
    
    # 保存详细结果到CSV
    results_df = pd.DataFrame(results)
    output_file = f'odd_even_validation_{num_periods}periods.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到: {output_file}")
    
    # 生成报告文件
    report_file = f'奇偶预测验证报告_{num_periods}期.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# 奇偶性预测模型 - 最近{num_periods}期验证报告\n\n")
        f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 验证方法\n")
        f.write("- 滚动预测：每次使用之前所有数据预测下一期的奇偶性\n")
        f.write(f"- 验证期数：最近{num_periods}期 (第{total_records-num_periods+1}期 至 第{total_records}期)\n")
        f.write(f"- 日期范围：{df['date'].iloc[total_records-num_periods]} 至 {df['date'].iloc[-1]}\n")
        f.write("- 使用模型：Gradient Boosting Classifier\n\n")
        
        f.write("## 总体准确率\n\n")
        f.write(f"**准确率: {correct_predictions}/{num_periods} = {accuracy*100:.2f}%**\n\n")
        
        # 统计预测奇数和偶数的准确率
        odd_results = [r for r in results if r['实际奇偶'] == '奇数']
        even_results = [r for r in results if r['实际奇偶'] == '偶数']
        
        odd_correct = sum(1 for r in odd_results if r['是否正确'] == '✅')
        even_correct = sum(1 for r in even_results if r['是否正确'] == '✅')
        
        f.write("## 分类统计\n\n")
        f.write("| 类别 | 总数 | 正确数 | 准确率 |\n")
        f.write("|------|------|--------|--------|\n")
        if len(odd_results) > 0:
            f.write(f"| 奇数 | {len(odd_results)} | {odd_correct} | {odd_correct/len(odd_results)*100:.2f}% |\n")
        if len(even_results) > 0:
            f.write(f"| 偶数 | {len(even_results)} | {even_correct} | {even_correct/len(even_results)*100:.2f}% |\n")
        f.write("\n")
        
        f.write("## 详细结果\n\n")
        f.write("| 期数 | 日期 | 实际数字 | 实际奇偶 | 预测奇偶 | 置信度 | 结果 |\n")
        f.write("|------|------|---------|---------|---------|--------|------|\n")
        
        for result in results:
            f.write(f"| {result['期数']} | {result['日期']} | {result['实际数字']} | ")
            f.write(f"{result['实际奇偶']} | {result['预测奇偶']} | {result['置信度']} | {result['是否正确']} |\n")
        
        f.write("\n## 结论\n\n")
        if accuracy >= 0.7:
            f.write(f"✅ **奇偶性预测准确率达到 {accuracy*100:.2f}%，模型表现优秀！**\n\n")
            f.write("该模型可以有效地预测下一期数字的奇偶性，可以作为辅助决策工具使用。\n")
        elif accuracy >= 0.6:
            f.write(f"✅ **奇偶性预测准确率为 {accuracy*100:.2f}%，模型表现良好。**\n\n")
            f.write("该模型具有一定的预测能力，可以作为参考。\n")
        else:
            f.write(f"⚠️ 奇偶性预测准确率为 {accuracy*100:.2f}%，模型还需要进一步优化。\n\n")
    
    print(f"报告已保存到: {report_file}")
    
    return results, accuracy


if __name__ == "__main__":
    # 验证最近30期
    validate_odd_even_prediction(num_periods=30)
