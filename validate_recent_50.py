"""
综合预测模型 - 最近50期的预测成功率验证
滚动预测：每次用之前的所有数据预测下一期
"""
import sys
sys.path.insert(0, 'd:\\AIagent')

import pandas as pd
from enhanced_predictor_v2 import EnhancedPredictor
from lucky_number_predictor import LuckyNumberPredictor
import os

def validate_recent_50_periods():
    """验证最近50期的预测成功率"""
    print("=" * 80)
    print("综合预测模型 - 最近50期滚动预测验证")
    print("=" * 80)
    
    # 读取完整数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total_records = len(df)
    
    print(f"\n数据集信息:")
    print(f"  总记录数: {total_records}")
    print(f"  最早日期: {df['date'].iloc[0]}")
    print(f"  最新日期: {df['date'].iloc[-1]}")
    print(f"  验证期数: 最近50期 (第{total_records-49}期 至 第{total_records}期)")
    
    print("\n" + "=" * 80)
    print("开始滚动预测...")
    print("=" * 80)
    
    results = []
    top3_hits = 0
    top5_hits = 0
    top10_hits = 0
    top15_hits = 0
    top20_hits = 0
    
    # 对最近50期进行滚动预测
    for i in range(50):
        test_index = total_records - 50 + i
        train_size = test_index
        
        period_num = test_index + 1
        actual_row = df.iloc[test_index]
        actual_number = actual_row['number']
        actual_date = actual_row['date']
        
        print(f"\n{'='*80}")
        print(f"预测第 {period_num} 期 ({actual_date}) - 进度: {i+1}/50")
        print(f"{'='*80}")
        print(f"使用前 {train_size} 期数据训练...")
        
        # 准备训练数据
        train_df = df.iloc[:train_size].copy()
        temp_file = f'data/temp_train_{i}.csv'
        train_df.to_csv(temp_file, index=False, encoding='utf-8-sig')
        
        try:
            # 训练3个模型
            predictors = []
            for model_type in ['gradient_boosting', 'lightgbm', 'xgboost']:
                pred = LuckyNumberPredictor()
                pred.load_data(temp_file, 'number', 'date', 'animal', 'element')
                pred.train_model(model_type, test_size=0.2)
                predictors.append(pred)
            
            # 创建增强预测器
            enhanced = EnhancedPredictor(predictors)
            predictions = enhanced.comprehensive_predict_v2(top_k=20)
            
            # 提取预测结果
            top20_numbers = [pred['number'] for pred in predictions[:20]]
            top15_numbers = top20_numbers[:15]
            top10_numbers = top20_numbers[:10]
            top5_numbers = top20_numbers[:5]
            top3_numbers = top20_numbers[:3]
            
            print(f"\n预测 Top 3:  {top3_numbers}")
            print(f"预测 Top 5:  {top5_numbers}")
            print(f"预测 Top 10: {top10_numbers}")
            print(f"预测 Top 15: {top15_numbers}")
            print(f"预测 Top 20: {top20_numbers}")
            print(f"实际数字:    {actual_number}")
            
            # 检查命中情况
            hit_top3 = actual_number in top3_numbers
            hit_top5 = actual_number in top5_numbers
            hit_top10 = actual_number in top10_numbers
            hit_top15 = actual_number in top15_numbers
            hit_top20 = actual_number in top20_numbers
            
            if hit_top3:
                rank = top3_numbers.index(actual_number) + 1
                result = f"✅ Top 3 命中 (排名第{rank})"
                top3_hits += 1
                top5_hits += 1
                top10_hits += 1
                top15_hits += 1
                top20_hits += 1
            elif hit_top5:
                rank = top5_numbers.index(actual_number) + 1
                result = f"✅ Top 5 命中 (排名第{rank})"
                top5_hits += 1
                top10_hits += 1
                top15_hits += 1
                top20_hits += 1
            elif hit_top10:
                rank = top10_numbers.index(actual_number) + 1
                result = f"✅ Top 10 命中 (排名第{rank})"
                top10_hits += 1
                top15_hits += 1
                top20_hits += 1
            elif hit_top15:
                rank = top15_numbers.index(actual_number) + 1
                result = f"✅ Top 15 命中 (排名第{rank})"
                top15_hits += 1
                top20_hits += 1
            elif hit_top20:
                rank = top20_numbers.index(actual_number) + 1
                result = f"✅ Top 20 命中 (排名第{rank})"
                top20_hits += 1
            else:
                result = "❌ 未命中"
            
            print(f"\n{result}")
            
            results.append({
                '期数': period_num,
                '日期': actual_date,
                '实际数字': actual_number,
                'Top3命中': hit_top3,
                'Top5命中': hit_top5,
                'Top10命中': hit_top10,
                'Top15命中': hit_top15,
                'Top20命中': hit_top20,
                '预测Top20': str(top20_numbers)
            })
            
            # 显示当前累计成功率
            current_count = i + 1
            print(f"\n当前累计成功率 ({current_count}期):")
            print(f"  Top 3:  {top3_hits}/{current_count} = {top3_hits/current_count*100:.2f}%")
            print(f"  Top 5:  {top5_hits}/{current_count} = {top5_hits/current_count*100:.2f}%")
            print(f"  Top 10: {top10_hits}/{current_count} = {top10_hits/current_count*100:.2f}%")
            print(f"  Top 15: {top15_hits}/{current_count} = {top15_hits/current_count*100:.2f}%")
            print(f"  Top 20: {top20_hits}/{current_count} = {top20_hits/current_count*100:.2f}%")
            
        except Exception as e:
            print(f"❌ 预测失败: {str(e)}")
            results.append({
                '期数': period_num,
                '日期': actual_date,
                '实际数字': actual_number,
                'Top3命中': False,
                'Top5命中': False,
                'Top10命中': False,
                'Top15命中': False,
                'Top20命中': False,
                '预测Top20': f"预测失败: {str(e)}"
            })
        
        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    # 生成最终报告
    print("\n" + "=" * 80)
    print("最近50期预测成功率统计")
    print("=" * 80)
    
    print(f"\nTop 3 命中:  {top3_hits}/50 = {top3_hits/50*100:.2f}%")
    print(f"Top 5 命中:  {top5_hits}/50 = {top5_hits/50*100:.2f}%")
    print(f"Top 10 命中: {top10_hits}/50 = {top10_hits/50*100:.2f}%")
    print(f"Top 15 命中: {top15_hits}/50 = {top15_hits/50*100:.2f}%")
    print(f"Top 20 命中: {top20_hits}/50 = {top20_hits/50*100:.2f}%")
    
    # 保存详细结果到CSV
    results_df = pd.DataFrame(results)
    output_file = 'validate_recent_50_results.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到: {output_file}")
    
    # 生成报告文件
    report_file = '最近50期验证报告.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 综合预测模型 - 最近50期验证报告\n\n")
        f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 验证方法\n")
        f.write("- 滚动预测：每次使用之前所有数据预测下一期\n")
        f.write(f"- 验证期数：最近50期 (第{total_records-49}期 至 第{total_records}期)\n")
        f.write(f"- 日期范围：{df['date'].iloc[total_records-50]} 至 {df['date'].iloc[-1]}\n\n")
        
        f.write("## 总体成功率\n\n")
        f.write(f"| 范围 | 命中次数 | 成功率 |\n")
        f.write(f"|------|---------|--------|\n")
        f.write(f"| Top 3  | {top3_hits}/50 | {top3_hits/50*100:.2f}% |\n")
        f.write(f"| Top 5  | {top5_hits}/50 | {top5_hits/50*100:.2f}% |\n")
        f.write(f"| Top 10 | {top10_hits}/50 | {top10_hits/50*100:.2f}% |\n")
        f.write(f"| Top 15 | {top15_hits}/50 | {top15_hits/50*100:.2f}% |\n")
        f.write(f"| Top 20 | {top20_hits}/50 | {top20_hits/50*100:.2f}% |\n\n")
        
        f.write("## 详细结果\n\n")
        f.write("| 期数 | 日期 | 实际数字 | Top3 | Top5 | Top10 | Top15 | Top20 |\n")
        f.write("|------|------|---------|------|------|-------|-------|-------|\n")
        
        for result in results:
            f.write(f"| {result['期数']} | {result['日期']} | {result['实际数字']} | ")
            f.write(f"{'✅' if result['Top3命中'] else '❌'} | ")
            f.write(f"{'✅' if result['Top5命中'] else '❌'} | ")
            f.write(f"{'✅' if result['Top10命中'] else '❌'} | ")
            f.write(f"{'✅' if result['Top15命中'] else '❌'} | ")
            f.write(f"{'✅' if result['Top20命中'] else '❌'} |\n")
        
        f.write("\n## 结论\n\n")
        if top15_hits / 50 >= 0.6:
            f.write(f"✅ **Top 15 成功率达到 {top15_hits/50*100:.2f}%，达到目标！**\n\n")
        else:
            f.write(f"⚠️ Top 15 成功率为 {top15_hits/50*100:.2f}%，还需继续优化。\n\n")
        
        if top20_hits / 50 >= 0.7:
            f.write(f"✅ **Top 20 成功率达到 {top20_hits/50*100:.2f}%，表现优异！**\n\n")
        else:
            f.write(f"⚠️ Top 20 成功率为 {top20_hits/50*100:.2f}%，有提升空间。\n\n")
    
    print(f"报告已保存到: {report_file}")
    
    return results, top3_hits, top5_hits, top10_hits, top15_hits, top20_hits


if __name__ == "__main__":
    validate_recent_50_periods()
