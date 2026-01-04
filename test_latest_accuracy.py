"""
测试最新数据的综合预测成功率
"""

import pandas as pd
import numpy as np
from lucky_number_predictor import LuckyNumberPredictor
from enhanced_predictor_v2 import EnhancedPredictor

def test_comprehensive_prediction():
    """测试综合预测在最新数据上的表现"""
    
    print("=" * 80)
    print("综合预测成功率测试 - 基于最新数据")
    print("=" * 80)
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"\n[数据集信息]")
    print(f"   总期数: {len(df)}")
    print(f"   日期范围: {df.iloc[0]['date']} - {df.iloc[-1]['date']}")
    
    # 最近的数据
    recent_10 = df.tail(10)['number'].tolist()
    print(f"   最近10期: {recent_10}")
    
    # 分析极端值
    extreme_count = sum(1 for n in recent_10 if n <= 10 or n >= 40)
    print(f"   极端值数量: {extreme_count}/10 ({extreme_count*10}%)")
    
    # 训练模型（使用前面的数据）
    print(f"\n[训练模型]")
    model_types = ['gradient_boosting', 'lightgbm', 'xgboost']
    predictors = []
    
    for i, model_type in enumerate(model_types, 1):
        print(f"   [{i}/3] {model_type}...", end='', flush=True)
        predictor = LuckyNumberPredictor()
        
        # 使用前面大部分数据训练
        train_df = df.iloc[:-10]  # 排除最后10期用于测试
        train_file = f'temp_train_{model_type}.csv'
        train_df.to_csv(train_file, index=False)
        
        predictor.load_data(train_file, 
                          number_column='number',
                          date_column='date', 
                          animal_column='animal',
                          element_column='element')
        
        # 检查并处理NaN
        if np.any(np.isnan(predictor.X)):
            print(f" (发现NaN, 使用0填充)...", end='', flush=True)
            predictor.X = np.nan_to_num(predictor.X, nan=0.0)
        if np.any(np.isnan(predictor.y)):
            predictor.y = np.nan_to_num(predictor.y, nan=0.0)
        
        predictor.train_model(model_type=model_type)
        predictors.append(predictor)
        print(" ✓")
    
    # 创建增强预测器
    enhanced = EnhancedPredictor(predictors)
    
    # 在最后10期上测试
    print(f"\n" + "=" * 80)
    print("测试最近10期")
    print("=" * 80)
    
    results = {
        'top5': 0, 'top10': 0, 'top15': 0, 'top20': 0,
        'details': []
    }
    
    total_periods = len(df)
    
    for idx in range(total_periods - 10, total_periods):
        period_num = idx + 1
        actual_number = df.iloc[idx]['number']
        actual_date = df.iloc[idx]['date']
        
        # 使用之前的数据预测
        test_df = df.iloc[:idx]
        test_file = f'temp_test_{idx}.csv'
        test_df.to_csv(test_file, index=False)
        
        print(f"\n第{period_num}期 ({actual_date}): 实际 = {actual_number}")
        
        try:
            # 重新加载数据到每个预测器
            for predictor in predictors:
                predictor.load_data(test_file,
                                  number_column='number',
                                  date_column='date',
                                  animal_column='animal',
                                  element_column='element')
                # 处理NaN
                if np.any(np.isnan(predictor.X)):
                    predictor.X = np.nan_to_num(predictor.X, nan=0.0)
                if np.any(np.isnan(predictor.y)):
                    predictor.y = np.nan_to_num(predictor.y, nan=0.0)
            
            # 预测 Top 20
            predictions = enhanced.comprehensive_predict_v2(top_k=20)
            
            predicted_numbers = [p['number'] for p in predictions]
            
            # 检查命中
            if actual_number in predicted_numbers:
                rank = predicted_numbers.index(actual_number) + 1
                
                # 确定级别
                if rank <= 5:
                    level = "[*] Top 5"
                    results['top5'] += 1
                    results['top10'] += 1
                    results['top15'] += 1
                    results['top20'] += 1
                elif rank <= 10:
                    level = "[v] Top 10"
                    results['top10'] += 1
                    results['top15'] += 1
                    results['top20'] += 1
                elif rank <= 15:
                    level = "[o] Top 15"
                    results['top15'] += 1
                    results['top20'] += 1
                else:
                    level = "[+] Top 20"
                    results['top20'] += 1
                
                print(f"   [HIT] 命中! 排名: {rank} {level}")
                print(f"   预测前10: {predicted_numbers[:10]}")
            else:
                print(f"   [MISS] 未命中")
                print(f"   预测前10: {predicted_numbers[:10]}")
            
            results['details'].append({
                'period': period_num,
                'date': actual_date,
                'actual': actual_number,
                'hit': actual_number in predicted_numbers,
                'rank': predicted_numbers.index(actual_number) + 1 if actual_number in predicted_numbers else -1
            })
            
        except Exception as e:
            print(f"   ⚠️ 预测失败: {str(e)}")
    
    # 统计结果
    print("\n" + "=" * 80)
    print("统计结果")
    print("=" * 80)
    
    total = len(results['details'])
    
    if total == 0:
        print("\n⚠️ 没有成功的预测记录")
        return results
    
    print(f"\n命中统计 (最近{total}期):")
    print(f"   Top 5:  {results['top5']}/{total} = {results['top5']/total*100:.1f}%")
    print(f"   Top 10: {results['top10']}/{total} = {results['top10']/total*100:.1f}%")
    print(f"   Top 15: {results['top15']}/{total} = {results['top15']/total*100:.1f}%")
    print(f"   Top 20: {results['top20']}/{total} = {results['top20']/total*100:.1f}%")
    
    # 对比理论随机
    print(f"\n对比理论随机:")
    random_rates = {
        'top5': 5/49*100,
        'top10': 10/49*100,
        'top15': 15/49*100,
        'top20': 20/49*100
    }
    
    for key in ['top5', 'top10', 'top15', 'top20']:
        actual = results[key]/total*100
        random = random_rates[key]
        improvement = actual / random if random > 0 else 0
        status = "[OK]" if improvement > 1.2 else "[WARN]"
        print(f"   {key.upper()}: {actual:.1f}% vs {random:.1f}% (提升{improvement:.2f}x) {status}")
    
    # 命中详情
    print(f"\n命中详情:")
    for detail in results['details']:
        if detail['hit']:
            rank = detail['rank']
            if rank <= 5:
                marker = "[*]"
            elif rank <= 10:
                marker = "[v]"
            elif rank <= 15:
                marker = "[o]"
            else:
                marker = "[+]"
            print(f"   第{detail['period']}期 ({detail['date']}): {detail['actual']} - 排名{rank} {marker}")
    
    # 清理临时文件
    print(f"\n[清理临时文件]")
    import os
    for f in os.listdir('.'):
        if f.startswith('temp_'):
            os.remove(f)
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)
    
    # 建议
    top15_rate = results['top15']/total*100
    top20_rate = results['top20']/total*100
    
    print(f"\n[建议]")
    if top20_rate >= 60:
        print(f"   [OK] Top 20 成功率 {top20_rate:.1f}% - 表现优秀！")
    elif top15_rate >= 50:
        print(f"   [OK] Top 15 成功率 {top15_rate:.1f}% - 建议使用 Top 15")
    elif top15_rate >= 40:
        print(f"   [WARN] Top 15 成功率 {top15_rate:.1f}% - 表现一般")
    else:
        print(f"   [WARN] 成功率较低，建议增加训练数据或调整策略")
    
    return results


if __name__ == '__main__':
    test_comprehensive_prediction()
