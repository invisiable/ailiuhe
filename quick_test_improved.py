"""
快速测试改进模型 - 先验证10期
"""

from improved_odd_even_predictor import ImprovedOddEvenPredictor
import pandas as pd
import os


def quick_test_improved(model_type='ensemble_voting', num_periods=10):
    """快速测试改进模型"""
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total_records = len(df)
    start_idx = total_records - num_periods
    
    predictions = []
    correct_count = 0
    
    print(f"\n{'='*80}")
    print(f"快速测试改进模型: {model_type}")
    print(f"验证期数: 最近 {num_periods} 期")
    print(f"{'='*80}\n")
    
    for i in range(num_periods):
        current_idx = start_idx + i
        train_df = df.iloc[:current_idx].copy()
        
        temp_csv = 'data/temp_train_quick.csv'
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
            
            status = '✅' if is_correct else '❌'
            print(f"{i+1:2d}. 第{current_idx+1}期 ({df.iloc[current_idx]['date']}) "
                  f"预测:{result['prediction']:4s} 实际:{actual_parity:4s} "
                  f"数字:{actual_number:2d} {status} "
                  f"置信度:{result['probability']*100:5.1f}%")
            
        except Exception as e:
            print(f"{i+1:2d}. 第{current_idx+1}期 预测失败: {e}")
        
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
    
    accuracy = correct_count / num_periods * 100
    print(f"\n{'='*80}")
    print(f"准确率: {correct_count}/{num_periods} = {accuracy:.2f}%")
    print(f"{'='*80}\n")
    
    return accuracy


if __name__ == "__main__":
    # 测试多个模型
    models = ['ensemble_voting', 'ensemble_stacking', 'gradient_boosting', 
              'neural_network', 'xgboost']
    
    results = {}
    
    for model in models:
        try:
            print(f"\n\n{'#'*80}")
            print(f"测试模型: {model}")
            print(f"{'#'*80}")
            acc = quick_test_improved(model_type=model, num_periods=10)
            results[model] = acc
        except Exception as e:
            print(f"模型 {model} 测试失败: {e}")
            results[model] = 0
    
    # 总结
    print("\n" + "="*80)
    print("模型性能对比（最近10期）")
    print("="*80)
    
    for model, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model:25s}: {acc:6.2f}%")
