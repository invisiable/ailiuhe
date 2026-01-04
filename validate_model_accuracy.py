"""
验证模型在历史数据上的预测成功率
使用滚动窗口方法：用前N个数据训练，预测第N+1个，然后对比实际值
"""
import pandas as pd
import numpy as np
from lucky_number_predictor import LuckyNumberPredictor
from collections import Counter

def validate_model_accuracy(model_type='random_forest', train_size=100, test_samples=20):
    """
    验证模型预测准确率
    
    Args:
        model_type: 模型类型
        train_size: 训练数据大小
        test_samples: 测试样本数量
    """
    print("="*80)
    print(f"模型预测准确率验证 - {model_type}")
    print("="*80)
    
    # 加载完整数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total_data = len(df)
    
    print(f"\n数据集信息:")
    print(f"  总数据量: {total_data}")
    print(f"  训练集大小: {train_size}")
    print(f"  测试样本数: {test_samples}")
    print(f"  测试范围: 第{train_size+1}期 到 第{train_size+test_samples}期")
    
    # 验证指标
    number_exact_matches = 0  # 数字完全匹配
    number_within_3 = 0       # 数字误差≤3
    number_within_5 = 0       # 数字误差≤5
    number_within_10 = 0      # 数字误差≤10
    animal_matches = 0        # 生肖匹配
    element_matches = 0       # 五行匹配
    full_matches = 0          # 数字+生肖+五行全匹配
    
    # Top 3命中率
    number_in_top3 = 0
    number_in_top5 = 0
    number_in_top10 = 0
    
    errors = []
    predictions_log = []
    
    print("\n开始滚动验证...\n")
    print(f"{'期数':<6} {'实际':<20} {'预测':<20} {'误差':<8} {'Top3命中':<10}")
    print("-"*80)
    
    for i in range(test_samples):
        test_index = train_size + i
        
        if test_index >= total_data:
            break
        
        # 使用前test_index个数据训练
        train_df = df.iloc[:test_index].copy()
        actual_row = df.iloc[test_index]
        
        # 保存临时训练数据
        temp_file = 'data/temp_train.csv'
        train_df.to_csv(temp_file, index=False, encoding='utf-8-sig')
        
        try:
            # 训练模型
            predictor = LuckyNumberPredictor()
            predictor.load_data(
                temp_file,
                number_column='number',
                date_column='date',
                animal_column='animal',
                element_column='element'
            )
            predictor.train_model(model_type, test_size=0.2)
            
            # 预测下一个
            pred = predictor.predict_next(n_predictions=1)[0]
            
            # 获取Top预测
            top10 = predictor.predict_top_probabilities(top_k=10)
            top10_numbers = [p['number'] for p in top10]
            
            # 实际值
            actual_number = actual_row['number']
            actual_animal = actual_row['animal']
            actual_element = actual_row['element']
            
            # 预测值
            pred_number = pred['number']
            pred_animal = pred['animal']
            pred_element = pred['element']
            
            # 计算误差
            error = abs(actual_number - pred_number)
            errors.append(error)
            
            # 判断匹配
            exact_match = (actual_number == pred_number)
            animal_match = (actual_animal == pred_animal)
            element_match = (actual_element == pred_element)
            
            if exact_match:
                number_exact_matches += 1
            if error <= 3:
                number_within_3 += 1
            if error <= 5:
                number_within_5 += 1
            if error <= 10:
                number_within_10 += 1
            if animal_match:
                animal_matches += 1
            if element_match:
                element_matches += 1
            if exact_match and animal_match and element_match:
                full_matches += 1
            
            # Top N命中
            in_top3 = actual_number in top10_numbers[:3]
            in_top5 = actual_number in top10_numbers[:5]
            in_top10 = actual_number in top10_numbers
            
            if in_top3:
                number_in_top3 += 1
            if in_top5:
                number_in_top5 += 1
            if in_top10:
                number_in_top10 += 1
            
            # 显示结果
            actual_str = f"{actual_number:2d} {actual_animal} {actual_element}"
            pred_str = f"{pred_number:2d} {pred_animal} {pred_element}"
            match_marker = "✓" if exact_match else " "
            top3_marker = "★" if in_top3 else ("☆" if in_top10 else " ")
            
            print(f"{test_index:<6d} {actual_str:<20} {pred_str:<20} {error:<8d} {top3_marker}{match_marker}")
            
            predictions_log.append({
                'period': test_index,
                'actual_number': actual_number,
                'pred_number': pred_number,
                'error': error,
                'actual_animal': actual_animal,
                'pred_animal': pred_animal,
                'actual_element': actual_element,
                'pred_element': pred_element,
                'in_top3': in_top3,
                'in_top10': in_top10
            })
            
        except Exception as e:
            print(f"{test_index:<6d} 预测失败: {e}")
            continue
    
    # 计算总体指标
    total_tests = len(predictions_log)
    
    print("\n" + "="*80)
    print("验证结果统计")
    print("="*80)
    
    print(f"\n【数字预测准确率】")
    print(f"  完全匹配: {number_exact_matches}/{total_tests} = {number_exact_matches/total_tests*100:.2f}%")
    print(f"  误差≤3:  {number_within_3}/{total_tests} = {number_within_3/total_tests*100:.2f}%")
    print(f"  误差≤5:  {number_within_5}/{total_tests} = {number_within_5/total_tests*100:.2f}%")
    print(f"  误差≤10: {number_within_10}/{total_tests} = {number_within_10/total_tests*100:.2f}%")
    
    print(f"\n【Top N 命中率】")
    print(f"  Top 3:  {number_in_top3}/{total_tests} = {number_in_top3/total_tests*100:.2f}%")
    print(f"  Top 5:  {number_in_top5}/{total_tests} = {number_in_top5/total_tests*100:.2f}%")
    print(f"  Top 10: {number_in_top10}/{total_tests} = {number_in_top10/total_tests*100:.2f}%")
    
    print(f"\n【生肖预测准确率】")
    print(f"  匹配数: {animal_matches}/{total_tests} = {animal_matches/total_tests*100:.2f}%")
    
    print(f"\n【五行预测准确率】")
    print(f"  匹配数: {element_matches}/{total_tests} = {element_matches/total_tests*100:.2f}%")
    
    print(f"\n【完全匹配率】(数字+生肖+五行)")
    print(f"  匹配数: {full_matches}/{total_tests} = {full_matches/total_tests*100:.2f}%")
    
    print(f"\n【误差统计】")
    if errors:
        print(f"  平均误差: {np.mean(errors):.2f}")
        print(f"  中位误差: {np.median(errors):.2f}")
        print(f"  最大误差: {max(errors)}")
        print(f"  最小误差: {min(errors)}")
        print(f"  标准差: {np.std(errors):.2f}")
    
    # 误差分布
    print(f"\n【误差分布】")
    error_ranges = [(0, 0), (1, 3), (4, 5), (6, 10), (11, 20), (21, 100)]
    for low, high in error_ranges:
        count = sum(1 for e in errors if low <= e <= high)
        if count > 0:
            range_str = f"{low}" if low == high else f"{low}-{high}"
            print(f"  误差 {range_str:>6}: {count:>3}次 ({count/total_tests*100:>5.1f}%)  {'█' * int(count/total_tests*50)}")
    
    return {
        'total_tests': total_tests,
        'number_exact': number_exact_matches,
        'number_within_3': number_within_3,
        'number_within_5': number_within_5,
        'number_within_10': number_within_10,
        'animal_matches': animal_matches,
        'element_matches': element_matches,
        'full_matches': full_matches,
        'top3_hits': number_in_top3,
        'top5_hits': number_in_top5,
        'top10_hits': number_in_top10,
        'mean_error': np.mean(errors) if errors else 0,
        'predictions_log': predictions_log
    }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("幸运数字预测模型 - 历史数据验证")
    print("="*80)
    
    # 测试所有模型
    models = [
        ('random_forest', '随机森林'),
        ('gradient_boosting', '梯度提升'),
        ('neural_network', '神经网络')
    ]
    
    results = {}
    
    for model_type, model_name in models:
        print(f"\n{'='*80}")
        print(f"测试模型: {model_name} ({model_type})")
        print(f"{'='*80}\n")
        
        try:
            result = validate_model_accuracy(
                model_type=model_type,
                train_size=100,
                test_samples=20
            )
            results[model_type] = result
        except Exception as e:
            print(f"模型 {model_name} 验证失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 对比所有模型
    print("\n" + "="*80)
    print("所有模型对比")
    print("="*80)
    
    print(f"\n{'模型':<20} {'完全匹配':<12} {'误差≤5':<12} {'Top3命中':<12} {'Top10命中':<12} {'平均误差'}")
    print("-"*80)
    
    for model_type, model_name in models:
        if model_type in results:
            r = results[model_type]
            total = r['total_tests']
            print(f"{model_name:<20} "
                  f"{r['number_exact']:>3}/{total:<3} ({r['number_exact']/total*100:>4.1f}%) "
                  f"{r['number_within_5']:>3}/{total:<3} ({r['number_within_5']/total*100:>4.1f}%) "
                  f"{r['top3_hits']:>3}/{total:<3} ({r['top3_hits']/total*100:>4.1f}%) "
                  f"{r['top10_hits']:>3}/{total:<3} ({r['top10_hits']/total*100:>4.1f}%) "
                  f"{r['mean_error']:>6.2f}")
    
    # 推荐最佳模型
    print("\n" + "="*80)
    print("推荐结论")
    print("="*80)
    
    if results:
        # 综合评分：Top3命中率40% + 误差≤5占比30% + 完全匹配20% + Top10命中率10%
        scores = {}
        for model_type in results:
            r = results[model_type]
            total = r['total_tests']
            score = (
                (r['top3_hits'] / total) * 0.40 +
                (r['number_within_5'] / total) * 0.30 +
                (r['number_exact'] / total) * 0.20 +
                (r['top10_hits'] / total) * 0.10
            )
            scores[model_type] = score
        
        best_model = max(scores.items(), key=lambda x: x[1])
        model_names_dict = {m[0]: m[1] for m in models}
        
        print(f"\n✓ 推荐模型: {model_names_dict[best_model[0]]} ({best_model[0]})")
        print(f"  综合评分: {best_model[1]*100:.2f}分")
        print(f"\n  推荐理由:")
        r = results[best_model[0]]
        total = r['total_tests']
        print(f"    - Top 3 命中率: {r['top3_hits']/total*100:.1f}%")
        print(f"    - 误差≤5准确率: {r['number_within_5']/total*100:.1f}%")
        print(f"    - 平均预测误差: {r['mean_error']:.2f}")
        print(f"    - Top 10 命中率: {r['top10_hits']/total*100:.1f}%")
    
    print("\n" + "="*80)
