"""
验证所有模型（包括新增的高级模型）的预测成功率
"""
import pandas as pd
import numpy as np
from lucky_number_predictor import LuckyNumberPredictor
import warnings
warnings.filterwarnings('ignore')

def quick_validate(model_type, model_name, train_size=100, test_samples=20):
    """快速验证单个模型"""
    try:
        df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
        
        number_exact = 0
        number_within_5 = 0
        number_within_10 = 0
        top3_hits = 0
        top10_hits = 0
        errors = []
        
        for i in range(test_samples):
            test_index = train_size + i
            if test_index >= len(df):
                break
            
            train_df = df.iloc[:test_index].copy()
            actual_row = df.iloc[test_index]
            
            temp_file = 'data/temp_train.csv'
            train_df.to_csv(temp_file, index=False, encoding='utf-8-sig')
            
            try:
                predictor = LuckyNumberPredictor()
                predictor.load_data(temp_file, number_column='number', date_column='date',
                                   animal_column='animal', element_column='element')
                predictor.train_model(model_type, test_size=0.2)
                
                pred = predictor.predict_next(n_predictions=1)[0]
                top10 = predictor.predict_top_probabilities(top_k=10)
                top10_numbers = [p['number'] for p in top10]
                
                actual_number = actual_row['number']
                pred_number = pred['number']
                error = abs(actual_number - pred_number)
                errors.append(error)
                
                if actual_number == pred_number:
                    number_exact += 1
                if error <= 5:
                    number_within_5 += 1
                if error <= 10:
                    number_within_10 += 1
                if actual_number in top10_numbers[:3]:
                    top3_hits += 1
                if actual_number in top10_numbers:
                    top10_hits += 1
                    
            except Exception as e:
                continue
        
        total = len(errors)
        if total == 0:
            return None
        
        return {
            'model_type': model_type,
            'model_name': model_name,
            'total': total,
            'exact': number_exact,
            'within_5': number_within_5,
            'within_10': number_within_10,
            'top3': top3_hits,
            'top10': top10_hits,
            'mean_error': np.mean(errors),
            'score': (top3_hits/total)*0.4 + (number_within_5/total)*0.3 + 
                    (number_exact/total)*0.2 + (top10_hits/total)*0.1
        }
    except Exception as e:
        print(f"  ✗ {model_name} 验证失败: {e}")
        return None


if __name__ == "__main__":
    print("="*80)
    print("全模型对比验证 - 寻找最佳预测模型")
    print("="*80)
    print("\n正在测试所有可用模型...\n")
    
    models = [
        ('gradient_boosting', '梯度提升'),
        ('random_forest', '随机森林'),
        ('xgboost', 'XGBoost'),
        ('lightgbm', 'LightGBM'),
        ('catboost', 'CatBoost'),
        ('ensemble', '集成模型'),
        ('neural_network', '神经网络'),
        ('svr', '支持向量机')
    ]
    
    results = []
    
    for model_type, model_name in models:
        print(f"测试 {model_name:<15} ", end='', flush=True)
        result = quick_validate(model_type, model_name, train_size=100, test_samples=20)
        if result:
            results.append(result)
            print(f"✓ 完成 (评分: {result['score']*100:.1f}分)")
        else:
            print(f"✗ 失败")
    
    if not results:
        print("\n没有模型通过验证！")
        exit(1)
    
    # 按评分排序
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n" + "="*80)
    print("验证结果排名")
    print("="*80)
    
    print(f"\n{'排名':<4} {'模型':<15} {'完全匹配':<10} {'误差≤5':<10} {'Top3命中':<10} {'Top10命中':<10} {'平均误差':<10} {'综合评分'}")
    print("-"*100)
    
    for i, r in enumerate(results, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal} {i:<2} {r['model_name']:<15} "
              f"{r['exact']:>2}/{r['total']:<2} ({r['exact']/r['total']*100:>4.1f}%) "
              f"{r['within_5']:>2}/{r['total']:<2} ({r['within_5']/r['total']*100:>4.1f}%) "
              f"{r['top3']:>2}/{r['total']:<2} ({r['top3']/r['total']*100:>4.1f}%) "
              f"{r['top10']:>2}/{r['total']:<2} ({r['top10']/r['total']*100:>4.1f}%) "
              f"{r['mean_error']:>6.2f}        "
              f"{r['score']*100:>5.1f}分")
    
    # 显示前三名的详细信息
    print("\n" + "="*80)
    print("🏆 Top 3 模型推荐")
    print("="*80)
    
    for i, r in enumerate(results[:3], 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        print(f"\n{medal} 第{i}名: {r['model_name']}")
        print(f"  综合评分: {r['score']*100:.1f}分")
        print(f"  完全匹配率: {r['exact']/r['total']*100:.1f}% ({r['exact']}/{r['total']})")
        print(f"  误差≤5准确率: {r['within_5']/r['total']*100:.1f}% ({r['within_5']}/{r['total']})")
        print(f"  误差≤10准确率: {r['within_10']/r['total']*100:.1f}% ({r['within_10']}/{r['total']})")
        print(f"  Top 3命中率: {r['top3']/r['total']*100:.1f}% ({r['top3']}/{r['total']})")
        print(f"  Top 10命中率: {r['top10']/r['total']*100:.1f}% ({r['top10']}/{r['total']})")
        print(f"  平均预测误差: {r['mean_error']:.2f}")
    
    # 推荐使用
    best = results[0]
    print("\n" + "="*80)
    print("💡 使用建议")
    print("="*80)
    print(f"\n推荐模型: {best['model_name']}")
    print(f"  - 在GUI中选择 '{best['model_name']}'")
    print(f"  - 预期准确率: Top 3命中 {best['top3']/best['total']*100:.0f}%, 误差≤5 准确率 {best['within_5']/best['total']*100:.0f}%")
    print(f"  - 综合表现最佳，评分 {best['score']*100:.1f}分")
    
    if len(results) > 1:
        second = results[1]
        print(f"\n备选模型: {second['model_name']}")
        print(f"  - 评分 {second['score']*100:.1f}分，仅次于最佳模型")
        print(f"  - 可作为对比参考")
    
    print("\n" + "="*80)
