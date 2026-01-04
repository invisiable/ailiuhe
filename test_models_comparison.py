"""
对比不同模型的预测效果
针对目标：数字3, 生肖兔, 五行金
"""
import pandas as pd
import numpy as np
from lucky_number_predictor import LuckyNumberPredictor

# 加载数据
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
print("="*70)
print("当前数据分析")
print("="*70)
print(f"总数据量: {len(df)} 条")
print(f"\n最后10条数据:")
print(df.tail(10))

last_row = df.iloc[-1]
print(f"\n最后一条: {last_row['date']} | 数字:{last_row['number']} | 生肖:{last_row['animal']} | 五行:{last_row['element']}")
print(f"\n目标预测: 数字:3 | 生肖:兔 | 五行:金")

# 测试不同模型
models = ['random_forest', 'gradient_boosting', 'neural_network']
results = {}

print("\n" + "="*70)
print("测试不同模型的预测结果")
print("="*70)

for model_type in models:
    print(f"\n【{model_type}】")
    print("-"*70)
    
    try:
        predictor = LuckyNumberPredictor()
        predictor.load_data(
            'data/lucky_numbers.csv',
            number_column='number',
            date_column='date',
            animal_column='animal',
            element_column='element'
        )
        
        # 训练模型
        train_results = predictor.train_model(model_type, test_size=0.2)
        print(f"训练完成 - MAE: {train_results['test_mae']:.4f}, R²: {train_results['test_r2']:.4f}")
        
        # 预测下一个数字
        next_pred = predictor.predict_next(n_predictions=1)[0]
        print(f"连续预测: 数字:{next_pred['number']}, 生肖:{next_pred['animal']}, 五行:{next_pred['element']}")
        
        # Top 3预测
        top3 = predictor.predict_top_probabilities(top_k=5)
        print(f"\nTop 5 预测:")
        for i, pred in enumerate(top3, 1):
            match_score = 0
            if pred['number'] == 3:
                match_score += 1
            if pred['element'] == '金':
                match_score += 1
            marker = " ✓✓✓" if match_score >= 2 else " ✓" if match_score == 1 else ""
            print(f"  {i}. 数字:{pred['number']:>2} 生肖:{pred['animal']} 五行:{pred['element']} 概率:{pred['probability']*100:>5.2f}%{marker}")
        
        # 检查是否预测到目标
        target_found = False
        target_rank = 0
        for i, pred in enumerate(top3, 1):
            if pred['number'] == 3 and pred['element'] == '金':
                target_found = True
                target_rank = i
                break
        
        results[model_type] = {
            'mae': train_results['test_mae'],
            'r2': train_results['test_r2'],
            'top1_number': top3[0]['number'],
            'top1_element': top3[0]['element'],
            'target_found': target_found,
            'target_rank': target_rank,
            'top3': top3
        }
        
    except Exception as e:
        print(f"错误: {e}")
        results[model_type] = None

# 总结对比
print("\n" + "="*70)
print("模型对比总结")
print("="*70)

print(f"\n{'模型':<20} {'MAE':<10} {'R²':<10} {'Top1数字':<10} {'Top1五行':<10} {'找到目标'}")
print("-"*70)

for model_type, result in results.items():
    if result:
        found = f"✓ 第{result['target_rank']}名" if result['target_found'] else "✗"
        print(f"{model_type:<20} {result['mae']:<10.4f} {result['r2']:<10.4f} {result['top1_number']:<10} {result['top1_element']:<10} {found}")

# 推荐最佳模型
print("\n" + "="*70)
print("推荐建议")
print("="*70)

valid_results = {k: v for k, v in results.items() if v is not None}
if valid_results:
    # 优先找到目标的模型
    found_models = [(k, v) for k, v in valid_results.items() if v['target_found']]
    
    if found_models:
        # 按排名和MAE排序
        best_model = min(found_models, key=lambda x: (x[1]['target_rank'], x[1]['mae']))
        print(f"\n✓ 推荐模型: {best_model[0]}")
        print(f"  原因: 在Top {best_model[1]['target_rank']}中找到目标 (3, 金), MAE={best_model[1]['mae']:.4f}")
    else:
        # 选择MAE最小的
        best_model = min(valid_results.items(), key=lambda x: x[1]['mae'])
        print(f"\n✓ 推荐模型: {best_model[0]}")
        print(f"  原因: 最低预测误差 MAE={best_model[1]['mae']:.4f}")
        print(f"  注意: 没有模型在Top 5中预测到目标值，可能需要:")
        print(f"    1. 增加训练数据量")
        print(f"    2. 调整特征工程")
        print(f"    3. 目标值(3,兔,金)可能不符合当前数据规律")

print("\n" + "="*70)
