"""
扩大Top K搜索范围，寻找数字3
"""
from lucky_number_predictor import LuckyNumberPredictor

predictor = LuckyNumberPredictor()
predictor.load_data(
    'data/lucky_numbers.csv',
    number_column='number',
    date_column='date',
    animal_column='animal',
    element_column='element'
)

print("测试不同模型在Top 30范围内能否找到数字3\n")
print("="*70)

for model_type in ['neural_network', 'random_forest', 'gradient_boosting']:
    print(f"\n【{model_type}】")
    
    predictor.train_model(model_type, test_size=0.2)
    top30 = predictor.predict_top_probabilities(top_k=30)
    
    found = False
    for i, pred in enumerate(top30, 1):
        if pred['number'] == 3:
            print(f"  ✓ 找到数字3! 排名第{i}, 概率{pred['probability']*100:.2f}%, 五行{pred['element']}")
            found = True
            break
    
    if not found:
        print(f"  ✗ Top 30中仍未找到数字3")
        # 显示最小的数字
        small_nums = sorted([p for p in top30 if p['number'] <= 10], key=lambda x: x['number'])
        if small_nums:
            print(f"  最小的预测数字:")
            for p in small_nums[:3]:
                print(f"    {p['number']}: 概率{p['probability']*100:.2f}%")

print("\n" + "="*70)
