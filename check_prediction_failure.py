"""
检查预测失效原因
"""

import pandas as pd
from zodiac_simple_smart import ZodiacSimpleSmart

# 加载数据
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
df['date'] = pd.to_datetime(df['date'])

# 检查最近300期的预测情况
all_animals = df['animal'].tolist()
all_dates = df['date'].dt.strftime('%Y/%m/%d').tolist()

test_start = len(all_animals) - 300
predictor = ZodiacSimpleSmart()

print("检查最近20期预测情况:")
print("="*70)

hit_count = 0
for i in range(test_start, min(test_start + 20, len(all_animals))):
    train_animals = all_animals[:i]
    actual_animal = all_animals[i]
    date = all_dates[i]
    
    # 预测
    prediction_result = predictor.predict_from_history(train_animals, top_n=5)
    predictions = prediction_result['top5']
    hit = actual_animal in predictions
    
    if hit:
        hit_count += 1
    
    print(f"{date}: 实际={actual_animal:2} | 预测={', '.join(predictions)} | {'✅' if hit else '❌'}")

print(f"\n命中率: {hit_count}/20 = {hit_count/20:.1%}")

# 再检查最后20期
print("\n检查最后20期预测情况:")
print("="*70)

hit_count = 0
for i in range(len(all_animals) - 20, len(all_animals)):
    train_animals = all_animals[:i]
    actual_animal = all_animals[i]
    date = all_dates[i]
    
    # 预测
    prediction_result = predictor.predict_from_history(train_animals, top_n=5)
    predictions = prediction_result['top5']
    hit = actual_animal in predictions
    
    if hit:
        hit_count += 1
    
    print(f"{date}: 实际={actual_animal:2} | 预测={', '.join(predictions)} | {'✅' if hit else '❌'}")

print(f"\n命中率: {hit_count}/20 = {hit_count/20:.1%}")

# 分析生肖分布
print("\n最近100期生肖分布:")
recent_animals = df.tail(100)['animal'].value_counts()
print(recent_animals)

print("\n最近300期生肖分布:")
recent_300 = df.tail(300)['animal'].value_counts()
print(recent_300)
