"""
分析数字序列规律，找出最适合的模型
不依赖生肖周期，只看数字本身的变化模式
"""
import pandas as pd
import numpy as np
from lucky_number_predictor import LuckyNumberPredictor
import matplotlib.pyplot as plt
from collections import Counter

# 加载数据
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values

print("="*70)
print("数字序列规律分析")
print("="*70)

# 1. 基本统计
print(f"\n【基本统计】")
print(f"数据量: {len(numbers)} 个")
print(f"范围: {numbers.min()} - {numbers.max()}")
print(f"平均值: {numbers.mean():.2f}")
print(f"标准差: {numbers.std():.2f}")
print(f"中位数: {np.median(numbers):.0f}")

# 2. 数字变化规律
changes = np.diff(numbers)
print(f"\n【数字变化规律】")
print(f"平均变化: {changes.mean():.2f}")
print(f"变化标准差: {changes.std():.2f}")
print(f"最大增幅: +{changes.max()}")
print(f"最大降幅: {changes.min()}")

# 统计变化频率
change_counter = Counter(changes)
print(f"\n最常见的变化:")
for change, count in sorted(change_counter.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  变化{change:+3d}: {count}次 ({count/len(changes)*100:.1f}%)")

# 3. 最近20个数字的趋势
recent_20 = numbers[-20:]
print(f"\n【最近20个数字】")
print(f"{list(recent_20)}")
print(f"平均值: {recent_20.mean():.2f}")
print(f"趋势: {recent_20[-1] - recent_20[0]:+d} (从{recent_20[0]}到{recent_20[-1]})")

# 4. 数字出现频率
print(f"\n【数字出现频率 Top 10】")
number_counter = Counter(numbers)
for num, count in number_counter.most_common(10):
    print(f"  数字{num:2d}: {count}次 ({count/len(numbers)*100:.1f}%)")

# 5. 区间分布
bins = [0, 10, 20, 30, 40, 50]
hist, _ = np.histogram(numbers, bins=bins)
print(f"\n【区间分布】")
for i in range(len(bins)-1):
    print(f"  {bins[i]:2d}-{bins[i+1]:2d}: {hist[i]:3d}个 ({hist[i]/len(numbers)*100:.1f}%)")

# 6. 最后10个数字详细分析
print(f"\n【最后10个数字详细】")
last_10 = numbers[-10:]
for i, num in enumerate(last_10):
    change = num - last_10[i-1] if i > 0 else 0
    print(f"  第{len(numbers)-10+i+1}期: {num:2d}  变化:{change:+3d}")

print(f"\n从最后数字 {numbers[-1]} 预测下一个:")
print(f"  - 如果延续平均变化 ({changes.mean():.2f}): 约 {numbers[-1] + changes.mean():.0f}")
print(f"  - 如果延续最近5期平均变化: 约 {numbers[-1] + changes[-5:].mean():.0f}")
print(f"  - 目标值: 3")

# 7. 测试不同模型 - 关注数字预测
print("\n" + "="*70)
print("测试不同模型预测数字准确度")
print("="*70)

models = ['random_forest', 'gradient_boosting', 'neural_network']
results = {}

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
        
        mae = train_results['test_mae']
        rmse = train_results['test_rmse']
        r2 = train_results['test_r2']
        
        print(f"MAE: {mae:.4f}  RMSE: {rmse:.4f}  R²: {r2:.4f}")
        
        # 分析预测值与实际值的偏差
        y_test = train_results['y_test']
        y_pred = train_results['y_pred']
        
        # 计算各区间的预测准确度
        errors = np.abs(y_test - y_pred)
        within_5 = np.sum(errors <= 5) / len(errors) * 100
        within_10 = np.sum(errors <= 10) / len(errors) * 100
        
        print(f"预测误差≤5的比例: {within_5:.1f}%")
        print(f"预测误差≤10的比例: {within_10:.1f}%")
        
        # 预测下一个数字
        next_pred = predictor.predict_next(n_predictions=1)[0]
        print(f"下一期预测: 数字 {next_pred['number']}")
        
        # Top 10预测（扩大范围寻找3）
        top10 = predictor.predict_top_probabilities(top_k=10)
        print(f"\nTop 10 预测数字:")
        
        found_3 = False
        for i, pred in enumerate(top10, 1):
            marker = " ★★★" if pred['number'] == 3 else ""
            if pred['number'] == 3:
                found_3 = True
            print(f"  {i:2d}. 数字:{pred['number']:2d}  概率:{pred['probability']*100:5.2f}%{marker}")
        
        if found_3:
            print(f"  ✓ 在Top 10中找到目标数字3！")
        else:
            print(f"  ✗ Top 10中未找到数字3")
        
        # 查看模型对小数字(1-10)的预测倾向
        small_numbers = [p for p in top10 if p['number'] <= 10]
        if small_numbers:
            print(f"\n小数字(≤10)预测:")
            for p in small_numbers:
                print(f"  数字{p['number']:2d}: {p['probability']*100:.2f}%")
        
        results[model_type] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'within_5': within_5,
            'within_10': within_10,
            'next_number': next_pred['number'],
            'found_3': found_3,
            'top10': top10,
            'small_number_count': len(small_numbers)
        }
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        results[model_type] = None

# 8. 综合对比
print("\n" + "="*70)
print("模型综合对比")
print("="*70)

print(f"\n{'模型':<20} {'MAE':<8} {'RMSE':<8} {'≤5误差':<10} {'≤10误差':<10} {'预测值':<8} {'找到3'}")
print("-"*70)

for model_type, result in results.items():
    if result:
        found = "✓" if result['found_3'] else "✗"
        print(f"{model_type:<20} {result['mae']:<8.2f} {result['rmse']:<8.2f} "
              f"{result['within_5']:<10.1f}% {result['within_10']:<10.1f}% "
              f"{result['next_number']:<8d} {found}")

# 9. 推荐
print("\n" + "="*70)
print("推荐结论")
print("="*70)

valid_results = {k: v for k, v in results.items() if v is not None}

if valid_results:
    # 按MAE排序找最佳模型
    best_by_mae = min(valid_results.items(), key=lambda x: x[1]['mae'])
    
    # 找到数字3的模型
    found_3_models = [(k, v) for k, v in valid_results.items() if v['found_3']]
    
    print(f"\n1. 【最准确的模型】: {best_by_mae[0]}")
    print(f"   MAE={best_by_mae[1]['mae']:.2f}, 预测值={best_by_mae[1]['next_number']}")
    
    if found_3_models:
        print(f"\n2. 【能预测到3的模型】:")
        for model, result in found_3_models:
            rank_3 = next(i for i, p in enumerate(result['top10'], 1) if p['number'] == 3)
            prob_3 = next(p['probability'] for p in result['top10'] if p['number'] == 3)
            print(f"   {model}: 排名第{rank_3}, 概率{prob_3*100:.2f}%")
    else:
        print(f"\n2. ⚠️ 没有模型在Top 10中预测到数字3")
        print(f"   原因分析:")
        print(f"   - 当前数字: {numbers[-1]} (木)")
        print(f"   - 最近趋势: 中高位波动 (平均{recent_20.mean():.0f})")
        print(f"   - 目标值3: 属于极小值，与当前趋势偏离较大")
        print(f"   - 建议: 需要等待数字趋势下降到10以下区间")

print("\n" + "="*70)
