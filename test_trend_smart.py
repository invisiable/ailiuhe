"""
测试v11.0实时趋势检测智能选择器
重点验证对"冷门爆发"和"高频变化"的检测能力
"""

import pandas as pd
from zodiac_trend_smart import ZodiacTrendSmart

def test_trend_smart():
    predictor = ZodiacTrendSmart()
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    
    print("="*110)
    print("v11.0实时趋势检测智能选择器 - 全面测试")
    print("="*110)
    
    # 重点测试最近30期
    test_ranges = [
        ('最近10期（问题期）', -10, None),
        ('最近20期', -20, None),
        ('最近30期', -30, None),
        ('全部100期', -100, None)
    ]
    
    for range_name, start, end in test_ranges:
        print(f"\n{range_name}:")
        print("-"*110)
        
        if end is None:
            test_data = df.iloc[start:]
        else:
            test_data = df.iloc[start:end]
        
        total = len(test_data)
        hit_top5 = 0
        hit_top2 = 0
        hit_top1 = 0
        
        model_usage = {}
        scenario_usage = {}
        
        for idx in range(total):
            animals = [str(a).strip() for a in df['animal'].values[:start + idx if start < 0 else idx]]
            
            if len(animals) < 30:  # v11.0需要至少30期数据
                continue
            
            prediction = predictor.predict_from_history(animals, top_n=5, debug=False)
            actual = str(test_data.iloc[idx]['animal']).strip()
            
            model_used = prediction['selected_model']
            scenario = prediction['scenario']
            
            model_usage[model_used] = model_usage.get(model_used, 0) + 1
            scenario_usage[scenario] = scenario_usage.get(scenario, 0) + 1
            
            if actual in prediction['top5']:
                rank = prediction['top5'].index(actual) + 1
                hit_top5 += 1
                if rank <= 2:
                    hit_top2 += 1
                if rank == 1:
                    hit_top1 += 1
        
        valid_count = sum(1 for idx in range(total) if len(df['animal'].values[:start + idx if start < 0 else idx]) >= 30)
        
        if valid_count > 0:
            top5_rate = (hit_top5 / valid_count * 100)
            top2_rate = (hit_top2 / valid_count * 100)
            top1_rate = (hit_top1 / valid_count * 100)
            
            print(f"TOP5命中: {hit_top5}/{valid_count} = {top5_rate:.1f}%")
            print(f"TOP2命中: {hit_top2}/{valid_count} = {top2_rate:.1f}%")
            print(f"TOP1命中: {hit_top1}/{valid_count} = {top1_rate:.1f}%")
            
            print(f"\n场景分布:")
            for scenario, count in sorted(scenario_usage.items(), key=lambda x: x[1], reverse=True):
                pct = count / valid_count * 100
                print(f"  {scenario:20s}: {count:3d}次 ({pct:5.1f}%)")
            
            print(f"\n模型使用:")
            for model, count in sorted(model_usage.items(), key=lambda x: x[1], reverse=True):
                pct = count / valid_count * 100
                print(f"  {model:20s}: {count:3d}次 ({pct:5.1f}%)")
    
    # 详细分析最近10期
    print(f"\n{'='*110}")
    print("【重点】最近10期详细分析（对比v10.0的20%命中率）:")
    print(f"{'='*110}")
    
    recent_10_data = df.iloc[-10:]
    total = len(df)
    
    print(f"\n{'序号':<4} {'日期':<12} {'实际':<4} {'预测TOP5':<30} {'模型':<20} {'场景':<20} {'结果':<10}")
    print("-"*110)
    
    hit_count = 0
    for idx in range(10):
        animals = [str(a).strip() for a in df['animal'].values[:total - 10 + idx]]
        
        prediction = predictor.predict_from_history(animals, top_n=5, debug=False)
        actual = str(recent_10_data.iloc[idx]['animal']).strip()
        date = recent_10_data.iloc[idx]['date']
        predicted_top5 = prediction['top5']
        model_used = prediction['selected_model']
        scenario = prediction['scenario']
        
        if actual in predicted_top5:
            rank = predicted_top5.index(actual) + 1
            hit_count += 1
            result_str = f"✓ TOP{rank}"
        else:
            result_str = "✗"
        
        predicted_str = ','.join(predicted_top5)
        print(f"{idx+1:<4} {date:<12} {actual:<4} {predicted_str:<30} {model_used:<20} {scenario:<20} {result_str:<10}")
    
    print(f"\nv11.0最近10期: {hit_count}/10 = {hit_count*10}%")
    print(f"v10.0最近10期: 2/10 = 20%")
    
    improvement = (hit_count - 2) * 10
    if improvement > 0:
        print(f"提升: +{improvement}% ✓")
    elif improvement < 0:
        print(f"下降: {improvement}%")
    else:
        print(f"持平")
    
    # 对比v10.0
    print(f"\n{'='*110}")
    print("与v10.0性能对比:")
    print(f"{'='*110}")
    
    print(f"\n测试区间       v11.0    v10.0    差异")
    print("-"*50)
    
    # 这里硬编码v10.0的结果作为对比
    comparisons = [
        ('最近10期', hit_count * 10, 20),
        ('最近30期', None, 43.3),  # 需要重新计算
        ('全部100期', None, 52.0)
    ]
    
    for test_name, v11_rate, v10_rate in comparisons:
        if v11_rate is not None:
            diff = v11_rate - v10_rate
            print(f"{test_name:12s}  {v11_rate:5.1f}%  {v10_rate:5.1f}%  {diff:+5.1f}%")
    
    print(f"\n{'='*110}")

if __name__ == '__main__':
    test_trend_smart()
