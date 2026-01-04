"""
测试简化版智能选择器 v10.0
"""

import pandas as pd
from zodiac_simple_smart import ZodiacSimpleSmart

def test_simple_smart():
    predictor = ZodiacSimpleSmart()
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    
    print("="*90)
    print("简化版智能选择器 v10.0 - 以v5.0为主，最小化切换")
    print("="*90)
    
    # 测试不同时期
    test_ranges = [
        ('最近10期', -10, None),
        ('最近20期', -20, None),
        ('最近50期', -50, None),
        ('全部100期', -100, None)
    ]
    
    model_usage = {}
    
    for range_name, start, end in test_ranges:
        print(f"\n{range_name}:")
        print("-"*90)
        
        if end is None:
            test_data = df.iloc[start:]
        else:
            test_data = df.iloc[start:end]
        
        total = len(test_data)
        hit_top5 = 0
        hit_top2 = 0
        hit_top1 = 0
        
        range_model_usage = {}
        
        for idx in range(total):
            animals = [str(a).strip() for a in df['animal'].values[:start + idx if start < 0 else idx]]
            
            if len(animals) < 10:
                continue
            
            prediction = predictor.predict_from_history(animals, top_n=5, debug=False)
            actual = str(test_data.iloc[idx]['animal']).strip()
            
            model_used = prediction['selected_model']
            range_model_usage[model_used] = range_model_usage.get(model_used, 0) + 1
            model_usage[model_used] = model_usage.get(model_used, 0) + 1
            
            if actual in prediction['top5']:
                rank = prediction['top5'].index(actual) + 1
                hit_top5 += 1
                if rank <= 2:
                    hit_top2 += 1
                if rank == 1:
                    hit_top1 += 1
        
        valid_count = total
        top5_rate = (hit_top5 / valid_count * 100) if valid_count > 0 else 0
        top2_rate = (hit_top2 / valid_count * 100) if valid_count > 0 else 0
        top1_rate = (hit_top1 / valid_count * 100) if valid_count > 0 else 0
        
        print(f"TOP5命中: {hit_top5}/{valid_count} = {top5_rate:.1f}%")
        print(f"TOP2命中: {hit_top2}/{valid_count} = {top2_rate:.1f}%")
        print(f"TOP1命中: {hit_top1}/{valid_count} = {top1_rate:.1f}%")
        
        print(f"\n模型使用:")
        for model_name, count in sorted(range_model_usage.items(), key=lambda x: x[1], reverse=True):
            usage_pct = count / valid_count * 100 if valid_count > 0 else 0
            print(f"  {model_name:15s}: {count:3d}次 ({usage_pct:5.1f}%)")
    
    print(f"\n{'='*90}")
    print("全局模型使用统计:")
    print(f"{'='*90}")
    total_usage = sum(model_usage.values())
    for model_name, count in sorted(model_usage.items(), key=lambda x: x[1], reverse=True):
        usage_pct = count / total_usage * 100 if total_usage > 0 else 0
        print(f"  {model_name:15s}: {count:4d}次 ({usage_pct:5.1f}%)")
    
    # 与v5.0对比
    print(f"\n{'='*90}")
    print("与之前版本对比 (最近100期):")
    print(f"{'='*90}")
    
    test_100 = df.iloc[-100:]
    hit_v10 = 0
    
    for idx in range(len(test_100)):
        animals = [str(a).strip() for a in df['animal'].values[:-100 + idx]]
        if len(animals) < 10:
            continue
        
        prediction = predictor.predict_from_history(animals, top_n=5, debug=False)
        actual = str(test_100.iloc[idx]['animal']).strip()
        
        if actual in prediction['top5']:
            hit_v10 += 1
    
    rate_v10 = hit_v10 / len(test_100) * 100
    
    print(f"v10.0简化智能:   {hit_v10}/100 = {rate_v10:.1f}%")
    print(f"v5.0平衡模型:    52/100 = 52.0%")
    print(f"v9.0智能选择器:  48/100 = 48.0%")
    
    diff = rate_v10 - 52.0
    if diff >= 0:
        print(f"\n相比v5.0: +{diff:.1f}%  ✓ 提升" if diff > 0 else f"\n相比v5.0: 持平")
    else:
        print(f"\n相比v5.0: {diff:.1f}%  ✗ 下降")
    
    print(f"\n{'='*90}")

if __name__ == '__main__':
    test_simple_smart()
