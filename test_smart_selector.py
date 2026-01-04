"""
测试智能模型选择器 v9.0
"""

import pandas as pd
from zodiac_smart_selector import ZodiacSmartSelector

def test_smart_selector():
    predictor = ZodiacSmartSelector()
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    
    print("="*90)
    print("智能模型选择器 v9.0 - 全面回测")
    print("="*90)
    
    # 测试不同时期的预测和模型选择
    test_ranges = [
        ('最近10期', -10, None),
        ('最近20期', -20, None),
        ('最近50期', -50, None),
        ('全部100期', -100, None),
        ('中期50期 (100-50期前)', -150, -50),
        ('早期50期 (150-100期前)', -200, -150)
    ]
    
    model_usage = {}
    
    for range_name, start, end in test_ranges:
        print(f"\n{'='*90}")
        print(f"测试区间: {range_name}")
        print(f"{'='*90}")
        
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
            
            if len(animals) < 10:  # 至少需要10期历史数据
                continue
            
            prediction = predictor.predict_from_history(animals, top_n=5, debug=False)
            actual = str(test_data.iloc[idx]['animal']).strip()
            
            # 统计模型使用情况
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
        
        # 打印结果
        valid_count = total if start >= 0 else total
        top5_rate = (hit_top5 / valid_count * 100) if valid_count > 0 else 0
        top2_rate = (hit_top2 / valid_count * 100) if valid_count > 0 else 0
        top1_rate = (hit_top1 / valid_count * 100) if valid_count > 0 else 0
        
        print(f"样本数: {valid_count}")
        print(f"TOP5命中: {hit_top5}/{valid_count} = {top5_rate:.1f}%")
        print(f"TOP2命中: {hit_top2}/{valid_count} = {top2_rate:.1f}%")
        print(f"TOP1命中: {hit_top1}/{valid_count} = {top1_rate:.1f}%")
        
        print(f"\n该区间模型使用统计:")
        for model_name, count in sorted(range_model_usage.items(), key=lambda x: x[1], reverse=True):
            usage_pct = count / valid_count * 100 if valid_count > 0 else 0
            print(f"  {predictor.models[model_name]['name']:15s}: {count:3d}次 ({usage_pct:5.1f}%)")
    
    # 总体模型使用统计
    print(f"\n{'='*90}")
    print("全局模型使用统计:")
    print(f"{'='*90}")
    total_usage = sum(model_usage.values())
    for model_name, count in sorted(model_usage.items(), key=lambda x: x[1], reverse=True):
        usage_pct = count / total_usage * 100 if total_usage > 0 else 0
        print(f"  {predictor.models[model_name]['name']:15s}: {count:4d}次 ({usage_pct:5.1f}%)")
    
    # 与v5.0对比
    print(f"\n{'='*90}")
    print("与v5.0平衡模型对比 (最近100期):")
    print(f"{'='*90}")
    
    # 重新测试最近100期获取详细数据
    test_100 = df.iloc[-100:]
    hit_v9 = 0
    
    for idx in range(len(test_100)):
        animals = [str(a).strip() for a in df['animal'].values[:-100 + idx]]
        if len(animals) < 10:
            continue
        
        prediction = predictor.predict_from_history(animals, top_n=5, debug=False)
        actual = str(test_100.iloc[idx]['animal']).strip()
        
        if actual in prediction['top5']:
            hit_v9 += 1
    
    rate_v9 = hit_v9 / len(test_100) * 100
    rate_v5 = 52.0  # v5.0已知成绩
    
    print(f"v9.0智能选择器: {hit_v9}/100 = {rate_v9:.1f}%")
    print(f"v5.0平衡模型:   52/100 = {rate_v5:.1f}%")
    
    diff = rate_v9 - rate_v5
    if diff > 0:
        print(f"提升: +{diff:.1f}%  ✓")
    elif diff < 0:
        print(f"下降: {diff:.1f}%  ✗")
    else:
        print(f"持平")
    
    print(f"\n{'='*90}")

if __name__ == '__main__':
    test_smart_selector()
