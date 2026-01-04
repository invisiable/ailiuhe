"""
展示v10.0智能模型选择器最近30期的预测结果
"""

import pandas as pd
from zodiac_simple_smart import ZodiacSimpleSmart

def show_recent_30_predictions():
    predictor = ZodiacSimpleSmart()
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    
    print("="*110)
    print("v10.0智能模型选择器 - 最近30期预测结果")
    print("="*110)
    
    total = len(df)
    test_data = df.iloc[-30:]  # 最近30期
    
    results = []
    hit_count = 0
    model_stats = {}
    
    print(f"\n{'序号':<4} {'日期':<12} {'实际':<4} {'预测TOP5':<30} {'模型':<18} {'结果':<10}")
    print("-"*110)
    
    for idx in range(30):
        # 用前面的数据预测当前期
        animals = [str(a).strip() for a in df['animal'].values[:total - 30 + idx]]
        
        if len(animals) < 10:
            continue
        
        # 获取预测
        prediction = predictor.predict_from_history(animals, top_n=5, debug=False)
        
        # 获取实际结果
        actual = str(test_data.iloc[idx]['animal']).strip()
        date = test_data.iloc[idx]['date']
        predicted_top5 = prediction['top5']
        model_used = prediction['selected_model']
        
        # 统计模型使用
        model_stats[model_used] = model_stats.get(model_used, 0) + 1
        
        # 判断是否命中
        if actual in predicted_top5:
            rank = predicted_top5.index(actual) + 1
            hit_count += 1
            result_str = f"✓ TOP{rank}"
        else:
            rank = 0
            result_str = "✗"
        
        results.append({
            'idx': idx + 1,
            'date': date,
            'actual': actual,
            'predicted': predicted_top5,
            'model': model_used,
            'hit': rank > 0,
            'rank': rank
        })
        
        # 打印每一期
        predicted_str = ','.join(predicted_top5)
        print(f"{idx+1:<4} {date:<12} {actual:<4} {predicted_str:<30} {model_used:<18} {result_str:<10}")
    
    # 统计汇总
    print("-"*110)
    print(f"\n总命中统计:")
    print(f"  TOP5命中: {hit_count}/30 = {hit_count/30*100:.1f}%")
    
    top2_count = sum(1 for r in results if r['hit'] and r['rank'] <= 2)
    top1_count = sum(1 for r in results if r['hit'] and r['rank'] == 1)
    print(f"  TOP2命中: {top2_count}/30 = {top2_count/30*100:.1f}%")
    print(f"  TOP1命中: {top1_count}/30 = {top1_count/30*100:.1f}%")
    
    print(f"\n模型使用统计:")
    for model, count in sorted(model_stats.items(), key=lambda x: x[1], reverse=True):
        pct = count / 30 * 100
        print(f"  {model:<18}: {count:2d}次 ({pct:5.1f}%)")
    
    print(f"\n{'='*110}")
    
    # 分段统计
    print("\n分段统计分析:")
    print("-"*110)
    
    segments = [
        ("最近10期", results[-10:]),
        ("中间10期", results[10:20]),
        ("早期10期", results[:10])
    ]
    
    for seg_name, seg_results in segments:
        seg_hit = sum(1 for r in seg_results if r['hit'])
        seg_top2 = sum(1 for r in seg_results if r['hit'] and r['rank'] <= 2)
        
        print(f"\n{seg_name}:")
        print(f"  TOP5: {seg_hit}/10 = {seg_hit*10}%")
        print(f"  TOP2: {seg_top2}/10 = {seg_top2*10}%")
        
        # 该段使用的模型
        seg_models = {}
        for r in seg_results:
            seg_models[r['model']] = seg_models.get(r['model'], 0) + 1
        
        print(f"  模型分布: ", end="")
        print(", ".join([f"{m}:{c}次" for m, c in seg_models.items()]))
    
    print(f"\n{'='*110}")

if __name__ == '__main__':
    show_recent_30_predictions()
