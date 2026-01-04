"""
展示v11.0实时趋势检测智能选择器最近30期的预测结果
"""

import pandas as pd
from zodiac_trend_smart import ZodiacTrendSmart

def show_v11_recent30():
    predictor = ZodiacTrendSmart()
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    
    print("="*120)
    print("v11.0实时趋势检测智能选择器 - 最近30期预测结果")
    print("="*120)
    
    total = len(df)
    test_data = df.iloc[-30:]  # 最近30期
    
    results = []
    hit_count = 0
    model_stats = {}
    scenario_stats = {}
    
    print(f"\n{'序号':<4} {'日期':<12} {'实际':<4} {'预测TOP5':<30} {'场景':<18} {'模型':<20} {'结果':<10}")
    print("-"*120)
    
    for idx in range(30):
        # 用前面的数据预测当前期
        animals = [str(a).strip() for a in df['animal'].values[:total - 30 + idx]]
        
        if len(animals) < 30:  # v11.0需要至少30期数据
            continue
        
        # 获取预测
        prediction = predictor.predict_from_history(animals, top_n=5, debug=False)
        
        # 获取实际结果
        actual = str(test_data.iloc[idx]['animal']).strip()
        date = test_data.iloc[idx]['date']
        predicted_top5 = prediction['top5']
        model_used = prediction['selected_model']
        scenario = prediction['scenario']
        
        # 统计模型和场景使用
        model_stats[model_used] = model_stats.get(model_used, 0) + 1
        scenario_stats[scenario] = scenario_stats.get(scenario, 0) + 1
        
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
            'scenario': scenario,
            'model': model_used,
            'hit': rank > 0,
            'rank': rank
        })
        
        # 打印每一期
        predicted_str = ','.join(predicted_top5)
        print(f"{idx+1:<4} {date:<12} {actual:<4} {predicted_str:<30} {scenario:<18} {model_used:<20} {result_str:<10}")
    
    # 统计汇总
    print("-"*120)
    print(f"\n总命中统计:")
    print(f"  TOP5命中: {hit_count}/30 = {hit_count/30*100:.1f}%")
    
    top2_count = sum(1 for r in results if r['hit'] and r['rank'] <= 2)
    top1_count = sum(1 for r in results if r['hit'] and r['rank'] == 1)
    print(f"  TOP2命中: {top2_count}/30 = {top2_count/30*100:.1f}%")
    print(f"  TOP1命中: {top1_count}/30 = {top1_count/30*100:.1f}%")
    
    print(f"\n场景识别统计:")
    for scenario, count in sorted(scenario_stats.items(), key=lambda x: x[1], reverse=True):
        pct = count / 30 * 100
        print(f"  {scenario:<18}: {count:2d}次 ({pct:5.1f}%)")
    
    print(f"\n模型使用统计:")
    for model, count in sorted(model_stats.items(), key=lambda x: x[1], reverse=True):
        pct = count / 30 * 100
        print(f"  {model:<20}: {count:2d}次 ({pct:5.1f}%)")
    
    print(f"\n{'='*120}")
    
    # 分段统计
    print("\n分段统计分析:")
    print("-"*120)
    
    segments = [
        ("最近10期", results[-10:]),
        ("中间10期", results[10:20]),
        ("早期10期", results[:10])
    ]
    
    for seg_name, seg_results in segments:
        seg_hit = sum(1 for r in seg_results if r['hit'])
        seg_top2 = sum(1 for r in seg_results if r['hit'] and r['rank'] <= 2)
        seg_top1 = sum(1 for r in seg_results if r['hit'] and r['rank'] == 1)
        
        print(f"\n{seg_name}:")
        print(f"  TOP5: {seg_hit}/10 = {seg_hit*10}%")
        print(f"  TOP2: {seg_top2}/10 = {seg_top2*10}%")
        print(f"  TOP1: {seg_top1}/10 = {seg_top1*10}%")
        
        # 该段使用的场景和模型
        seg_scenarios = {}
        seg_models = {}
        for r in seg_results:
            seg_scenarios[r['scenario']] = seg_scenarios.get(r['scenario'], 0) + 1
            seg_models[r['model']] = seg_models.get(r['model'], 0) + 1
        
        print(f"  场景: ", end="")
        print(", ".join([f"{s}:{c}次" for s, c in sorted(seg_scenarios.items(), key=lambda x: x[1], reverse=True)]))
        print(f"  模型: ", end="")
        print(", ".join([f"{m}:{c}次" for m, c in sorted(seg_models.items(), key=lambda x: x[1], reverse=True)]))
    
    # 与v10.0对比
    print(f"\n{'='*120}")
    print("与v10.0性能对比:")
    print("-"*120)
    
    print(f"\n{'期数':<15} {'v11.0':<10} {'v10.0':<10} {'差异':<10}")
    print("-"*50)
    
    # 最近10期
    recent10_v11 = sum(1 for r in results[-10:] if r['hit'])
    print(f"{'最近10期':<15} {recent10_v11*10:>5}%     {'20':>5}%     {recent10_v11*10-20:>+5.0f}%")
    
    # 最近30期
    recent30_v11 = hit_count / 30 * 100
    print(f"{'最近30期':<15} {recent30_v11:>5.1f}%     {'43.3':>5}%     {recent30_v11-43.3:>+5.1f}%")
    
    print(f"\n{'='*120}")
    
    # 趋势检测效果展示
    print("\n【趋势检测效果展示】- 最近10期:")
    print("-"*120)
    
    for idx in range(20, 30):
        animals = [str(a).strip() for a in df['animal'].values[:total - 30 + idx]]
        prediction = predictor.predict_from_history(animals, top_n=5, debug=False)
        
        info = prediction['scenario_info']
        actual = str(test_data.iloc[idx]['animal']).strip()
        date = test_data.iloc[idx]['date']
        
        print(f"\n{date} - 实际: {actual}")
        
        if info.get('burst_zodiacs'):
            print(f"  检测到爆发: {info['burst_zodiacs']}")
            for z in info['burst_zodiacs']:
                trend = info['trend_analysis'][z]
                print(f"    {z}: 前20期{trend['prev']}次 → 最近10期{trend['recent']}次")
        
        if info.get('hot_zodiacs'):
            print(f"  持续热门: {info['hot_zodiacs']}")
        
        if info.get('recent_hot'):
            print(f"  最近高频: {info['recent_hot']}")
        
        print(f"  预测TOP5: {', '.join(prediction['top5'])}")
        hit_mark = "✓" if actual in prediction['top5'] else "✗"
        print(f"  结果: {hit_mark}")
    
    print(f"\n{'='*120}")

if __name__ == '__main__':
    show_v11_recent30()
