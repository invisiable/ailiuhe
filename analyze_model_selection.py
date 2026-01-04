"""
详细分析模型选择情况 - 找出最优模型组合
"""

import pandas as pd
from zodiac_smart_selector import ZodiacSmartSelector

def detailed_model_analysis():
    predictor = ZodiacSmartSelector()
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    
    print("="*110)
    print("详细分析：各模型在不同时期的表现")
    print("="*110)
    
    # 分析最近100期
    test_data = df.iloc[-100:]
    
    # 记录每个模型的成功/失败案例
    model_performance = {
        'ultra_cold': {'hit': 0, 'miss': 0, 'periods': []},
        'balanced': {'hit': 0, 'miss': 0, 'periods': []},
        'v5_ultra_hybrid': {'hit': 0, 'miss': 0, 'periods': []},
        'gap_focus': {'hit': 0, 'miss': 0, 'periods': []},
        'hot_aware': {'hit': 0, 'miss': 0, 'periods': []},
        'diversity': {'hit': 0, 'miss': 0, 'periods': []}
    }
    
    for idx in range(len(test_data)):
        animals = [str(a).strip() for a in df['animal'].values[:-100 + idx]]
        if len(animals) < 10:
            continue
        
        prediction = predictor.predict_from_history(animals, top_n=5, debug=False)
        actual = str(test_data.iloc[idx]['animal']).strip()
        model_used = prediction['selected_model']
        date = test_data.iloc[idx]['date']
        
        if actual in prediction['top5']:
            model_performance[model_used]['hit'] += 1
            model_performance[model_used]['periods'].append((date, actual, True))
        else:
            model_performance[model_used]['miss'] += 1
            model_performance[model_used]['periods'].append((date, actual, False))
    
    # 打印各模型表现
    print("\n各模型命中统计:")
    print("-"*110)
    for model_name in ['ultra_cold', 'balanced', 'v5_ultra_hybrid', 'gap_focus', 'hot_aware', 'diversity']:
        perf = model_performance[model_name]
        total = perf['hit'] + perf['miss']
        if total > 0:
            rate = perf['hit'] / total * 100
            model_display = predictor.models[model_name]['name']
            print(f"{model_display:15s}: {perf['hit']:2d}/{total:2d} = {rate:5.1f}%  (命中{perf['hit']}, 未中{perf['miss']})")
    
    # 分析最近10期使用的模型
    print(f"\n{'='*110}")
    print("最近10期详细分析:")
    print(f"{'='*110}")
    
    recent_10 = []
    for idx in range(90, 100):
        animals = [str(a).strip() for a in df['animal'].values[:-100 + idx]]
        prediction = predictor.predict_from_history(animals, top_n=5, debug=False)
        actual = str(test_data.iloc[idx]['animal']).strip()
        
        hit = actual in prediction['top5']
        rank = prediction['top5'].index(actual) + 1 if hit else 0
        
        recent_10.append({
            'date': test_data.iloc[idx]['date'],
            'actual': actual,
            'predicted': prediction['top5'],
            'model': prediction['model_name'],
            'hit': hit,
            'rank': rank
        })
    
    for item in recent_10:
        hit_mark = f"✓ TOP{item['rank']}" if item['hit'] else "✗"
        print(f"日期{item['date']} | 实际:{item['actual']:2s} | 预测:{','.join(item['predicted'])} | 模型:{item['model']:15s} | {hit_mark}")
    
    hit_count = sum(1 for item in recent_10 if item['hit'])
    print(f"\n最近10期命中: {hit_count}/10 = {hit_count*10}%")
    
    # 寻找最优模型权重组合
    print(f"\n{'='*110}")
    print("建议优化方向:")
    print(f"{'='*110}")
    
    # 统计哪个模型在最近表现最好
    recent_20_model_perf = {}
    for idx in range(80, 100):
        animals = [str(a).strip() for a in df['animal'].values[:-100 + idx]]
        if len(animals) < 10:
            continue
        
        prediction = predictor.predict_from_history(animals, top_n=5, debug=False)
        actual = str(test_data.iloc[idx]['animal']).strip()
        model_used = prediction['selected_model']
        
        if model_used not in recent_20_model_perf:
            recent_20_model_perf[model_used] = {'hit': 0, 'total': 0}
        
        recent_20_model_perf[model_used]['total'] += 1
        if actual in prediction['top5']:
            recent_20_model_perf[model_used]['hit'] += 1
    
    print("\n最近20期各模型表现:")
    for model_name, perf in recent_20_model_perf.items():
        rate = perf['hit'] / perf['total'] * 100 if perf['total'] > 0 else 0
        model_display = predictor.models[model_name]['name']
        print(f"  {model_display:15s}: {perf['hit']}/{perf['total']} = {rate:.1f}%")
    
    # 计算如果强制使用某个模型的效果
    print(f"\n{'='*110}")
    print("模拟：如果全程使用单一模型的效果（最近100期）")
    print(f"{'='*110}")
    
    for test_model in ['ultra_cold', 'balanced', 'v5_ultra_hybrid', 'gap_focus', 'hot_aware', 'diversity']:
        hit_count = 0
        for idx in range(len(test_data)):
            animals = [str(a).strip() for a in df['animal'].values[:-100 + idx]]
            if len(animals) < 10:
                continue
            
            # 强制使用指定模型的权重
            weights = predictor.models[test_model]['weights']
            
            # 手动计算预测
            strategies_scores = {
                'ultra_cold': predictor._ultra_cold_strategy(animals),
                'anti_hot': predictor._anti_hot_strategy(animals),
                'gap': predictor._gap_analysis(animals),
                'rotation': predictor._rotation_advanced(animals),
                'absence_penalty': predictor._continuous_absence_penalty(animals),
                'diversity': predictor._diversity_boost(animals),
                'similarity': predictor._historical_similarity(animals)
            }
            
            if 'hot_momentum' in weights:
                strategies_scores['hot_momentum'] = predictor._hot_momentum_strategy(animals)
            
            final_scores = {}
            for zodiac in predictor.zodiacs:
                score = 0.0
                for strategy_name, strategy_scores in strategies_scores.items():
                    weight = weights.get(strategy_name, 0)
                    score += strategy_scores.get(zodiac, 0) * weight
                final_scores[zodiac] = score
            
            sorted_zodiacs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            top5 = [z for z, s in sorted_zodiacs[:5]]
            
            actual = str(test_data.iloc[idx]['animal']).strip()
            if actual in top5:
                hit_count += 1
        
        rate = hit_count / len(test_data) * 100
        model_display = predictor.models[test_model]['name']
        print(f"  {model_display:15s}: {hit_count}/100 = {rate:.1f}%")

if __name__ == '__main__':
    detailed_model_analysis()
