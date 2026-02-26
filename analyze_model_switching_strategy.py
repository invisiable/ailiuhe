"""
分析连续失败时切换模型的策略效果

研究问题：
1. 当连续3期预测失败时，是否应该切换预测模型？
2. 不同的模型切换策略对成功率的影响？
"""

import pandas as pd
from ensemble_zodiac_predictor import EnsembleZodiacPredictor

def analyze_consecutive_failures(csv_file='data/lucky_numbers.csv', test_periods=200):
    """分析连续失败期间使用的模型"""
    print("=" * 80)
    print("📊 连续失败期间的模型分析")
    print("=" * 80)
    print()
    
    # 加载数据
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    all_animals = [str(a).strip() for a in df['animal'].tolist()]
    
    # 初始化预测器
    predictor = EnsembleZodiacPredictor()
    
    # 测试最后test_periods期
    start_idx = len(all_animals) - test_periods
    predictions = []
    actual_animals = []
    selected_models = []
    
    print(f"开始分析最后{test_periods}期数据...\n")
    
    # 逐期预测
    for i in range(start_idx, len(all_animals)):
        history = all_animals[:i]
        result = predictor.predict_from_history(history, top_n=5, debug=False)
        
        predictions.append(result['top4'])
        actual_animals.append(all_animals[i])
        selected_models.append(result['selected_model'])
    
    # 分析连续失败情况
    consecutive_failures = []
    current_streak = 0
    current_models = []
    
    for i, (pred_top4, actual) in enumerate(zip(predictions, actual_animals)):
        hit = actual in pred_top4
        
        if not hit:
            current_streak += 1
            current_models.append(selected_models[i])
        else:
            if current_streak >= 3:
                consecutive_failures.append({
                    'length': current_streak,
                    'models': current_models.copy(),
                    'end_index': start_idx + i
                })
            current_streak = 0
            current_models = []
    
    # 如果最后还有未结束的连续失败
    if current_streak >= 3:
        consecutive_failures.append({
            'length': current_streak,
            'models': current_models.copy(),
            'end_index': len(all_animals)
        })
    
    # 统计结果
    print(f"发现 {len(consecutive_failures)} 次连续3期以上的失败")
    print()
    
    # 分析每次连续失败
    model_consistency = []
    for i, failure in enumerate(consecutive_failures):
        print(f"失败#{i+1}: 连续{failure['length']}期未中")
        print(f"  使用的模型: {' → '.join(failure['models'])}")
        
        # 检查模型是否一致
        unique_models = set(failure['models'])
        if len(unique_models) == 1:
            print(f"  ⚠️  模型一直是: {list(unique_models)[0]} (未切换)")
            model_consistency.append('same')
        else:
            print(f"  ✓ 模型有切换: {len(unique_models)}个不同模型")
            model_consistency.append('switched')
        print()
    
    # 统计
    same_count = model_consistency.count('same')
    switched_count = model_consistency.count('switched')
    
    print("=" * 80)
    print("📈 统计结果")
    print("=" * 80)
    print(f"连续失败时模型未切换: {same_count}次 ({same_count/len(consecutive_failures)*100:.1f}%)")
    print(f"连续失败时模型有切换: {switched_count}次 ({switched_count/len(consecutive_failures)*100:.1f}%)")
    print()
    
    # 结论
    if same_count > switched_count:
        print("💡 发现: 大部分连续失败时，集成预测器选择了同一个模型")
        print("   建议: 可以尝试强制切换模型策略")
    else:
        print("💡 发现: 集成预测器已经在连续失败时自动切换模型")
        print("   建议: 当前的自适应机制已经在工作")
    print()
    
    return consecutive_failures, selected_models, predictions, actual_animals


def test_model_switching_strategy(csv_file='data/lucky_numbers.csv', test_periods=200):
    """测试强制模型切换策略的效果"""
    print("=" * 80)
    print("🔄 测试强制模型切换策略")
    print("=" * 80)
    print()
    
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    all_animals = [str(a).strip() for a in df['animal'].tolist()]
    
    predictor = EnsembleZodiacPredictor()
    start_idx = len(all_animals) - test_periods
    
    # 策略A: 基准 - 不强制切换（使用集成预测器的默认行为）
    print("策略A: 基准策略（集成预测器默认行为）")
    baseline_hits = 0
    baseline_predictions = []
    
    for i in range(start_idx, len(all_animals)):
        history = all_animals[:i]
        result = predictor.predict_from_history(history, top_n=5, debug=False)
        actual = all_animals[i]
        hit = actual in result['top4']
        baseline_predictions.append((result['top4'], result['selected_model'], hit))
        if hit:
            baseline_hits += 1
    
    baseline_rate = baseline_hits / len(baseline_predictions) * 100
    print(f"  命中率: {baseline_rate:.2f}% ({baseline_hits}/{len(baseline_predictions)})")
    print()
    
    # 策略B: 连续3期失败后，强制轮换模型
    print("策略B: 连续3期失败后强制轮换模型")
    strategy_b_hits = 0
    consecutive_losses = 0
    blocked_model = None
    blocked_until = 0
    
    for i in range(start_idx, len(all_animals)):
        history = all_animals[:i]
        
        # 如果有被阻止的模型且还在阻止期内，使用排除法
        if blocked_model and i < blocked_until:
            # 强制使用non-blocked模型
            result = predictor.predict_from_history(history, top_n=5, debug=False)
            
            # 如果选中的是被阻止的模型，切换到另一个
            if result['selected_model'] == blocked_model:
                # 强制切换：如果是v10，用optimized；如果是optimized，用v10
                if 'v10' in blocked_model.lower():
                    # 手动使用optimized predictor
                    top4_pred = predictor.optimized_predictor.predict_from_history(history, top_n=5)
                    result = {'top4': top4_pred, 'selected_model': 'optimized'}
                else:
                    # 手动使用v10 predictor
                    top4_pred = predictor.v10_predictor.predict_from_history(history, top_n=5)
                    result = {'top4': top4_pred, 'selected_model': 'v10'}
        else:
            result = predictor.predict_from_history(history, top_n=5, debug=False)
        
        actual = all_animals[i]
        hit = actual in result['top4']
        
        if hit:
            strategy_b_hits += 1
            consecutive_losses = 0
            blocked_model = None
        else:
            consecutive_losses += 1
            # 连续3期失败，阻止当前模型3期
            if consecutive_losses >= 3:
                blocked_model = result['selected_model']
                blocked_until = i + 3
                print(f"  期数{i-start_idx+1}: 连续{consecutive_losses}期失败，阻止模型'{blocked_model}' 3期")
    
    strategy_b_rate = strategy_b_hits / test_periods * 100
    print(f"  命中率: {strategy_b_rate:.2f}% ({strategy_b_hits}/{test_periods})")
    improvement_b = strategy_b_rate - baseline_rate
    print(f"  改进: {improvement_b:+.2f}%")
    print()
    
    # 策略C: 追踪每个模型的近期表现，连续3期失败后切换到表现最好的
    print("策略C: 基于近期表现智能切换")
    strategy_c_hits = 0
    consecutive_losses_c = 0
    model_performance = {'v10': [], 'optimized': []}  # 记录最近10期表现
    
    for i in range(start_idx, len(all_animals)):
        history = all_animals[:i]
        
        # 如果连续3期失败，选择近期表现更好的模型
        if consecutive_losses_c >= 3:
            # 计算每个模型的近期成功率
            v10_recent_rate = sum(model_performance['v10'][-10:]) / max(len(model_performance['v10'][-10:]), 1)
            opt_recent_rate = sum(model_performance['optimized'][-10:]) / max(len(model_performance['optimized'][-10:]), 1)
            
            # 选择表现更好的模型
            if v10_recent_rate > opt_recent_rate:
                top4_pred = predictor.v10_predictor.predict_from_history(history, top_n=5)
                result = {'top4': top4_pred, 'selected_model': 'v10'}
                forced_model = 'v10'
            else:
                top4_pred = predictor.optimized_predictor.predict_from_history(history, top_n=5)
                result = {'top4': top4_pred, 'selected_model': 'optimized'}
                forced_model = 'optimized'
            
            # print(f"  期数{i-start_idx+1}: 强制使用'{forced_model}' (近期成功率: v10={v10_recent_rate:.2f}, opt={opt_recent_rate:.2f})")
        else:
            result = predictor.predict_from_history(history, top_n=5, debug=False)
        
        actual = all_animals[i]
        hit = actual in result['top4']
        
        # 记录模型表现
        model_name = result['selected_model']
        if 'v10' in model_name.lower():
            model_performance['v10'].append(1 if hit else 0)
        else:
            model_performance['optimized'].append(1 if hit else 0)
        
        if hit:
            strategy_c_hits += 1
            consecutive_losses_c = 0
        else:
            consecutive_losses_c += 1
    
    strategy_c_rate = strategy_c_hits / test_periods * 100
    print(f"  命中率: {strategy_c_rate:.2f}% ({strategy_c_hits}/{test_periods})")
    improvement_c = strategy_c_rate - baseline_rate
    print(f"  改进: {improvement_c:+.2f}%")
    print()
    
    # 总结
    print("=" * 80)
    print("📊 策略对比总结")
    print("=" * 80)
    print(f"基准策略（默认）:   {baseline_rate:.2f}%")
    print(f"策略B（强制轮换）:  {strategy_b_rate:.2f}% ({improvement_b:+.2f}%)")
    print(f"策略C（智能切换）:  {strategy_c_rate:.2f}% ({improvement_c:+.2f}%)")
    print()
    
    if improvement_b > 0 or improvement_c > 0:
        best_strategy = "策略C" if improvement_c > improvement_b else "策略B"
        best_improvement = max(improvement_b, improvement_c)
        print(f"✅ {best_strategy}表现最好，命中率提升{best_improvement:+.2f}%")
        print(f"💡 建议: 可以将模型切换策略集成到GUI中")
    else:
        print(f"⚠️  强制切换策略未能提升命中率")
        print(f"💡 建议: 保持集成预测器的默认自适应行为")
    print()
    
    return {
        'baseline': baseline_rate,
        'strategy_b': strategy_b_rate,
        'strategy_c': strategy_c_rate
    }


if __name__ == '__main__':
    # 分析连续失败情况
    failures, models, predictions, actuals = analyze_consecutive_failures(test_periods=200)
    
    print()
    
    # 测试模型切换策略
    results = test_model_switching_strategy(test_periods=200)
    
    print()
    print("=" * 80)
    print("🎯 结论与建议")
    print("=" * 80)
    print()
    print("1. 如果策略B或C有明显改进（>1%），建议集成到GUI")
    print("2. 如果改进不明显（<1%），说明集成预测器已经足够智能")
    print("3. 可以考虑将模型切换策略作为可选功能，让用户选择是否启用")
    print()
