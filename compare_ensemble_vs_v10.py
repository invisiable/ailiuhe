"""
比较 EnsembleZodiacPredictor vs ZodiacSimpleSmart(v10) 的预测成功率
"""

import pandas as pd
from ensemble_zodiac_predictor import EnsembleZodiacPredictor
from zodiac_simple_smart import ZodiacSimpleSmart

def compare_models(data_file='data/lucky_numbers.csv', test_periods=100):
    """比较两个模型的预测成功率"""
    
    print("="*80)
    print("生肖预测模型对比验证")
    print("="*80)
    print(f"验证期数: 最近{test_periods}期\n")
    
    # 读取数据
    df = pd.read_csv(data_file, encoding='utf-8-sig')
    print(f"数据加载: {len(df)}期历史数据")
    print(f"最新期: {df.iloc[-1]['date']} - {df.iloc[-1]['animal']}\n")
    
    # 创建两个预测器
    ensemble = EnsembleZodiacPredictor()
    v10 = ZodiacSimpleSmart()
    
    # 测试数据范围
    start_idx = len(df) - test_periods
    
    # 记录结果
    ensemble_results = {
        'top3': [],
        'top4': [],
        'top5': [],
        'model_used': []
    }
    
    v10_results = {
        'top3': [],
        'top4': [],
        'top5': [],
        'model_used': []
    }
    
    actuals = []
    
    print("开始滚动验证...\n")
    
    for i in range(start_idx, len(df)):
        # 使用i之前的数据进行预测
        train_animals = [str(a).strip() for a in df['animal'].iloc[:i].tolist()]
        
        # Ensemble预测
        ensemble_pred = ensemble.predict_from_history(train_animals, top_n=5, debug=False)
        ensemble_results['top3'].append(ensemble_pred['top3'])
        ensemble_results['top4'].append(ensemble_pred['top4'])
        ensemble_results['top5'].append(ensemble_pred['top5'])
        ensemble_results['model_used'].append(ensemble_pred['selected_model'])
        
        # V10预测
        v10_pred = v10.predict_from_history(train_animals, top_n=5, debug=False)
        v10_results['top3'].append(v10_pred['top5'][:3])
        v10_results['top4'].append(v10_pred['top5'][:4])
        v10_results['top5'].append(v10_pred['top5'])
        v10_results['model_used'].append(v10_pred['selected_model'])
        
        # 实际结果
        actual = str(df.iloc[i]['animal']).strip()
        actuals.append(actual)
        
        if (i - start_idx + 1) % 20 == 0:
            print(f"  已处理 {i - start_idx + 1}/{test_periods} 期...")
    
    print(f"\n✅ 验证完成！\n")
    
    # 计算命中率
    def calc_hit_rate(predictions, actuals):
        hits = sum(1 for pred, actual in zip(predictions, actuals) if actual in pred)
        return hits, hits / len(actuals) * 100 if len(actuals) > 0 else 0
    
    # Ensemble结果
    print("="*80)
    print("【EnsembleZodiacPredictor 结果】")
    print("="*80)
    
    ens_top3_hits, ens_top3_rate = calc_hit_rate(ensemble_results['top3'], actuals)
    ens_top4_hits, ens_top4_rate = calc_hit_rate(ensemble_results['top4'], actuals)
    ens_top5_hits, ens_top5_rate = calc_hit_rate(ensemble_results['top5'], actuals)
    
    print(f"TOP3 命中: {ens_top3_hits}/{test_periods} = {ens_top3_rate:.2f}%")
    print(f"TOP4 命中: {ens_top4_hits}/{test_periods} = {ens_top4_rate:.2f}%")
    print(f"TOP5 命中: {ens_top5_hits}/{test_periods} = {ens_top5_rate:.2f}%")
    
    # 统计模型使用情况
    from collections import Counter
    ens_model_count = Counter(ensemble_results['model_used'])
    print(f"\n模型使用统计:")
    for model, count in ens_model_count.most_common():
        print(f"  {model}: {count}次 ({count/test_periods*100:.1f}%)")
    
    # V10结果
    print("\n" + "="*80)
    print("【ZodiacSimpleSmart (v10.0) 结果】")
    print("="*80)
    
    v10_top3_hits, v10_top3_rate = calc_hit_rate(v10_results['top3'], actuals)
    v10_top4_hits, v10_top4_rate = calc_hit_rate(v10_results['top4'], actuals)
    v10_top5_hits, v10_top5_rate = calc_hit_rate(v10_results['top5'], actuals)
    
    print(f"TOP3 命中: {v10_top3_hits}/{test_periods} = {v10_top3_rate:.2f}%")
    print(f"TOP4 命中: {v10_top4_hits}/{test_periods} = {v10_top4_rate:.2f}%")
    print(f"TOP5 命中: {v10_top5_hits}/{test_periods} = {v10_top5_rate:.2f}%")
    
    # 统计模型使用情况
    v10_model_count = Counter(v10_results['model_used'])
    print(f"\n模型使用统计:")
    for model, count in v10_model_count.most_common():
        print(f"  {model}: {count}次 ({count/test_periods*100:.1f}%)")
    
    # 对比总结
    print("\n" + "="*80)
    print("【对比总结】")
    print("="*80)
    
    print(f"\nTOP3 对比:")
    print(f"  Ensemble: {ens_top3_rate:.2f}%")
    print(f"  v10:      {v10_top3_rate:.2f}%")
    print(f"  差异:     {ens_top3_rate - v10_top3_rate:+.2f}% {'🏆 Ensemble胜出' if ens_top3_rate > v10_top3_rate else '🏆 v10胜出' if v10_top3_rate > ens_top3_rate else '⚖️ 平局'}")
    
    print(f"\nTOP4 对比:")
    print(f"  Ensemble: {ens_top4_rate:.2f}%")
    print(f"  v10:      {v10_top4_rate:.2f}%")
    print(f"  差异:     {ens_top4_rate - v10_top4_rate:+.2f}% {'🏆 Ensemble胜出' if ens_top4_rate > v10_top4_rate else '🏆 v10胜出' if v10_top4_rate > ens_top4_rate else '⚖️ 平局'}")
    
    print(f"\nTOP5 对比:")
    print(f"  Ensemble: {ens_top5_rate:.2f}%")
    print(f"  v10:      {v10_top5_rate:.2f}%")
    print(f"  差异:     {ens_top5_rate - v10_top5_rate:+.2f}% {'🏆 Ensemble胜出' if ens_top5_rate > v10_top5_rate else '🏆 v10胜出' if v10_top5_rate > ens_top5_rate else '⚖️ 平局'}")
    
    # 计算综合得分（加权平均）
    ens_score = ens_top3_rate * 0.2 + ens_top4_rate * 0.3 + ens_top5_rate * 0.5
    v10_score = v10_top3_rate * 0.2 + v10_top4_rate * 0.3 + v10_top5_rate * 0.5
    
    print(f"\n综合得分（加权: TOP3=20%, TOP4=30%, TOP5=50%）:")
    print(f"  Ensemble: {ens_score:.2f}分")
    print(f"  v10:      {v10_score:.2f}分")
    print(f"  差异:     {ens_score - v10_score:+.2f}分")
    
    if ens_score > v10_score:
        print(f"\n🏆 总体胜出: EnsembleZodiacPredictor")
        print(f"   优势: {ens_score - v10_score:.2f}分")
    elif v10_score > ens_score:
        print(f"\n🏆 总体胜出: ZodiacSimpleSmart (v10.0)")
        print(f"   优势: {v10_score - ens_score:.2f}分")
    else:
        print(f"\n⚖️ 两个模型综合表现相当")
    
    print("\n" + "="*80)
    
    # 详细对比表格
    print("\n详细对比表（最近20期）:")
    print("-" * 100)
    print(f"{'期数':<6} {'日期':<12} {'实际':<8} {'Ensemble-TOP4':<25} {'v10-TOP4':<25} {'Ens':<4} {'v10':<4}")
    print("-" * 100)
    
    for i in range(max(0, test_periods-20), test_periods):
        idx = start_idx + i
        actual = actuals[i]
        date_str = df.iloc[idx]['date']
        
        ens_top4 = ensemble_results['top4'][i]
        v10_top4 = v10_results['top4'][i]
        
        ens_hit = "✓" if actual in ens_top4 else "✗"
        v10_hit = "✓" if actual in v10_top4 else "✗"
        
        ens_str = ','.join(ens_top4)
        v10_str = ','.join(v10_top4)
        
        print(f"第{idx+1:<4}期 {date_str:<12} {actual:<8} {ens_str:<25} {v10_str:<25} {ens_hit:<4} {v10_hit:<4}")
    
    print("-" * 100)
    
    return {
        'ensemble': {
            'top3': ens_top3_rate,
            'top4': ens_top4_rate,
            'top5': ens_top5_rate,
            'score': ens_score
        },
        'v10': {
            'top3': v10_top3_rate,
            'top4': v10_top4_rate,
            'top5': v10_top5_rate,
            'score': v10_score
        }
    }

if __name__ == '__main__':
    # 比较最近100期
    results = compare_models(test_periods=100)
