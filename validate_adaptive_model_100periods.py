"""
验证自适应预测器在最近100期的效果
对比三个模型：旧模型、重训练v2.0、自适应v3.0
"""

import pandas as pd
import numpy as np
from adaptive_zodiac_predictor import AdaptiveZodiacPredictor
from retrained_zodiac_predictor import RetrainedZodiacPredictor
from ensemble_zodiac_predictor import EnsembleZodiacPredictor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def validate_adaptive_model(csv_file, start_period, end_period):
    """验证自适应模型"""
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    predictor = AdaptiveZodiacPredictor(train_window=50, monitor_window=10)
    
    results = []
    hits = 0
    strategy_changes = []
    
    print(f"\n{'='*80}")
    print(f"自适应模型v3.0 - 第{start_period}-{end_period}期验证")
    print(f"{'='*80}\n")
    
    for i, period in enumerate(range(start_period, end_period + 1)):
        # 使用period之前的数据预测
        history_df = df.iloc[:period-1]
        animals = [str(a).strip() for a in history_df['animal'].values]
        
        # 预测
        pred_result = predictor.predict_from_history(animals, top_n=4, debug=False)
        top4_pred = pred_result['top4']
        
        # 记录策略变化
        if not strategy_changes or strategy_changes[-1]['strategy'] != pred_result['strategy']:
            strategy_changes.append({
                'period': period,
                'strategy': pred_result['strategy'],
                'pattern': pred_result['pattern']
            })
        
        # 获取实际结果
        actual_row = df.iloc[period-1]
        actual_zodiac = str(actual_row['animal']).strip()
        
        # 判断命中
        is_hit = actual_zodiac in top4_pred
        if is_hit:
            hits += 1
        
        # 更新性能监控
        predictor.update_performance(is_hit)
        
        results.append({
            'period': period,
            'date': actual_row['date'],
            'actual_zodiac': actual_zodiac,
            'predicted_top4': ', '.join(top4_pred),
            'is_hit': is_hit,
            'strategy': pred_result['strategy'],
            'pattern': pred_result['pattern']
        })
        
        # 每10期显示进度
        if (i + 1) % 10 == 0:
            current_rate = hits / (i + 1) * 100
            stats = predictor.get_performance_stats()
            recent_rate = stats['recent_rate'] if stats else 0
            print(f"已验证 {i+1}/100 期 | 整体命中率: {current_rate:.1f}% | "
                  f"最近{predictor.monitor_window}期: {recent_rate:.1f}%")
    
    hit_rate = hits / len(results) * 100
    
    print(f"\n{'='*80}")
    print(f"【验证结果】")
    print(f"命中: {hits}/{len(results)}")
    print(f"命中率: {hit_rate:.1f}%")
    print(f"{'='*80}\n")
    
    # 显示策略切换
    print("【策略自动切换记录】\n")
    for change in strategy_changes:
        print(f"第{change['period']:>3}期起: {change['strategy']} (模式: {change['pattern']})")
    
    return {
        'results': results,
        'hits': hits,
        'total': len(results),
        'hit_rate': hit_rate,
        'strategy_changes': strategy_changes
    }


def comprehensive_comparison(csv_file='data/lucky_numbers.csv'):
    """三模型全面对比"""
    print("="*80)
    print("三模型全面对比 - 最近100期（第289-388期）")
    print("="*80)
    
    # 模型1: 旧模型（之前已验证）
    print("\n【模型1: 集成预测器（旧模型）】")
    print("使用之前验证结果...")
    old_hits = 42
    old_total = 100
    old_rate = 42.0
    
    # 模型2: 重训练v2.0（之前已验证）
    print("\n【模型2: 重训练v2.0】")
    print("使用之前验证结果...")
    retrained_hits = 47
    retrained_total = 100
    retrained_rate = 47.0
    
    # 模型3: 自适应v3.0（新验证）
    print("\n【模型3: 自适应v3.0（带动态重训练）】")
    adaptive_result = validate_adaptive_model(csv_file, 289, 388)
    
    # 综合对比
    print("\n" + "="*80)
    print("【综合对比总结】")
    print("="*80 + "\n")
    
    models_data = [
        {'name': '旧模型(集成预测器)', 'hits': old_hits, 'total': old_total, 'rate': old_rate},
        {'name': '重训练v2.0', 'hits': retrained_hits, 'total': retrained_total, 'rate': retrained_rate},
        {'name': '自适应v3.0', 'hits': adaptive_result['hits'], 'total': adaptive_result['total'], 'rate': adaptive_result['hit_rate']}
    ]
    
    print(f"{'模型':<20} {'命中':<15} {'命中率':<12} {'vs理论值':<12} {'盈利(固定1倍)'}")
    print("-"*80)
    
    single_bet = 16
    single_win = 46
    
    for model in models_data:
        profit = model['hits'] * single_win - model['total'] * single_bet
        vs_theory = model['rate'] - 33.3
        print(f"{model['name']:<20} {model['hits']}/{model['total']:<10} "
              f"{model['rate']:.1f}%{'':<7} {vs_theory:>+5.1f}%{'':<5} {profit:>+5}元")
    
    # 找出最佳模型
    best_model = max(models_data, key=lambda x: x['rate'])
    print(f"\n🏆 最佳模型: {best_model['name']} (命中率 {best_model['rate']:.1f}%)")
    
    # 分段对比（自适应模型）
    print("\n【自适应模型分段分析】（每20期）\n")
    
    results = adaptive_result['results']
    for i in range(0, 100, 20):
        segment = results[i:i+20]
        hits = sum(1 for r in segment if r['is_hit'])
        rate = hits / len(segment) * 100
        start_p = segment[0]['period']
        end_p = segment[-1]['period']
        
        bar = '█' * int(rate / 5)
        print(f"第{start_p:>3}-{end_p:<3}期: {hits:>2}/{len(segment)} ({rate:>5.1f}%) {bar}")
    
    # 可视化
    visualize_three_models(old_rate, retrained_rate, adaptive_result['hit_rate'])
    
    # 保存结果
    results_df = pd.DataFrame(adaptive_result['results'])
    results_df.to_csv('adaptive_model_validation_100periods.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 自适应模型详细结果已保存: adaptive_model_validation_100periods.csv")
    
    # 生成报告
    save_comprehensive_report(models_data, adaptive_result)
    
    return models_data


def visualize_three_models(old_rate, retrained_rate, adaptive_rate):
    """三模型可视化对比"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['旧模型\n(集成预测器)', '重训练v2.0\n(固定策略)', '自适应v3.0\n(动态策略)']
    rates = [old_rate, retrained_rate, adaptive_rate]
    colors = ['#ff6b6b', '#4ecdc4', '#95e1d3']
    
    bars = ax.bar(models, rates, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # 添加理论值线
    ax.axhline(y=33.3, color='gray', linestyle='--', linewidth=2, label='理论值 33.3%')
    
    # 添加数值标签
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # 显示vs理论值
        vs_theory = rate - 33.3
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{vs_theory:+.1f}%',
                ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    
    ax.set_ylabel('命中率 (%)', fontsize=12, fontweight='bold')
    ax.set_title('三模型命中率对比（100期）', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(rates) * 1.15)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('three_models_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 三模型对比图表已保存: three_models_comparison.png")
    plt.close()


def save_comprehensive_report(models_data, adaptive_result):
    """保存综合报告"""
    report = []
    report.append("="*80)
    report.append("生肖预测器三模型全面对比报告")
    report.append("="*80)
    report.append("")
    report.append("验证期数: 第289-388期（共100期）")
    report.append("日期范围: 2025年10月至2026年2月")
    report.append("")
    report.append("【模型对比】")
    report.append("")
    
    for model in models_data:
        profit = model['hits'] * 46 - model['total'] * 16
        roi = profit / (model['total'] * 16) * 100
        report.append(f"{model['name']}:")
        report.append(f"  命中率: {model['rate']:.1f}% ({model['hits']}/{model['total']})")
        report.append(f"  vs理论值: {model['rate'] - 33.3:+.1f}%")
        report.append(f"  盈利: {profit:+d}元 (ROI: {roi:.1f}%)")
        report.append("")
    
    best = max(models_data, key=lambda x: x['rate'])
    report.append(f"🏆 最佳模型: {best['name']} (命中率 {best['rate']:.1f}%)")
    report.append("")
    
    report.append("【自适应模型特色】")
    report.append("")
    report.append("1. 动态训练窗口: 使用最近50期数据，快速适应变化")
    report.append("2. 自动策略切换: 根据数据模式自动调整权重")
    report.append("3. 实时监控: 追踪最近10期表现，及时预警")
    report.append("")
    
    report.append("策略切换记录:")
    for change in adaptive_result['strategy_changes']:
        report.append(f"  第{change['period']}期起: {change['strategy']} (模式: {change['pattern']})")
    
    report.append("")
    report.append("【结论】")
    report.append("")
    
    if best['name'] == '自适应v3.0':
        report.append("✅ 自适应模型表现最佳，成功验证动态重训练机制有效性")
        report.append("✅ 推荐在实战中使用自适应v3.0模型")
    else:
        report.append(f"⚠️  {best['name']}表现最佳")
        report.append(f"ℹ️  自适应模型命中率:{adaptive_result['hit_rate']:.1f}%，仍在优化中")
    
    report.append("")
    report.append("="*80)
    
    with open('three_models_comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"📄 综合对比报告已保存: three_models_comparison_report.txt")


if __name__ == '__main__':
    comprehensive_comparison()
