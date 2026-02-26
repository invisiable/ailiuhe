"""
验证重训练模型在最近100期的效果
更全面地评估模型稳定性和长期表现
"""

import pandas as pd
import numpy as np
from retrained_zodiac_predictor import RetrainedZodiacPredictor
from ensemble_zodiac_predictor import EnsembleZodiacPredictor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def validate_model_100periods(predictor, csv_file, start_period, end_period, model_name):
    """验证模型在100期的命中率"""
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    
    results = []
    hits = 0
    
    print(f"\n{'='*80}")
    print(f"{model_name} - 第{start_period}-{end_period}期验证（共{end_period-start_period+1}期）")
    print(f"{'='*80}\n")
    
    # 分段显示，避免输出过长
    display_periods = list(range(start_period, end_period + 1))
    
    for i, period in enumerate(display_periods):
        # 使用period之前的数据预测
        history_df = df.iloc[:period-1]
        animals = [str(a).strip() for a in history_df['animal'].values]
        
        # 预测
        pred_result = predictor.predict_from_history(animals, top_n=4, debug=False)
        top4_pred = pred_result['top4']
        
        # 获取实际结果
        actual_row = df.iloc[period-1]
        actual_zodiac = str(actual_row['animal']).strip()
        actual_date = actual_row['date']
        actual_number = actual_row['number']
        
        # 判断命中
        is_hit = actual_zodiac in top4_pred
        if is_hit:
            hits += 1
        
        results.append({
            'period': period,
            'date': actual_date,
            'actual_number': actual_number,
            'actual_zodiac': actual_zodiac,
            'predicted_top4': ', '.join(top4_pred),
            'is_hit': is_hit
        })
        
        # 每10期显示一次进度
        if (i + 1) % 10 == 0:
            current_hit_rate = hits / (i + 1) * 100
            print(f"已验证 {i+1}/{len(display_periods)} 期，当前命中率: {current_hit_rate:.1f}%")
    
    hit_rate = hits / len(results) * 100
    
    print(f"\n{'='*80}")
    print(f"【最终验证结果】")
    print(f"命中: {hits}/{len(results)}")
    print(f"命中率: {hit_rate:.1f}%")
    print(f"理论命中率: 33.3% (4/12生肖)")
    print(f"与理论值差距: {hit_rate - 33.3:+.1f}%")
    print(f"{'='*80}\n")
    
    return {
        'model_name': model_name,
        'results': results,
        'hits': hits,
        'total': len(results),
        'hit_rate': hit_rate
    }


def analyze_by_segments(results, segment_size=20):
    """分段分析命中率趋势"""
    print(f"\n【分段分析】（每{segment_size}期）\n")
    
    segments = []
    for i in range(0, len(results), segment_size):
        segment = results[i:i+segment_size]
        hits = sum(1 for r in segment if r['is_hit'])
        hit_rate = hits / len(segment) * 100
        
        start_period = segment[0]['period']
        end_period = segment[-1]['period']
        
        segments.append({
            'range': f"{start_period}-{end_period}",
            'hits': hits,
            'total': len(segment),
            'hit_rate': hit_rate
        })
        
        bar = '█' * int(hit_rate / 5)
        print(f"第{start_period:>3}-{end_period:<3}期: {hits:>2}/{len(segment)} ({hit_rate:>5.1f}%) {bar}")
    
    return segments


def compare_models_100periods(csv_file='data/lucky_numbers.csv'):
    """对比新旧模型在100期上的表现"""
    print("="*80)
    print("新旧模型对比验证 - 最近100期（第289-388期）")
    print("="*80)
    
    # 旧模型
    print("\n【开始验证旧模型】")
    old_predictor = EnsembleZodiacPredictor()
    old_result = validate_model_100periods(old_predictor, csv_file, 289, 388, '旧模型(集成预测器)')
    
    # 分段分析
    print("\n旧模型分段分析:")
    old_segments = analyze_by_segments(old_result['results'], segment_size=20)
    
    # 新模型
    print("\n" + "="*80)
    print("【开始验证新模型】")
    new_predictor = RetrainedZodiacPredictor()
    new_result = validate_model_100periods(new_predictor, csv_file, 289, 388, '新模型(重训练v2.0)')
    
    # 分段分析
    print("\n新模型分段分析:")
    new_segments = analyze_by_segments(new_result['results'], segment_size=20)
    
    # 对比总结
    print("\n" + "="*80)
    print("【对比总结】")
    print("="*80)
    
    print(f"\n旧模型(集成预测器):")
    print(f"  命中率: {old_result['hit_rate']:.1f}% ({old_result['hits']}/{old_result['total']})")
    print(f"  与理论值差距: {old_result['hit_rate'] - 33.3:+.1f}%")
    
    print(f"\n新模型(重训练v2.0):")
    print(f"  命中率: {new_result['hit_rate']:.1f}% ({new_result['hits']}/{new_result['total']})")
    print(f"  与理论值差距: {new_result['hit_rate'] - 33.3:+.1f}%")
    
    improvement = new_result['hit_rate'] - old_result['hit_rate']
    print(f"\n提升幅度: {improvement:+.1f}%")
    
    # 固定1倍投注收益对比
    print(f"\n【固定1倍投注效果对比（100期）】")
    single_bet = 16  # 4个生肖×4元
    single_win = 46  # 单次中奖金额
    total_periods = old_result['total']
    
    # 旧模型
    old_total_bet = single_bet * total_periods
    old_total_win = old_result['hits'] * single_win
    old_profit = old_total_win - old_total_bet
    old_roi = (old_profit / old_total_bet) * 100
    
    print(f"\n旧模型:")
    print(f"  总投注: {old_total_bet}元")
    print(f"  总中奖: {old_total_win}元")
    print(f"  净盈利: {old_profit:+d}元")
    print(f"  ROI: {old_roi:.1f}%")
    
    # 新模型
    new_total_bet = single_bet * total_periods
    new_total_win = new_result['hits'] * single_win
    new_profit = new_total_win - new_total_bet
    new_roi = (new_profit / new_total_bet) * 100
    
    print(f"\n新模型:")
    print(f"  总投注: {new_total_bet}元")
    print(f"  总中奖: {new_total_win}元")
    print(f"  净盈利: {new_profit:+d}元")
    print(f"  ROI: {new_roi:.1f}%")
    
    profit_improvement = new_profit - old_profit
    print(f"\n收益提升: {profit_improvement:+d}元")
    
    # 可视化对比
    visualize_comparison(old_segments, new_segments, old_result, new_result)
    
    # 保存详细结果
    results_df = pd.DataFrame(new_result['results'])
    results_df.to_csv('retrained_model_validation_100periods.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 新模型详细结果已保存: retrained_model_validation_100periods.csv")
    
    # 保存对比报告
    save_comparison_report(old_result, new_result, old_segments, new_segments)
    
    return {
        'old_model': old_result,
        'new_model': new_result,
        'improvement': improvement
    }


def visualize_comparison(old_segments, new_segments, old_result, new_result):
    """可视化对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1: 分段命中率对比
    ax1 = axes[0]
    segment_labels = [s['range'] for s in old_segments]
    old_rates = [s['hit_rate'] for s in old_segments]
    new_rates = [s['hit_rate'] for s in new_segments]
    
    x = np.arange(len(segment_labels))
    width = 0.35
    
    ax1.bar(x - width/2, old_rates, width, label='旧模型', color='#ff6b6b', alpha=0.7)
    ax1.bar(x + width/2, new_rates, width, label='新模型', color='#4ecdc4', alpha=0.7)
    ax1.axhline(y=33.3, color='gray', linestyle='--', label='理论值33.3%')
    
    ax1.set_xlabel('期数范围', fontsize=11)
    ax1.set_ylabel('命中率 (%)', fontsize=11)
    ax1.set_title('分段命中率对比（每20期）', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(segment_labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 图2: 整体对比
    ax2 = axes[1]
    models = ['旧模型\n(集成预测器)', '新模型\n(重训练v2.0)', '理论值']
    rates = [old_result['hit_rate'], new_result['hit_rate'], 33.3]
    colors = ['#ff6b6b', '#4ecdc4', '#95a5a6']
    
    bars = ax2.bar(models, rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('命中率 (%)', fontsize=11)
    ax2.set_title('整体命中率对比（100期）', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(rates) * 1.2)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('retrained_model_comparison_100periods.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 对比图表已保存: retrained_model_comparison_100periods.png")
    plt.close()


def save_comparison_report(old_result, new_result, old_segments, new_segments):
    """保存对比报告"""
    report = []
    report.append("="*80)
    report.append("生肖预测器重训练效果报告 - 100期验证")
    report.append("="*80)
    report.append("")
    report.append(f"验证期数: 第289-388期（共100期）")
    report.append(f"日期范围: 2025年10月至2026年2月")
    report.append("")
    report.append("【整体对比】")
    report.append("")
    report.append(f"旧模型(集成预测器):")
    report.append(f"  命中率: {old_result['hit_rate']:.1f}% ({old_result['hits']}/{old_result['total']})")
    report.append(f"  与理论值(33.3%)差距: {old_result['hit_rate'] - 33.3:+.1f}%")
    report.append("")
    report.append(f"新模型(重训练v2.0):")
    report.append(f"  命中率: {new_result['hit_rate']:.1f}% ({new_result['hits']}/{new_result['total']})")
    report.append(f"  与理论值(33.3%)差距: {new_result['hit_rate'] - 33.3:+.1f}%")
    report.append("")
    report.append(f"提升幅度: {new_result['hit_rate'] - old_result['hit_rate']:+.1f}%")
    report.append("")
    report.append("【盈利对比】（固定1倍投注）")
    report.append("")
    
    single_bet = 16
    single_win = 46
    
    old_profit = old_result['hits'] * single_win - old_result['total'] * single_bet
    new_profit = new_result['hits'] * single_win - new_result['total'] * single_bet
    
    report.append(f"旧模型: {old_profit:+d}元")
    report.append(f"新模型: {new_profit:+d}元")
    report.append(f"收益提升: {new_profit - old_profit:+d}元")
    report.append("")
    report.append("【分段分析】（每20期）")
    report.append("")
    report.append(f"{'期数范围':<15} {'旧模型':<12} {'新模型':<12} {'差距'}")
    report.append("-"*50)
    
    for old_seg, new_seg in zip(old_segments, new_segments):
        diff = new_seg['hit_rate'] - old_seg['hit_rate']
        report.append(f"{old_seg['range']:<15} {old_seg['hit_rate']:>5.1f}%      {new_seg['hit_rate']:>5.1f}%      {diff:+5.1f}%")
    
    report.append("")
    report.append("【结论与建议】")
    report.append("")
    
    if new_result['hit_rate'] > 33.3:
        report.append("✅ 新模型命中率超过理论值，表现优秀")
    elif new_result['hit_rate'] > old_result['hit_rate']:
        report.append("✅ 新模型相比旧模型有所改进")
        if new_result['hit_rate'] < 33.3:
            report.append("⚠️  但仍低于理论值，建议:")
            report.append("   - 继续优化权重配置")
            report.append("   - 增加更多特征工程")
            report.append("   - 考虑引入机器学习模型")
    else:
        report.append("❌ 新模型表现不如旧模型")
        report.append("   建议回退到旧模型或进一步调整")
    
    report.append("")
    report.append("="*80)
    
    with open('retrained_model_report_100periods.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"📄 对比报告已保存: retrained_model_report_100periods.txt")


if __name__ == '__main__':
    compare_models_100periods()
