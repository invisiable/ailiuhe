"""
验证混合自适应预测器（v3.1）在100期的表现
最终四模型对比
"""

import pandas as pd
import numpy as np
from hybrid_adaptive_predictor import HybridAdaptivePredictor


def validate_hybrid_model(csv_file, start_period, end_period):
    """验证混合模型"""
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    predictor = HybridAdaptivePredictor()
    
    results = []
    hits = 0
    mode_switches = []
    
    print(f"\n{'='*80}")
    print(f"混合自适应模型v3.1 - 第{start_period}-{end_period}期验证")
    print(f"{'='*80}\n")
    
    prev_mode = None
    
    for i, period in enumerate(range(start_period, end_period + 1)):
        # 使用period之前的数据预测
        history_df = df.iloc[:period-1]
        animals = [str(a).strip() for a in history_df['animal'].values]
        
        # 预测
        pred_result = predictor.predict_from_history(animals, top_n=4, debug=False)
        top4_pred = pred_result['top4']
        current_mode = pred_result['mode']
        
        # 记录模式切换
        if prev_mode and current_mode != prev_mode:
            mode_switches.append({
                'period': period,
                'from': prev_mode,
                'to': current_mode,
                'predictor': pred_result['predictor']
            })
        prev_mode = current_mode
        
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
            'mode': current_mode,
            'predictor': pred_result['predictor']
        })
        
        # 每10期显示进度
        if (i + 1) % 10 == 0:
            current_rate = hits / (i + 1) * 100
            status = predictor.get_status()
            recent_rate = status['recent_rate'] if status.get('recent_total') else 0
            print(f"已验证 {i+1}/100 期 | 整体: {current_rate:.1f}% | "
                  f"最近10期: {recent_rate:.1f}% | 模式: {current_mode}")
    
    hit_rate = hits / len(results) * 100
    
    print(f"\n{'='*80}")
    print(f"【验证结果】")
    print(f"命中: {hits}/{len(results)}")
    print(f"命中率: {hit_rate:.1f}%")
    print(f"{'='*80}\n")
    
    # 显示模式切换
    print("【模式切换记录】\n")
    if mode_switches:
        for switch in mode_switches:
            print(f"第{switch['period']:>3}期: {switch['from']} → {switch['to']}")
            print(f"       切换为: {switch['predictor']}")
    else:
        print("全程使用同一模式，无切换")
    
    # 分模式统计
    print("\n【分模式统计】\n")
    stable_results = [r for r in results if r['mode'] == 'stable']
    adaptive_results = [r for r in results if r['mode'] == 'adaptive']
    
    if stable_results:
        stable_hits = sum(1 for r in stable_results if r['is_hit'])
        stable_rate = stable_hits / len(stable_results) * 100
        print(f"稳定模式: {stable_hits}/{len(stable_results)} ({stable_rate:.1f}%)")
    
    if adaptive_results:
        adaptive_hits = sum(1 for r in adaptive_results if r['is_hit'])
        adaptive_rate = adaptive_hits / len(adaptive_results) * 100
        print(f"自适应模式: {adaptive_hits}/{len(adaptive_results)} ({adaptive_rate:.1f}%)")
    
    return {
        'results': results,
        'hits': hits,
        'total': len(results),
        'hit_rate': hit_rate,
        'mode_switches': mode_switches,
        'stable_stats': {'hits': stable_hits, 'total': len(stable_results), 'rate': stable_rate} if stable_results else None,
        'adaptive_stats': {'hits': adaptive_hits, 'total': len(adaptive_results), 'rate': adaptive_rate} if adaptive_results else None
    }


def final_comparison():
    """最终四模型对比"""
    print("="*80)
    print("最终四模型对比 - 最近100期（第289-388期）")
    print("="*80)
    
    # 前三个模型的结果（已知）
    models_summary = [
        {'name': '旧模型(集成预测器)', 'hits': 42, 'total': 100, 'rate': 42.0},
        {'name': '重训练v2.0', 'hits': 47, 'total': 100, 'rate': 47.0},
        {'name': '自适应v3.0', 'hits': 27, 'total': 100, 'rate': 27.0},
    ]
    
    # 验证混合模型
    print("\n【验证混合自适应模型v3.1】")
    hybrid_result = validate_hybrid_model('data/lucky_numbers.csv', 289, 388)
    
    models_summary.append({
        'name': '混合自适应v3.1',
        'hits': hybrid_result['hits'],
        'total': hybrid_result['total'],
        'rate': hybrid_result['hit_rate']
    })
    
    # 综合对比
    print("\n" + "="*80)
    print("【最终对比】")
    print("="*80 + "\n")
    
    print(f"{'模型':<20} {'命中':<15} {'命中率':<12} {'vs理论':<12} {'盈利'}")
    print("-"*75)
    
    single_bet = 16
    single_win = 46
    
    for model in models_summary:
        profit = model['hits'] * single_win - model['total'] * single_bet
        roi = profit / (model['total'] * single_bet) * 100
        vs_theory = model['rate'] - 33.3
        print(f"{model['name']:<20} {model['hits']}/{model['total']:<10} "
              f"{model['rate']:.1f}%{'':<7} {vs_theory:>+5.1f}%{'':<5} {profit:>+5}元")
    
    # 找出最佳
    best = max(models_summary, key=lambda x: x['rate'])
    print(f"\n🏆 最佳模型: {best['name']} (命中率 {best['rate']:.1f}%)")
    
    # 混合模型详细分析
    if hybrid_result['stable_stats'] and hybrid_result['adaptive_stats']:
        print("\n【混合模型详细分析】\n")
        stable = hybrid_result['stable_stats']
        adaptive = hybrid_result['adaptive_stats']
        
        print(f"稳定模式期数: {stable['total']} ({stable['total']/100*100:.0f}%)")
        print(f"  命中率: {stable['rate']:.1f}% ({stable['hits']}/{stable['total']})")
        
        print(f"\n自适应模式期数: {adaptive['total']} ({adaptive['total']/100*100:.0f}%)")
        print(f"  命中率: {adaptive['rate']:.1f}% ({adaptive['hits']}/{adaptive['total']})")
        
        print(f"\n模式切换次数: {len(hybrid_result['mode_switches'])}次")
    
    # 保存结果
    results_df = pd.DataFrame(hybrid_result['results'])
    results_df.to_csv('hybrid_adaptive_validation_100periods.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 混合模型详细结果已保存: hybrid_adaptive_validation_100periods.csv")
    
    # 生成最终报告
    save_final_report(models_summary, hybrid_result)
    
    return models_summary


def save_final_report(models_summary, hybrid_result):
    """生成最终报告"""
    report = []
    report.append("="*80)
    report.append("生肖预测器最终报告 - 四模型对比")
    report.append("="*80)
    report.append("")
    report.append("验证期数: 第289-388期（共100期）")
    report.append("日期范围: 2025年10月至2026年2月")
    report.append("")
    report.append("【模型对比】")
    report.append("")
    
    for model in models_summary:
        profit = model['hits'] * 46 - model['total'] * 16
        roi = profit / (model['total'] * 16) * 100
        report.append(f"{model['name']}:")
        report.append(f"  命中率: {model['rate']:.1f}% ({model['hits']}/{model['total']})")
        report.append(f"  vs理论值: {model['rate'] - 33.3:+.1f}%")
        report.append(f"  盈利: {profit:+d}元 (ROI: {roi:.1f}%)")
        report.append("")
    
    best = max(models_summary, key=lambda x: x['rate'])
    report.append(f"🏆 最佳模型: {best['name']} (命中率 {best['rate']:.1f}%)")
    report.append("")
    
    if hybrid_result['stable_stats'] and hybrid_result['adaptive_stats']:
        report.append("【混合自适应模型v3.1特性】")
        report.append("")
        report.append("核心机制:")
        report.append("1. 默认使用重训练v2.0的稳定策略（47%命中率）")
        report.append("2. 检测到数据模式剧变时，自动切换到自适应模式")
        report.append("3. 切换冷却机制，避免频繁切换导致不稳定")
        report.append("")
        
        stable = hybrid_result['stable_stats']
        adaptive = hybrid_result['adaptive_stats']
        
        report.append(f"稳定模式: {stable['total']}期，命中率{stable['rate']:.1f}%")
        report.append(f"自适应模式: {adaptive['total']}期，命中率{adaptive['rate']:.1f}%")
        report.append(f"模式切换: {len(hybrid_result['mode_switches'])}次")
        report.append("")
        
        if hybrid_result['mode_switches']:
            report.append("切换记录:")
            for switch in hybrid_result['mode_switches']:
                report.append(f"  第{switch['period']}期: {switch['from']} → {switch['to']}")
    
    report.append("")
    report.append("【最终建议】")
    report.append("")
    
    if best['name'] == '混合自适应v3.1':
        report.append("✅ 混合自适应模型表现最佳，推荐使用")
    elif best['name'] == '重训练v2.0':
        report.append("✅ 重训练v2.0模型稳定可靠，推荐使用")
        report.append("💡 混合自适应v3.1可作为备选，在数据剧变时启用")
    else:
        report.append(f"✅ {best['name']}表现最佳")
    
    report.append("")
    report.append("实战策略:")
    report.append("1. 使用表现最佳的模型进行预测")
    report.append("2. 固定1倍投注（4个生肖×4元=16元/期）")
    report.append("3. 每10期回顾命中率，低于30%时考虑更换模型")
    report.append("4. 严格止损：连续5期未中，暂停投注并重新分析")
    report.append("")
    report.append("="*80)
    
    with open('final_four_models_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"📄 最终报告已保存: final_four_models_report.txt")


if __name__ == '__main__':
    final_comparison()
