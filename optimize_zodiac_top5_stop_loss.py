"""
生肖TOP5止损策略优化测试
目标：通过调整止损参数（连续N期失败后暂停N期），将ROI提升到50%

测试策略：
- 连续N期失败 → 暂停投注N期 → 自动恢复投注
- 测试不同的N值：2, 3, 4, 5, 6, 7, 8, 9, 10
- 基础倍投：斐波那契数列 (1,1,2,3,5,8,13...)
"""

import pandas as pd
import numpy as np
from zodiac_simple_smart import ZodiacSimpleSmart


def calculate_stop_loss_strategy(hit_records, n_threshold, base_bet=20, win_amount=47):
    """
    计算止损策略结果
    
    Args:
        hit_records: 命中记录列表 (True/False)
        n_threshold: N值，连续N期失败后暂停N期
        base_bet: 基础投注金额
        win_amount: 命中奖励金额
    
    Returns:
        包含统计指标的字典
    """
    # 斐波那契数列
    def fibonacci_multiplier(losses):
        fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        if losses < len(fib):
            return fib[losses]
        return fib[-1]
    
    total_profit = 0
    total_investment = 0
    consecutive_losses = 0
    max_consecutive_losses = 0
    max_bet = base_bet
    max_drawdown = 0
    peak_balance = 0
    
    # 止损相关
    is_paused = False
    pause_remaining = 0  # 剩余暂停期数
    paused_periods = 0  # 总暂停期数
    actual_betting_periods = 0  # 实际投注期数
    hits = 0
    
    for i, hit in enumerate(hit_records):
        if is_paused:
            # 暂停期间不投注
            pause_remaining -= 1
            paused_periods += 1
            
            if pause_remaining <= 0:
                # 暂停期满，恢复投注，重置倍数
                is_paused = False
                consecutive_losses = 0
            
            # 更新最大回撤
            drawdown = peak_balance - total_profit
            max_drawdown = max(max_drawdown, drawdown)
            
            continue
        
        # 正常投注期
        multiplier = fibonacci_multiplier(consecutive_losses)
        current_bet = base_bet * multiplier
        total_investment += current_bet
        actual_betting_periods += 1
        max_bet = max(max_bet, current_bet)
        
        if hit:
            # 命中
            profit = win_amount * multiplier - current_bet
            total_profit += profit
            consecutive_losses = 0
            hits += 1
        else:
            # 未中
            total_profit -= current_bet
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            
            # 检查是否触发止损
            if consecutive_losses >= n_threshold:
                is_paused = True
                pause_remaining = n_threshold
                consecutive_losses = 0  # 重置连败计数
        
        # 更新最大回撤
        peak_balance = max(peak_balance, total_profit)
        drawdown = peak_balance - total_profit
        max_drawdown = max(max_drawdown, drawdown)
    
    roi = (total_profit / total_investment * 100) if total_investment > 0 else 0
    hit_rate = (hits / actual_betting_periods) if actual_betting_periods > 0 else 0
    
    return {
        'total_profit': total_profit,
        'total_investment': total_investment,
        'roi': roi,
        'hit_rate': hit_rate * 100,
        'hits': hits,
        'actual_betting_periods': actual_betting_periods,
        'paused_periods': paused_periods,
        'max_consecutive_losses': max_consecutive_losses,
        'max_bet': max_bet,
        'max_drawdown': max_drawdown,
    }


def test_stop_loss_strategies():
    """测试不同N值的止损策略"""
    print("="*100)
    print("🐉 生肖TOP5止损策略优化测试")
    print("="*100)
    print("目标：通过调整止损参数N，将ROI提升到50%")
    print("策略：连续N期失败 → 暂停投注N期 → 自动恢复投注\n")
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"✅ 数据加载完成: {len(df)}期\n")
    
    # 测试最近200期
    test_periods = min(200, len(df))
    start_idx = len(df) - test_periods
    
    print(f"测试期数: {test_periods}期 (第{start_idx+1}期 ~ 第{len(df)}期)")
    print(f"日期范围: {df.iloc[start_idx]['date']} ~ {df.iloc[-1]['date']}\n")
    
    # 使用v10.0生成预测
    print("🔄 生成TOP5预测...")
    predictor = ZodiacSimpleSmart()
    predictions_top5 = []
    hit_records = []
    
    for i in range(start_idx, len(df)):
        train_animals = df['animal'].iloc[:i].tolist()
        result = predictor.predict_from_history(train_animals, top_n=5, debug=False)
        top5 = result['top5']
        predictions_top5.append(top5)
        
        actual = df.iloc[i]['animal']
        hit = actual in top5
        hit_records.append(hit)
        
        if (i - start_idx + 1) % 50 == 0:
            print(f"  已处理 {i - start_idx + 1}/{test_periods} 期...")
    
    hits = sum(hit_records)
    hit_rate = hits / len(hit_records)
    print(f"✅ 预测完成！命中率: {hit_rate*100:.2f}% ({hits}/{len(hit_records)})\n")
    
    # 基准策略：纯斐波那契倍投（无止损）
    print("="*100)
    print("【基准策略】纯斐波那契倍投（无止损）")
    print("="*100)
    
    baseline_result = calculate_stop_loss_strategy(hit_records, n_threshold=999, base_bet=20, win_amount=47)
    
    print(f"总投入: {baseline_result['total_investment']:.2f}元")
    print(f"总收益: {baseline_result['total_profit']:+.2f}元")
    print(f"ROI: {baseline_result['roi']:+.2f}% ⭐")
    print(f"命中率: {baseline_result['hit_rate']:.2f}%")
    print(f"实际投注期数: {baseline_result['actual_betting_periods']}期")
    print(f"最大连败: {baseline_result['max_consecutive_losses']}期")
    print(f"最大单期投入: {baseline_result['max_bet']:.2f}元")
    print(f"最大回撤: {baseline_result['max_drawdown']:.2f}元\n")
    
    # 测试不同的N值
    print("="*100)
    print("【止损策略测试】连续N期失败→暂停N期→自动恢复")
    print("="*100)
    print()
    
    n_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    results = []
    
    for n in n_values:
        result = calculate_stop_loss_strategy(hit_records, n_threshold=n, base_bet=20, win_amount=47)
        results.append((n, result))
        
        print(f"【N={n}】连续{n}期失败→暂停{n}期")
        print(f"  总投入: {result['total_investment']:.2f}元  |  总收益: {result['total_profit']:+.2f}元  |  ROI: {result['roi']:+.2f}%", end="")
        
        # 标记是否达到50%
        if result['roi'] >= 50:
            print(" ⭐⭐⭐ 达标！")
        elif result['roi'] >= baseline_result['roi']:
            print(" ⭐ 优于基准")
        else:
            print()
        
        print(f"  命中率: {result['hit_rate']:.2f}%")
        print(f"  实际投注: {result['actual_betting_periods']}期  |  暂停: {result['paused_periods']}期  |  暂停率: {result['paused_periods']/test_periods*100:.1f}%")
        print(f"  最大连败: {result['max_consecutive_losses']}期  |  最大投入: {result['max_bet']:.2f}元  |  最大回撤: {result['max_drawdown']:.2f}元")
        print(f"  vs基准: ROI {result['roi'] - baseline_result['roi']:+.2f}%  |  收益 {result['total_profit'] - baseline_result['total_profit']:+.2f}元")
        print()
    
    # 按ROI排序
    print("="*100)
    print("【结果排名】按ROI排序")
    print("="*100)
    
    # 添加基准策略
    all_results = [('基准', baseline_result)] + results
    sorted_results = sorted(all_results, key=lambda x: x[1]['roi'], reverse=True)
    
    print(f"{'排名':<6} {'策略':<20} {'ROI':<12} {'收益':<15} {'投入':<15} {'暂停率':<12} {'最大回撤':<15}")
    print("-" * 100)
    
    for i, (strategy, result) in enumerate(sorted_results):
        rank = f"#{i+1}"
        pause_rate = result['paused_periods'] / test_periods * 100 if 'paused_periods' in result else 0
        marker = " ⭐⭐⭐" if result['roi'] >= 50 else " ⭐" if i == 0 else ""
        
        print(f"{rank:<6} {strategy:<20} {result['roi']:>+6.2f}%{marker:<6} {result['total_profit']:>+10.2f}元 "
              f"{result['total_investment']:>10.2f}元 {pause_rate:>8.1f}% {result['max_drawdown']:>10.2f}元")
    
    print()
    
    # 结论
    print("="*100)
    print("【分析结论】")
    print("="*100)
    
    best_strategy = sorted_results[0]
    best_name = best_strategy[0]
    best_result = best_strategy[1]
    
    print(f"🏆 最优策略: {best_name}")
    print(f"   ROI: {best_result['roi']:+.2f}%")
    print(f"   总收益: {best_result['total_profit']:+.2f}元")
    print(f"   vs基准: {best_result['roi'] - baseline_result['roi']:+.2f}%")
    print()
    
    if best_result['roi'] >= 50:
        print("✅ 成功达到50% ROI目标！")
    else:
        print(f"⚠️ 未能达到50% ROI目标")
        print(f"   当前最优: {best_result['roi']:.2f}%")
        print(f"   差距: {50 - best_result['roi']:.2f}%")
    
    print()
    print("💡 发现：")
    
    # 判断止损是否有效
    improvement_count = sum(1 for _, r in results if r['roi'] > baseline_result['roi'])
    
    if improvement_count == 0:
        print("   • 所有止损策略均低于基准策略，止损机制不适用于该场景")
        print("   • 建议：保持纯斐波那契倍投，不使用止损")
    elif improvement_count < len(results) / 2:
        print("   • 大部分止损策略低于基准，止损收益有限")
        print("   • 原因：暂停期间错过了恢复机会，降低了整体收益")
    else:
        print("   • 部分止损策略优于基准，N值选择很重要")
        print(f"   • 最优N值: {best_name}")
    
    # 分析ROI瓶颈
    print()
    print("📊 ROI提升瓶颈分析：")
    print(f"   • 当前命中率: {hit_rate*100:.2f}%")
    print(f"   • 理论命中率: 41.67% (5/12生肖)")
    print(f"   • 盈亏比: 1:1.35 (赢27元 vs 输20元)")
    print()
    
    # 计算达到50% ROI需要的命中率
    # 假设斐波那契倍投，简化计算：ROI ≈ (命中率 * 1.35 - (1-命中率)) / 1
    # 50% ROI需要: 0.5 = hit_rate * 1.35 - 1 + hit_rate
    # 0.5 = 2.35 * hit_rate - 1
    # hit_rate = 1.5 / 2.35 = 63.8%
    required_hit_rate = (1 + 0.5) / (1 + 27/20) * 100
    print(f"   要达到50% ROI，需要命中率: ~{required_hit_rate:.1f}%")
    print(f"   当前差距: {required_hit_rate - hit_rate*100:.1f}%")
    print()
    print("   结论：止损策略无法突破命中率瓶颈")
    print("        提升ROI需要从模型预测准确率入手，而非投注策略")
    
    print()
    print("="*100)
    print("测试完成！")
    print("="*100)
    
    return sorted_results


if __name__ == '__main__':
    test_stop_loss_strategies()
