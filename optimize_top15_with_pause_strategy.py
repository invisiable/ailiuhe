"""
综合TOP15投注策略优化 - 智能暂停机制
N期不中则暂停投注N期，自动恢复后继续
目标：将ROI从33%提升到50%+
"""

import pandas as pd
import numpy as np
from top15_predictor import Top15Predictor
from betting_strategy import BettingStrategy

def test_pause_strategy(pause_threshold=3, pause_periods=3, test_periods=200):
    """
    测试智能暂停策略
    
    Args:
        pause_threshold: 连续不中多少期后触发暂停
        pause_periods: 暂停多少期
        test_periods: 测试期数
    """
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # 获取最近的测试数据
    test_data = df.tail(test_periods + 100).copy()  # 多取100期用于训练
    numbers = test_data['number'].values
    
    # 初始化预测器
    predictor = Top15Predictor()
    
    # 斐波那契数列生成器
    def get_fibonacci_multiplier(consecutive_losses):
        """获取斐波那契倍数"""
        if consecutive_losses == 0:
            return 1
        fib = [1, 1]
        for i in range(2, consecutive_losses + 1):
            fib.append(fib[-1] + fib[-2])
            if fib[-1] > 100:  # 限制最大倍数
                break
        return fib[min(consecutive_losses, len(fib)-1)]
    
    # 统计数据
    results = []
    consecutive_misses = 0
    consecutive_losses_for_bet = 0  # 用于计算斐波那契倍数
    pause_remaining = 0  # 剩余暂停期数
    total_paused_periods = 0  # 总暂停期数
    
    for i in range(100, len(numbers)):
        period_num = i - 99  # 期数从1开始
        
        # 如果处于暂停期
        if pause_remaining > 0:
            results.append({
                'period': period_num,
                'actual': numbers[i],
                'predicted': [],
                'hit': False,
                'paused': True,
                'bet_amount': 0,
                'reward': 0,
                'profit': 0,
                'consecutive_misses': consecutive_misses,
                'pause_remaining': pause_remaining
            })
            pause_remaining -= 1
            total_paused_periods += 1
            continue
        
        # 正常投注
        train_data = numbers[:i]
        predictions = predictor.predict(train_data)
        actual = numbers[i]
        
        # 判断是否命中
        hit = actual in predictions
        
        # 计算投注和收益
        if hit:
            multiplier = get_fibonacci_multiplier(consecutive_losses_for_bet)
            bet = 15 * multiplier
            reward = 47 * multiplier
            profit = reward - bet
            consecutive_misses = 0
            consecutive_losses_for_bet = 0  # 命中后重置
        else:
            multiplier = get_fibonacci_multiplier(consecutive_losses_for_bet)
            bet = 15 * multiplier
            reward = 0
            profit = -bet
            consecutive_misses += 1
            consecutive_losses_for_bet += 1
            
            # 检查是否需要触发暂停
            if consecutive_misses >= pause_threshold:
                pause_remaining = pause_periods
                consecutive_misses = 0  # 重置连续不中计数
                consecutive_losses_for_bet = 0  # 暂停期间重置倍投
        
        results.append({
            'period': period_num,
            'actual': actual,
            'predicted': predictions,
            'hit': hit,
            'paused': False,
            'bet_amount': bet,
            'reward': reward,
            'profit': profit,
            'consecutive_misses': consecutive_misses,
            'pause_remaining': 0
        })
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 计算总体统计
    active_periods = results_df[~results_df['paused']]
    total_periods = len(results_df)
    active_count = len(active_periods)
    paused_count = total_paused_periods
    
    hits = active_periods['hit'].sum()
    hit_rate = hits / active_count if active_count > 0 else 0
    
    total_cost = active_periods['bet_amount'].sum()
    total_reward = active_periods['reward'].sum()
    total_profit = active_periods['profit'].sum()
    
    roi = (total_profit / total_cost * 100) if total_cost > 0 else 0
    avg_profit = total_profit / total_periods if total_periods > 0 else 0
    
    # 计算最大连续不中（仅计算活跃期）
    max_consecutive = 0
    current_consecutive = 0
    for _, row in results_df.iterrows():
        if row['paused']:
            continue
        if not row['hit']:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    # 计算最大回撤
    cumulative_profit = 0
    max_profit = 0
    max_drawdown = 0
    
    for _, row in results_df.iterrows():
        cumulative_profit += row['profit']
        max_profit = max(max_profit, cumulative_profit)
        drawdown = max_profit - cumulative_profit
        max_drawdown = max(max_drawdown, drawdown)
    
    return {
        'pause_threshold': pause_threshold,
        'pause_periods': pause_periods,
        'total_periods': total_periods,
        'active_periods': active_count,
        'paused_periods': paused_count,
        'hits': hits,
        'hit_rate': hit_rate,
        'total_cost': total_cost,
        'total_reward': total_reward,
        'total_profit': total_profit,
        'roi': roi,
        'avg_profit_per_period': avg_profit,
        'max_consecutive_misses': max_consecutive,
        'max_drawdown': max_drawdown,
        'results_df': results_df
    }

def compare_pause_strategies(test_periods=200):
    """
    对比不同的暂停策略参数
    """
    print("=" * 80)
    print("综合TOP15投注策略优化 - 智能暂停机制测试")
    print("=" * 80)
    print(f"测试期数: {test_periods}期")
    print(f"基础投注: 15元/期 (15个数字 × 1元)")
    print(f"中奖奖励: 47元")
    print(f"倍投方式: 斐波那契数列")
    print("=" * 80)
    print()
    
    # 测试不同的参数组合
    strategies = []
    
    # 基准策略（不暂停）
    print("测试基准策略（无暂停机制）...")
    baseline = test_pause_strategy(pause_threshold=999, pause_periods=0, test_periods=test_periods)
    strategies.append({
        'name': '基准策略（无暂停）',
        'params': 'N/A',
        **baseline
    })
    print(f"✓ 基准策略: ROI={baseline['roi']:.2f}%, 命中率={baseline['hit_rate']*100:.2f}%\n")
    
    # 测试不同的暂停参数
    test_configs = [
        (3, 3, "3期不中暂停3期"),
        (3, 5, "3期不中暂停5期"),
        (4, 4, "4期不中暂停4期"),
        (4, 6, "4期不中暂停6期"),
        (5, 5, "5期不中暂停5期"),
        (5, 8, "5期不中暂停8期"),
        (6, 6, "6期不中暂停6期"),
        (6, 10, "6期不中暂停10期"),
    ]
    
    for threshold, periods, desc in test_configs:
        print(f"测试策略: {desc}...")
        result = test_pause_strategy(
            pause_threshold=threshold,
            pause_periods=periods,
            test_periods=test_periods
        )
        strategies.append({
            'name': desc,
            'params': f"{threshold}-{periods}",
            **result
        })
        print(f"✓ {desc}: ROI={result['roi']:.2f}%, 命中率={result['hit_rate']*100:.2f}%, "
              f"暂停{result['paused_periods']}期\n")
    
    # 创建对比表
    print("\n" + "=" * 120)
    print("策略对比结果汇总")
    print("=" * 120)
    
    # 表头
    print(f"{'策略名称':<20} {'ROI':<10} {'命中率':<10} {'总收益':<12} "
          f"{'总投注':<12} {'活跃期':<8} {'暂停期':<8} {'最大连不中':<12} {'最大回撤':<12}")
    print("-" * 120)
    
    # 数据行
    for s in strategies:
        print(f"{s['name']:<20} "
              f"{s['roi']:>8.2f}% "
              f"{s['hit_rate']*100:>8.2f}% "
              f"{s['total_profit']:>10.2f}元 "
              f"{s['total_cost']:>10.2f}元 "
              f"{s['active_periods']:>6}期 "
              f"{s['paused_periods']:>6}期 "
              f"{s['max_consecutive_misses']:>10}期 "
              f"{s['max_drawdown']:>10.2f}元")
    
    print("=" * 120)
    
    # 找出最优策略
    best_roi = max(strategies, key=lambda x: x['roi'])
    best_profit = max(strategies, key=lambda x: x['total_profit'])
    
    print("\n" + "=" * 80)
    print("最优策略分析")
    print("=" * 80)
    
    print(f"\n🏆 ROI最高策略: {best_roi['name']}")
    print(f"   - ROI: {best_roi['roi']:.2f}%")
    print(f"   - 总收益: {best_roi['total_profit']:.2f}元")
    print(f"   - 命中率: {best_roi['hit_rate']*100:.2f}%")
    print(f"   - 活跃期数: {best_roi['active_periods']}期")
    print(f"   - 暂停期数: {best_roi['paused_periods']}期")
    print(f"   - 总投注: {best_roi['total_cost']:.2f}元")
    print(f"   - 最大连不中: {best_roi['max_consecutive_misses']}期")
    print(f"   - 最大回撤: {best_roi['max_drawdown']:.2f}元")
    
    if best_profit['name'] != best_roi['name']:
        print(f"\n💰 总收益最高策略: {best_profit['name']}")
        print(f"   - 总收益: {best_profit['total_profit']:.2f}元")
        print(f"   - ROI: {best_profit['roi']:.2f}%")
        print(f"   - 命中率: {best_profit['hit_rate']*100:.2f}%")
    
    # 对比基准
    baseline_strategy = strategies[0]
    print(f"\n📊 相对基准策略的提升:")
    print(f"   - ROI提升: {best_roi['roi'] - baseline_strategy['roi']:.2f}个百分点 "
          f"({baseline_strategy['roi']:.2f}% → {best_roi['roi']:.2f}%)")
    print(f"   - 收益提升: {best_roi['total_profit'] - baseline_strategy['total_profit']:.2f}元 "
          f"({baseline_strategy['total_profit']:.2f}元 → {best_roi['total_profit']:.2f}元)")
    print(f"   - 投注减少: {baseline_strategy['total_cost'] - best_roi['total_cost']:.2f}元 "
          f"({baseline_strategy['total_cost']:.2f}元 → {best_roi['total_cost']:.2f}元)")
    print(f"   - 暂停期数: {best_roi['paused_periods']}期")
    
    # 检查是否达到目标
    print(f"\n🎯 目标达成情况:")
    if best_roi['roi'] >= 50:
        print(f"   ✅ ROI达到50%目标！当前ROI: {best_roi['roi']:.2f}%")
    else:
        print(f"   ⚠️  ROI未达到50%目标，当前最高: {best_roi['roi']:.2f}%")
        print(f"   📈 距离目标还差: {50 - best_roi['roi']:.2f}个百分点")
    
    print("=" * 80)
    
    # 保存详细结果
    save_detailed_results(strategies, best_roi)
    
    return strategies, best_roi

def save_detailed_results(strategies, best_strategy):
    """保存详细的测试结果"""
    # 保存汇总数据
    summary_data = []
    for s in strategies:
        summary_data.append({
            '策略名称': s['name'],
            '参数': s['params'],
            'ROI(%)': f"{s['roi']:.2f}",
            '命中率(%)': f"{s['hit_rate']*100:.2f}",
            '总收益(元)': f"{s['total_profit']:.2f}",
            '总投注(元)': f"{s['total_cost']:.2f}",
            '活跃期数': s['active_periods'],
            '暂停期数': s['paused_periods'],
            '最大连不中': s['max_consecutive_misses'],
            '最大回撤(元)': f"{s['max_drawdown']:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('top15_pause_strategy_summary.csv', index=False, encoding='utf-8-sig')
    print(f"\n✓ 汇总结果已保存至: top15_pause_strategy_summary.csv")
    
    # 保存最优策略的详细投注记录
    if 'results_df' in best_strategy:
        best_df = best_strategy['results_df'].copy()
        best_df['predicted'] = best_df['predicted'].apply(lambda x: str(x))
        best_df.to_csv('top15_best_pause_strategy_details.csv', index=False, encoding='utf-8-sig')
        print(f"✓ 最优策略详细记录已保存至: top15_best_pause_strategy_details.csv")

if __name__ == "__main__":
    # 运行对比测试
    strategies, best = compare_pause_strategies(test_periods=200)
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
