"""
综合TOP15投注策略优化 - 动态倍投优化版
目标：通过智能倍投管理将ROI从42%提升到50%+
"""

import pandas as pd
import numpy as np
from top15_predictor import Top15Predictor

def get_fibonacci_multiplier(consecutive_losses):
    """获取标准斐波那契倍数"""
    if consecutive_losses == 0:
        return 1
    fib = [1, 1]
    for i in range(2, consecutive_losses + 1):
        fib.append(fib[-1] + fib[-2])
        if fib[-1] > 100:
            break
    return fib[min(consecutive_losses, len(fib)-1)]

def get_aggressive_multiplier(consecutive_losses, current_profit):
    """
    激进倍投策略
    - 盈利状态：使用标准斐波那契
    - 亏损状态：加速倍投（1.5倍斐波那契）
    """
    base_multiplier = get_fibonacci_multiplier(consecutive_losses)
    
    # 如果当前处于亏损，加速倍投
    if current_profit < 0:
        return int(base_multiplier * 1.5)
    else:
        return base_multiplier

def get_smart_multiplier(consecutive_losses, current_profit, total_periods):
    """
    智能倍投策略 - 根据多个因素动态调整
    
    核心逻辑：
    1. 早期（前30期）：保守倍投，建立利润基础
    2. 中期（31-150期）：标准倍投，稳定增长
    3. 后期（151+期）：如果达到目标利润则保守，否则激进追赶
    4. 盈利状态：标准倍投
    5. 亏损状态：激进倍投
    """
    base_multiplier = get_fibonacci_multiplier(consecutive_losses)
    
    # 目标利润率：期望每期平均盈利
    target_profit_per_period = 15  # 每期期望赚15元，对应50% ROI
    expected_profit = total_periods * target_profit_per_period
    
    # 早期（前30期）：保守策略，限制最大倍数
    if total_periods <= 30:
        return min(base_multiplier, 5)  # 最多5倍
    
    # 后期如果利润不足，使用激进策略
    if total_periods > 100:
        if current_profit < expected_profit * 0.7:  # 利润不足预期的70%
            # 激进倍投：1.5倍斐波那契
            return int(base_multiplier * 1.5)
    
    # 亏损状态：加速回本
    if current_profit < 0:
        return int(base_multiplier * 1.3)
    
    # 默认：标准倍投
    return base_multiplier

def get_tiered_multiplier(consecutive_losses, win_rate, total_periods):
    """
    分层倍投策略 - 根据当前命中率动态调整
    
    逻辑：
    - 命中率高（>40%）：保守倍投，保护利润
    - 命中率中（30-40%）：标准倍投
    - 命中率低（<30%）：激进倍投，追赶收益
    """
    base_multiplier = get_fibonacci_multiplier(consecutive_losses)
    
    if total_periods < 20:
        return base_multiplier  # 早期样本不足，使用默认
    
    # 根据命中率调整
    if win_rate > 0.40:
        # 高命中率：保守倍投
        return min(base_multiplier, 8)
    elif win_rate < 0.30:
        # 低命中率：激进倍投
        return int(base_multiplier * 1.5)
    else:
        # 中等命中率：标准倍投
        return base_multiplier

def test_multiplier_strategy(strategy_name, multiplier_func, test_periods=200):
    """
    测试特定的倍投策略
    
    Args:
        strategy_name: 策略名称
        multiplier_func: 倍数计算函数
        test_periods: 测试期数
    """
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # 获取测试数据
    test_data = df.tail(test_periods + 100).copy()
    numbers = test_data['number'].values
    
    # 初始化预测器
    predictor = Top15Predictor()
    
    # 统计数据
    results = []
    consecutive_losses = 0
    current_profit = 0
    total_hits = 0
    total_active = 0
    
    for i in range(100, len(numbers)):
        period_num = i - 99
        
        # 预测
        train_data = numbers[:i]
        predictions = predictor.predict(train_data)
        actual = numbers[i]
        
        # 判断是否命中
        hit = actual in predictions
        total_active += 1
        if hit:
            total_hits += 1
        
        # 计算当前命中率
        win_rate = total_hits / total_active if total_active > 0 else 0
        
        # 根据策略计算倍数
        if strategy_name == '标准斐波那契':
            multiplier = get_fibonacci_multiplier(consecutive_losses)
        elif strategy_name == '激进倍投':
            multiplier = get_aggressive_multiplier(consecutive_losses, current_profit)
        elif strategy_name == '智能倍投':
            multiplier = get_smart_multiplier(consecutive_losses, current_profit, total_active)
        elif strategy_name == '分层倍投':
            multiplier = get_tiered_multiplier(consecutive_losses, win_rate, total_active)
        else:
            multiplier = get_fibonacci_multiplier(consecutive_losses)
        
        # 计算投注和收益
        bet = 15 * multiplier
        if hit:
            reward = 47 * multiplier
            profit = reward - bet
            consecutive_losses = 0
        else:
            reward = 0
            profit = -bet
            consecutive_losses += 1
        
        current_profit += profit
        
        results.append({
            'period': period_num,
            'actual': actual,
            'predicted': str(predictions),
            'hit': hit,
            'multiplier': multiplier,
            'bet_amount': bet,
            'reward': reward,
            'profit': profit,
            'cumulative_profit': current_profit,
            'consecutive_losses': consecutive_losses,
            'win_rate': win_rate
        })
    
    # 计算统计信息
    results_df = pd.DataFrame(results)
    
    hits = results_df['hit'].sum()
    hit_rate = hits / len(results_df)
    
    total_cost = results_df['bet_amount'].sum()
    total_reward = results_df['reward'].sum()
    total_profit = results_df['profit'].sum()
    
    roi = (total_profit / total_cost * 100) if total_cost > 0 else 0
    avg_profit = total_profit / len(results_df)
    
    # 最大连续不中
    max_consecutive = results_df['consecutive_losses'].max()
    
    # 最大回撤
    cumulative = results_df['cumulative_profit'].values
    max_profit = np.maximum.accumulate(cumulative)
    drawdown = max_profit - cumulative
    max_drawdown = drawdown.max()
    
    return {
        'strategy_name': strategy_name,
        'hits': hits,
        'hit_rate': hit_rate,
        'total_cost': total_cost,
        'total_reward': total_reward,
        'total_profit': total_profit,
        'roi': roi,
        'avg_profit': avg_profit,
        'max_consecutive': max_consecutive,
        'max_drawdown': max_drawdown,
        'results_df': results_df
    }

def compare_multiplier_strategies(test_periods=200):
    """对比不同的倍投策略"""
    print("=" * 80)
    print("综合TOP15投注策略优化 - 动态倍投对比测试")
    print("=" * 80)
    print(f"测试期数: {test_periods}期")
    print(f"基础投注: 15元/期 (15个数字 × 1元)")
    print(f"中奖奖励: 47元")
    print("=" * 80)
    print()
    
    strategies = [
        ('标准斐波那契', None),
        ('激进倍投', get_aggressive_multiplier),
        ('智能倍投', get_smart_multiplier),
        ('分层倍投', get_tiered_multiplier),
    ]
    
    results = []
    
    for strategy_name, _ in strategies:
        print(f"测试策略: {strategy_name}...")
        result = test_multiplier_strategy(strategy_name, None, test_periods)
        results.append(result)
        print(f"✓ {strategy_name}: ROI={result['roi']:.2f}%, "
              f"命中率={result['hit_rate']*100:.2f}%, "
              f"总收益={result['total_profit']:.2f}元\n")
    
    # 显示对比表
    print("\n" + "=" * 100)
    print("倍投策略对比结果")
    print("=" * 100)
    
    print(f"{'策略名称':<15} {'ROI':<10} {'命中率':<10} {'总收益':<12} "
          f"{'总投注':<12} {'最大连不中':<12} {'最大回撤':<12}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['strategy_name']:<15} "
              f"{r['roi']:>8.2f}% "
              f"{r['hit_rate']*100:>8.2f}% "
              f"{r['total_profit']:>10.2f}元 "
              f"{r['total_cost']:>10.2f}元 "
              f"{r['max_consecutive']:>10}期 "
              f"{r['max_drawdown']:>10.2f}元")
    
    print("=" * 100)
    
    # 找出最优策略
    best = max(results, key=lambda x: x['roi'])
    
    print(f"\n🏆 最优策略: {best['strategy_name']}")
    print(f"   - ROI: {best['roi']:.2f}%")
    print(f"   - 总收益: {best['total_profit']:.2f}元")
    print(f"   - 命中率: {best['hit_rate']*100:.2f}%")
    print(f"   - 总投注: {best['total_cost']:.2f}元")
    print(f"   - 最大连不中: {best['max_consecutive']}期")
    print(f"   - 最大回撤: {best['max_drawdown']:.2f}元")
    
    # 检查目标
    print(f"\n🎯 目标达成情况:")
    if best['roi'] >= 50:
        print(f"   ✅ ROI达到50%目标！当前: {best['roi']:.2f}%")
    else:
        print(f"   ⚠️ ROI未达到50%，当前: {best['roi']:.2f}%")
        print(f"   📈 距离目标还差: {50 - best['roi']:.2f}个百分点")
    
    # 保存结果
    summary = pd.DataFrame([{
        '策略名称': r['strategy_name'],
        'ROI(%)': f"{r['roi']:.2f}",
        '命中率(%)': f"{r['hit_rate']*100:.2f}",
        '总收益(元)': f"{r['total_profit']:.2f}",
        '总投注(元)': f"{r['total_cost']:.2f}",
        '最大连不中': r['max_consecutive'],
        '最大回撤(元)': f"{r['max_drawdown']:.2f}"
    } for r in results])
    
    summary.to_csv('top15_multiplier_strategies_comparison.csv', index=False, encoding='utf-8-sig')
    print(f"\n✓ 结果已保存至: top15_multiplier_strategies_comparison.csv")
    
    return results, best

if __name__ == "__main__":
    results, best = compare_multiplier_strategies(test_periods=200)
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
