"""
综合TOP15投注策略终极优化
结合：智能倍投 + 风险控制暂停 + 动态恢复
目标：ROI 50%+
"""

import pandas as pd
import numpy as np
from top15_predictor import Top15Predictor

def get_fibonacci_multiplier(consecutive_losses):
    """获取斐波那契倍数"""
    if consecutive_losses == 0:
        return 1
    fib = [1, 1]
    for i in range(2, consecutive_losses + 1):
        fib.append(fib[-1] + fib[-2])
        if fib[-1] > 100:
            break
    return fib[min(consecutive_losses, len(fib)-1)]

def test_ultimate_strategy(
    max_bet_multiplier=13,  # 最大允许倍数（斐波那契第9项）
    profit_threshold=500,  # 目标利润阈值
    pause_on_high_risk=True,  # 高风险时暂停
    pause_periods=3,  # 暂停期数
    adaptive_recovery=True,  # 自适应恢复
    test_periods=200
):
    """
    终极优化策略
    
    核心逻辑：
    1. 使用标准斐波那契倍投
    2. 当倍数达到阈值（如13倍）时，暂停投注N期避免高风险
    3. 暂停后恢复，重置倍数为1
    4. 如果累计利润达到目标，降低风险（使用较小最大倍数）
    5. 自适应恢复：根据当前盈利状态决定暂停时长
    
    Args:
        max_bet_multiplier: 最大倍数限制，超过则触发暂停
        profit_threshold: 利润目标，达到后降低风险
        pause_on_high_risk: 是否在高风险时暂停
        pause_periods: 基础暂停期数
        adaptive_recovery: 是否使用自适应恢复
        test_periods: 测试期数
    """
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    test_data = df.tail(test_periods + 100).copy()
    numbers = test_data['number'].values
    
    # 初始化
    predictor = Top15Predictor()
    
    results = []
    consecutive_losses = 0
    current_profit = 0
    pause_remaining = 0
    total_paused = 0
    risk_mode = 'normal'  # normal, conservative, aggressive
    
    for i in range(100, len(numbers)):
        period_num = i - 99
        
        # 检查是否处于暂停期
        if pause_remaining > 0:
            results.append({
                'period': period_num,
                'actual': numbers[i],
                'predicted': [],
                'hit': False,
                'paused': True,
                'multiplier': 0,
                'bet_amount': 0,
                'reward': 0,
                'profit': 0,
                'cumulative_profit': current_profit,
                'consecutive_losses': consecutive_losses,
                'pause_remaining': pause_remaining,
                'risk_mode': risk_mode
            })
            pause_remaining -= 1
            total_paused += 1
            
            # 暂停结束时重置倍投
            if pause_remaining == 0:
                consecutive_losses = 0
            
            continue
        
        # 正常投注
        train_data = numbers[:i]
        predictions = predictor.predict(train_data)
        actual = numbers[i]
        hit = actual in predictions
        
        # 计算倍数
        multiplier = get_fibonacci_multiplier(consecutive_losses)
        
        # 风险控制：根据当前盈利状态调整最大倍数
        if current_profit >= profit_threshold:
            # 达到目标利润，降低风险
            risk_mode = 'conservative'
            effective_max = max(5, max_bet_multiplier // 2)
        elif current_profit < -500:
            # 亏损较大，稍微激进
            risk_mode = 'aggressive'
            effective_max = max_bet_multiplier + 3
        else:
            risk_mode = 'normal'
            effective_max = max_bet_multiplier
        
        # 限制倍数
        multiplier = min(multiplier, effective_max)
        
        # 高风险暂停判断
        trigger_pause = False
        if pause_on_high_risk and multiplier >= max_bet_multiplier:
            trigger_pause = True
            # 自适应暂停时长
            if adaptive_recovery:
                if current_profit > 0:
                    # 盈利状态：短暂停
                    pause_remaining = max(1, pause_periods - 1)
                elif current_profit < -300:
                    # 亏损状态：长暂停
                    pause_remaining = pause_periods + 2
                else:
                    pause_remaining = pause_periods
            else:
                pause_remaining = pause_periods
        
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
            'paused': False,
            'multiplier': multiplier,
            'bet_amount': bet,
            'reward': reward,
            'profit': profit,
            'cumulative_profit': current_profit,
            'consecutive_losses': consecutive_losses,
            'pause_remaining': 0,
            'risk_mode': risk_mode,
            'triggered_pause': trigger_pause
        })
        
        # 如果触发暂停，重置倍投计数
        if trigger_pause:
            consecutive_losses = 0
    
    # 统计结果
    results_df = pd.DataFrame(results)
    
    active_periods = results_df[~results_df['paused']]
    
    hits = active_periods['hit'].sum()
    hit_rate = hits / len(active_periods) if len(active_periods) > 0 else 0
    
    total_cost = active_periods['bet_amount'].sum()
    total_reward = active_periods['reward'].sum()
    total_profit = active_periods['profit'].sum()
    
    roi = (total_profit / total_cost * 100) if total_cost > 0 else 0
    
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
    
    cumulative = results_df['cumulative_profit'].values
    max_profit = np.maximum.accumulate(cumulative)
    drawdown = max_profit - cumulative
    max_drawdown = drawdown.max()
    
    return {
        'max_bet_multiplier': max_bet_multiplier,
        'profit_threshold': profit_threshold,
        'pause_periods': pause_periods,
        'adaptive_recovery': adaptive_recovery,
        'total_periods': len(results_df),
        'active_periods': len(active_periods),
        'paused_periods': total_paused,
        'hits': hits,
        'hit_rate': hit_rate,
        'total_cost': total_cost,
        'total_reward': total_reward,
        'total_profit': total_profit,
        'roi': roi,
        'max_consecutive': max_consecutive,
        'max_drawdown': max_drawdown,
        'results_df': results_df
    }

def compare_ultimate_strategies(test_periods=200):
    """对比不同参数配置"""
    print("=" * 80)
    print("综合TOP15终极优化策略测试")
    print("=" * 80)
    print(f"测试期数: {test_periods}期")
    print("=" * 80)
    print()
    
    # 测试配置
    configs = [
        {
            'name': '基准（无风控）',
            'max_bet_multiplier': 999,
            'pause_on_high_risk': False,
        },
        {
            'name': '保守（5倍限制+3期暂停）',
            'max_bet_multiplier': 5,
            'pause_on_high_risk': True,
            'pause_periods': 3,
        },
        {
            'name': '平衡（8倍限制+3期暂停）',
            'max_bet_multiplier': 8,
            'pause_on_high_risk': True,
            'pause_periods': 3,
        },
        {
            'name': '激进（13倍限制+2期暂停）',
            'max_bet_multiplier': 13,
            'pause_on_high_risk': True,
            'pause_periods': 2,
        },
        {
            'name': '智能（13倍+自适应暂停）',
            'max_bet_multiplier': 13,
            'pause_on_high_risk': True,
            'pause_periods': 3,
            'adaptive_recovery': True,
        },
        {
            'name': '极致（21倍限制+1期暂停）',
            'max_bet_multiplier': 21,
            'pause_on_high_risk': True,
            'pause_periods': 1,
        },
    ]
    
    results = []
    
    for config in configs:
        name = config.pop('name')
        print(f"测试策略: {name}...")
        
        config.setdefault('profit_threshold', 500)
        config.setdefault('pause_on_high_risk', False)
        config.setdefault('pause_periods', 3)
        config.setdefault('adaptive_recovery', False)
        
        result = test_ultimate_strategy(**config, test_periods=test_periods)
        result['name'] = name
        results.append(result)
        
        print(f"✓ {name}: ROI={result['roi']:.2f}%, "
              f"收益={result['total_profit']:.2f}元, "
              f"暂停={result['paused_periods']}期\n")
    
    # 显示对比
    print("\n" + "=" * 110)
    print("终极策略对比结果")
    print("=" * 110)
    
    print(f"{'策略名称':<25} {'ROI':<10} {'总收益':<12} {'总投注':<12} "
          f"{'活跃期':<8} {'暂停期':<8} {'最大回撤':<12}")
    print("-" * 110)
    
    for r in results:
        print(f"{r['name']:<25} "
              f"{r['roi']:>8.2f}% "
              f"{r['total_profit']:>10.2f}元 "
              f"{r['total_cost']:>10.2f}元 "
              f"{r['active_periods']:>6}期 "
              f"{r['paused_periods']:>6}期 "
              f"{r['max_drawdown']:>10.2f}元")
    
    print("=" * 110)
    
    # 找出最优
    best_roi = max(results, key=lambda x: x['roi'])
    best_profit = max(results, key=lambda x: x['total_profit'])
    
    print(f"\n🏆 ROI最高: {best_roi['name']} - {best_roi['roi']:.2f}%")
    print(f"💰 收益最高: {best_profit['name']} - {best_profit['total_profit']:.2f}元")
    
    print(f"\n🎯 目标达成情况:")
    if best_roi['roi'] >= 50:
        print(f"   ✅ ROI达到50%目标！当前: {best_roi['roi']:.2f}%")
    else:
        print(f"   ⚠️ ROI: {best_roi['roi']:.2f}% (目标50%，差{50-best_roi['roi']:.2f}%)")
    
    # 保存
    summary = pd.DataFrame([{
        '策略名称': r['name'],
        '最大倍数限制': r['max_bet_multiplier'],
        'ROI(%)': f"{r['roi']:.2f}",
        '总收益(元)': f"{r['total_profit']:.2f}",
        '总投注(元)': f"{r['total_cost']:.2f}",
        '活跃期数': r['active_periods'],
        '暂停期数': r['paused_periods'],
        '最大回撤(元)': f"{r['max_drawdown']:.2f}"
    } for r in results])
    
    summary.to_csv('top15_ultimate_strategy_comparison.csv', index=False, encoding='utf-8-sig')
    print(f"\n✓ 结果已保存至: top15_ultimate_strategy_comparison.csv")
    
    # 保存最优策略详情
    if best_roi['results_df'] is not None:
        best_df = best_roi['results_df'].copy()
        best_df['predicted'] = best_df['predicted'].apply(lambda x: str(x) if x else '')
        best_df.to_csv('top15_ultimate_best_details.csv', index=False, encoding='utf-8-sig')
        print(f"✓ 最优策略详情已保存至: top15_ultimate_best_details.csv")
    
    return results, best_roi

if __name__ == "__main__":
    results, best = compare_ultimate_strategies(test_periods=200)
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
