"""
综合TOP15 vs 精准TOP15 - 多期数对比分析
对比最近300期、200期、100期的收益表现
"""

import pandas as pd
import numpy as np
from top15_predictor import Top15Predictor
from precise_top15_predictor import PreciseTop15Predictor
from betting_strategy import BettingStrategy

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

def backtest_strategy(predictor, predictor_name, test_periods, df):
    """
    回测指定策略
    
    Args:
        predictor: 预测器实例
        predictor_name: 预测器名称
        test_periods: 测试期数
        df: 数据DataFrame
    """
    start_idx = len(df) - test_periods
    
    # 统计数据
    results = []
    consecutive_losses = 0
    current_profit = 0
    
    for i in range(start_idx, len(df)):
        period_num = i - start_idx + 1
        
        # 预测
        train_data = df.iloc[:i]['number'].values
        
        if predictor_name == '综合TOP15':
            analysis = predictor.get_analysis(train_data)
            predictions = analysis['top15']
        else:  # 精准TOP15
            predictions = predictor.predict(train_data)
        
        actual = df.iloc[i]['number']
        
        # 判断命中
        hit = actual in predictions
        
        # 计算倍数和收益
        multiplier = get_fibonacci_multiplier(consecutive_losses)
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
            'hit': hit,
            'multiplier': multiplier,
            'bet': bet,
            'profit': profit,
            'cumulative_profit': current_profit
        })
    
    # 统计结果
    hits = sum(1 for r in results if r['hit'])
    hit_rate = hits / len(results)
    total_cost = sum(r['bet'] for r in results)
    total_profit = current_profit
    roi = (total_profit / total_cost * 100) if total_cost > 0 else 0
    
    max_consecutive = 0
    current_consecutive = 0
    for r in results:
        if not r['hit']:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    # 最大回撤
    cumulative_profits = [r['cumulative_profit'] for r in results]
    max_profit = np.maximum.accumulate(cumulative_profits)
    drawdowns = max_profit - cumulative_profits
    max_drawdown = np.max(drawdowns)
    
    return {
        'periods': test_periods,
        'hits': hits,
        'hit_rate': hit_rate,
        'total_cost': total_cost,
        'total_profit': total_profit,
        'roi': roi,
        'max_consecutive': max_consecutive,
        'max_drawdown': max_drawdown,
        'avg_profit_per_period': total_profit / len(results)
    }

def compare_strategies_multiple_periods():
    """对比多个时间窗口的表现"""
    print("=" * 100)
    print("综合TOP15 vs 精准TOP15 - 多期数收益对比分析")
    print("=" * 100)
    print()
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"数据总期数: {len(df)}期")
    print(f"测试窗口: 最近100期、最近200期、最近{len(df)-100}期（最大可用）")
    print(f"投注规则: 15元/期，命中奖励47元，斐波那契倍投")
    print()
    
    # 初始化预测器
    predictor1 = Top15Predictor()
    predictor2 = PreciseTop15Predictor()
    
    # 测试不同期数（包括最大可用期数）
    max_periods = len(df) - 100  # 保留100期用于初始训练
    test_periods_list = [100, 200, max_periods]
    
    all_results = []
    
    for test_periods in test_periods_list:
        if test_periods > len(df) - 100:
            print(f"⚠️ 跳过{test_periods}期测试（数据不足）\n")
            continue
        
        print(f"{'='*100}")
        print(f"测试窗口: 最近{test_periods}期")
        print(f"{'='*100}")
        print()
        
        # 策略1：综合TOP15
        print(f"测试策略1：综合TOP15...")
        result1 = backtest_strategy(predictor1, '综合TOP15', test_periods, df)
        print(f"✓ 完成: ROI={result1['roi']:.2f}%, 收益={result1['total_profit']:+.2f}元\n")
        
        # 策略2：精准TOP15
        print(f"测试策略2：精准TOP15...")
        result2 = backtest_strategy(predictor2, '精准TOP15', test_periods, df)
        print(f"✓ 完成: ROI={result2['roi']:.2f}%, 收益={result2['total_profit']:+.2f}元\n")
        
        # 计算差异
        roi_diff = result1['roi'] - result2['roi']
        profit_diff = result1['total_profit'] - result2['total_profit']
        hit_rate_diff = (result1['hit_rate'] - result2['hit_rate']) * 100
        
        all_results.append({
            'periods': test_periods,
            'strategy1': result1,
            'strategy2': result2,
            'roi_diff': roi_diff,
            'profit_diff': profit_diff,
            'hit_rate_diff': hit_rate_diff
        })
        
        # 显示对比
        print(f"{'策略对比':-^100}")
        print(f"{'指标':<20} {'综合TOP15':<25} {'精准TOP15':<25} {'差异':<25}")
        print("-" * 100)
        print(f"{'命中率':<20} {result1['hit_rate']*100:>8.2f}% {'':<15} {result2['hit_rate']*100:>8.2f}% {'':<15} {hit_rate_diff:>+8.2f}% {'':<12}")
        print(f"{'ROI':<20} {result1['roi']:>8.2f}% {'':<15} {result2['roi']:>8.2f}% {'':<15} {roi_diff:>+8.2f}% {'':<12}")
        print(f"{'总收益':<20} {result1['total_profit']:>8.2f}元 {'':<14} {result2['total_profit']:>8.2f}元 {'':<14} {profit_diff:>+8.2f}元 {'':<11}")
        print(f"{'总投注':<20} {result1['total_cost']:>8.2f}元 {'':<14} {result2['total_cost']:>8.2f}元 {'':<14} {result1['total_cost']-result2['total_cost']:>+8.2f}元 {'':<11}")
        print(f"{'最大连不中':<20} {result1['max_consecutive']:>8}期 {'':<15} {result2['max_consecutive']:>8}期 {'':<15} {result1['max_consecutive']-result2['max_consecutive']:>+8}期 {'':<12}")
        print(f"{'最大回撤':<20} {result1['max_drawdown']:>8.2f}元 {'':<14} {result2['max_drawdown']:>8.2f}元 {'':<14} {result1['max_drawdown']-result2['max_drawdown']:>+8.2f}元 {'':<11}")
        print(f"{'每期平均收益':<20} {result1['avg_profit_per_period']:>8.2f}元 {'':<14} {result2['avg_profit_per_period']:>8.2f}元 {'':<14} {result1['avg_profit_per_period']-result2['avg_profit_per_period']:>+8.2f}元 {'':<11}")
        
        # 判断优劣
        if roi_diff > 0:
            winner = "✅ 综合TOP15全面优于精准TOP15"
        else:
            winner = "⚠️ 精准TOP15表现更好"
        
        print()
        print(f"结论: {winner}")
        print(f"  - ROI优势: {roi_diff:+.2f}个百分点")
        print(f"  - 收益优势: {profit_diff:+.2f}元")
        print(f"  - 命中率优势: {hit_rate_diff:+.2f}个百分点")
        print()
        print()
    
    # 综合对比表
    print("=" * 100)
    print("综合对比汇总表")
    print("=" * 100)
    print()
    
    print(f"{'测试期数':<12} {'策略':<15} {'命中率':<12} {'ROI':<12} {'总收益':<12} {'总投注':<12} {'最大回撤':<12}")
    print("-" * 100)
    
    for result in all_results:
        periods = result['periods']
        r1 = result['strategy1']
        r2 = result['strategy2']
        
        print(f"{periods}期 {'':<6} {'综合TOP15':<15} {r1['hit_rate']*100:>6.2f}% {'':<4} {r1['roi']:>+8.2f}% {'':<2} {r1['total_profit']:>+9.2f}元 {r1['total_cost']:>9.2f}元 {r1['max_drawdown']:>9.2f}元")
        print(f"{'':<12} {'精准TOP15':<15} {r2['hit_rate']*100:>6.2f}% {'':<4} {r2['roi']:>+8.2f}% {'':<2} {r2['total_profit']:>+9.2f}元 {r2['total_cost']:>9.2f}元 {r2['max_drawdown']:>9.2f}元")
        print(f"{'':<12} {'差异':<15} {result['hit_rate_diff']:>+6.2f}% {'':<4} {result['roi_diff']:>+8.2f}% {'':<2} {result['profit_diff']:>+9.2f}元 {r1['total_cost']-r2['total_cost']:>+9.2f}元 {r1['max_drawdown']-r2['max_drawdown']:>+9.2f}元")
        print("-" * 100)
    
    # 趋势分析
    print()
    print("=" * 100)
    print("趋势分析")
    print("=" * 100)
    print()
    
    if len(all_results) >= 2:
        print("【综合TOP15 vs 精准TOP15 在不同时间窗口的表现】")
        print()
        
        for i, result in enumerate(all_results, 1):
            periods = result['periods']
            roi_diff = result['roi_diff']
            profit_diff = result['profit_diff']
            
            if roi_diff > 5:
                status = "✅ 显著优势"
            elif roi_diff > 0:
                status = "✓ 略有优势"
            else:
                status = "⚠️ 劣势"
            
            print(f"{i}. 最近{periods}期: 综合TOP15 ROI高出{roi_diff:+.2f}%, 多赚{profit_diff:+.2f}元 - {status}")
        
        print()
        print("【一致性分析】")
        
        # 检查是否所有窗口都是综合TOP15更优
        all_better = all(r['roi_diff'] > 0 for r in all_results)
        
        if all_better:
            print("✅ 综合TOP15在所有测试窗口中均表现优于精准TOP15")
            print("✅ 策略稳定性强，适合长期使用")
            
            # 计算平均优势
            avg_roi_diff = np.mean([r['roi_diff'] for r in all_results])
            avg_profit_diff = np.mean([r['profit_diff'] for r in all_results])
            
            print(f"\n平均优势:")
            print(f"  - ROI平均高出: {avg_roi_diff:.2f}个百分点")
            print(f"  - 平均多赚: {avg_profit_diff:.2f}元/窗口")
        else:
            print("⚠️ 不同时间窗口表现不一致，需谨慎选择")
    
    # 保存结果
    summary_data = []
    for result in all_results:
        periods = result['periods']
        r1 = result['strategy1']
        r2 = result['strategy2']
        
        summary_data.append({
            '测试期数': f"{periods}期",
            '策略': '综合TOP15',
            '命中率(%)': f"{r1['hit_rate']*100:.2f}",
            'ROI(%)': f"{r1['roi']:.2f}",
            '总收益(元)': f"{r1['total_profit']:.2f}",
            '总投注(元)': f"{r1['total_cost']:.2f}",
            '最大连不中': r1['max_consecutive'],
            '最大回撤(元)': f"{r1['max_drawdown']:.2f}",
            '每期平均收益(元)': f"{r1['avg_profit_per_period']:.2f}"
        })
        
        summary_data.append({
            '测试期数': f"{periods}期",
            '策略': '精准TOP15',
            '命中率(%)': f"{r2['hit_rate']*100:.2f}",
            'ROI(%)': f"{r2['roi']:.2f}",
            '总收益(元)': f"{r2['total_profit']:.2f}",
            '总投注(元)': f"{r2['total_cost']:.2f}",
            '最大连不中': r2['max_consecutive'],
            '最大回撤(元)': f"{r2['max_drawdown']:.2f}",
            '每期平均收益(元)': f"{r2['avg_profit_per_period']:.2f}"
        })
        
        summary_data.append({
            '测试期数': f"{periods}期",
            '策略': '差异',
            '命中率(%)': f"{result['hit_rate_diff']:+.2f}",
            'ROI(%)': f"{result['roi_diff']:+.2f}",
            '总收益(元)': f"{result['profit_diff']:+.2f}",
            '总投注(元)': f"{r1['total_cost']-r2['total_cost']:+.2f}",
            '最大连不中': r1['max_consecutive']-r2['max_consecutive'],
            '最大回撤(元)': f"{r1['max_drawdown']-r2['max_drawdown']:+.2f}",
            '每期平均收益(元)': f"{r1['avg_profit_per_period']-r2['avg_profit_per_period']:+.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('top15_strategies_comparison_multi_periods.csv', index=False, encoding='utf-8-sig')
    
    print()
    print("=" * 100)
    print(f"✓ 对比结果已保存至: top15_strategies_comparison_multi_periods.csv")
    print("=" * 100)

if __name__ == "__main__":
    compare_strategies_multiple_periods()
    
    print("\n测试完成！")
