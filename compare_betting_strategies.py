"""
对比不同投注策略的收益率
测试：固定倍数、凯利公式、反向马丁格尔、激进马丁格尔、达朗贝尔等策略
"""

import pandas as pd
import numpy as np
from betting_strategy import BettingStrategy
from top15_predictor import Top15Predictor

def compare_all_strategies():
    """对比所有投注策略"""
    print("="*100)
    print("投注策略收益率对比测试")
    print("="*100)
    
    # 加载数据
    df = pd.read_csv('lucky_numbers - 副本.csv')
    
    # 使用最近100期测试
    test_periods = 100
    start_idx = len(df) - test_periods
    
    predictor = Top15Predictor()
    predictions = []
    actuals = []
    
    print(f"\n生成{test_periods}期TOP15预测...\n")
    
    for i in range(start_idx, len(df)):
        train_data = df.iloc[:i]['number'].values
        analysis = predictor.get_analysis(train_data)
        predictions.append(analysis['top15'])
        actuals.append(df.iloc[i]['number'])
    
    # 计算实际命中率
    actual_hit_rate = sum(1 for i in range(len(actuals)) if actuals[i] in predictions[i]) / len(actuals)
    
    # 创建投注策略实例
    betting = BettingStrategy(base_bet=15, win_reward=45, loss_penalty=15)
    
    # 定义所有策略
    strategies = {
        'fixed': '固定1倍投注（最保守）',
        'dalembert': '达朗贝尔渐进式（保守）',
        'kelly': '凯利公式动态（优化）',
        'fibonacci': '斐波那契数列（稳健）',
        'martingale': '马丁格尔翻倍（标准）',
        'reverse': '反向马丁格尔（趋势）',
        'aggressive': '激进马丁格尔（高风险）'
    }
    
    print("开始回测各策略...\n")
    
    results = {}
    for strategy_type, strategy_name in strategies.items():
        result = betting.simulate_strategy(
            predictions, actuals, strategy_type, hit_rate=actual_hit_rate
        )
        results[strategy_type] = {
            'name': strategy_name,
            'result': result
        }
    
    # 打印对比结果
    print("="*100)
    print("策略对比结果")
    print("="*100)
    print(f"测试期数: {test_periods}期")
    print(f"实际命中率: {actual_hit_rate*100:.2f}%\n")
    
    # 表头
    print(f"{'策略名称':<25} {'命中率':<10} {'总收益':<12} {'ROI':<10} {'最大回撤':<12} {'最大连亏':<10} {'风险评级'}")
    print("-"*100)
    
    # 按ROI排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['result']['roi'], reverse=True)
    
    for strategy_type, data in sorted_results:
        r = data['result']
        name = data['name']
        
        # 风险评级
        if r['max_drawdown'] < -100:
            risk = "⚠⚠⚠ 极高"
        elif r['max_drawdown'] < -50:
            risk = "⚠⚠ 高"
        elif r['max_drawdown'] < -30:
            risk = "⚠ 中"
        else:
            risk = "✓ 低"
        
        print(f"{name:<25} {r['hit_rate']*100:>7.2f}%  {r['total_profit']:>+10.2f}元  "
              f"{r['roi']:>+7.2f}%  {r['max_drawdown']:>+10.2f}元  "
              f"{r['max_consecutive_losses']:>7}期  {risk}")
    
    # 详细分析最佳策略
    print("\n" + "="*100)
    print("最佳策略详细分析")
    print("="*100)
    
    best_strategy_type = sorted_results[0][0]
    best_data = sorted_results[0][1]
    best_result = best_data['result']
    
    print(f"\n🏆 最佳策略: {best_data['name']}")
    print(f"\n【基础统计】")
    print(f"  测试期数: {best_result['total_periods']}")
    print(f"  命中次数: {best_result['wins']}")
    print(f"  未中次数: {best_result['losses']}")
    print(f"  命中率: {best_result['hit_rate']*100:.2f}%")
    
    print(f"\n【财务统计】")
    print(f"  总投注: {best_result['total_cost']:.2f}元")
    print(f"  总奖励: {best_result['total_reward']:.2f}元")
    print(f"  总收益: {best_result['total_profit']:+.2f}元")
    print(f"  平均每期收益: {best_result['avg_profit_per_period']:+.2f}元")
    print(f"  投资回报率: {best_result['roi']:+.2f}%")
    
    print(f"\n【风险指标】")
    print(f"  最大连续亏损: {best_result['max_consecutive_losses']}期")
    print(f"  最大回撤: {best_result['max_drawdown']:.2f}元")
    print(f"  最终余额: {best_result['final_balance']:+.2f}元")
    
    # 倍数使用统计
    mult_stats = {}
    for period in best_result['history']:
        mult = period['multiplier']
        mult_stats[mult] = mult_stats.get(mult, 0) + 1
    
    print(f"\n【倍数使用分布】")
    for mult in sorted(mult_stats.keys()):
        percentage = mult_stats[mult] / len(best_result['history']) * 100
        print(f"  {mult}倍: {mult_stats[mult]}期 ({percentage:.1f}%)")
    
    # 对比最差策略
    print("\n" + "="*100)
    print("最差策略分析")
    print("="*100)
    
    worst_data = sorted_results[-1][1]
    worst_result = worst_data['result']
    
    print(f"\n⚠️ 最差策略: {worst_data['name']}")
    print(f"  总收益: {worst_result['total_profit']:+.2f}元")
    print(f"  ROI: {worst_result['roi']:+.2f}%")
    print(f"  最大回撤: {worst_result['max_drawdown']:.2f}元")
    print(f"  最大连续亏损: {worst_result['max_consecutive_losses']}期")
    
    # 策略建议
    print("\n" + "="*100)
    print("策略选择建议")
    print("="*100)
    
    print(f"\n基于当前命中率 {actual_hit_rate*100:.2f}%：\n")
    
    if actual_hit_rate >= 0.6:
        print("  ✓ 命中率较高，建议使用：")
        print("    1. 凯利公式（最优风险收益比）")
        print("    2. 达朗贝尔（稳健增长）")
        print("    3. 反向马丁格尔（扩大连胜收益）")
    elif actual_hit_rate >= 0.5:
        print("  ⚠ 命中率中等，建议使用：")
        print("    1. 固定倍数（控制风险）")
        print("    2. 达朗贝尔（温和倍投）")
        print("    3. 避免激进策略")
    else:
        print("  ⚠⚠ 命中率偏低，建议：")
        print("    1. 仅使用固定倍数")
        print("    2. 不建议任何倍投策略")
        print("    3. 优先改进预测模型")
    
    print("\n风险偏好建议：")
    print("  - 保守型：固定倍数、达朗贝尔")
    print("  - 稳健型：凯利公式、斐波那契")
    print("  - 激进型：马丁格尔、反向马丁格尔")
    print("  - 高风险：激进马丁格尔（不推荐）")
    
    print("\n" + "="*100)
    print("✅ 测试完成！")
    print("="*100)
    
    return results, sorted_results

if __name__ == "__main__":
    results, sorted_results = compare_all_strategies()
