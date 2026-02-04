"""
测试增强的投注策略 - 包含倍投规则和基于成功率的最佳策略
"""

import pandas as pd
from betting_strategy import BettingStrategy
from top15_predictor import Top15Predictor

def test_enhanced_strategy():
    """测试增强的投注策略功能"""
    print("="*80)
    print("增强投注策略测试 - 倍投规则 + 成功率策略")
    print("="*80)
    
    df = pd.read_csv('lucky_numbers - 副本.csv')
    
    # 测试场景1：连续亏损的情况
    print("\n【场景1：模拟连续亏损3期的倍投建议】")
    print("-"*80)
    
    betting = BettingStrategy(base_bet=15, win_reward=47, loss_penalty=15)
    
    consecutive_losses = 3
    total_loss = 45  # 3期 × 15元
    
    recommendation = betting.generate_next_bet_recommendation(
        consecutive_losses=consecutive_losses,
        total_loss=total_loss,
        strategy_type='dalembert'
    )
    
    print(f"当前状态：连续亏损 {consecutive_losses} 期")
    print(f"累计亏损：{total_loss:.2f}元")
    print(f"\n倍投规则说明：")
    print(f"  - 基础投注：15元（15个数字 × 1元）")
    print(f"  - 达朗贝尔策略：每连续亏损1期，倍数+1")
    print(f"  - 当前建议倍数：{recommendation['recommended_multiplier']}倍")
    print(f"  - 当前建议投注：{recommendation['recommended_bet']:.2f}元")
    
    # 计算需要多少倍才能覆盖亏损
    min_multiplier = int(total_loss / 30) + 1  # 30是单倍命中的净收益
    print(f"\n理论覆盖亏损所需倍数：{min_multiplier}倍")
    print(f"  - 计算方式：累计亏损({total_loss}元) ÷ 单倍净收益(30元) + 1 = {min_multiplier}倍")
    
    if recommendation['potential_profit_if_win'] >= total_loss:
        print(f"  ✓ 如果下期命中，可完全覆盖累计亏损并盈利")
        print(f"    命中收益：{recommendation['potential_profit_if_win']:.2f}元")
        print(f"    覆盖后盈利：{recommendation['potential_profit_if_win'] - total_loss:.2f}元")
    else:
        recovery = total_loss - recommendation['potential_profit_if_win']
        print(f"  ⚠ 如果下期命中，仍有{recovery:.2f}元未覆盖")
    
    # 测试场景2：不同命中率下的最佳策略
    print("\n\n【场景2：不同命中率下的最佳投注策略】")
    print("-"*80)
    
    hit_rates = [0.7, 0.55, 0.45, 0.35]
    
    for hit_rate in hit_rates:
        print(f"\n命中率：{hit_rate*100:.0f}%")
        
        # 计算期望收益
        expected_profit = hit_rate * 30 - (1 - hit_rate) * 15
        print(f"  期望收益/期：{expected_profit:+.2f}元")
        
        # 策略建议
        if hit_rate >= 0.6:
            strategy = "✓ 积极投注"
            max_mult = "最高5倍"
            control = "单期最高不超过5倍"
        elif hit_rate >= 0.5:
            strategy = "⚠ 谨慎投注"
            max_mult = "最高3倍"
            control = "连续亏损3期应考虑暂停"
        elif hit_rate >= 0.4:
            strategy = "⚠⚠ 保守投注"
            max_mult = "仅1倍"
            control = "连续亏损2期应立即停止"
        else:
            strategy = "❌ 暂停投注"
            max_mult = "不建议"
            control = "重新评估预测模型"
        
        print(f"  策略建议：{strategy}")
        print(f"  倍投上限：{max_mult}")
        print(f"  风险控制：{control}")
        
        if expected_profit > 0:
            periods_to_100 = 100 / expected_profit
            print(f"  预计{periods_to_100:.0f}期可盈利100元")
        else:
            print(f"  ❌ 长期期望为负，不建议持续投注")
    
    # 测试场景3：实际数据回测
    print("\n\n【场景3：实际数据回测 - 验证倍投效果】")
    print("-"*80)
    
    test_periods = 30
    start_idx = len(df) - test_periods
    
    predictor = Top15Predictor()
    predictions = []
    actuals = []
    
    for i in range(start_idx, len(df)):
        train_data = df.iloc[:i]['number'].values
        analysis = predictor.get_analysis(train_data)
        predictions.append(analysis['top15'])
        actuals.append(df.iloc[i]['number'])
    
    result = betting.simulate_strategy(predictions, actuals, 'dalembert')
    
    print(f"\n回测结果（{test_periods}期）：")
    print(f"  命中率：{result['hit_rate']*100:.2f}%")
    print(f"  总收益：{result['total_profit']:+.2f}元")
    print(f"  ROI：{result['roi']:+.2f}%")
    print(f"  最大连续亏损：{result['max_consecutive_losses']}期")
    print(f"  最大回撤：{result['max_drawdown']:.2f}元")
    
    # 显示倍投效果
    print(f"\n倍投效果分析：")
    max_mult_used = max([p['multiplier'] for p in result['history']])
    print(f"  最高使用倍数：{max_mult_used}倍")
    
    # 统计各倍数使用次数
    mult_stats = {}
    for p in result['history']:
        mult = p['multiplier']
        mult_stats[mult] = mult_stats.get(mult, 0) + 1
    
    print(f"  倍数分布：")
    for mult in sorted(mult_stats.keys()):
        print(f"    {mult}倍：{mult_stats[mult]}期")
    
    # 最后5期详情
    print(f"\n最后5期详情：")
    print(f"{'期号':<6} {'倍数':<6} {'投注':<10} {'结果':<6} {'盈亏':<12}")
    print("-"*50)
    for period in result['history'][-5:]:
        print(f"{period['period']:<6} {period['multiplier']:<6} "
              f"{period['bet_amount']:<10.2f} {period['result']:<6} "
              f"{period['profit']:>+12.2f}")
    
    print("\n" + "="*80)
    print("✅ 测试完成！")
    print("="*80)

if __name__ == "__main__":
    test_enhanced_strategy()
