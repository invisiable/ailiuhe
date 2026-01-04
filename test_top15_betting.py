"""
测试修改后的投注策略 - 购买全部TOP15
"""

import pandas as pd
from betting_strategy import BettingStrategy
from top15_predictor import Top15Predictor

def test_top15_betting():
    """测试TOP15投注策略"""
    print("="*70)
    print("测试修改后的投注策略：购买TOP15全部15个数字")
    print("="*70)
    
    # 加载历史数据
    df = pd.read_csv('lucky_numbers - 副本.csv')
    
    # 使用最近30期测试
    test_periods = 30
    start_idx = len(df) - test_periods
    
    predictor = Top15Predictor()
    predictions_top15 = []
    actuals = []
    
    print(f"\n生成{test_periods}期TOP15预测...\n")
    
    for i in range(start_idx, len(df)):
        train_data = df.iloc[:i]['number'].values
        analysis = predictor.get_analysis(train_data)
        top15 = analysis['top15']
        predictions_top15.append(top15)
        
        actual = df.iloc[i]['number']
        actuals.append(actual)
        
        # 显示前5期
        if i - start_idx < 5:
            hit = "✓" if actual in top15 else "✗"
            print(f"期号 {i+1}: TOP15={top15[:5]}... 实际={actual} {hit}")
    
    # 创建投注策略（每期15个数字，每个1元）
    betting = BettingStrategy(base_bet=15, win_reward=45, loss_penalty=15)
    
    print(f"\n执行投注策略回测...\n")
    
    # 使用达朗贝尔策略
    result = betting.simulate_strategy(predictions_top15, actuals, 'dalembert')
    
    print("="*70)
    print("回测结果")
    print("="*70)
    print(f"测试期数: {result['total_periods']}")
    print(f"命中次数: {result['wins']}")
    print(f"未中次数: {result['losses']}")
    print(f"命中率: {result['hit_rate']*100:.2f}%")
    print(f"\n总投注: {result['total_cost']:.2f}元")
    print(f"总收益: {result['total_profit']:+.2f}元")
    print(f"投资回报率: {result['roi']:+.2f}%")
    print(f"最大回撤: {result['max_drawdown']:.2f}元")
    
    # 显示最近5期详情
    print(f"\n最近5期详情：")
    print(f"{'期号':<6} {'倍数':<6} {'投注':<10} {'结果':<6} {'盈亏':<12}")
    print("-"*50)
    for period in result['history'][-5:]:
        print(f"{period['period']:<6} {period['multiplier']:<6} "
              f"{period['bet_amount']:<10.2f} {period['result']:<6} "
              f"{period['profit']:>+12.2f}")
    
    # 生成下期建议
    print(f"\n" + "="*70)
    print("下期投注建议")
    print("="*70)
    
    # 使用全部数据预测下期
    all_data = df['number'].values
    next_analysis = predictor.get_analysis(all_data)
    next_top15 = next_analysis['top15']
    
    print(f"\n预测方法: ⭐综合预测Top15 (60%成功率)")
    print(f"当前趋势: {next_analysis['trend']}")
    print(f"\nTOP15预测: {next_top15}")
    print(f"建议购买: 全部15个数字")
    
    # 计算建议倍数
    last_periods = result['history'][-5:]
    consecutive_losses = 0
    total_loss = 0
    
    for period in reversed(last_periods):
        if period['result'] == 'LOSS':
            consecutive_losses += 1
            total_loss += period.get('loss', 0)
        else:
            break
    
    recommendation = betting.generate_next_bet_recommendation(
        consecutive_losses=consecutive_losses,
        total_loss=total_loss,
        strategy_type='dalembert'
    )
    
    print(f"\n连续亏损: {consecutive_losses}期")
    print(f"建议倍数: {recommendation['recommended_multiplier']}倍")
    print(f"投注金额: {recommendation['recommended_bet']:.2f}元 "
          f"({recommendation['bet_per_number']:.2f}元/号 × 15号)")
    print(f"\n如果命中: +{recommendation['potential_profit_if_win']:.2f}元 ✓")
    print(f"如果未中: -{recommendation['potential_loss_if_miss']:.2f}元")
    
    print("\n✅ 测试完成！")

if __name__ == "__main__":
    test_top15_betting()
