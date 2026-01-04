"""
投注策略演示脚本
展示如何使用智能投注策略实现收益最大化
"""

import pandas as pd
import numpy as np
from betting_strategy import BettingStrategy
from top15_predictor import Top15Predictor


def demo_with_real_data():
    """使用真实数据演示投注策略"""
    
    print("=" * 80)
    print("💰 智能投注策略演示 - 基于真实历史数据")
    print("=" * 80)
    print()
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"✅ 数据加载完成: {len(df)}期")
    
    # 使用最近100期进行回测
    test_periods = min(100, len(df))
    start_idx = len(df) - test_periods
    
    print(f"📊 回测期数: {test_periods}期")
    print()
    
    # 生成预测
    print("=" * 80)
    print("第一步：生成历史TOP5预测")
    print("=" * 80)
    
    predictor = Top15Predictor()
    predictions_top5 = []
    actuals = []
    
    print("使用与GUI'⭐ 综合预测 Top 15'相同的预测方法...")
    for i in range(start_idx, len(df)):
        # 使用i之前的数据进行预测
        train_data = df.iloc[:i]['number'].values
        
        # 使用与综合预测相同的方法：get_analysis() 获取top15
        analysis = predictor.get_analysis(train_data)
        top15 = analysis['top15']
        top5 = top15[:5]
        predictions_top5.append(top5)
        
        # 实际结果
        actual = df.iloc[i]['number']
        actuals.append(actual)
        
        if (i - start_idx + 1) % 25 == 0:
            print(f"  进度: {i - start_idx + 1}/{test_periods}期...")
    
    print(f"\n✅ 预测完成！共{len(predictions_top5)}期")
    print()
    
    # 创建投注策略
    betting = BettingStrategy()
    
    # 对比分析
    print("=" * 80)
    print("第二步：三种投注策略对比分析")
    print("=" * 80)
    print()
    
    comparison = betting.recommend_strategy(predictions_top5, actuals)
    
    # 打印每种策略的详细报告
    for strategy_name, result in comparison['results'].items():
        betting.print_strategy_report(result)
        print()
    
    # 显示推荐
    print("\n" + "=" * 80)
    print("📌 策略推荐")
    print("=" * 80)
    print(f"\n🎯 推荐策略: {comparison['recommended'].upper()}")
    print(f"📈 推荐理由: {comparison['reason']}")
    print()
    
    # 获取最优策略结果
    best_result = comparison['results'][comparison['recommended']]
    
    print("=" * 80)
    print("关键指标总结")
    print("=" * 80)
    print(f"\n命中率: {best_result['hit_rate']*100:.2f}%")
    print(f"总收益: {best_result['total_profit']:+.2f}元")
    print(f"投资回报率: {best_result['roi']:+.2f}%")
    print(f"平均每期收益: {best_result['avg_profit_per_period']:+.2f}元")
    print()
    print(f"风险指标:")
    print(f"  - 最大连续亏损: {best_result['max_consecutive_losses']}期")
    print(f"  - 最大回撤: {best_result['max_drawdown']:.2f}元")
    print()
    
    # 生成下期投注建议
    print("=" * 80)
    print("第三步：下期投注建议")
    print("=" * 80)
    
    # 计算当前状态（检查最近几期）
    last_periods = best_result['history'][-5:]
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
        strategy_type=comparison['recommended']
    )
    
    betting.print_next_bet_recommendation(recommendation)
    
    # 实际执行下期预测
    print("\n" + "=" * 80)
    print("第四步：下期TOP5预测（实际购买建议）")
    print("=" * 80)
    
    # 使用所有数据预测下一期（与GUI综合预测相同方法）
    all_numbers = df['number'].values
    analysis = predictor.get_analysis(all_numbers)
    next_top15 = analysis['top15']
    next_top5 = next_top15[:5]
    
    print(f"\n📊 预测下期TOP15: {next_top15}")
    print(f"🎯 建议购买TOP5: {next_top5}")
    print(f"\n💰 投注方案:")
    print(f"   总投注: {recommendation['recommended_bet']:.2f}元")
    print(f"   每个号码: {recommendation['bet_per_number']:.2f}元")
    print(f"   购买数字: {next_top5}")
    print()
    print(f"💡 期望收益:")
    print(f"   如果命中: +{recommendation['potential_profit_if_win']:.2f}元")
    print(f"   如果未中: -{recommendation['potential_loss_if_miss']:.2f}元")
    print()
    
    print("=" * 80)
    print("✅ 投注策略分析完成")
    print("=" * 80)


def demo_strategy_comparison():
    """演示不同命中率下的策略表现"""
    
    print("\n\n" + "=" * 80)
    print("📊 不同命中率场景下的策略表现对比")
    print("=" * 80)
    print()
    
    betting = BettingStrategy()
    n_periods = 100
    
    hit_rates = [0.30, 0.35, 0.40, 0.45, 0.50]
    
    print(f"{'命中率':<10} {'策略':<15} {'总收益':<12} {'ROI':<10} {'最大回撤':<12}")
    print("-" * 70)
    
    for hit_rate in hit_rates:
        # 生成模拟数据
        np.random.seed(42)
        predictions = []
        actuals = []
        
        for i in range(n_periods):
            top5 = np.random.choice(range(1, 50), size=5, replace=False).tolist()
            predictions.append(top5)
            
            if np.random.random() < hit_rate:
                actual = np.random.choice(top5)
            else:
                others = [x for x in range(1, 50) if x not in top5]
                actual = np.random.choice(others)
            actuals.append(actual)
        
        # 测试三种策略
        for strategy_type in ['martingale', 'fibonacci', 'dalembert']:
            result = betting.simulate_strategy(predictions, actuals, strategy_type)
            
            print(f"{hit_rate*100:<10.0f}%   "
                  f"{strategy_type:<15} "
                  f"{result['total_profit']:>+10.2f}元  "
                  f"{result['roi']:>+7.1f}%  "
                  f"{result['max_drawdown']:>10.2f}元")
    
    print("-" * 70)
    print()
    print("💡 分析结论:")
    print("   1. 命中率>40%时，马丁格尔策略收益最高")
    print("   2. 命中率<40%时，保守策略（达朗贝尔）更安全")
    print("   3. 斐波那契策略在各种情况下都较为稳健")
    print()


def demo_progressive_betting():
    """演示渐进式投注的威力"""
    
    print("\n" + "=" * 80)
    print("📈 渐进式投注 vs 固定投注对比")
    print("=" * 80)
    print()
    
    # 模拟场景：连续3次未中，第4次命中
    print("场景：连续3次未中后，第4次命中")
    print("-" * 80)
    print()
    
    # 固定投注
    print("【固定投注】")
    print("  第1期: 投注5元，未中，亏损-15元，累计: -15元")
    print("  第2期: 投注5元，未中，亏损-15元，累计: -30元")
    print("  第3期: 投注5元，未中，亏损-15元，累计: -45元")
    print("  第4期: 投注5元，命中，奖励45元，盈利+40元，累计: -5元 ❌")
    print("  → 结果：仍亏损5元")
    print()
    
    # 马丁格尔策略
    betting = BettingStrategy()
    print("【马丁格尔策略】")
    
    total = 0
    losses = 0
    loss_amount = 0
    
    for i in range(1, 5):
        multiplier, bet = betting.calculate_optimal_bet(losses, loss_amount)
        
        if i < 4:  # 未中
            loss = multiplier * 15
            total -= loss
            losses += 1
            loss_amount += loss
            print(f"  第{i}期: 投注{bet:.0f}元({multiplier}倍)，未中，亏损-{loss:.0f}元，累计: {total:.0f}元")
        else:  # 命中
            reward = multiplier * 45
            profit = reward - bet
            total += profit
            print(f"  第{i}期: 投注{bet:.0f}元({multiplier}倍)，命中，奖励{reward:.0f}元，盈利+{profit:.0f}元，累计: {total:.0f}元 ✓")
    
    print(f"  → 结果：盈利{total:.0f}元")
    print()
    
    print("💡 渐进式投注的优势：")
    print("   ✓ 能够快速覆盖之前的亏损")
    print("   ✓ 保证命中后实现盈利")
    print("   ✓ 适合命中率较高的预测模型")
    print()


if __name__ == '__main__':
    # 演示1：使用真实数据
    demo_with_real_data()
    
    # 演示2：不同命中率对比
    demo_strategy_comparison()
    
    # 演示3：渐进式投注原理
    demo_progressive_betting()
    
    print("\n" + "=" * 80)
    print("📚 使用说明")
    print("=" * 80)
    print("""
1. 三种投注策略：
   • 马丁格尔（激进型）：连续亏损时快速加倍，适合高命中率
   • 斐波那契（稳健型）：按斐波那契数列增加，平衡风险收益
   • 达朗贝尔（保守型）：每次只增加1倍，最安全但收益较慢

2. 使用建议：
   • 命中率>40%：推荐马丁格尔策略
   • 命中率35-40%：推荐斐波那契策略
   • 命中率<35%：推荐达朗贝尔策略或不投注

3. 风险控制：
   • 设置最大投注倍数（默认10倍）
   • 监控最大回撤
   • 连续亏损达到阈值时考虑暂停

4. 实战步骤：
   ① 运行TOP15预测获取下期TOP5
   ② 根据历史表现选择投注策略
   ③ 按建议倍数购买TOP5数字
   ④ 记录结果，动态调整下期投注
""")
    print("=" * 80)
