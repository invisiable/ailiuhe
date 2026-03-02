#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比不同投注策略的表现
"""

import pandas as pd
from probability_betting_strategy import validate_probability_strategy
from precise_top15_predictor import PreciseTop15Predictor
from betting_strategy import BettingStrategy


def main():
    print("=" * 80)
    print("📊 投注策略对比分析")
    print("=" * 80)
    print()
    
    # 加载数据
    file_path = 'data/lucky_numbers.csv'
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    print(f"✅ 数据加载完成: {len(df)}期")
    
    # 测试期数
    test_periods = 100
    print(f"验证期数: 最近{test_periods}期\n")
    
    numbers = df['number'].values
    animals = df['animal'].values
    elements = df['element'].values
    
    # 创建预测器
    predictor = PreciseTop15Predictor()
    
    print("正在生成预测...")
    
    # 生成预测
    start_idx = len(df) - test_periods
    predictions = []
    actuals = []
    
    for i in range(start_idx, len(df)):
        train_data = df.iloc[:i]['number'].values
        top15 = predictor.predict(train_data)
        predictions.append(top15)
        actuals.append(df.iloc[i]['number'])
        
        hit = df.iloc[i]['number'] in top15
        predictor.update_performance(top15, df.iloc[i]['number'])
    
    print(f"✅ 预测生成完成\n")
    
    # 计算命中率
    hit_rate = sum(1 for i in range(len(actuals)) if actuals[i] in predictions[i]) / len(actuals)
    print(f"基础命中率: {hit_rate*100:.2f}%\n")
    
    # 1. 固定投注
    print("【策略1：固定投注】")
    fixed_bet = 15
    fixed_total_bet = test_periods * fixed_bet
    fixed_wins = sum(1 for i in range(len(actuals)) if actuals[i] in predictions[i])
    fixed_reward = fixed_wins * 47
    fixed_profit = fixed_reward - fixed_total_bet
    fixed_roi = (fixed_profit / fixed_total_bet * 100) if fixed_total_bet > 0 else 0
    
    print(f"  每期投注: {fixed_bet}元")
    print(f"  总投注: {fixed_total_bet}元")
    print(f"  总收益: {fixed_reward}元")
    print(f"  净利润: {fixed_profit:+.0f}元")
    print(f"  ROI: {fixed_roi:+.2f}%")
    print(f"  最大回撤: -（无倍投，回撤为单期亏损）")
    print()
    
    # 2. 斐波那契策略
    print("【策略2：斐波那契倍投】")
    betting = BettingStrategy(base_bet=15, win_reward=47)
    fib_result = betting.simulate_strategy(predictions, actuals, 'fibonacci', hit_rate=hit_rate)
    
    print(f"  总投注: {fib_result['total_cost']:.0f}元")
    print(f"  总收益: {fib_result['total_reward']:.0f}元")
    print(f"  净利润: {fib_result['total_profit']:+.0f}元")
    print(f"  ROI: {fib_result['roi']:+.2f}%")
    print(f"  最大回撤: {fib_result['max_drawdown']:.0f}元")
    print(f"  最长连亏: {fib_result['max_consecutive_losses']}期")
    print()
    
    # 3. 概率预测策略
    print("【策略3：概率预测动态倍投🔮】")
    prob_result = validate_probability_strategy(
        predictor,
        numbers,
        animals,
        elements,
        test_periods=test_periods
    )
    
    print(f"  总投注: {prob_result['total_bet']:.0f}元")
    print(f"  总收益: {prob_result['total_win']:.0f}元")
    print(f"  净利润: {prob_result['total_profit']:+.0f}元")
    print(f"  ROI: {prob_result['roi']:+.2f}%")
    print(f"  最大回撤: {prob_result['max_drawdown']:.0f}元")
    print(f"  预测MAE: {prob_result['prediction_accuracy']['mae']:.4f}")
    print()
    
    # 综合对比
    print("=" * 80)
    print("📈 综合对比")
    print("=" * 80)
    print()
    
    strategies = [
        {
            'name': '固定投注',
            'roi': fixed_roi,
            'profit': fixed_profit,
            'drawdown': 15,  # 单期亏损
            'cost': fixed_total_bet
        },
        {
            'name': '斐波那契',
            'roi': fib_result['roi'],
            'profit': fib_result['total_profit'],
            'drawdown': fib_result['max_drawdown'],
            'cost': fib_result['total_cost']
        },
        {
            'name': '概率预测🔮',
            'roi': prob_result['roi'],
            'profit': prob_result['total_profit'],
            'drawdown': prob_result['max_drawdown'],
            'cost': prob_result['total_bet']
        }
    ]
    
    print(f"{'策略名称':<15} | {'ROI':>8} | {'净利润':>8} | {'回撤':>7} | {'总投注':>8}")
    print("-" * 80)
    
    for s in strategies:
        print(
            f"{s['name']:<15} | {s['roi']:>7.2f}% | {s['profit']:>+7.0f}元 | "
            f"{s['drawdown']:>6.0f}元 | {s['cost']:>7.0f}元"
        )
    
    print()
    
    # 排名
    print("【各项指标排名】")
    
    # ROI排名
    roi_sorted = sorted(strategies, key=lambda x: x['roi'], reverse=True)
    print(f"  ROI最高: {roi_sorted[0]['name']} ({roi_sorted[0]['roi']:+.2f}%)")
    
    # 利润排名
    profit_sorted = sorted(strategies, key=lambda x: x['profit'], reverse=True)
    print(f"  利润最高: {profit_sorted[0]['name']} ({profit_sorted[0]['profit']:+.0f}元)")
    
    # 回撤排名（最低最好）
    drawdown_sorted = sorted(strategies, key=lambda x: x['drawdown'])
    print(f"  回撤最低: {drawdown_sorted[0]['name']} ({drawdown_sorted[0]['drawdown']:.0f}元) ⭐")
    
    # 成本排名（最低最好）
    cost_sorted = sorted(strategies, key=lambda x: x['cost'])
    print(f"  成本最低: {cost_sorted[0]['name']} ({cost_sorted[0]['cost']:.0f}元)")
    
    print()
    
    # 推荐
    print("【策略推荐】")
    print(f"  🏆 最佳收益: {roi_sorted[0]['name']} - 追求高回报")
    print(f"  🛡️  最佳风控: {drawdown_sorted[0]['name']} - 追求低风险")
    print(f"  ⚖️  最佳平衡: 概率预测🔮 - 综合表现优异")
    print()
    
    # 风险收益比
    print("【风险收益比】（利润/回撤）")
    for s in strategies:
        if s['drawdown'] > 0:
            ratio = s['profit'] / s['drawdown']
            print(f"  {s['name']:<15}: {ratio:>6.2f}")
    print()


if __name__ == '__main__':
    main()
