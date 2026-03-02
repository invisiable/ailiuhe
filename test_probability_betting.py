#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试概率预测动态倍投策略
"""

import pandas as pd
from probability_betting_strategy import validate_probability_strategy
from precise_top15_predictor import PreciseTop15Predictor


def main():
    print("=" * 70)
    print("🔮 测试概率预测动态倍投策略")
    print("=" * 70)
    print()
    
    # 加载数据
    file_path = 'data/lucky_numbers.csv'
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    print(f"✅ 数据加载完成: {len(df)}期")
    print()
    
    # 提取数据
    numbers = df['number'].values
    animals = df['animal'].values
    elements = df['element'].values
    
    # 创建预测器
    predictor = PreciseTop15Predictor()
    
    # 测试不同期数
    test_periods_list = [50, 100]
    
    for test_periods in test_periods_list:
        print(f"\n{'='*70}")
        print(f"验证期数: {test_periods}期")
        print(f"{'='*70}\n")
        
        result = validate_probability_strategy(
            predictor,
            numbers,
            animals,
            elements,
            test_periods=test_periods
        )
        
        print(f"【基础统计】")
        print(f"  总期数: {result['total_periods']}")
        print(f"  命中: {result['wins']} | 未中: {result['losses']}")
        print(f"  命中率: {result['hit_rate']*100:.2f}%")
        print()
        
        print(f"【财务统计】")
        print(f"  总投注: {result['total_bet']:.0f}元")
        print(f"  总收益: {result['total_win']:.0f}元")
        print(f"  净利润: {result['total_profit']:+.0f}元")
        print(f"  ROI: {result['roi']:+.2f}%")
        print(f"  最大回撤: {result['max_drawdown']:.0f}元")
        print()
        
        # 预测准确性
        if result['prediction_accuracy']:
            acc = result['prediction_accuracy']
            print(f"【预测准确性】")
            print(f"  MAE: {acc['mae']:.4f}")
            print(f"  RMSE: {acc['rmse']:.4f}")
            print()
            
            if 'calibration' in acc and acc['calibration']:
                print(f"【概率校准度】")
                print(f"  范围          | 次数 | 预测概率 | 实际命中率 | 偏差")
                print(f"  {'-'*60}")
                for cal in acc['calibration']:
                    print(
                        f"  {cal['range']:>12} | {cal['count']:>4} | "
                        f"{cal['avg_predicted']:>7.1%} | {cal['avg_actual']:>9.1%} | "
                        f"{cal['bias']:>+6.1%}"
                    )
                print()
        
        # 对比固定投注
        fixed_bet_total = test_periods * 15
        fixed_profit = result['total_win'] - fixed_bet_total
        fixed_roi = (fixed_profit / fixed_bet_total * 100) if fixed_bet_total > 0 else 0
        
        print(f"【对比固定投注】")
        print(f"  固定投注: {fixed_bet_total}元")
        print(f"  固定利润: {fixed_profit:+.0f}元")
        print(f"  固定ROI: {fixed_roi:+.2f}%")
        print()
        print(f"  概率策略 vs 固定投注:")
        diff_profit = result['total_profit'] - fixed_profit
        diff_roi = result['roi'] - fixed_roi
        if diff_profit > 0:
            print(f"  ✅ 概率策略更优: 收益增加{diff_profit:+.0f}元, ROI提升{diff_roi:+.2f}%")
        else:
            print(f"  ⚠️  固定投注更优: 收益减少{abs(diff_profit):.0f}元, ROI降低{abs(diff_roi):.2f}%")
        print()
        
        # 显示最近10期详情
        print(f"【最近10期详情】")
        print(f"  期数  概率   倍数   投注   结果  利润")
        print(f"  {'-'*45}")
        for h in result['history'][-10:]:
            hit_mark = '✓' if h['hit'] else '✗'
            print(
                f"  {h['period']:>4}  {h['predicted_prob']:>5.1%}  {h['multiplier']:>5.1f}  "
                f"{h['bet']:>5.0f}元  {hit_mark:>3}  {h['profit']:>+6.0f}元"
            )
        print()


if __name__ == '__main__':
    main()
