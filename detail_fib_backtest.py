#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
纯Fibonacci 10倍限制 - 最近300期详细回测
"""

import pandas as pd
from precise_top15_predictor import PreciseTop15Predictor

# Fibonacci序列
fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

def main():
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    
    config = {
        'max_multiplier': 10,
        'base_bet': 15,
        'win_reward': 45
    }
    
    test_periods = 300
    start_idx = len(df) - test_periods
    
    print('=' * 100)
    print('纯Fibonacci 10倍限制 - 最近300期详细回测')
    print('=' * 100)
    print(f'数据范围: {df.iloc[start_idx]["date"]} ~ {df.iloc[-1]["date"]}')
    print(f'配置: 倍数上限={config["max_multiplier"]}x, 基础投注={config["base_bet"]}元, 中奖奖励={config["win_reward"]}元')
    print('=' * 100)
    
    predictor = PreciseTop15Predictor()
    
    fib_index = 0
    total_bet = 0
    total_win = 0
    balance = 0
    min_balance = 0
    max_drawdown = 0
    hits = 0
    
    # 详细记录
    details = []
    
    print(f"\n{'期号':>4} {'日期':>12} {'实际':>4} {'倍数':>6} {'投注':>8} {'结果':>6} {'盈亏':>8} {'累计':>10} {'回撤':>8}")
    print('-' * 100)
    
    for i in range(start_idx, len(df)):
        period_num = i - start_idx + 1
        date = df.iloc[i]['date']
        actual = df.iloc[i]['number']
        
        # 预测
        train_data = df.iloc[:i]['number'].values
        predictions = predictor.predict(train_data)
        hit = actual in predictions
        
        # 计算倍数（纯Fibonacci，上限10倍）
        if fib_index < len(fib_sequence):
            multiplier = min(fib_sequence[fib_index], config['max_multiplier'])
        else:
            multiplier = config['max_multiplier']
        
        # 投注
        bet = config['base_bet'] * multiplier
        total_bet += bet
        
        if hit:
            hits += 1
            win = config['win_reward'] * multiplier
            total_win += win
            profit = win - bet
            balance += profit
            result = '✅命中'
            fib_index = 0  # 重置
        else:
            profit = -bet
            balance += profit
            result = '❌未中'
            fib_index += 1
            if balance < min_balance:
                min_balance = balance
                max_drawdown = abs(min_balance)
        
        # 输出详情
        print(f"{period_num:>4} {date:>12} {actual:>4} {multiplier:>5.1f}x {bet:>7.0f}元 {result:>6} {profit:>+7.0f}元 {balance:>+9.0f}元 {max_drawdown:>7.0f}元")
        
        details.append({
            'period': period_num,
            'date': date,
            'actual': actual,
            'multiplier': multiplier,
            'bet': bet,
            'hit': hit,
            'profit': profit,
            'balance': balance,
            'drawdown': max_drawdown
        })
    
    print('-' * 100)
    
    # 统计
    roi = (balance / total_bet * 100) if total_bet > 0 else 0
    hit_rate = hits / test_periods * 100
    
    print(f"\n{'=' * 100}")
    print('【统计汇总】')
    print(f"{'=' * 100}")
    print(f"测试期数: {test_periods}期")
    print(f"命中次数: {hits}次 ({hit_rate:.1f}%)")
    print(f"总投注: {total_bet:.0f}元")
    print(f"总收益: {balance:+.0f}元")
    print(f"ROI: {roi:.2f}%")
    print(f"最大回撤: {max_drawdown:.0f}元")
    
    # 10倍投注统计
    ten_x_periods = [d for d in details if d['multiplier'] >= 10]
    ten_x_hits = [d for d in ten_x_periods if d['hit']]
    
    print(f"\n【10倍投注详情】")
    print(f"10倍投注次数: {len(ten_x_periods)}次")
    print(f"10倍命中次数: {len(ten_x_hits)}次 ({len(ten_x_hits)/len(ten_x_periods)*100:.1f}%)" if ten_x_periods else "无10倍投注")
    
    if ten_x_periods:
        print(f"\n{'期号':>4} {'日期':>12} {'实际':>4} {'结果':>6} {'盈亏':>8}")
        print('-' * 50)
        for d in ten_x_periods:
            result = '✅命中' if d['hit'] else '❌未中'
            print(f"{d['period']:>4} {d['date']:>12} {d['actual']:>4} {result:>6} {d['profit']:>+7.0f}元")
    
    # 高倍投注统计
    high_mult_periods = [d for d in details if d['multiplier'] >= 8]
    high_mult_hits = [d for d in high_mult_periods if d['hit']]
    
    print(f"\n【≥8倍投注详情】")
    print(f"高倍投注次数: {len(high_mult_periods)}次")
    print(f"高倍命中次数: {len(high_mult_hits)}次 ({len(high_mult_hits)/len(high_mult_periods)*100:.1f}%)" if high_mult_periods else "无高倍投注")
    
    if high_mult_periods:
        print(f"\n{'期号':>4} {'日期':>12} {'倍数':>6} {'实际':>4} {'结果':>6} {'盈亏':>8}")
        print('-' * 60)
        for d in high_mult_periods:
            result = '✅命中' if d['hit'] else '❌未中'
            print(f"{d['period']:>4} {d['date']:>12} {d['multiplier']:>5.1f}x {d['actual']:>4} {result:>6} {d['profit']:>+7.0f}元")


if __name__ == '__main__':
    main()
