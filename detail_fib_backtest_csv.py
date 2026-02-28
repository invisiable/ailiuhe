#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
纯Fibonacci 10倍限制 - 最近300期详细回测 (输出CSV)
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
    
    print(f'纯Fibonacci 10倍限制 - 最近300期详细回测')
    print(f'数据范围: {df.iloc[start_idx]["date"]} ~ {df.iloc[-1]["date"]}')
    print(f'配置: 倍数上限={config["max_multiplier"]}x, 基础投注={config["base_bet"]}元, 中奖奖励={config["win_reward"]}元\n')
    
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
            result = '命中'
            fib_index = 0  # 重置
        else:
            profit = -bet
            balance += profit
            result = '未中'
            fib_index += 1
            if balance < min_balance:
                min_balance = balance
                max_drawdown = abs(min_balance)
        
        details.append({
            '期号': period_num,
            '日期': date,
            '实际号码': actual,
            '投注倍数': multiplier,
            '投注金额': bet,
            '结果': result,
            '盈亏': profit,
            '累计盈亏': balance,
            '最大回撤': max_drawdown,
            'Fib索引': fib_index
        })
    
    # 保存为CSV
    result_df = pd.DataFrame(details)
    result_df.to_csv('fib_300periods_detail.csv', index=False, encoding='utf-8-sig')
    print(f'详细数据已保存到: fib_300periods_detail.csv ({len(details)}期)')
    
    # 统计
    roi = (balance / total_bet * 100) if total_bet > 0 else 0
    hit_rate = hits / test_periods * 100
    
    print(f'\n统计汇总:')
    print(f'  测试期数: {test_periods}期')
    print(f'  命中次数: {hits}次 ({hit_rate:.1f}%)')
    print(f'  总投注: {total_bet:.0f}元')
    print(f'  总收益: {balance:+.0f}元')
    print(f'  ROI: {roi:.2f}%')
    print(f'  最大回撤: {max_drawdown:.0f}元')
    
    # 10倍投注统计
    ten_x_periods = [d for d in details if d['投注倍数'] >= 10]
    ten_x_hits = [d for d in ten_x_periods if d['结果'] == '命中']
    
    print(f'\n10倍投注统计:')
    print(f'  10倍投注次数: {len(ten_x_periods)}次')
    print(f'  10倍命中次数: {len(ten_x_hits)}次 ({len(ten_x_hits)/len(ten_x_periods)*100:.1f}%)' if ten_x_periods else '  无10倍投注')
    
    # 高倍投注统计
    high_mult_periods = [d for d in details if d['投注倍数'] >= 8]
    high_mult_hits = [d for d in high_mult_periods if d['结果'] == '命中']
    
    print(f'\n≥8倍投注统计:')
    print(f'  高倍投注次数: {len(high_mult_periods)}次')
    print(f'  高倍命中次数: {len(high_mult_hits)}次 ({len(high_mult_hits)/len(high_mult_periods)*100:.1f}%)' if high_mult_periods else '  无高倍投注')
    
    # 保存10倍和高倍详情
    if ten_x_periods:
        ten_x_df = pd.DataFrame(ten_x_periods)
        ten_x_df.to_csv('fib_10x_detail.csv', index=False, encoding='utf-8-sig')
        print(f'\n10倍投注详情已保存到: fib_10x_detail.csv')
    
    if high_mult_periods:
        high_df = pd.DataFrame(high_mult_periods)
        high_df.to_csv('fib_high_mult_detail.csv', index=False, encoding='utf-8-sig')
        print(f'≥8倍投注详情已保存到: fib_high_mult_detail.csv')


if __name__ == '__main__':
    main()
