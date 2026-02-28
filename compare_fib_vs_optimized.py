#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比: 纯Fibonacci 10倍限制 vs 优化配置
"""

import pandas as pd
from precise_top15_predictor import PreciseTop15Predictor

# Fibonacci序列
fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

def backtest_config(name, use_dynamic, config, df, test_periods):
    predictor = PreciseTop15Predictor()
    
    fib_index = 0
    recent_results = []
    total_bet = 0
    total_win = 0
    balance = 0
    min_balance = 0
    max_drawdown = 0
    hits = 0
    ten_x_count = 0
    high_mult_count = 0
    
    start_idx = len(df) - test_periods
    
    for i in range(start_idx, len(df)):
        train_data = df.iloc[:i]['number'].values
        predictions = predictor.predict(train_data)
        actual = df.iloc[i]['number']
        hit = actual in predictions
        
        # 计算基础倍数
        if fib_index < len(fib_sequence):
            base_mult = min(fib_sequence[fib_index], config['max_multiplier'])
        else:
            base_mult = min(fib_sequence[-1], config['max_multiplier'])
        
        # 动态调整
        if use_dynamic and len(recent_results) >= config['lookback']:
            rate = sum(recent_results) / len(recent_results)
            if rate >= config['good_thresh']:
                multiplier = min(base_mult * config['boost_mult'], config['max_multiplier'])
            elif rate <= config['bad_thresh']:
                multiplier = max(base_mult * config['reduce_mult'], 1)
            else:
                multiplier = base_mult
        else:
            multiplier = base_mult
        
        # 投注
        bet = config['base_bet'] * multiplier
        total_bet += bet
        
        if hit:
            hits += 1
            win = config['win_reward'] * multiplier
            total_win += win
            balance += (win - bet)
            fib_index = 0
        else:
            balance -= bet
            fib_index += 1
            if balance < min_balance:
                min_balance = balance
                max_drawdown = abs(min_balance)
        
        if multiplier >= 10:
            ten_x_count += 1
        if multiplier >= 8:
            high_mult_count += 1
        
        # 更新历史
        recent_results.append(1 if hit else 0)
        if len(recent_results) > config.get('lookback', 10):
            recent_results.pop(0)
    
    roi = (balance / total_bet * 100) if total_bet > 0 else 0
    hit_rate = hits / test_periods * 100
    return {
        'name': name, 
        'roi': roi, 
        'profit': balance, 
        'drawdown': max_drawdown, 
        'ten_x': ten_x_count,
        'high_mult': high_mult_count,
        'total_bet': total_bet,
        'hit_rate': hit_rate
    }

def main():
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    test_periods = min(300, len(df) - 50)
    
    configs = [
        ('纯Fibonacci 10倍限制 (无动态)', False, {
            'max_multiplier': 10, 'base_bet': 15, 'win_reward': 45,
            'lookback': 10, 'good_thresh': 0.30, 'bad_thresh': 0.20, 
            'boost_mult': 1.0, 'reduce_mult': 1.0
        }),
        ('优化配置 (lookback=10, boost=1.2x)', True, {
            'max_multiplier': 10, 'base_bet': 15, 'win_reward': 45,
            'lookback': 10, 'good_thresh': 0.30, 'bad_thresh': 0.20, 
            'boost_mult': 1.2, 'reduce_mult': 0.5
        }),
        ('原版配置 (lookback=8, boost=1.5x)', True, {
            'max_multiplier': 10, 'base_bet': 15, 'win_reward': 45,
            'lookback': 8, 'good_thresh': 0.35, 'bad_thresh': 0.20, 
            'boost_mult': 1.5, 'reduce_mult': 0.6
        }),
    ]
    
    print('=' * 90)
    print('纯Fibonacci 10倍限制 vs 动态调整配置 对比')
    print('=' * 90)
    print(f'数据: {len(df)}期, 测试: {test_periods}期\n')
    
    results = []
    for name, use_dynamic, cfg in configs:
        print(f'测试: {name}...')
        r = backtest_config(name, use_dynamic, cfg, df, test_periods)
        results.append(r)
    
    print('\n' + '=' * 90)
    print('【回测结果对比】')
    print('=' * 90)
    
    header = f"{'配置':<40}{'ROI':>8}{'净收益':>10}{'回撤':>8}{'10x':>6}{'总投注':>10}"
    print(header)
    print('-' * 90)
    
    for r in results:
        line = f"{r['name']:<40}{r['roi']:>7.2f}%{r['profit']:>+9.0f}元{r['drawdown']:>7.0f}元{r['ten_x']:>5}次{r['total_bet']:>9.0f}元"
        print(line)
    
    print('\n' + '=' * 90)
    print('【对比分析】')
    print('=' * 90)
    
    baseline = results[0]  # 纯Fibonacci
    optimized = results[1]  # 优化配置
    original = results[2]   # 原版配置
    
    print(f"\n📊 纯Fibonacci 10倍限制 vs 优化配置:")
    print(f"   ROI: {baseline['roi']:.2f}% → {optimized['roi']:.2f}% ({optimized['roi']-baseline['roi']:+.2f}%)")
    print(f"   收益: {baseline['profit']:+.0f}元 → {optimized['profit']:+.0f}元 ({optimized['profit']-baseline['profit']:+.0f}元)")
    print(f"   回撤: {baseline['drawdown']:.0f}元 → {optimized['drawdown']:.0f}元 ({optimized['drawdown']-baseline['drawdown']:+.0f}元)")
    
    print(f"\n📊 纯Fibonacci 10倍限制 vs 原版配置:")
    print(f"   ROI: {baseline['roi']:.2f}% → {original['roi']:.2f}% ({original['roi']-baseline['roi']:+.2f}%)")
    print(f"   收益: {baseline['profit']:+.0f}元 → {original['profit']:+.0f}元 ({original['profit']-baseline['profit']:+.0f}元)")
    print(f"   回撤: {baseline['drawdown']:.0f}元 → {original['drawdown']:.0f}元 ({original['drawdown']-baseline['drawdown']:+.0f}元)")
    
    print('\n' + '=' * 90)
    print('【结论】')
    print('=' * 90)
    
    # 找最优
    best_roi = max(results, key=lambda x: x['roi'])
    best_profit = max(results, key=lambda x: x['profit'])
    best_drawdown = min(results, key=lambda x: x['drawdown'])
    
    print(f"\n🏆 最高ROI: {best_roi['name']}")
    print(f"   ROI {best_roi['roi']:.2f}%, 收益 {best_roi['profit']:+.0f}元, 回撤 {best_roi['drawdown']:.0f}元")
    
    print(f"\n🏆 最低回撤: {best_drawdown['name']}")
    print(f"   ROI {best_drawdown['roi']:.2f}%, 收益 {best_drawdown['profit']:+.0f}元, 回撤 {best_drawdown['drawdown']:.0f}元")
    
    # 综合评分
    for r in results:
        max_roi = max(x['roi'] for x in results)
        min_roi = min(x['roi'] for x in results)
        max_profit = max(x['profit'] for x in results)
        min_profit = min(x['profit'] for x in results)
        max_dd = max(x['drawdown'] for x in results)
        min_dd = min(x['drawdown'] for x in results)
        
        if max_roi > min_roi:
            roi_score = (r['roi'] - min_roi) / (max_roi - min_roi) * 100
        else:
            roi_score = 50
        if max_profit > min_profit:
            profit_score = (r['profit'] - min_profit) / (max_profit - min_profit) * 100
        else:
            profit_score = 50
        if max_dd > min_dd:
            dd_score = (max_dd - r['drawdown']) / (max_dd - min_dd) * 100
        else:
            dd_score = 50
        
        r['score'] = roi_score * 0.4 + profit_score * 0.3 + dd_score * 0.3
    
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
    
    print(f"\n📈 综合评分排名 (ROI 40% + 收益 30% + 低回撤 30%):")
    for i, r in enumerate(results_sorted):
        marker = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        print(f"   {marker} {r['name']:<45} ({r['score']:.1f}分)")
    
    winner = results_sorted[0]
    print(f"\n✅ 推荐使用: {winner['name']}")
    print(f"   ROI: {winner['roi']:.2f}%, 收益: {winner['profit']:+.0f}元, 回撤: {winner['drawdown']:.0f}元")


if __name__ == '__main__':
    main()
