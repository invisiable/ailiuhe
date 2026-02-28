#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化配置 vs 原版动态配置 - 详细对比
"""

import pandas as pd
from precise_top15_predictor import PreciseTop15Predictor

# Fibonacci序列
fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

def backtest_dynamic_config(config, df, test_periods):
    """回测动态配置，返回每期详情"""
    
    predictor = PreciseTop15Predictor()
    
    fib_index = 0
    recent_results = []
    total_bet = 0
    total_win = 0
    balance = 0
    min_balance = 0
    max_drawdown = 0
    hits = 0
    
    details = []
    start_idx = len(df) - test_periods
    
    for i in range(start_idx, len(df)):
        period_num = i - start_idx + 1
        date = df.iloc[i]['date']
        actual = df.iloc[i]['number']
        
        # 预测
        train_data = df.iloc[:i]['number'].values
        predictions = predictor.predict(train_data)
        hit = actual in predictions
        
        # 计算基础倍数
        if fib_index < len(fib_sequence):
            base_mult = min(fib_sequence[fib_index], config['max_multiplier'])
        else:
            base_mult = config['max_multiplier']
        
        # 动态调整（基于投注前的历史数据）
        if len(recent_results) >= config['lookback']:
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
            profit = win - bet
            balance += profit
            result = '命中'
            fib_index = 0
        else:
            profit = -bet
            balance += profit
            result = '未中'
            fib_index += 1
            if balance < min_balance:
                min_balance = balance
                max_drawdown = abs(min_balance)
        
        recent_rate = sum(recent_results) / len(recent_results) if recent_results else 0
        
        details.append({
            '期号': period_num,
            '日期': date,
            '实际号码': actual,
            '基础倍数': base_mult,
            '投注倍数': multiplier,
            '投注金额': bet,
            '结果': result,
            '盈亏': profit,
            '累计盈亏': balance,
            '最大回撤': max_drawdown,
            '近期命中率': recent_rate,
            'Fib索引': fib_index
        })
        
        # 更新历史（在投注后）
        recent_results.append(1 if hit else 0)
        if len(recent_results) > config['lookback']:
            recent_results.pop(0)
    
    return details, {
        'total_bet': total_bet,
        'balance': balance,
        'max_drawdown': max_drawdown,
        'hits': hits,
        'hit_rate': hits / test_periods * 100,
        'roi': (balance / total_bet * 100) if total_bet > 0 else 0
    }


def main():
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    test_periods = 300
    
    # 两个配置
    configs = {
        '原版配置': {
            'max_multiplier': 10, 'base_bet': 15, 'win_reward': 45,
            'lookback': 8, 'good_thresh': 0.35, 'bad_thresh': 0.20,
            'boost_mult': 1.5, 'reduce_mult': 0.6
        },
        '优化配置': {
            'max_multiplier': 10, 'base_bet': 15, 'win_reward': 45,
            'lookback': 10, 'good_thresh': 0.30, 'bad_thresh': 0.20,
            'boost_mult': 1.2, 'reduce_mult': 0.5
        }
    }
    
    print('=' * 100)
    print('优化配置 vs 原版动态配置 - 详细对比')
    print('=' * 100)
    print(f'数据: {len(df)}期, 测试: {test_periods}期')
    print(f'时间范围: {df.iloc[len(df)-test_periods]["date"]} ~ {df.iloc[-1]["date"]}')
    
    print('\n【配置参数对比】')
    print(f'{"参数":<15} {"原版配置":<20} {"优化配置":<20}')
    print('-' * 55)
    print(f'{"回看窗口":<15} {configs["原版配置"]["lookback"]}期{"":<17} {configs["优化配置"]["lookback"]}期')
    print(f'{"增强阈值":<15} {configs["原版配置"]["good_thresh"]*100:.0f}%{"":<18} {configs["优化配置"]["good_thresh"]*100:.0f}%')
    print(f'{"降低阈值":<15} {configs["原版配置"]["bad_thresh"]*100:.0f}%{"":<18} {configs["优化配置"]["bad_thresh"]*100:.0f}%')
    print(f'{"增强倍数":<15} {configs["原版配置"]["boost_mult"]}x{"":<17} {configs["优化配置"]["boost_mult"]}x')
    print(f'{"降低倍数":<15} {configs["原版配置"]["reduce_mult"]}x{"":<17} {configs["优化配置"]["reduce_mult"]}x')
    
    # 回测两个配置
    results = {}
    all_details = {}
    for name, cfg in configs.items():
        print(f'\n测试: {name}...')
        details, stats = backtest_dynamic_config(cfg, df, test_periods)
        results[name] = stats
        all_details[name] = details
    
    # 统计对比
    print('\n' + '=' * 100)
    print('【性能指标对比】')
    print('=' * 100)
    
    orig = results['原版配置']
    opt = results['优化配置']
    
    print(f'\n{"指标":<15} {"原版配置":<15} {"优化配置":<15} {"差异":<15}')
    print('-' * 60)
    print(f'{"命中率":<15} {orig["hit_rate"]:.1f}%{"":<12} {opt["hit_rate"]:.1f}%{"":<12} {opt["hit_rate"]-orig["hit_rate"]:+.1f}%')
    print(f'{"ROI":<15} {orig["roi"]:.2f}%{"":<11} {opt["roi"]:.2f}%{"":<11} {opt["roi"]-orig["roi"]:+.2f}%')
    print(f'{"净收益":<15} {orig["balance"]:+.0f}元{"":<10} {opt["balance"]:+.0f}元{"":<10} {opt["balance"]-orig["balance"]:+.0f}元')
    print(f'{"最大回撤":<15} {orig["max_drawdown"]:.0f}元{"":<11} {opt["max_drawdown"]:.0f}元{"":<11} {opt["max_drawdown"]-orig["max_drawdown"]:+.0f}元')
    print(f'{"总投注":<15} {orig["total_bet"]:.0f}元{"":<10} {opt["total_bet"]:.0f}元{"":<10} {opt["total_bet"]-orig["total_bet"]:+.0f}元')
    
    # 找出倍数差异的期数
    print('\n' + '=' * 100)
    print('【投注倍数差异期数】')
    print('=' * 100)
    
    diff_periods = []
    for i in range(test_periods):
        orig_d = all_details['原版配置'][i]
        opt_d = all_details['优化配置'][i]
        if abs(orig_d['投注倍数'] - opt_d['投注倍数']) > 0.01:
            diff_periods.append({
                '期号': orig_d['期号'],
                '日期': orig_d['日期'],
                '实际': orig_d['实际号码'],
                '结果': orig_d['结果'],
                '原版倍数': orig_d['投注倍数'],
                '优化倍数': opt_d['投注倍数'],
                '倍数差': opt_d['投注倍数'] - orig_d['投注倍数'],
                '原版盈亏': orig_d['盈亏'],
                '优化盈亏': opt_d['盈亏'],
                '盈亏差': opt_d['盈亏'] - orig_d['盈亏']
            })
    
    print(f'\n差异期数: {len(diff_periods)}/{test_periods} ({len(diff_periods)/test_periods*100:.1f}%)')
    
    if diff_periods:
        print(f'\n{"期号":>4} {"日期":>12} {"实际":>4} {"结果":>4} {"原版倍数":>8} {"优化倍数":>8} {"倍数差":>8} {"原版盈亏":>10} {"优化盈亏":>10} {"盈亏差":>8}')
        print('-' * 100)
        
        total_profit_diff = 0
        for d in diff_periods:
            line = (f"{d['期号']:>4} {d['日期']:>12} {d['实际']:>4} {d['结果']:>4} "
                    f"{d['原版倍数']:>7.1f}x {d['优化倍数']:>7.1f}x {d['倍数差']:>+7.1f}x "
                    f"{d['原版盈亏']:>+9.0f}元 {d['优化盈亏']:>+9.0f}元 {d['盈亏差']:>+7.0f}元")
            print(line)
            total_profit_diff += d['盈亏差']
        
        print('-' * 100)
        print(f'{"累计盈亏差异:":>60} {total_profit_diff:>+17.0f}元')
    
    # ≥8倍投注对比
    print('\n' + '=' * 100)
    print('【≥8倍投注对比】')
    print('=' * 100)
    
    orig_high = [d for d in all_details['原版配置'] if d['投注倍数'] >= 8]
    opt_high = [d for d in all_details['优化配置'] if d['投注倍数'] >= 8]
    
    orig_high_hits = len([d for d in orig_high if d['结果'] == '命中'])
    opt_high_hits = len([d for d in opt_high if d['结果'] == '命中'])
    
    print(f'\n原版配置: ≥8倍投注 {len(orig_high)}次, 命中 {orig_high_hits}次 ({orig_high_hits/len(orig_high)*100:.1f}%)' if orig_high else '\n原版配置: 无≥8倍投注')
    print(f'优化配置: ≥8倍投注 {len(opt_high)}次, 命中 {opt_high_hits}次 ({opt_high_hits/len(opt_high)*100:.1f}%)' if opt_high else '优化配置: 无≥8倍投注')
    
    # 10倍投注对比
    orig_10x = [d for d in all_details['原版配置'] if d['投注倍数'] >= 10]
    opt_10x = [d for d in all_details['优化配置'] if d['投注倍数'] >= 10]
    
    orig_10x_hits = len([d for d in orig_10x if d['结果'] == '命中'])
    opt_10x_hits = len([d for d in opt_10x if d['结果'] == '命中'])
    
    print(f'\n原版配置: 10倍投注 {len(orig_10x)}次, 命中 {orig_10x_hits}次 ({orig_10x_hits/len(orig_10x)*100:.1f}%)' if orig_10x else '\n原版配置: 无10倍投注')
    print(f'优化配置: 10倍投注 {len(opt_10x)}次, 命中 {opt_10x_hits}次 ({opt_10x_hits/len(opt_10x)*100:.1f}%)' if opt_10x else '优化配置: 无10倍投注')
    
    # 结论
    print('\n' + '=' * 100)
    print('【结论】')
    print('=' * 100)
    
    if opt['roi'] > orig['roi'] and opt['max_drawdown'] < orig['max_drawdown']:
        print('\n✅ 优化配置在ROI和回撤两方面都优于原版')
    elif opt['roi'] > orig['roi']:
        print('\n⚠️ 优化配置ROI更高，但回撤也更高')
    elif opt['max_drawdown'] < orig['max_drawdown']:
        print('\n⚠️ 优化配置回撤更低，但ROI也更低')
    else:
        print('\n❌ 优化配置表现不如原版')
    
    print(f'\nROI提升: {orig["roi"]:.2f}% → {opt["roi"]:.2f}% ({opt["roi"]-orig["roi"]:+.2f}%, 相对{(opt["roi"]-orig["roi"])/orig["roi"]*100 if orig["roi"] != 0 else 0:+.1f}%)')
    print(f'回撤降低: {orig["max_drawdown"]:.0f}元 → {opt["max_drawdown"]:.0f}元 ({opt["max_drawdown"]-orig["max_drawdown"]:+.0f}元, 相对{(opt["max_drawdown"]-orig["max_drawdown"])/orig["max_drawdown"]*100 if orig["max_drawdown"] != 0 else 0:+.1f}%)')
    print(f'收益提升: {orig["balance"]:+.0f}元 → {opt["balance"]:+.0f}元 ({opt["balance"]-orig["balance"]:+.0f}元)')
    
    # 保存详细对比数据
    diff_df = pd.DataFrame(diff_periods)
    if not diff_df.empty:
        diff_df.to_csv('config_diff_detail.csv', index=False, encoding='utf-8-sig')
        print(f'\n差异期数详情已保存到: config_diff_detail.csv')
    
    # 保存两个配置的完整详情
    for name, details in all_details.items():
        filename = f'dynamic_{name.replace("配置", "")}_detail.csv'
        pd.DataFrame(details).to_csv(filename, index=False, encoding='utf-8-sig')
        print(f'{name}详情已保存到: {filename}')


if __name__ == '__main__':
    main()
