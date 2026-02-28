"""
精简参数优化 - 快速找到最优配置
"""

import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor
from itertools import product
import time


def backtest_strategy(config, df, test_periods=300):
    """回测单个策略配置"""
    
    start_idx = len(df) - test_periods
    predictor = PreciseTop15Predictor()
    fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    
    recent_results = []
    fib_index = 0
    balance = 0
    min_balance = 0
    total_bet = 0
    hit_10x_count = 0
    
    for i in range(start_idx, len(df)):
        train_data = df.iloc[:i]['number'].values
        predictions = predictor.predict(train_data)
        actual = df.iloc[i]['number']
        hit = actual in predictions
        
        predictor.update_performance(predictions, actual)
        
        # 正确时序
        if fib_index >= len(fib_sequence):
            base_mult = min(fib_sequence[-1], config['max_multiplier'])
        else:
            base_mult = min(fib_sequence[fib_index], config['max_multiplier'])
        
        if len(recent_results) >= config['lookback']:
            recent_hits = sum(recent_results[-config['lookback']:])
            rate = recent_hits / config['lookback']
            
            if rate >= config['good_thresh']:
                multiplier = min(base_mult * config['boost_mult'], config['max_multiplier'])
            elif rate <= config['bad_thresh']:
                multiplier = max(base_mult * config['reduce_mult'], 1)
            else:
                multiplier = base_mult
        else:
            multiplier = base_mult
        
        bet = config['base_bet'] * multiplier
        total_bet += bet
        
        if hit:
            profit = config['win_reward'] * multiplier - bet
            balance += profit
            fib_index = 0
        else:
            profit = -bet
            balance += profit
            fib_index += 1
            if balance < min_balance:
                min_balance = balance
        
        recent_results.append(1 if hit else 0)
        if multiplier >= 10:
            hit_10x_count += 1
    
    roi = (balance / total_bet * 100) if total_bet > 0 else 0
    
    return {
        'roi': roi,
        'balance': balance,
        'max_drawdown': abs(min_balance),
        'total_bet': total_bet,
        'hit_10x_count': hit_10x_count
    }


def quick_optimization():
    """快速参数优化"""
    
    print("=" * 120)
    print("精简参数优化")
    print("=" * 120)
    print()
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    test_periods = 300
    
    print(f"数据: {df.iloc[-test_periods]['date']} ~ {df.iloc[-1]['date']} ({test_periods}期)")
    print()
    
    # 精简参数空间
    param_space = {
        'base_bet': [15],
        'win_reward': [45],
        'max_multiplier': [8, 10],
        'lookback': [5, 6, 8, 10],
        'good_thresh': [0.25, 0.30, 0.35, 0.40],
        'bad_thresh': [0.15, 0.20, 0.25],
        'boost_mult': [1.0, 1.2, 1.5],
        'reduce_mult': [0.5, 0.6, 0.8, 1.0]
    }
    
    combinations = [dict(zip(param_space.keys(), v)) for v in product(*param_space.values())]
    print(f"参数组合: {len(combinations)}个")
    print()
    
    # 测试
    all_results = []
    start_time = time.time()
    
    for idx, config in enumerate(combinations, 1):
        if idx % 100 == 0:
            print(f"进度: {idx}/{len(combinations)}")
        
        result = backtest_strategy(config, df, test_periods)
        all_results.append({'config': config, 'metrics': result})
    
    elapsed = time.time() - start_time
    print(f"\n✅ 完成，耗时 {elapsed:.1f}秒")
    print()
    
    # 排序
    sorted_by_roi = sorted(all_results, key=lambda x: x['metrics']['roi'], reverse=True)
    
    print("=" * 120)
    print("【TOP 15 - 最高ROI】")
    print("=" * 120)
    print(f"{'#':<3} {'ROI':<8} {'收益':<9} {'回撤':<8} {'10x':<5} {'上限':<5} {'窗口':<5} {'增强阈':<8} {'降低阈':<8} {'增强倍':<8} {'降低倍':<8}")
    print("-" * 120)
    
    for rank, item in enumerate(sorted_by_roi[:15], 1):
        c = item['config']
        m = item['metrics']
        print(f"{rank:<3} {m['roi']:<8.2f} {m['balance']:<+9.0f} {m['max_drawdown']:<8.0f} "
              f"{m['hit_10x_count']:<5} {c['max_multiplier']:<5} {c['lookback']:<5} "
              f"{c['good_thresh']:<8.2f} {c['bad_thresh']:<8.2f} {c['boost_mult']:<8.2f} {c['reduce_mult']:<8.2f}")
    
    print()
    
    # 按综合评分（ROI高 + 回撤低）
    for item in all_results:
        m = item['metrics']
        item['score'] = m['roi'] * 2 - m['max_drawdown'] / 100
    
    sorted_by_score = sorted(all_results, key=lambda x: x['score'], reverse=True)
    
    print("=" * 120)
    print("【TOP 15 - 综合最优（ROI×2 - 回撤/100）】")
    print("=" * 120)
    print(f"{'#':<3} {'评分':<8} {'ROI':<8} {'收益':<9} {'回撤':<8} {'10x':<5} {'上限':<5} {'窗口':<5} {'增强阈':<8} {'降低阈':<8} {'增强倍':<8} {'降低倍':<8}")
    print("-" * 120)
    
    for rank, item in enumerate(sorted_by_score[:15], 1):
        c = item['config']
        m = item['metrics']
        print(f"{rank:<3} {item['score']:<8.2f} {m['roi']:<8.2f} {m['balance']:<+9.0f} {m['max_drawdown']:<8.0f} "
              f"{m['hit_10x_count']:<5} {c['max_multiplier']:<5} {c['lookback']:<5} "
              f"{c['good_thresh']:<8.2f} {c['bad_thresh']:<8.2f} {c['boost_mult']:<8.2f} {c['reduce_mult']:<8.2f}")
    
    print()
    
    # 对比基准
    print("=" * 120)
    print("【配置对比】")
    print("=" * 120)
    print()
    
    # 原始配置
    original = {'base_bet': 15, 'win_reward': 45, 'max_multiplier': 10, 'lookback': 8,
                'good_thresh': 0.35, 'bad_thresh': 0.20, 'boost_mult': 1.5, 'reduce_mult': 0.6}
    orig_result = backtest_strategy(original, df, test_periods)
    
    best = sorted_by_score[0]
    
    print(f"【原始配置】")
    print(f"  参数: 上限{original['max_multiplier']}x 窗口{original['lookback']} 增强阈{original['good_thresh']*100:.0f}% "
          f"降低阈{original['bad_thresh']*100:.0f}% 增强×{original['boost_mult']} 降低×{original['reduce_mult']}")
    print(f"  表现: ROI {orig_result['roi']:.2f}% | 收益 {orig_result['balance']:+.0f}元 | 回撤 {orig_result['max_drawdown']:.0f}元")
    print()
    
    print(f"【综合最优配置】")
    c = best['config']
    m = best['metrics']
    print(f"  参数: 上限{c['max_multiplier']}x 窗口{c['lookback']} 增强阈{c['good_thresh']*100:.0f}% "
          f"降低阈{c['bad_thresh']*100:.0f}% 增强×{c['boost_mult']} 降低×{c['reduce_mult']}")
    print(f"  表现: ROI {m['roi']:.2f}% | 收益 {m['balance']:+.0f}元 | 回撤 {m['max_drawdown']:.0f}元")
    print()
    
    print(f"【改进幅度】")
    print(f"  ROI: {orig_result['roi']:.2f}% → {m['roi']:.2f}% ({m['roi']-orig_result['roi']:+.2f}%)")
    print(f"  收益: {orig_result['balance']:+.0f}元 → {m['balance']:+.0f}元 ({m['balance']-orig_result['balance']:+.0f}元)")
    print(f"  回撤: {orig_result['max_drawdown']:.0f}元 → {m['max_drawdown']:.0f}元 ({m['max_drawdown']-orig_result['max_drawdown']:+.0f}元)")
    print()
    
    # 推荐配置
    print("=" * 120)
    print("【🎯 推荐配置】")
    print("=" * 120)
    print()
    print(f"倍数上限: {c['max_multiplier']}x")
    print(f"回看窗口: {c['lookback']}期")
    print(f"增强阈值: {c['good_thresh']*100:.0f}%")
    print(f"降低阈值: {c['bad_thresh']*100:.0f}%")
    print(f"增强倍数: {c['boost_mult']}x")
    print(f"降低倍数: {c['reduce_mult']}x")
    print()

    return best


if __name__ == '__main__':
    best = quick_optimization()
