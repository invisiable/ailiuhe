"""
基于修复后的正确时序，全面优化投注策略参数
目标：提高收益率，降低回撤
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
    
    # 获取Fibonacci配置
    if config.get('fib_cap_index'):
        fib_sequence = [1, 1, 2, 3, 5, 8]
        # 后续都用8
        fib_sequence.extend([8] * 10)
    else:
        fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    
    # 策略状态
    recent_results = []
    fib_index = 0
    balance = 0
    min_balance = 0
    max_drawdown = 0
    total_bet = 0
    
    results = []
    hit_10x_count = 0
    hit_limit_count = 0
    
    for i in range(start_idx, len(df)):
        train_data = df.iloc[:i]['number'].values
        predictions = predictor.predict(train_data)
        actual = df.iloc[i]['number']
        hit = actual in predictions
        
        predictor.update_performance(predictions, actual)
        
        # ===== 正确时序：先计算倍数，再更新历史 =====
        
        # 1. 计算基础倍数
        if fib_index >= len(fib_sequence):
            base_mult = min(fib_sequence[-1], config['max_multiplier'])
        else:
            base_mult = min(fib_sequence[fib_index], config['max_multiplier'])
        
        # 2. 动态调整（基于投注前的历史）
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
        
        # 3. 投注和结算
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
                max_drawdown = abs(min_balance)
        
        # 4. 更新历史（在投注结算之后）
        recent_results.append(1 if hit else 0)
        
        # 统计
        if multiplier >= config['max_multiplier']:
            hit_limit_count += 1
            if multiplier >= 10:
                hit_10x_count += 1
        
        results.append({
            'hit': hit,
            'multiplier': multiplier,
            'profit': profit,
            'balance': balance
        })
    
    # 计算统计指标
    hits = sum(1 for r in results if r['hit'])
    hit_rate = hits / len(results) if results else 0
    roi = (balance / total_bet * 100) if total_bet > 0 else 0
    
    # 连败统计
    max_consecutive_losses = 0
    current_losses = 0
    for r in results:
        if not r['hit']:
            current_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
        else:
            current_losses = 0
    
    # 收益波动（标准差）
    profits = [r['profit'] for r in results]
    profit_std = np.std(profits) if profits else 0
    
    # 风险调整收益（夏普比率的简化版）
    avg_profit = balance / len(results) if results else 0
    sharpe_like = avg_profit / profit_std if profit_std > 0 else 0
    
    return {
        'hit_rate': hit_rate,
        'roi': roi,
        'balance': balance,
        'max_drawdown': max_drawdown,
        'total_bet': total_bet,
        'hit_limit_count': hit_limit_count,
        'hit_10x_count': hit_10x_count,
        'max_consecutive_losses': max_consecutive_losses,
        'profit_std': profit_std,
        'sharpe_like': sharpe_like,
        'results': results
    }


def optimize_strategy():
    """优化策略参数"""
    
    print("=" * 120)
    print("投注策略参数优化（基于正确时序）")
    print("=" * 120)
    print()
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    test_periods = 300
    
    print(f"数据范围: 最近{test_periods}期")
    print(f"时间范围: {df.iloc[-test_periods]['date']} ~ {df.iloc[-1]['date']}")
    print()
    
    # 定义参数空间
    param_space = {
        'base_bet': [15],  # 固定
        'win_reward': [45],  # 固定
        'max_multiplier': [8, 10],  # 关键参数
        'lookback': [5, 6, 8, 10],  # 回看窗口
        'good_thresh': [0.30, 0.35, 0.40],  # 增强阈值
        'bad_thresh': [0.15, 0.20, 0.25],  # 降低阈值
        'boost_mult': [1.2, 1.5, 1.8],  # 增强倍数
        'reduce_mult': [0.5, 0.6, 0.7],  # 降低倍数
        'fib_cap_index': [None, 6]  # Fibonacci上限索引
    }
    
    # 生成所有组合
    keys = param_space.keys()
    values = param_space.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    print(f"参数组合总数: {len(combinations)}")
    print()
    print("开始测试...（预计需要几分钟）")
    print()
    
    # 测试所有组合
    all_results = []
    start_time = time.time()
    
    for idx, config in enumerate(combinations, 1):
        if idx % 50 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / idx * (len(combinations) - idx)
            print(f"进度: {idx}/{len(combinations)} ({idx/len(combinations)*100:.1f}%) - 预计剩余{eta/60:.1f}分钟")
        
        result = backtest_strategy(config, df, test_periods)
        
        all_results.append({
            'config': config,
            'metrics': result
        })
    
    elapsed = time.time() - start_time
    print()
    print(f"✅ 测试完成，耗时 {elapsed/60:.1f} 分钟")
    print()
    
    # 排序和分析
    print("=" * 120)
    print("【优化结果】按不同指标排序")
    print("=" * 120)
    print()
    
    # 1. 按ROI排序
    sorted_by_roi = sorted(all_results, key=lambda x: x['metrics']['roi'], reverse=True)
    
    print("【TOP 10 - 最高ROI】")
    print(f"{'排名':<4} {'ROI':<8} {'收益':<8} {'回撤':<8} {'命中率':<8} {'10x':<5} "
          f"{'上限':<5} {'窗口':<5} {'增强阈':<8} {'降低阈':<8} {'增强倍':<8} {'降低倍':<8} {'Fib上限':<8}")
    print("-" * 120)
    
    for rank, item in enumerate(sorted_by_roi[:10], 1):
        cfg = item['config']
        m = item['metrics']
        fib_str = f"Index{cfg['fib_cap_index']}" if cfg['fib_cap_index'] else "无限制"
        
        print(f"{rank:<4} {m['roi']:<8.2f} {m['balance']:<+8.0f} {m['max_drawdown']:<8.0f} "
              f"{m['hit_rate']*100:<8.2f} {m['hit_10x_count']:<5} "
              f"{cfg['max_multiplier']:<5} {cfg['lookback']:<5} {cfg['good_thresh']:<8.2f} "
              f"{cfg['bad_thresh']:<8.2f} {cfg['boost_mult']:<8.2f} {cfg['reduce_mult']:<8.2f} {fib_str:<8}")
    
    print()
    
    # 2. 按最低回撤排序
    sorted_by_drawdown = sorted(all_results, key=lambda x: x['metrics']['max_drawdown'])
    
    print("【TOP 10 - 最低回撤】")
    print(f"{'排名':<4} {'回撤':<8} {'收益':<8} {'ROI':<8} {'命中率':<8} {'10x':<5} "
          f"{'上限':<5} {'窗口':<5} {'增强阈':<8} {'降低阈':<8} {'增强倍':<8} {'降低倍':<8} {'Fib上限':<8}")
    print("-" * 120)
    
    for rank, item in enumerate(sorted_by_drawdown[:10], 1):
        cfg = item['config']
        m = item['metrics']
        fib_str = f"Index{cfg['fib_cap_index']}" if cfg['fib_cap_index'] else "无限制"
        
        print(f"{rank:<4} {m['max_drawdown']:<8.0f} {m['balance']:<+8.0f} {m['roi']:<8.2f} "
              f"{m['hit_rate']*100:<8.2f} {m['hit_10x_count']:<5} "
              f"{cfg['max_multiplier']:<5} {cfg['lookback']:<5} {cfg['good_thresh']:<8.2f} "
              f"{cfg['bad_thresh']:<8.2f} {cfg['boost_mult']:<8.2f} {cfg['reduce_mult']:<8.2f} {fib_str:<8}")
    
    print()
    
    # 3. 按风险调整收益排序
    sorted_by_sharpe = sorted(all_results, key=lambda x: x['metrics']['sharpe_like'], reverse=True)
    
    print("【TOP 10 - 最佳风险调整收益】")
    print(f"{'排名':<4} {'夏普':<8} {'ROI':<8} {'回撤':<8} {'命中率':<8} {'10x':<5} "
          f"{'上限':<5} {'窗口':<5} {'增强阈':<8} {'降低阈':<8} {'增强倍':<8} {'降低倍':<8} {'Fib上限':<8}")
    print("-" * 120)
    
    for rank, item in enumerate(sorted_by_sharpe[:10], 1):
        cfg = item['config']
        m = item['metrics']
        fib_str = f"Index{cfg['fib_cap_index']}" if cfg['fib_cap_index'] else "无限制"
        
        print(f"{rank:<4} {m['sharpe_like']:<8.4f} {m['roi']:<8.2f} {m['max_drawdown']:<8.0f} "
              f"{m['hit_rate']*100:<8.2f} {m['hit_10x_count']:<5} "
              f"{cfg['max_multiplier']:<5} {cfg['lookback']:<5} {cfg['good_thresh']:<8.2f} "
              f"{cfg['bad_thresh']:<8.2f} {cfg['boost_mult']:<8.2f} {cfg['reduce_mult']:<8.2f} {fib_str:<8}")
    
    print()
    
    # 4. 综合评分（ROI和低回撤的平衡）
    for item in all_results:
        m = item['metrics']
        # 综合评分 = ROI * 10 - 回撤/100 + 夏普*100
        score = m['roi'] * 10 - m['max_drawdown'] / 100 + m['sharpe_like'] * 100
        item['综合得分'] = score
    
    sorted_by_score = sorted(all_results, key=lambda x: x['综合得分'], reverse=True)
    
    print("【TOP 10 - 综合最优（平衡收益和风险）】")
    print(f"{'排名':<4} {'综合分':<8} {'ROI':<8} {'收益':<8} {'回撤':<8} {'命中率':<8} {'10x':<5} "
          f"{'上限':<5} {'窗口':<5} {'增强阈':<8} {'降低阈':<8} {'增强倍':<8} {'降低倍':<8} {'Fib上限':<8}")
    print("-" * 120)
    
    for rank, item in enumerate(sorted_by_score[:10], 1):
        cfg = item['config']
        m = item['metrics']
        fib_str = f"Index{cfg['fib_cap_index']}" if cfg['fib_cap_index'] else "无限制"
        
        print(f"{rank:<4} {item['综合得分']:<8.1f} {m['roi']:<8.2f} {m['balance']:<+8.0f} "
              f"{m['max_drawdown']:<8.0f} {m['hit_rate']*100:<8.2f} {m['hit_10x_count']:<5} "
              f"{cfg['max_multiplier']:<5} {cfg['lookback']:<5} {cfg['good_thresh']:<8.2f} "
              f"{cfg['bad_thresh']:<8.2f} {cfg['boost_mult']:<8.2f} {cfg['reduce_mult']:<8.2f} {fib_str:<8}")
    
    print()
    
    # 5. 推荐配置
    print("=" * 120)
    print("【推荐配置】")
    print("=" * 120)
    print()
    
    best_overall = sorted_by_score[0]
    best_roi = sorted_by_roi[0]
    best_drawdown = sorted_by_drawdown[0]
    
    # 当前配置（修复前的基准）
    current_config = {
        'base_bet': 15,
        'win_reward': 45,
        'max_multiplier': 10,
        'lookback': 8,
        'good_thresh': 0.35,
        'bad_thresh': 0.20,
        'boost_mult': 1.5,
        'reduce_mult': 0.6,
        'fib_cap_index': None
    }
    current_result = backtest_strategy(current_config, df, test_periods)
    
    configs_to_show = [
        ("当前配置（已修复时序）", current_config, current_result),
        ("综合最优", best_overall['config'], best_overall['metrics']),
        ("最高ROI", best_roi['config'], best_roi['metrics']),
        ("最低回撤", best_drawdown['config'], best_drawdown['metrics'])
    ]
    
    for name, cfg, metrics in configs_to_show:
        print(f"【{name}】")
        print(f"  参数配置:")
        print(f"    倍数上限: {cfg['max_multiplier']}x")
        print(f"    回看窗口: {cfg['lookback']}期")
        print(f"    增强阈值: {cfg['good_thresh']*100:.0f}% (×{cfg['boost_mult']})")
        print(f"    降低阈值: {cfg['bad_thresh']*100:.0f}% (×{cfg['reduce_mult']})")
        fib_str = f"限制在Index {cfg['fib_cap_index']}" if cfg['fib_cap_index'] else "无限制"
        print(f"    Fibonacci: {fib_str}")
        print()
        print(f"  表现指标:")
        print(f"    命中率: {metrics['hit_rate']*100:.2f}%")
        print(f"    ROI: {metrics['roi']:.2f}%")
        print(f"    净收益: {metrics['balance']:+.0f}元")
        print(f"    最大回撤: {metrics['max_drawdown']:.0f}元")
        print(f"    总投注: {metrics['total_bet']:.0f}元")
        print(f"    触达上限次数: {metrics['hit_limit_count']}次")
        print(f"    10倍次数: {metrics['hit_10x_count']}次")
        print(f"    最长连败: {metrics['max_consecutive_losses']}期")
        print(f"    风险调整收益: {metrics['sharpe_like']:.4f}")
        print()
    
    # 对比改进
    print("=" * 120)
    print("【改进对比】综合最优 vs 当前配置")
    print("=" * 120)
    print()
    
    best = best_overall['metrics']
    curr = current_result
    
    print(f"ROI提升: {current_result['roi']:.2f}% → {best['roi']:.2f}% ({best['roi']-curr['roi']:+.2f}%)")
    print(f"收益提升: {curr['balance']:+.0f}元 → {best['balance']:+.0f}元 ({best['balance']-curr['balance']:+.0f}元)")
    print(f"回撤降低: {curr['max_drawdown']:.0f}元 → {best['max_drawdown']:.0f}元 ({best['max_drawdown']-curr['max_drawdown']:+.0f}元)")
    print(f"10倍减少: {curr['hit_10x_count']}次 → {best['hit_10x_count']}次 ({best['hit_10x_count']-curr['hit_10x_count']:+}次)")
    print(f"风险调整收益: {curr['sharpe_like']:.4f} → {best['sharpe_like']:.4f} ({best['sharpe_like']-curr['sharpe_like']:+.4f})")
    print()
    
    return sorted_by_score[0]


if __name__ == '__main__':
    best_config = optimize_strategy()
