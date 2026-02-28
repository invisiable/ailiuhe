"""
全面参数优化 - 扩大搜索范围
目标：找到真正最优的动态投注方案
"""

import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor
from itertools import product
import time


def backtest_strategy(config, df, test_periods=300):
    """回测单个策略配置 - 正确时序"""
    
    start_idx = len(df) - test_periods
    predictor = PreciseTop15Predictor()
    
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
    
    for i in range(start_idx, len(df)):
        train_data = df.iloc[:i]['number'].values
        predictions = predictor.predict(train_data)
        actual = df.iloc[i]['number']
        hit = actual in predictions
        
        predictor.update_performance(predictions, actual)
        
        # ===== 正确时序：先计算倍数（基于投注前历史），再更新历史 =====
        
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
    
    # 收益波动
    profits = [r['profit'] for r in results]
    profit_std = np.std(profits) if profits else 0
    avg_profit = balance / len(results) if results else 0
    sharpe_like = avg_profit / profit_std if profit_std > 0 else 0
    
    return {
        'hit_rate': hit_rate,
        'roi': roi,
        'balance': balance,
        'max_drawdown': max_drawdown,
        'total_bet': total_bet,
        'hit_10x_count': hit_10x_count,
        'profit_std': profit_std,
        'sharpe_like': sharpe_like
    }


def comprehensive_optimization():
    """全面参数优化"""
    
    print("=" * 120)
    print("全面参数优化 - 扩大搜索范围")
    print("=" * 120)
    print()
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    test_periods = 300
    
    print(f"数据范围: 最近{test_periods}期")
    print(f"时间范围: {df.iloc[-test_periods]['date']} ~ {df.iloc[-1]['date']}")
    print()
    
    # 扩大参数空间
    param_space = {
        'base_bet': [15],
        'win_reward': [45],
        'max_multiplier': [8, 10, 12],  # 增加12倍选项
        'lookback': [4, 5, 6, 7, 8, 10, 12],  # 更多窗口选项
        'good_thresh': [0.25, 0.30, 0.35, 0.40, 0.45, 0.50],  # 更多阈值
        'bad_thresh': [0.10, 0.15, 0.20, 0.25],  # 更多阈值
        'boost_mult': [1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0],  # 包含1.0（无增强）
        'reduce_mult': [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]  # 包含1.0（无降低）
    }
    
    # 生成所有组合
    keys = param_space.keys()
    values = param_space.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    print(f"参数组合总数: {len(combinations)}")
    print()
    print("开始测试...")
    print()
    
    # 测试所有组合
    all_results = []
    start_time = time.time()
    
    for idx, config in enumerate(combinations, 1):
        if idx % 500 == 0:
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
    print(f"✅ 测试完成，耗时 {elapsed/60:.1f} 分钟，共测试 {len(combinations)} 组配置")
    print()
    
    # ===== 排序和分析 =====
    
    # 1. 按ROI排序
    sorted_by_roi = sorted(all_results, key=lambda x: x['metrics']['roi'], reverse=True)
    
    print("=" * 130)
    print("【TOP 20 - 最高ROI】")
    print("=" * 130)
    print(f"{'#':<3} {'ROI':<7} {'收益':<8} {'回撤':<7} {'投注':<8} {'10x':<4} "
          f"{'上限':<4} {'窗口':<4} {'增强阈':<6} {'降低阈':<6} {'增强倍':<6} {'降低倍':<6}")
    print("-" * 130)
    
    for rank, item in enumerate(sorted_by_roi[:20], 1):
        cfg = item['config']
        m = item['metrics']
        
        print(f"{rank:<3} {m['roi']:<7.2f} {m['balance']:<+8.0f} {m['max_drawdown']:<7.0f} "
              f"{m['total_bet']:<8.0f} {m['hit_10x_count']:<4} "
              f"{cfg['max_multiplier']:<4} {cfg['lookback']:<4} "
              f"{cfg['good_thresh']:<6.2f} {cfg['bad_thresh']:<6.2f} "
              f"{cfg['boost_mult']:<6.2f} {cfg['reduce_mult']:<6.2f}")
    
    print()
    
    # 2. 按最低回撤排序（只看收益>0的）
    profitable = [r for r in all_results if r['metrics']['balance'] > 0]
    sorted_by_drawdown = sorted(profitable, key=lambda x: x['metrics']['max_drawdown'])
    
    print("=" * 130)
    print("【TOP 20 - 最低回撤（仅盈利配置）】")
    print("=" * 130)
    print(f"{'#':<3} {'回撤':<7} {'ROI':<7} {'收益':<8} {'投注':<8} {'10x':<4} "
          f"{'上限':<4} {'窗口':<4} {'增强阈':<6} {'降低阈':<6} {'增强倍':<6} {'降低倍':<6}")
    print("-" * 130)
    
    for rank, item in enumerate(sorted_by_drawdown[:20], 1):
        cfg = item['config']
        m = item['metrics']
        
        print(f"{rank:<3} {m['max_drawdown']:<7.0f} {m['roi']:<7.2f} {m['balance']:<+8.0f} "
              f"{m['total_bet']:<8.0f} {m['hit_10x_count']:<4} "
              f"{cfg['max_multiplier']:<4} {cfg['lookback']:<4} "
              f"{cfg['good_thresh']:<6.2f} {cfg['bad_thresh']:<6.2f} "
              f"{cfg['boost_mult']:<6.2f} {cfg['reduce_mult']:<6.2f}")
    
    print()
    
    # 3. 综合评分（平衡收益与风险）
    for item in all_results:
        m = item['metrics']
        # 综合评分 = ROI * 收益系数 / 回撤风险
        # 更重视ROI和低回撤
        if m['max_drawdown'] > 0:
            risk_adj_score = m['roi'] * (1 + m['balance']/1000) / (m['max_drawdown']/500)
        else:
            risk_adj_score = m['roi'] * 100
        item['score'] = risk_adj_score
    
    sorted_by_score = sorted(all_results, key=lambda x: x['score'], reverse=True)
    
    print("=" * 130)
    print("【TOP 20 - 综合最优（平衡ROI与回撤）】")
    print("=" * 130)
    print(f"{'#':<3} {'综合分':<7} {'ROI':<7} {'收益':<8} {'回撤':<7} {'10x':<4} "
          f"{'上限':<4} {'窗口':<4} {'增强阈':<6} {'降低阈':<6} {'增强倍':<6} {'降低倍':<6}")
    print("-" * 130)
    
    for rank, item in enumerate(sorted_by_score[:20], 1):
        cfg = item['config']
        m = item['metrics']
        
        print(f"{rank:<3} {item['score']:<7.2f} {m['roi']:<7.2f} {m['balance']:<+8.0f} "
              f"{m['max_drawdown']:<7.0f} {m['hit_10x_count']:<4} "
              f"{cfg['max_multiplier']:<4} {cfg['lookback']:<4} "
              f"{cfg['good_thresh']:<6.2f} {cfg['bad_thresh']:<6.2f} "
              f"{cfg['boost_mult']:<6.2f} {cfg['reduce_mult']:<6.2f}")
    
    print()
    
    # ===== 最优配置详情 =====
    print("=" * 130)
    print("【最优配置详情】")
    print("=" * 130)
    print()
    
    # 当前GUI配置（原始）
    original_config = {
        'base_bet': 15, 'win_reward': 45, 'max_multiplier': 10,
        'lookback': 8, 'good_thresh': 0.35, 'bad_thresh': 0.20,
        'boost_mult': 1.5, 'reduce_mult': 0.6
    }
    original_result = backtest_strategy(original_config, df, test_periods)
    
    # 之前优化的配置
    prev_optimized = {
        'base_bet': 15, 'win_reward': 45, 'max_multiplier': 10,
        'lookback': 10, 'good_thresh': 0.30, 'bad_thresh': 0.20,
        'boost_mult': 1.2, 'reduce_mult': 0.5
    }
    prev_result = backtest_strategy(prev_optimized, df, test_periods)
    
    best_roi = sorted_by_roi[0]
    best_drawdown = sorted_by_drawdown[0]
    best_score = sorted_by_score[0]
    
    configs_to_compare = [
        ("原始配置（修复前参数）", original_config, original_result),
        ("上次优化配置", prev_optimized, prev_result),
        ("最高ROI配置", best_roi['config'], best_roi['metrics']),
        ("最低回撤配置", best_drawdown['config'], best_drawdown['metrics']),
        ("综合最优配置", best_score['config'], best_score['metrics'])
    ]
    
    for name, cfg, metrics in configs_to_compare:
        print(f"【{name}】")
        print(f"  参数: 上限{cfg['max_multiplier']}x | 窗口{cfg['lookback']}期 | "
              f"增强阈{cfg['good_thresh']*100:.0f}% | 降低阈{cfg['bad_thresh']*100:.0f}% | "
              f"增强×{cfg['boost_mult']} | 降低×{cfg['reduce_mult']}")
        print(f"  表现: ROI {metrics['roi']:.2f}% | 收益 {metrics['balance']:+.0f}元 | "
              f"回撤 {metrics['max_drawdown']:.0f}元 | 投注 {metrics['total_bet']:.0f}元 | "
              f"10倍 {metrics['hit_10x_count']}次")
        print()
    
    # ===== 对比改进 =====
    print("=" * 130)
    print("【对比分析】")
    print("=" * 130)
    print()
    
    best = best_score['metrics']
    orig = original_result
    
    print(f"综合最优 vs 原始配置:")
    print(f"  ROI: {orig['roi']:.2f}% → {best['roi']:.2f}% ({best['roi']-orig['roi']:+.2f}%)")
    print(f"  收益: {orig['balance']:+.0f}元 → {best['balance']:+.0f}元 ({best['balance']-orig['balance']:+.0f}元)")
    print(f"  回撤: {orig['max_drawdown']:.0f}元 → {best['max_drawdown']:.0f}元 ({best['max_drawdown']-orig['max_drawdown']:+.0f}元)")
    print(f"  投注: {orig['total_bet']:.0f}元 → {best['total_bet']:.0f}元 ({best['total_bet']-orig['total_bet']:+.0f}元)")
    print()
    
    # ===== 推荐方案 =====
    print("=" * 130)
    print("【推荐的最优动态投注方案】")
    print("=" * 130)
    print()
    
    # 选择综合最优
    best_cfg = best_score['config']
    best_m = best_score['metrics']
    
    print(f"📌 推荐配置:")
    print(f"   倍数上限: {best_cfg['max_multiplier']}x")
    print(f"   回看窗口: {best_cfg['lookback']}期")
    print(f"   增强阈值: {best_cfg['good_thresh']*100:.0f}% (命中率≥此值时增强)")
    print(f"   降低阈值: {best_cfg['bad_thresh']*100:.0f}% (命中率≤此值时降低)")
    print(f"   增强倍数: {best_cfg['boost_mult']}x")
    print(f"   降低倍数: {best_cfg['reduce_mult']}x")
    print()
    
    print(f"📊 预期表现:")
    print(f"   ROI: {best_m['roi']:.2f}%")
    print(f"   净收益: {best_m['balance']:+.0f}元")
    print(f"   最大回撤: {best_m['max_drawdown']:.0f}元")
    print(f"   总投注: {best_m['total_bet']:.0f}元")
    print(f"   10倍触发: {best_m['hit_10x_count']}次")
    print()
    
    # 保存结果
    return {
        'best_roi': best_roi,
        'best_drawdown': best_drawdown,
        'best_score': best_score,
        'original': {'config': original_config, 'metrics': original_result},
        'prev_optimized': {'config': prev_optimized, 'metrics': prev_result}
    }


if __name__ == '__main__':
    results = comprehensive_optimization()
