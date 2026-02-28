#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接对比原版配置和优化配置
确保使用相同的代码路径进行公平对比
"""

import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor

def backtest_config(config_name, config, df, test_periods):
    """回测指定配置"""
    
    # Fibonacci序列
    fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    
    class SmartDynamicStrategy:
        def __init__(self, cfg):
            self.cfg = cfg
            self.fib_index = 0
            self.recent_results = []
            self.total_bet = 0
            self.total_win = 0
            self.balance = 0
            self.min_balance = 0
            self.max_drawdown = 0
        
        def get_base_multiplier(self):
            if self.fib_index >= len(fib_sequence):
                return min(fib_sequence[-1], self.cfg['max_multiplier'])
            return min(fib_sequence[self.fib_index], self.cfg['max_multiplier'])
        
        def get_recent_rate(self):
            if len(self.recent_results) == 0:
                return 0.33
            return sum(self.recent_results) / len(self.recent_results)
        
        def process_period(self, hit):
            # 【正确时序】先计算倍数（基于投注前的历史），再更新历史
            
            # 获取基础倍数
            base_mult = self.get_base_multiplier()
            
            # 根据最近命中率计算动态倍数（使用投注前的历史数据）
            if len(self.recent_results) >= self.cfg['lookback']:
                rate = self.get_recent_rate()
                if rate >= self.cfg['good_thresh']:
                    multiplier = min(base_mult * self.cfg['boost_mult'], self.cfg['max_multiplier'])
                elif rate <= self.cfg['bad_thresh']:
                    multiplier = max(base_mult * self.cfg['reduce_mult'], 1)
                else:
                    multiplier = base_mult
            else:
                multiplier = base_mult
            
            # 计算投注和收益
            bet = self.cfg['base_bet'] * multiplier
            self.total_bet += bet
            
            if hit:
                win = self.cfg['win_reward'] * multiplier
                self.total_win += win
                profit = win - bet
                self.balance += profit
                self.fib_index = 0
            else:
                profit = -bet
                self.balance += profit
                self.fib_index += 1
                
                if self.balance < self.min_balance:
                    self.min_balance = self.balance
                    self.max_drawdown = abs(self.min_balance)
            
            # 添加结果到历史（在投注和结算之后）
            self.recent_results.append(1 if hit else 0)
            if len(self.recent_results) > self.cfg['lookback']:
                self.recent_results.pop(0)
            
            return multiplier
    
    # 初始化预测器和策略
    predictor = PreciseTop15Predictor()
    strategy = SmartDynamicStrategy(config)
    
    start_idx = len(df) - test_periods
    hits = 0
    high_mult_hits = 0
    high_mult_count = 0
    ten_x_count = 0
    
    for i in range(start_idx, len(df)):
        # 预测
        train_data = df.iloc[:i]['number'].values
        predictions = predictor.predict(train_data)
        
        # 获取实际开奖号码
        actual = df.iloc[i]['number']
        hit = actual in predictions
        
        # 处理该期
        multiplier = strategy.process_period(hit)
        
        if hit:
            hits += 1
        
        if multiplier >= 8:
            high_mult_count += 1
            if hit:
                high_mult_hits += 1
        
        if multiplier >= 10:
            ten_x_count += 1
    
    # 计算结果
    hit_rate = hits / test_periods * 100
    roi = (strategy.balance / strategy.total_bet * 100) if strategy.total_bet > 0 else 0
    
    return {
        'name': config_name,
        'hit_rate': hit_rate,
        'roi': roi,
        'profit': strategy.balance,
        'drawdown': strategy.max_drawdown,
        'total_bet': strategy.total_bet,
        'high_mult_count': high_mult_count,
        'high_mult_hits': high_mult_hits,
        'ten_x_count': ten_x_count
    }


def main():
    print("=" * 80)
    print("原版配置 vs 优化配置 直接对比验证")
    print("=" * 80)
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"\n数据总量: {len(df)}期")
    print(f"时间范围: {df.iloc[0]['date']} ~ {df.iloc[-1]['date']}")
    
    test_periods = min(300, len(df) - 50)
    print(f"测试期数: {test_periods}期\n")
    
    # 定义两个配置
    configs = [
        {
            'name': '原版配置 (lookback=8, boost=1.5x)',
            'params': {
                'lookback': 8,
                'good_thresh': 0.35,
                'bad_thresh': 0.20,
                'boost_mult': 1.5,
                'reduce_mult': 0.6,
                'max_multiplier': 10,
                'base_bet': 15,
                'win_reward': 45
            }
        },
        {
            'name': '优化配置 (lookback=10, boost=1.2x)',
            'params': {
                'lookback': 10,
                'good_thresh': 0.30,
                'bad_thresh': 0.20,
                'boost_mult': 1.2,
                'reduce_mult': 0.5,
                'max_multiplier': 10,
                'base_bet': 15,
                'win_reward': 45
            }
        },
        {
            'name': '保守配置 (lookback=10, boost=1.0x)',
            'params': {
                'lookback': 10,
                'good_thresh': 0.35,
                'bad_thresh': 0.20,
                'boost_mult': 1.0,
                'reduce_mult': 0.6,
                'max_multiplier': 10,
                'base_bet': 15,
                'win_reward': 45
            }
        },
        {
            'name': '平衡配置 (lookback=8, boost=1.2x)',
            'params': {
                'lookback': 8,
                'good_thresh': 0.35,
                'bad_thresh': 0.20,
                'boost_mult': 1.2,
                'reduce_mult': 0.5,
                'max_multiplier': 10,
                'base_bet': 15,
                'win_reward': 45
            }
        }
    ]
    
    # 测试所有配置
    results = []
    for cfg in configs:
        print(f"测试: {cfg['name']}...")
        result = backtest_config(cfg['name'], cfg['params'], df, test_periods)
        results.append(result)
    
    # 输出结果对比
    print("\n" + "=" * 80)
    print("【回测结果对比】")
    print("=" * 80)
    
    print(f"\n{'配置名称':<30} {'命中率':>8} {'ROI':>10} {'净收益':>10} {'最大回撤':>10} {'10x次数':>8}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['name']:<30} {r['hit_rate']:>7.1f}% {r['roi']:>9.2f}% {r['profit']:>+9.0f}元 {r['drawdown']:>9.0f}元 {r['ten_x_count']:>7}次")
    
    # 找出最优配置
    best_roi = max(results, key=lambda x: x['roi'])
    best_profit = max(results, key=lambda x: x['profit'])
    best_drawdown = min(results, key=lambda x: x['drawdown'])
    
    print("\n" + "=" * 80)
    print("【最优配置总结】")
    print("=" * 80)
    print(f"\n最高ROI: {best_roi['name']}")
    print(f"  → ROI {best_roi['roi']:.2f}%, 收益 {best_roi['profit']:+.0f}元, 回撤 {best_roi['drawdown']:.0f}元")
    
    print(f"\n最高收益: {best_profit['name']}")
    print(f"  → ROI {best_profit['roi']:.2f}%, 收益 {best_profit['profit']:+.0f}元, 回撤 {best_profit['drawdown']:.0f}元")
    
    print(f"\n最低回撤: {best_drawdown['name']}")
    print(f"  → ROI {best_drawdown['roi']:.2f}%, 收益 {best_drawdown['profit']:+.0f}元, 回撤 {best_drawdown['drawdown']:.0f}元")
    
    # 原版vs优化的详细对比
    original = results[0]
    optimized = results[1]
    
    print("\n" + "=" * 80)
    print("【原版 vs 优化 详细对比】")
    print("=" * 80)
    
    roi_change = optimized['roi'] - original['roi']
    profit_change = optimized['profit'] - original['profit']
    drawdown_change = optimized['drawdown'] - original['drawdown']
    
    print(f"\nROI变化: {original['roi']:.2f}% → {optimized['roi']:.2f}% ({roi_change:+.2f}%)")
    print(f"收益变化: {original['profit']:+.0f}元 → {optimized['profit']:+.0f}元 ({profit_change:+.0f}元)")
    print(f"回撤变化: {original['drawdown']:.0f}元 → {optimized['drawdown']:.0f}元 ({drawdown_change:+.0f}元)")
    
    if optimized['roi'] > original['roi'] and optimized['drawdown'] < original['drawdown']:
        print("\n✅ 优化配置在ROI和回撤两方面都优于原版")
        print("   建议使用优化配置 (lookback=10, good_thresh=0.30, boost_mult=1.2, reduce_mult=0.5)")
    elif optimized['roi'] > original['roi']:
        print("\n⚠️ 优化配置ROI更高，但回撤更大")
        if drawdown_change > original['drawdown'] * 0.2:
            print("   回撤增加超过20%，建议考虑保守配置")
        else:
            print("   回撤增加可接受，优化配置仍有优势")
    elif optimized['drawdown'] < original['drawdown']:
        print("\n⚠️ 优化配置回撤更低，但ROI降低")
        print("   如果追求稳定，可以使用优化配置")
    else:
        print("\n❌ 优化配置表现不如原版")
        print("   建议恢复原版配置 (lookback=8, good_thresh=0.35, boost_mult=1.5, reduce_mult=0.6)")
    
    # 给出推荐
    print("\n" + "=" * 80)
    print("【推荐配置】")
    print("=" * 80)
    
    # 按综合评分排序（ROI权重0.5 + 收益0.3 + 低回撤0.2）
    for r in results:
        max_roi = max(x['roi'] for x in results)
        min_roi = min(x['roi'] for x in results)
        max_profit = max(x['profit'] for x in results)
        min_profit = min(x['profit'] for x in results)
        max_dd = max(x['drawdown'] for x in results)
        min_dd = min(x['drawdown'] for x in results)
        
        roi_score = (r['roi'] - min_roi) / (max_roi - min_roi + 0.001) * 100
        profit_score = (r['profit'] - min_profit) / (max_profit - min_profit + 0.001) * 100
        dd_score = (max_dd - r['drawdown']) / (max_dd - min_dd + 0.001) * 100
        
        r['score'] = roi_score * 0.4 + profit_score * 0.3 + dd_score * 0.3
    
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
    
    print(f"\n综合评分排名（ROI 40% + 收益 30% + 低回撤 30%）:")
    for i, r in enumerate(results_sorted):
        print(f"  {i+1}. {r['name']:<40} (综合分: {r['score']:.1f})")
    
    best = results_sorted[0]
    print(f"\n🏆 推荐使用: {best['name']}")
    print(f"   ROI: {best['roi']:.2f}%, 收益: {best['profit']:+.0f}元, 回撤: {best['drawdown']:.0f}元")


if __name__ == '__main__':
    main()
