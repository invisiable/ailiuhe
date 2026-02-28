"""
优化智能动态倍投策略参数
目标：提升ROI并降低最大回撤
"""

import pandas as pd
import numpy as np
from itertools import product
from precise_top15_predictor import PreciseTop15Predictor

class BettingStrategyBase:
    """投注策略基类（含10倍限制）"""
    
    def __init__(self, base_bet=15, win_reward=47, max_multiplier=10):
        self.base_bet = base_bet
        self.win_reward = win_reward
        self.max_multiplier = max_multiplier
        self.fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        self.reset()
    
    def reset(self):
        self.consecutive_miss = 0
        self.fib_index = 0
        self.total_bet = 0
        self.total_win = 0
        self.balance = 0
        self.max_drawdown = 0
        self.min_balance = 0
    
    def get_base_multiplier(self):
        """获取基础Fibonacci倍数（含限制）"""
        if self.fib_index >= len(self.fib_sequence):
            return min(self.fib_sequence[-1], self.max_multiplier)
        return min(self.fib_sequence[self.fib_index], self.max_multiplier)
    
    def update_balance(self, hit, multiplier):
        """更新余额和统计"""
        bet = self.base_bet * multiplier
        self.total_bet += bet
        
        if hit:
            win = self.win_reward * multiplier
            self.total_win += win
            self.balance += (win - bet)
            self.consecutive_miss = 0
            self.fib_index = 0
            return bet, win, win - bet
        else:
            self.balance -= bet
            self.consecutive_miss += 1
            self.fib_index += 1
            
            if self.balance < self.min_balance:
                self.min_balance = self.balance
                self.max_drawdown = abs(self.min_balance)
            
            return bet, 0, -bet


class SmartDynamicStrategy(BettingStrategyBase):
    """智能动态倍投策略"""
    
    def __init__(self, lookback=12, good_thresh=0.35, bad_thresh=0.20, 
                 boost_mult=1.2, reduce_mult=0.8, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.good_thresh = good_thresh
        self.bad_thresh = bad_thresh
        self.boost_mult = boost_mult
        self.reduce_mult = reduce_mult
        self.recent_results = []
    
    def get_recent_rate(self):
        if len(self.recent_results) == 0:
            return 0.33
        return sum(self.recent_results) / len(self.recent_results)
    
    def process_period(self, prediction, actual):
        """处理一期投注"""
        hit = actual in prediction
        
        self.recent_results.append(1 if hit else 0)
        if len(self.recent_results) > self.lookback:
            self.recent_results.pop(0)
        
        base_mult = self.get_base_multiplier()
        
        if len(self.recent_results) >= self.lookback:
            rate = self.get_recent_rate()
            if rate >= self.good_thresh:
                multiplier = min(base_mult * self.boost_mult, self.max_multiplier)
            elif rate <= self.bad_thresh:
                multiplier = max(base_mult * self.reduce_mult, 1)
            else:
                multiplier = base_mult
        else:
            multiplier = base_mult
        
        bet, win, profit = self.update_balance(hit, multiplier)
        
        return {
            'multiplier': multiplier,
            'bet': bet,
            'win': win,
            'profit': profit,
            'hit': hit
        }


def test_parameters(lookback, good_thresh, bad_thresh, boost_mult, reduce_mult, 
                    numbers, predictor, test_periods=300):
    """测试一组参数"""
    
    strategy = SmartDynamicStrategy(
        lookback=lookback,
        good_thresh=good_thresh,
        bad_thresh=bad_thresh,
        boost_mult=boost_mult,
        reduce_mult=reduce_mult,
        base_bet=15,
        win_reward=47,
        max_multiplier=10
    )
    
    total = len(numbers)
    start = total - test_periods
    hits = 0
    hit_10x_count = 0
    
    for i in range(start, total):
        history = numbers[:i]
        actual = numbers[i]
        
        if len(history) >= 30:
            predictions = predictor.predict(history)
        else:
            predictions = list(range(1, 16))
        
        result = strategy.process_period(predictions, actual)
        
        if result['hit']:
            hits += 1
        
        if result['multiplier'] >= 10:
            hit_10x_count += 1
    
    hit_rate = hits / test_periods * 100
    roi = (strategy.balance / strategy.total_bet * 100) if strategy.total_bet > 0 else 0
    
    return {
        'lookback': lookback,
        'good_thresh': good_thresh,
        'bad_thresh': bad_thresh,
        'boost_mult': boost_mult,
        'reduce_mult': reduce_mult,
        'hits': hits,
        'hit_rate': hit_rate,
        'total_bet': strategy.total_bet,
        'total_win': strategy.total_win,
        'balance': strategy.balance,
        'roi': roi,
        'max_drawdown': strategy.max_drawdown,
        'hit_10x_count': hit_10x_count
    }


def optimize_parameters():
    """优化参数"""
    
    print("="*100)
    print("智能动态倍投策略参数优化")
    print("="*100)
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv')
    numbers = df['number'].tolist()
    predictor = PreciseTop15Predictor()
    
    # 基准策略结果（用于对比）
    baseline = {
        'roi': 15.10,
        'balance': 1709,
        'max_drawdown': 775
    }
    
    print(f"\n【基准策略】")
    print(f"  参数: lookback=12, good_thresh=0.35, bad_thresh=0.20, boost=1.2, reduce=0.8")
    print(f"  ROI: {baseline['roi']:.2f}%")
    print(f"  净利润: {baseline['balance']:.0f}元")
    print(f"  最大回撤: {baseline['max_drawdown']:.0f}元")
    
    # 定义参数网格
    param_grid = {
        'lookback': [8, 10, 12, 15, 18],
        'good_thresh': [0.30, 0.35, 0.40, 0.45],
        'bad_thresh': [0.15, 0.20, 0.25],
        'boost_mult': [1.1, 1.2, 1.3, 1.5],
        'reduce_mult': [0.6, 0.7, 0.8, 0.9]
    }
    
    # 计算总组合数
    total_combinations = 1
    for v in param_grid.values():
        total_combinations *= len(v)
    
    print(f"\n开始测试 {total_combinations} 组参数组合...")
    print(f"筛选条件: ROI > {baseline['roi']:.2f}% 且 回撤 < {baseline['max_drawdown']:.0f}元")
    print("="*100)
    
    # 测试所有组合
    results = []
    count = 0
    better_count = 0
    
    for lookback, good_thresh, bad_thresh, boost_mult, reduce_mult in product(
        param_grid['lookback'],
        param_grid['good_thresh'],
        param_grid['bad_thresh'],
        param_grid['boost_mult'],
        param_grid['reduce_mult']
    ):
        # 跳过不合理的参数组合
        if bad_thresh >= good_thresh:
            continue
        
        count += 1
        
        result = test_parameters(
            lookback, good_thresh, bad_thresh, boost_mult, reduce_mult,
            numbers, predictor
        )
        
        results.append(result)
        
        # 检查是否优于基准
        if result['roi'] > baseline['roi'] and result['max_drawdown'] < baseline['max_drawdown']:
            better_count += 1
            roi_improve = result['roi'] - baseline['roi']
            drawdown_reduce = baseline['max_drawdown'] - result['max_drawdown']
            print(f"✅ 找到更优参数组合 #{better_count}:")
            print(f"   lookback={result['lookback']}, good={result['good_thresh']:.2f}, "
                  f"bad={result['bad_thresh']:.2f}, boost={result['boost_mult']:.1f}, reduce={result['reduce_mult']:.1f}")
            print(f"   ROI: {result['roi']:.2f}% (+{roi_improve:.2f}%), "
                  f"利润: {result['balance']:.0f}元, 回撤: {result['max_drawdown']:.0f}元 (-{drawdown_reduce:.0f}元)")
        
        if count % 100 == 0:
            print(f"已测试: {count}/{total_combinations} 组合...")
    
    print(f"\n测试完成！共测试 {count} 组参数组合")
    print(f"找到 {better_count} 组优于基准的参数配置")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    
    # 筛选优于基准的结果
    better_results = results_df[
        (results_df['roi'] > baseline['roi']) & 
        (results_df['max_drawdown'] < baseline['max_drawdown'])
    ].copy()
    
    if len(better_results) > 0:
        # 计算综合得分（ROI提升 + 回撤降低的归一化得分）
        better_results['roi_improve'] = better_results['roi'] - baseline['roi']
        better_results['drawdown_reduce'] = baseline['max_drawdown'] - better_results['max_drawdown']
        better_results['score'] = (
            better_results['roi_improve'] / baseline['roi'] * 100 +  # ROI提升百分比
            better_results['drawdown_reduce'] / baseline['max_drawdown'] * 100  # 回撤降低百分比
        )
        
        # 按综合得分排序
        better_results = better_results.sort_values('score', ascending=False)
        
        print("\n" + "="*100)
        print("【TOP 10 最优参数配置】")
        print("="*100)
        
        for idx, row in better_results.head(10).iterrows():
            rank = better_results.index.get_loc(idx) + 1
            roi_improve = row['roi'] - baseline['roi']
            roi_improve_pct = roi_improve / baseline['roi'] * 100
            drawdown_reduce = baseline['max_drawdown'] - row['max_drawdown']
            drawdown_reduce_pct = drawdown_reduce / baseline['max_drawdown'] * 100
            profit_improve = row['balance'] - baseline['balance']
            
            print(f"\n【排名 #{rank}】综合得分: {row['score']:.2f}")
            print(f"  参数配置:")
            print(f"    回看期数: {int(row['lookback'])}期")
            print(f"    增强阈值: {row['good_thresh']:.2f} → 倍数×{row['boost_mult']:.1f}")
            print(f"    降低阈值: {row['bad_thresh']:.2f} → 倍数×{row['reduce_mult']:.1f}")
            print(f"  性能指标:")
            print(f"    ROI: {row['roi']:.2f}% (基准+{roi_improve:.2f}%, 提升{roi_improve_pct:.1f}%)")
            print(f"    净利润: {row['balance']:.0f}元 (基准+{profit_improve:.0f}元)")
            print(f"    最大回撤: {row['max_drawdown']:.0f}元 (基准-{drawdown_reduce:.0f}元, 降低{drawdown_reduce_pct:.1f}%)")
            print(f"    触及10倍上限: {int(row['hit_10x_count'])}次")
            print(f"    总投入: {row['total_bet']:.0f}元")
        
        # 保存所有优于基准的结果
        better_results.to_csv('smart_dynamic_optimization_results.csv', index=False, encoding='utf-8-sig')
        print(f"\n所有{len(better_results)}组优化结果已保存到: smart_dynamic_optimization_results.csv")
    
    else:
        print("\n❌ 未找到优于基准策略的参数组合")
        
        # 显示最接近的配置
        results_df['roi_rank'] = results_df['roi'].rank(ascending=False)
        results_df['drawdown_rank'] = results_df['max_drawdown'].rank(ascending=True)
        results_df['avg_rank'] = (results_df['roi_rank'] + results_df['drawdown_rank']) / 2
        
        best_compromise = results_df.nsmallest(5, 'avg_rank')
        
        print("\n【TOP 5 折中方案】")
        for idx, row in best_compromise.iterrows():
            print(f"\n参数: lookback={int(row['lookback'])}, good={row['good_thresh']:.2f}, "
                  f"bad={row['bad_thresh']:.2f}, boost={row['boost_mult']:.1f}, reduce={row['reduce_mult']:.1f}")
            print(f"  ROI: {row['roi']:.2f}% ({row['roi']-baseline['roi']:+.2f}%)")
            print(f"  回撤: {row['max_drawdown']:.0f}元 ({row['max_drawdown']-baseline['max_drawdown']:+.0f}元)")
    
    # 保存所有测试结果
    results_df.to_csv('smart_dynamic_all_test_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n所有测试结果已保存到: smart_dynamic_all_test_results.csv")
    
    return better_results if len(better_results) > 0 else best_compromise


if __name__ == "__main__":
    better_results = optimize_parameters()
    
    print("\n" + "="*100)
    print("✅ 参数优化完成！")
    print("="*100)
