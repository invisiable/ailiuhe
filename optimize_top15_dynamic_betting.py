"""
精准TOP15投注逻辑优化V3：智能动态倍投策略
根据近期命中率和连败情况，动态调整倍投系数，实现收益和风险的平衡
"""

import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor
from itertools import product

class SmartDynamicBettingStrategy:
    """智能动态倍投策略"""
    
    def __init__(self, predictor, base_bet=15, win_reward=47,
                 lookback_window=10, good_threshold=0.4, bad_threshold=0.25,
                 boost_multiplier=1.5, reduce_multiplier=0.7):
        """
        参数:
        - lookback_window: 回看最近N期表现
        - good_threshold: 命中率>此值时增强倍投
        - bad_threshold: 命中率<此值时降低倍投
        - boost_multiplier: 增强系数
        - reduce_multiplier: 降低系数
        """
        self.predictor = predictor
        self.base_bet = base_bet
        self.win_reward = win_reward
        self.lookback_window = lookback_window
        self.good_threshold = good_threshold
        self.bad_threshold = bad_threshold
        self.boost_multiplier = boost_multiplier
        self.reduce_multiplier = reduce_multiplier
        
        # Fibonacci序列
        self.fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        
        self.reset()
    
    def reset(self):
        """重置状态"""
        self.consecutive_miss = 0
        self.fib_index = 0
        self.total_bet = 0
        self.total_win = 0
        self.balance = 0
        self.max_drawdown = 0
        self.min_balance = 0
        
        # 近期表现跟踪
        self.recent_results = []  # 最近N期的命中情况
        self.boost_periods = 0
        self.reduce_periods = 0
    
    def get_recent_hit_rate(self):
        """计算近期命中率"""
        if len(self.recent_results) == 0:
            return 0.35  # 默认命中率
        return sum(self.recent_results) / len(self.recent_results)
    
    def get_dynamic_multiplier(self):
        """获取动态调整后的倍投倍数"""
        # 基础Fibonacci倍数
        if self.fib_index >= len(self.fib_sequence):
            base_fib = self.fib_sequence[-1]
        else:
            base_fib = self.fib_sequence[self.fib_index]
        
        # 如果历史不足，使用基础倍数
        if len(self.recent_results) < self.lookback_window:
            return base_fib
        
        # 计算近期命中率
        recent_rate = self.get_recent_hit_rate()
        
        # 动态调整
        if recent_rate >= self.good_threshold:
            # 表现好，增强倍投
            adjusted = base_fib * self.boost_multiplier
            return min(adjusted, self.fib_sequence[-1])  # 不超过最大档位
        elif recent_rate <= self.bad_threshold:
            # 表现差，降低倍投
            adjusted = base_fib * self.reduce_multiplier
            return max(adjusted, 1)  # 不低于1倍
        else:
            # 表现正常，保持原倍数
            return base_fib
    
    def process_period(self, prediction, actual):
        """处理一期投注"""
        hit = actual in prediction
        
        # 动态倍投
        multiplier = self.get_dynamic_multiplier()
        bet = self.base_bet * multiplier
        self.total_bet += bet
        
        # 记录近期表现
        self.recent_results.append(1 if hit else 0)
        if len(self.recent_results) > self.lookback_window:
            self.recent_results.pop(0)
        
        # 判断是否在增强/降低状态
        recent_rate = self.get_recent_hit_rate()
        is_boosted = recent_rate >= self.good_threshold and len(self.recent_results) >= self.lookback_window
        is_reduced = recent_rate <= self.bad_threshold and len(self.recent_results) >= self.lookback_window
        
        if is_boosted:
            self.boost_periods += 1
        if is_reduced:
            self.reduce_periods += 1
        
        if hit:
            # 命中
            win = self.win_reward * multiplier
            self.total_win += win
            self.balance += (win - bet)
            self.consecutive_miss = 0
            self.fib_index = 0
        else:
            # 未命中
            self.balance -= bet
            self.consecutive_miss += 1
            self.fib_index += 1
        
        # 更新最大回撤
        if self.balance < self.min_balance:
            self.min_balance = self.balance
            self.max_drawdown = abs(self.min_balance)
        
        return {
            'bet': bet,
            'win': win if hit else 0,
            'hit': hit,
            'multiplier': multiplier,
            'recent_rate': recent_rate,
            'is_boosted': is_boosted,
            'is_reduced': is_reduced,
            'balance': self.balance
        }
    
    def get_stats(self):
        """获取统计数据"""
        roi = (self.balance / self.total_bet * 100) if self.total_bet > 0 else 0
        return {
            'total_bet': self.total_bet,
            'total_win': self.total_win,
            'balance': self.balance,
            'roi': roi,
            'max_drawdown': self.max_drawdown,
            'boost_periods': self.boost_periods,
            'reduce_periods': self.reduce_periods
        }


def backtest_strategy(numbers, predictor, lookback, good_thresh, bad_thresh,
                      boost_mult, reduce_mult, test_periods=200):
    """回测特定参数的策略"""
    total = len(numbers)
    start = total - test_periods
    
    strategy = SmartDynamicBettingStrategy(
        predictor=predictor,
        lookback_window=lookback,
        good_threshold=good_thresh,
        bad_threshold=bad_thresh,
        boost_multiplier=boost_mult,
        reduce_multiplier=reduce_mult
    )
    
    results = []
    hit_count = 0
    
    for i in range(start, total):
        history = numbers[:i]
        actual = numbers[i]
        prediction = predictor.predict(history)
        
        result = strategy.process_period(prediction, actual)
        
        if result['hit']:
            hit_count += 1
        
        results.append({
            'period': i - start + 1,
            'actual': actual,
            'hit': result['hit'],
            'bet': result['bet'],
            'multiplier': result['multiplier'],
            'recent_rate': result['recent_rate'],
            'is_boosted': result['is_boosted'],
            'is_reduced': result['is_reduced'],
            'balance': result['balance']
        })
    
    stats = strategy.get_stats()
    stats['hit_count'] = hit_count
    stats['hit_rate'] = hit_count / test_periods * 100
    
    return stats, results


def test_all_combinations(test_periods=200):
    """测试所有参数组合"""
    print("="*100)
    print("精准TOP15投注逻辑优化V3：智能动态倍投策略回测")
    print("="*100)
    
    df = pd.read_csv('data/lucky_numbers.csv')
    numbers = df['number'].tolist()
    
    predictor = PreciseTop15Predictor()
    
    # 首先测试原始策略
    print("\n【基准策略】原始Fibonacci倍投")
    print("-" * 100)
    
    baseline_stats, _ = backtest_strategy(
        numbers, predictor,
        lookback=10,
        good_thresh=1.0,  # 永不增强
        bad_thresh=0.0,   # 永不降低
        boost_mult=1.0,
        reduce_mult=1.0,
        test_periods=test_periods
    )
    
    print(f"命中率: {baseline_stats['hit_rate']:.2f}%")
    print(f"总投入: {baseline_stats['total_bet']:.0f}元")
    print(f"总收益: {baseline_stats['total_win']:.0f}元")
    print(f"净利润: {baseline_stats['balance']:.0f}元")
    print(f"ROI: {baseline_stats['roi']:+.2f}%")
    print(f"最大回撤: {baseline_stats['max_drawdown']:.0f}元")
    
    # 测试不同的动态参数组合
    print("\n" + "="*100)
    print("【优化策略】测试不同动态参数组合")
    print("="*100)
    
    # 参数范围
    lookback_options = [8, 10, 12, 15]
    good_threshold_options = [0.35, 0.40, 0.45]
    bad_threshold_options = [0.20, 0.25, 0.30]
    boost_multiplier_options = [1.2, 1.3, 1.5]
    reduce_multiplier_options = [0.6, 0.7, 0.8]
    
    all_results = []
    total_tests = (len(lookback_options) * len(good_threshold_options) * 
                   len(bad_threshold_options) * len(boost_multiplier_options) * 
                   len(reduce_multiplier_options))
    
    print(f"总测试数: {total_tests}个策略\n")
    
    test_count = 0
    for lookback in lookback_options:
        for good_thresh in good_threshold_options:
            for bad_thresh in bad_threshold_options:
                if bad_thresh >= good_thresh:
                    continue  # 跳过无效参数
                
                for boost_mult in boost_multiplier_options:
                    for reduce_mult in reduce_multiplier_options:
                        test_count += 1
                        if test_count % 20 == 0:
                            print(f"进度: {test_count}/{total_tests}...")
                        
                        stats, _ = backtest_strategy(
                            numbers, predictor,
                            lookback=lookback,
                            good_thresh=good_thresh,
                            bad_thresh=bad_thresh,
                            boost_mult=boost_mult,
                            reduce_mult=reduce_mult,
                            test_periods=test_periods
                        )
                        
                        stats['lookback'] = lookback
                        stats['good_threshold'] = good_thresh
                        stats['bad_threshold'] = bad_thresh
                        stats['boost_multiplier'] = boost_mult
                        stats['reduce_multiplier'] = reduce_mult
                        all_results.append(stats)
    
    print(f"\n实际测试: {len(all_results)}个有效策略")
    
    # 转为DataFrame
    results_df = pd.DataFrame(all_results)
    
    # 按ROI排序
    results_df = results_df.sort_values('roi', ascending=False)
    
    # 显示前10名
    print("\n【TOP 10最优策略】(按ROI排序)")
    print("-" * 100)
    print(f"{'排名':<5} {'回看期':<8} {'好阈值':<8} {'差阈值':<8} {'增强':<7} {'降低':<7} {'ROI':<10} {'回撤':<10}")
    print("-" * 100)
    
    for idx, row in results_df.head(10).iterrows():
        rank = results_df.index.get_loc(idx) + 1
        print(f"{rank:<5} {row['lookback']:<8.0f} {row['good_threshold']:<8.2f} "
              f"{row['bad_threshold']:<8.2f} {row['boost_multiplier']:<7.1f} "
              f"{row['reduce_multiplier']:<7.1f} {row['roi']:<9.2f}% {row['max_drawdown']:<10.0f}元")
    
    # 找到最优策略
    best = results_df.iloc[0]
    
    print("\n" + "="*100)
    print("【最优策略详情】")
    print("="*100)
    print(f"\n参数配置:")
    print(f"  回看窗口: 最近{best['lookback']:.0f}期")
    print(f"  命中率>{best['good_threshold']:.2f}时: 增强倍投至{best['boost_multiplier']:.1f}倍")
    print(f"  命中率<{best['bad_threshold']:.2f}时: 降低倍投至{best['reduce_multiplier']:.1f}倍")
    
    print(f"\n性能指标:")
    print(f"  命中率: {best['hit_rate']:.2f}%")
    print(f"  总投入: {best['total_bet']:.0f}元")
    print(f"  净利润: {best['balance']:.0f}元")
    print(f"  ROI: {best['roi']:+.2f}%")
    print(f"  最大回撤: {best['max_drawdown']:.0f}元")
    print(f"  增强期数: {best['boost_periods']:.0f}期")
    print(f"  降低期数: {best['reduce_periods']:.0f}期")
    
    print(f"\n与基准对比:")
    roi_improve = best['roi'] - baseline_stats['roi']
    drawdown_change = best['max_drawdown'] - baseline_stats['max_drawdown']
    profit_improve = best['balance'] - baseline_stats['balance']
    
    print(f"  ROI: {baseline_stats['roi']:.2f}% → {best['roi']:.2f}% ({roi_improve:+.2f}%)")
    print(f"  最大回撤: {baseline_stats['max_drawdown']:.0f}元 → {best['max_drawdown']:.0f}元 ({drawdown_change:+.0f}元)")
    print(f"  净利润: {baseline_stats['balance']:.0f}元 → {best['balance']:.0f}元 ({profit_improve:+.0f}元)")
    
    if roi_improve > 0 and drawdown_change < 0:
        print(f"\n🎉 成功！找到双优策略：ROI提升{roi_improve:.2f}%，回撤减少{-drawdown_change:.0f}元")
    elif roi_improve > 0:
        print(f"\n✅ ROI提升{roi_improve:.2f}%（回撤变化{drawdown_change:+.0f}元）")
    elif drawdown_change < 0:
        print(f"\n⚠️ 回撤减少{-drawdown_change:.0f}元，但ROI下降{-roi_improve:.2f}%")
    else:
        print(f"\n❌ 未找到优于基准的策略")
    
    # 详细回测最优策略
    print("\n" + "="*100)
    print("【最优策略详细回测】")
    print("="*100)
    
    _, detail_results = backtest_strategy(
        numbers, predictor,
        lookback=int(best['lookback']),
        good_thresh=best['good_threshold'],
        bad_thresh=best['bad_threshold'],
        boost_mult=best['boost_multiplier'],
        reduce_mult=best['reduce_multiplier'],
        test_periods=test_periods
    )
    
    detail_df = pd.DataFrame(detail_results)
    
    # 保存详细结果
    detail_df.to_csv('top15_dynamic_betting_detail.csv', index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到: top15_dynamic_betting_detail.csv")
    
    # 分析动态调整效果
    boosted_periods = detail_df[detail_df['is_boosted'] == True]
    reduced_periods = detail_df[detail_df['is_reduced'] == True]
    
    print(f"\n动态调整分析:")
    print(f"  增强期数: {len(boosted_periods)}期")
    if len(boosted_periods) > 0:
        print(f"    增强期命中率: {boosted_periods['hit'].mean()*100:.2f}%")
        print(f"    平均倍数: {boosted_periods['multiplier'].mean():.2f}")
    
    print(f"  降低期数: {len(reduced_periods)}期")
    if len(reduced_periods) > 0:
        print(f"    降低期命中率: {reduced_periods['hit'].mean()*100:.2f}%")
        print(f"    平均倍数: {reduced_periods['multiplier'].mean():.2f}")
    
    return results_df, best, baseline_stats


def analyze_best_strategies(results_df, baseline_stats):
    """分析最优策略"""
    print("\n" + "="*100)
    print("【综合优化分析】")
    print("="*100)
    
    # 筛选双优策略
    double_better = results_df[
        (results_df['roi'] > baseline_stats['roi']) &
        (results_df['max_drawdown'] < baseline_stats['max_drawdown'])
    ]
    
    better_roi = results_df[results_df['roi'] > baseline_stats['roi']]
    better_drawdown = results_df[results_df['max_drawdown'] < baseline_stats['max_drawdown']]
    
    print(f"\n策略分类统计:")
    print(f"  总测试策略数: {len(results_df)}")
    print(f"  ROI优于基准: {len(better_roi)}个 ({len(better_roi)/len(results_df)*100:.1f}%)")
    print(f"  回撤优于基准: {len(better_drawdown)}个 ({len(better_drawdown)/len(results_df)*100:.1f}%)")
    print(f"  🎯 双优策略: {len(double_better)}个")
    
    if len(double_better) > 0:
        print("\n✅ 找到同时提升ROI和降低回撤的策略！")
        print("-" * 100)
        
        for idx, (_, row) in enumerate(double_better.head(5).iterrows(), 1):
            roi_gain = row['roi'] - baseline_stats['roi']
            drawdown_save = baseline_stats['max_drawdown'] - row['max_drawdown']
            
            print(f"\n策略{idx}:")
            print(f"  参数: 回看{row['lookback']:.0f}期, 好>{row['good_threshold']:.2f}×{row['boost_multiplier']:.1f}, "
                  f"差<{row['bad_threshold']:.2f}×{row['reduce_multiplier']:.1f}")
            print(f"  ROI提升: {roi_gain:+.2f}% ({row['roi']:.2f}%)")
            print(f"  回撤减少: {drawdown_save:.0f}元 ({row['max_drawdown']:.0f}元)")
    
    # 风险收益比
    results_df['reward_risk_ratio'] = results_df['roi'] / (results_df['max_drawdown'] + 1)
    baseline_ratio = baseline_stats['roi'] / (baseline_stats['max_drawdown'] + 1)
    
    best_ratio = results_df.nlargest(1, 'reward_risk_ratio').iloc[0]
    
    print(f"\n【风险收益比分析】")
    print(f"  基准策略: {baseline_ratio:.4f}")
    print(f"  最优策略: {best_ratio['reward_risk_ratio']:.4f}")
    ratio_improve = (best_ratio['reward_risk_ratio'] - baseline_ratio) / baseline_ratio * 100
    print(f"  提升幅度: {ratio_improve:+.2f}%")


if __name__ == "__main__":
    print("\n" + "="*100)
    print("开始回测 - 测试期数: 200期")
    print("="*100)
    
    results_df, best, baseline_stats = test_all_combinations(test_periods=200)
    
    # 综合分析
    analyze_best_strategies(results_df, baseline_stats)
    
    # 保存所有结果
    results_df.to_csv('top15_dynamic_all_strategies.csv', index=False, encoding='utf-8-sig')
    print("\n" + "="*100)
    print("所有策略结果已保存到: top15_dynamic_all_strategies.csv")
    print("="*100)
