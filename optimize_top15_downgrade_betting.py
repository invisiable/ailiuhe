"""
精准TOP15投注逻辑优化V2：降档倍投策略
在连续N期失败后，降低倍投档位而非暂停，既控制风险又不错过盈利机会
"""

import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor
from itertools import product

class DowngradeBettingStrategy:
    """降档倍投策略"""
    
    def __init__(self, predictor, base_bet=15, win_reward=47,
                 trigger_after_n_miss=5, downgrade_to_level=0, 
                 downgrade_duration=3):
        """
        参数:
        - trigger_after_n_miss: 连续失败N期后触发降档
        - downgrade_to_level: 降低到Fibonacci第几档(0=第1档)
        - downgrade_duration: 降档持续N期后恢复
        """
        self.predictor = predictor
        self.base_bet = base_bet
        self.win_reward = win_reward
        self.trigger_after_n_miss = trigger_after_n_miss
        self.downgrade_to_level = downgrade_to_level
        self.downgrade_duration = downgrade_duration
        
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
        
        # 降档状态
        self.is_downgraded = False
        self.downgrade_remaining = 0
        self.downgrade_triggered_count = 0
    
    def get_current_multiplier(self):
        """获取当前倍投倍数"""
        if self.is_downgraded:
            return self.fib_sequence[self.downgrade_to_level]
        
        if self.fib_index >= len(self.fib_sequence):
            return self.fib_sequence[-1]
        return self.fib_sequence[self.fib_index]
    
    def process_period(self, prediction, actual):
        """处理一期投注"""
        hit = actual in prediction
        
        # 降档期间计数
        if self.is_downgraded:
            self.downgrade_remaining -= 1
            if self.downgrade_remaining <= 0:
                self.is_downgraded = False
        
        # 正常投注（包括降档投注）
        multiplier = self.get_current_multiplier()
        bet = self.base_bet * multiplier
        self.total_bet += bet
        
        if hit:
            # 命中
            win = self.win_reward * multiplier
            self.total_win += win
            self.balance += (win - bet)
            self.consecutive_miss = 0
            
            # 命中后根据是否降档决定fib_index
            if not self.is_downgraded:
                self.fib_index = 0
            # 如果在降档期间命中，继续使用降档倍数，但不重置
        else:
            # 未命中
            self.balance -= bet
            self.consecutive_miss += 1
            
            # 如果不在降档期，正常递增
            if not self.is_downgraded:
                self.fib_index += 1
                
                # 判断是否触发降档
                if self.consecutive_miss >= self.trigger_after_n_miss:
                    self.is_downgraded = True
                    self.downgrade_remaining = self.downgrade_duration
                    self.downgrade_triggered_count += 1
                    self.fib_index = self.downgrade_to_level  # 重置到降档档位
        
        # 更新最大回撤
        if self.balance < self.min_balance:
            self.min_balance = self.balance
            self.max_drawdown = abs(self.min_balance)
        
        return {
            'bet': bet,
            'win': win if hit else 0,
            'hit': hit,
            'downgraded': self.is_downgraded,
            'multiplier': multiplier,
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
            'downgrade_triggered': self.downgrade_triggered_count
        }


def backtest_strategy(numbers, predictor, trigger_after, downgrade_to, duration, test_periods=200):
    """回测特定参数的策略"""
    total = len(numbers)
    start = total - test_periods
    
    strategy = DowngradeBettingStrategy(
        predictor=predictor,
        trigger_after_n_miss=trigger_after,
        downgrade_to_level=downgrade_to,
        downgrade_duration=duration
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
            'downgraded': result['downgraded'],
            'balance': result['balance']
        })
    
    stats = strategy.get_stats()
    stats['hit_count'] = hit_count
    stats['hit_rate'] = hit_count / test_periods * 100
    
    return stats, results


def test_all_combinations(test_periods=200):
    """测试所有参数组合"""
    print("="*100)
    print("精准TOP15投注逻辑优化V2：降档倍投策略回测")
    print("="*100)
    
    df = pd.read_csv('data/lucky_numbers.csv')
    numbers = df['number'].tolist()
    
    predictor = PreciseTop15Predictor()
    
    # 首先测试原始策略（不降档）
    print("\n【基准策略】原始Fibonacci倍投（不降档）")
    print("-" * 100)
    
    baseline_stats, _ = backtest_strategy(
        numbers, predictor,
        trigger_after=999,  # 永不触发
        downgrade_to=0,
        duration=0,
        test_periods=test_periods
    )
    
    print(f"命中率: {baseline_stats['hit_rate']:.2f}%")
    print(f"总投入: {baseline_stats['total_bet']:.0f}元")
    print(f"总收益: {baseline_stats['total_win']:.0f}元")
    print(f"净利润: {baseline_stats['balance']:.0f}元")
    print(f"ROI: {baseline_stats['roi']:+.2f}%")
    print(f"最大回撤: {baseline_stats['max_drawdown']:.0f}元")
    
    # 测试不同的降档参数组合
    print("\n" + "="*100)
    print("【优化策略】测试不同降档参数组合")
    print("="*100)
    
    # 参数范围
    trigger_options = [4, 5, 6, 7]  # 连续失败N期后降档
    downgrade_to_options = [0, 1, 2]  # 降低到第几档 (0=1倍, 1=1倍, 2=2倍)
    duration_options = [2, 3, 4, 5]  # 降档持续N期
    
    all_results = []
    
    for trigger, downgrade_to, duration in product(trigger_options, downgrade_to_options, duration_options):
        stats, _ = backtest_strategy(
            numbers, predictor,
            trigger_after=trigger,
            downgrade_to=downgrade_to,
            duration=duration,
            test_periods=test_periods
        )
        
        stats['trigger_after'] = trigger
        stats['downgrade_to'] = downgrade_to
        stats['duration'] = duration
        all_results.append(stats)
    
    # 转为DataFrame
    results_df = pd.DataFrame(all_results)
    
    # 按ROI排序
    results_df = results_df.sort_values('roi', ascending=False)
    
    # 显示前10名
    print("\n【TOP 10最优策略】(按ROI排序)")
    print("-" * 100)
    print(f"{'排名':<5} {'触发连败':<10} {'降至档位':<10} {'持续期数':<10} {'命中率':<10} {'ROI':<12} {'最大回撤':<12} {'触发次数':<10}")
    print("-" * 100)
    
    for idx, row in results_df.head(10).iterrows():
        rank = results_df.index.get_loc(idx) + 1
        fib_level_display = f"Fib[{int(row['downgrade_to'])}]"
        print(f"{rank:<5} {row['trigger_after']:<10.0f} {fib_level_display:<10} "
              f"{row['duration']:<10.0f} {row['hit_rate']:<10.2f}% "
              f"{row['roi']:<11.2f}% {row['max_drawdown']:<12.0f}元 "
              f"{row['downgrade_triggered']:<10.0f}")
    
    # 找到最优策略
    best = results_df.iloc[0]
    
    print("\n" + "="*100)
    print("【最优策略详情】")
    print("="*100)
    print(f"\n参数配置:")
    print(f"  连续失败{best['trigger_after']:.0f}期后 → 降低到Fibonacci第{best['downgrade_to']:.0f}档")
    print(f"  降档持续{best['duration']:.0f}期后恢复正常倍投")
    print(f"  Fibonacci档位: {[1,1,2,3,5,8,13,21,34,55,89,144]}")
    
    print(f"\n性能指标:")
    print(f"  命中率: {best['hit_rate']:.2f}%")
    print(f"  总投入: {best['total_bet']:.0f}元")
    print(f"  净利润: {best['balance']:.0f}元")
    print(f"  ROI: {best['roi']:+.2f}%")
    print(f"  最大回撤: {best['max_drawdown']:.0f}元")
    print(f"  降档触发: {best['downgrade_triggered']:.0f}次")
    
    print(f"\n与基准对比:")
    roi_improve = best['roi'] - baseline_stats['roi']
    drawdown_reduce = baseline_stats['max_drawdown'] - best['max_drawdown']
    profit_improve = best['balance'] - baseline_stats['balance']
    
    print(f"  ROI: {baseline_stats['roi']:.2f}% → {best['roi']:.2f}% ({roi_improve:+.2f}%)")
    print(f"  最大回撤: {baseline_stats['max_drawdown']:.0f}元 → {best['max_drawdown']:.0f}元 ({drawdown_reduce:+.0f}元)")
    print(f"  净利润: {baseline_stats['balance']:.0f}元 → {best['balance']:.0f}元 ({profit_improve:+.0f}元)")
    
    if roi_improve > 0 and drawdown_reduce > 0:
        print(f"\n✅ 找到更优策略！ROI提升{roi_improve:.2f}%，回撤减少{drawdown_reduce:.0f}元")
    elif roi_improve > 0:
        print(f"\n⚠️ ROI提升{roi_improve:.2f}%，但回撤增加{-drawdown_reduce:.0f}元")
    elif drawdown_reduce > 0:
        print(f"\n⚠️ 回撤减少{drawdown_reduce:.0f}元，但ROI下降{-roi_improve:.2f}%")
    else:
        print(f"\n❌ 未找到双优策略（ROI和回撤同时改善）")
    
    # 详细回测最优策略
    print("\n" + "="*100)
    print("【最优策略详细回测】")
    print("="*100)
    
    _, detail_results = backtest_strategy(
        numbers, predictor,
        trigger_after=int(best['trigger_after']),
        downgrade_to=int(best['downgrade_to']),
        duration=int(best['duration']),
        test_periods=test_periods
    )
    
    detail_df = pd.DataFrame(detail_results)
    
    # 保存详细结果
    detail_df.to_csv('top15_downgrade_betting_detail.csv', index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到: top15_downgrade_betting_detail.csv")
    
    # 分析降档效果
    downgraded_bets = detail_df[detail_df['downgraded'] == True]
    print(f"\n降档投注分析:")
    print(f"  降档期数: {len(downgraded_bets)}期")
    print(f"  降档期间命中: {downgraded_bets['hit'].sum()}次")
    if len(downgraded_bets) > 0:
        print(f"  降档期平均倍数: {downgraded_bets['multiplier'].mean():.2f}")
    
    # 连续失败分析
    consecutive_miss = 0
    miss_segments = []
    
    for _, row in detail_df.iterrows():
        if not row['hit']:
            consecutive_miss += 1
        else:
            if consecutive_miss > 0:
                miss_segments.append(consecutive_miss)
            consecutive_miss = 0
    
    if miss_segments:
        print(f"\n连续失败期段分析:")
        print(f"  最大连续失败: {max(miss_segments)}期")
        print(f"  平均连续失败: {np.mean(miss_segments):.2f}期")
        print(f"  ≥5期长连败: {len([s for s in miss_segments if s >= 5])}次")
    
    return results_df, best, baseline_stats


def analyze_best_strategies(results_df, baseline_stats):
    """分析综合最优策略"""
    print("\n" + "="*100)
    print("【综合优化分析】")
    print("="*100)
    
    # 筛选：ROI > 基准 AND 回撤 < 基准
    better_roi = results_df[results_df['roi'] > baseline_stats['roi']]
    better_drawdown = results_df[results_df['max_drawdown'] < baseline_stats['max_drawdown']]
    
    # 双优策略
    double_better = results_df[
        (results_df['roi'] > baseline_stats['roi']) &
        (results_df['max_drawdown'] < baseline_stats['max_drawdown'])
    ]
    
    print(f"\n策略分类统计:")
    print(f"  总测试策略数: {len(results_df)}")
    print(f"  ROI优于基准: {len(better_roi)}个")
    print(f"  回撤优于基准: {len(better_drawdown)}个")
    print(f"  双优策略(ROI↑且回撤↓): {len(double_better)}个")
    
    if len(double_better) > 0:
        print("\n✅ 找到双优策略！")
        print("-" * 100)
        print(f"{'排名':<5} {'触发连败':<10} {'降至档位':<10} {'持续期数':<10} {'ROI':<12} {'最大回撤':<12}")
        print("-" * 100)
        
        for idx, (_, row) in enumerate(double_better.head(10).iterrows(), 1):
            fib_level = f"Fib[{int(row['downgrade_to'])}]"
            print(f"{idx:<5} {row['trigger_after']:<10.0f} {fib_level:<10} "
                  f"{row['duration']:<10.0f} {row['roi']:<11.2f}% {row['max_drawdown']:<12.0f}元")
        
        best_double = double_better.iloc[0]
        roi_gain = best_double['roi'] - baseline_stats['roi']
        drawdown_save = baseline_stats['max_drawdown'] - best_double['max_drawdown']
        
        print(f"\n【推荐策略】")
        print(f"  连续失败{best_double['trigger_after']:.0f}期 → 降至Fib[{best_double['downgrade_to']:.0f}]档 → 持续{best_double['duration']:.0f}期")
        print(f"  ROI提升: {roi_gain:+.2f}%")
        print(f"  回撤减少: {drawdown_save:.0f}元")
    else:
        print("\n⚠️ 未找到同时提升ROI和降低回撤的策略")
        print("\n备选方案:")
        
        if len(better_roi) > 0:
            best_roi = better_roi.iloc[0]
            print(f"\n  【方案A】最高ROI策略: {best_roi['roi']:.2f}%")
            print(f"    但回撤增加: {best_roi['max_drawdown'] - baseline_stats['max_drawdown']:.0f}元")
        
        if len(better_drawdown) > 0:
            best_drawdown = better_drawdown.nsmallest(1, 'max_drawdown').iloc[0]
            print(f"\n  【方案B】最低回撤策略: {best_drawdown['max_drawdown']:.0f}元")
            print(f"    但ROI降至: {best_drawdown['roi']:.2f}%")
    
    # 风险收益比分析
    results_df['reward_risk_ratio'] = results_df['roi'] / (results_df['max_drawdown'] + 1)
    baseline_ratio = baseline_stats['roi'] / (baseline_stats['max_drawdown'] + 1)
    
    best_ratio = results_df.nlargest(1, 'reward_risk_ratio').iloc[0]
    
    print(f"\n【风险收益比分析】")
    print(f"  基准策略: {baseline_ratio:.4f}")
    print(f"  最优策略: {best_ratio['reward_risk_ratio']:.4f}")
    print(f"  提升幅度: {(best_ratio['reward_risk_ratio'] - baseline_ratio) / baseline_ratio * 100:+.2f}%")


if __name__ == "__main__":
    print("\n" + "="*100)
    print("开始回测 - 测试期数: 200期")
    print("="*100)
    
    results_df, best, baseline_stats = test_all_combinations(test_periods=200)
    
    # 综合分析
    analyze_best_strategies(results_df, baseline_stats)
    
    # 保存所有结果
    results_df.to_csv('top15_downgrade_all_strategies.csv', index=False, encoding='utf-8-sig')
    print("\n" + "="*100)
    print("所有策略结果已保存到: top15_downgrade_all_strategies.csv")
    print("="*100)
