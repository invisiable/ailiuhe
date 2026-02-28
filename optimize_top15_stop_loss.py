"""
精准TOP15投注逻辑优化：智能止损暂停策略
测试不同的止损暂停参数，找到最优方案
"""

import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor
from itertools import product

class SmartStopLossStrategy:
    """智能止损暂停策略"""
    
    def __init__(self, predictor, base_bet=15, win_reward=47, 
                 stop_after_n_miss=5, pause_n_periods=3):
        """
        参数:
        - predictor: 预测器
        - base_bet: 基础投注额
        - win_reward: 中奖奖励
        - stop_after_n_miss: 连续失败N期后触发止损
        - pause_n_periods: 暂停N期
        """
        self.predictor = predictor
        self.base_bet = base_bet
        self.win_reward = win_reward
        self.stop_after_n_miss = stop_after_n_miss
        self.pause_n_periods = pause_n_periods
        
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
        
        # 止损暂停状态
        self.is_paused = False
        self.pause_remaining = 0
        self.pause_virtual_hits = []  # 暂停期间的虚拟命中记录
    
    def get_current_multiplier(self):
        """获取当前倍投倍数"""
        if self.fib_index >= len(self.fib_sequence):
            return self.fib_sequence[-1]
        return self.fib_sequence[self.fib_index]
    
    def should_bet(self):
        """判断是否应该投注"""
        return not self.is_paused
    
    def process_period(self, prediction, actual):
        """处理一期投注"""
        hit = actual in prediction
        
        # 如果在暂停期
        if self.is_paused:
            self.pause_remaining -= 1
            
            # 虚拟跟踪命中情况
            if hit:
                self.pause_virtual_hits.append(True)
                # 如果暂停期间命中，立即取消暂停
                self.is_paused = False
                self.pause_remaining = 0
                self.consecutive_miss = 0
                self.fib_index = 0
                return {
                    'bet': 0,
                    'win': 0,
                    'hit': hit,
                    'paused': True,
                    'pause_cancelled': True,
                    'pause_remaining': 0,
                    'balance': self.balance
                }
            else:
                self.pause_virtual_hits.append(False)
            
            # 暂停期结束
            if self.pause_remaining <= 0:
                self.is_paused = False
                self.consecutive_miss = 0
                self.fib_index = 0
            
            return {
                'bet': 0,
                'win': 0,
                'hit': hit,
                'paused': True,
                'pause_cancelled': False,                'pause_remaining': self.pause_remaining,                'balance': self.balance
            }
        
        # 正常投注
        multiplier = self.get_current_multiplier()
        bet = self.base_bet * multiplier
        self.total_bet += bet
        
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
            
            # 判断是否触发止损
            if self.consecutive_miss >= self.stop_after_n_miss:
                self.is_paused = True
                self.pause_remaining = self.pause_n_periods
                self.pause_virtual_hits = []
        
        # 更新最大回撤
        if self.balance < self.min_balance:
            self.min_balance = self.balance
            self.max_drawdown = abs(self.min_balance)
        
        return {
            'bet': bet,
            'win': self.total_win - (self.total_bet - bet) if hit else 0,
            'hit': hit,
            'paused': False,
            'pause_cancelled': False,
            'pause_remaining': 0,
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
            'max_drawdown': self.max_drawdown
        }


def backtest_strategy(numbers, predictor, stop_after_n, pause_n, test_periods=200):
    """回测特定参数的策略"""
    total = len(numbers)
    start = total - test_periods
    
    strategy = SmartStopLossStrategy(
        predictor=predictor,
        stop_after_n_miss=stop_after_n,
        pause_n_periods=pause_n
    )
    
    results = []
    hit_count = 0
    pause_triggered_count = 0
    pause_cancelled_count = 0
    
    for i in range(start, total):
        history = numbers[:i]
        actual = numbers[i]
        prediction = predictor.predict(history)
        
        result = strategy.process_period(prediction, actual)
        
        if result['hit']:
            hit_count += 1
        
        if result['paused'] and result['pause_remaining'] == pause_n - 1:
            pause_triggered_count += 1
        
        if result['pause_cancelled']:
            pause_cancelled_count += 1
        
        results.append({
            'period': i - start + 1,
            'actual': actual,
            'hit': result['hit'],
            'bet': result['bet'],
            'balance': result['balance'],
            'paused': result['paused']
        })
    
    stats = strategy.get_stats()
    stats['hit_count'] = hit_count
    stats['hit_rate'] = hit_count / test_periods * 100
    stats['pause_triggered'] = pause_triggered_count
    stats['pause_cancelled'] = pause_cancelled_count
    
    return stats, results


def test_all_combinations(test_periods=200):
    """测试所有参数组合"""
    print("="*100)
    print("精准TOP15投注逻辑优化：智能止损暂停策略回测")
    print("="*100)
    
    df = pd.read_csv('data/lucky_numbers.csv')
    numbers = df['number'].tolist()
    
    predictor = PreciseTop15Predictor()
    
    # 首先测试原始策略（不止损）
    print("\n【基准策略】原始Fibonacci倍投（不止损）")
    print("-" * 100)
    
    baseline_stats, _ = backtest_strategy(
        numbers, predictor, 
        stop_after_n=999,  # 永不触发
        pause_n=0, 
        test_periods=test_periods
    )
    
    print(f"命中率: {baseline_stats['hit_rate']:.2f}%")
    print(f"总投入: {baseline_stats['total_bet']:.0f}元")
    print(f"总收益: {baseline_stats['total_win']:.0f}元")
    print(f"净利润: {baseline_stats['balance']:.0f}元")
    print(f"ROI: {baseline_stats['roi']:+.2f}%")
    print(f"最大回撤: {baseline_stats['max_drawdown']:.0f}元")
    
    # 测试不同的止损参数组合
    print("\n" + "="*100)
    print("【优化策略】测试不同止损参数组合")
    print("="*100)
    
    # 参数范围
    stop_after_options = [3, 4, 5, 6, 7]  # 连续失败N期后止损
    pause_options = [1, 2, 3, 4, 5]  # 暂停N期
    
    all_results = []
    
    for stop_after, pause in product(stop_after_options, pause_options):
        stats, _ = backtest_strategy(
            numbers, predictor,
            stop_after_n=stop_after,
            pause_n=pause,
            test_periods=test_periods
        )
        
        stats['stop_after'] = stop_after
        stats['pause'] = pause
        all_results.append(stats)
    
    # 转为DataFrame
    results_df = pd.DataFrame(all_results)
    
    # 按ROI排序
    results_df = results_df.sort_values('roi', ascending=False)
    
    # 显示前10名
    print("\n【TOP 10最优策略】(按ROI排序)")
    print("-" * 100)
    print(f"{'排名':<5} {'连败→止损':<12} {'暂停期数':<10} {'命中率':<10} {'ROI':<12} {'最大回撤':<12} {'触发次数':<10} {'取消次数':<10}")
    print("-" * 100)
    
    for idx, row in results_df.head(10).iterrows():
        rank = results_df.index.get_loc(idx) + 1
        print(f"{rank:<5} {row['stop_after']:<12.0f} {row['pause']:<10.0f} "
              f"{row['hit_rate']:<10.2f}% {row['roi']:<11.2f}% "
              f"{row['max_drawdown']:<12.0f}元 {row['pause_triggered']:<10.0f} "
              f"{row['pause_cancelled']:<10.0f}")
    
    # 找到最优策略
    best = results_df.iloc[0]
    
    print("\n" + "="*100)
    print("【最优策略详情】")
    print("="*100)
    print(f"\n参数配置:")
    print(f"  连续失败{best['stop_after']:.0f}期后 → 暂停{best['pause']:.0f}期")
    print(f"  暂停期间虚拟跟踪，如命中则立即恢复投注")
    
    print(f"\n性能指标:")
    print(f"  命中率: {best['hit_rate']:.2f}%")
    print(f"  总投入: {best['total_bet']:.0f}元")
    print(f"  净利润: {best['balance']:.0f}元")
    print(f"  ROI: {best['roi']:+.2f}%")
    print(f"  最大回撤: {best['max_drawdown']:.0f}元")
    print(f"  止损触发: {best['pause_triggered']:.0f}次")
    print(f"  暂停取消: {best['pause_cancelled']:.0f}次")
    
    print(f"\n与基准对比:")
    roi_improve = best['roi'] - baseline_stats['roi']
    drawdown_reduce = baseline_stats['max_drawdown'] - best['max_drawdown']
    profit_improve = best['balance'] - baseline_stats['balance']
    
    print(f"  ROI改善: {baseline_stats['roi']:.2f}% → {best['roi']:.2f}% ({roi_improve:+.2f}%)")
    print(f"  最大回撤: {baseline_stats['max_drawdown']:.0f}元 → {best['max_drawdown']:.0f}元 ({drawdown_reduce:+.0f}元)")
    print(f"  净利润: {baseline_stats['balance']:.0f}元 → {best['balance']:.0f}元 ({profit_improve:+.0f}元)")
    
    if roi_improve > 0 and drawdown_reduce > 0:
        print(f"\n✅ 找到更优策略！ROI提升{roi_improve:.2f}%，回撤减少{drawdown_reduce:.0f}元")
    elif roi_improve > 0:
        print(f"\n⚠️ ROI提升{roi_improve:.2f}%，但回撤增加{-drawdown_reduce:.0f}元")
    elif drawdown_reduce > 0:
        print(f"\n⚠️ 回撤减少{drawdown_reduce:.0f}元，但ROI下降{-roi_improve:.2f}%")
    else:
        print(f"\n❌ 未找到更优策略")
    
    # 详细回测最优策略
    print("\n" + "="*100)
    print("【最优策略详细回测】")
    print("="*100)
    
    _, detail_results = backtest_strategy(
        numbers, predictor,
        stop_after_n=int(best['stop_after']),
        pause_n=int(best['pause']),
        test_periods=test_periods
    )
    
    detail_df = pd.DataFrame(detail_results)
    
    # 保存详细结果
    detail_df.to_csv('top15_optimized_stop_loss_detail.csv', index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到: top15_optimized_stop_loss_detail.csv")
    
    # 显示关键期数
    print("\n连续失败期段分析:")
    consecutive_miss = 0
    miss_segments = []
    
    for _, row in detail_df.iterrows():
        if not row['hit'] and not row['paused']:
            consecutive_miss += 1
        else:
            if consecutive_miss > 0:
                miss_segments.append(consecutive_miss)
            consecutive_miss = 0
    
    if miss_segments:
        print(f"  最大连续失败: {max(miss_segments)}期")
        print(f"  平均连续失败: {np.mean(miss_segments):.2f}期")
        print(f"  ≥5期长连败: {len([s for s in miss_segments if s >= 5])}次")
    
    # 暂停期分析
    pause_periods = detail_df[detail_df['paused'] == True]
    print(f"\n暂停期分析:")
    print(f"  总暂停期数: {len(pause_periods)}期")
    print(f"  节省投入: {len(pause_periods) * 15:.0f}元起")
    
    return results_df, best, baseline_stats


def analyze_risk_reward(results_df, baseline_stats):
    """分析风险收益平衡"""
    print("\n" + "="*100)
    print("【风险收益平衡分析】")
    print("="*100)
    
    # 计算夏普比率的简化版本（收益/回撤比）
    results_df['reward_risk_ratio'] = results_df['roi'] / (results_df['max_drawdown'] + 1)
    
    # 按风险收益比排序
    top_risk_reward = results_df.nlargest(5, 'reward_risk_ratio')
    
    print("\n【最佳风险收益比TOP 5】")
    print("-" * 100)
    print(f"{'排名':<5} {'连败→止损':<12} {'暂停期数':<10} {'ROI':<12} {'最大回撤':<12} {'收益/回撤比':<15}")
    print("-" * 100)
    
    for idx, (_, row) in enumerate(top_risk_reward.iterrows(), 1):
        print(f"{idx:<5} {row['stop_after']:<12.0f} {row['pause']:<10.0f} "
              f"{row['roi']:<11.2f}% {row['max_drawdown']:<12.0f}元 "
              f"{row['reward_risk_ratio']:<15.4f}")
    
    baseline_ratio = baseline_stats['roi'] / (baseline_stats['max_drawdown'] + 1)
    best_ratio_row = top_risk_reward.iloc[0]
    
    print(f"\n基准策略收益/回撤比: {baseline_ratio:.4f}")
    print(f"最优策略收益/回撤比: {best_ratio_row['reward_risk_ratio']:.4f}")
    print(f"改善幅度: {(best_ratio_row['reward_risk_ratio'] - baseline_ratio) / baseline_ratio * 100:+.2f}%")


if __name__ == "__main__":
    # 200期回测
    print("\n" + "="*100)
    print("开始回测 - 测试期数: 200期")
    print("="*100)
    
    results_df, best, baseline_stats = test_all_combinations(test_periods=200)
    
    # 风险收益分析
    analyze_risk_reward(results_df, baseline_stats)
    
    # 保存所有结果
    results_df.to_csv('top15_stop_loss_all_strategies.csv', index=False, encoding='utf-8-sig')
    print("\n" + "="*100)
    print("所有策略结果已保存到: top15_stop_loss_all_strategies.csv")
    print("="*100)
