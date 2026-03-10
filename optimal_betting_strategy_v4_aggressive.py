"""
TOP15最优智能投注策略 v4.2 激进版
基于命中率规律优化的激进版本（高风险高收益）

核心改进：
1. 完全依据连败次数（连败1次是黄金投注期）
2. 激进倍投策略：连败越多，倍数越高
3. 智能暂停策略（连胜后暂停，连败1次后命中不暂停）
4. 暂停期结果同步：暂停期命中→连败0，未中→连败1
5. 移除12期命中率逻辑（分析显示效果不显著）

⚠️ 与v4.1的区别：
- v4.1稳健版：连败0次×0.8, 1次×1.5, 2-3次×0.9, 4+次×1.0
- v4.2激进版：连败0次×0.5, 1次×2.0, 2-3次×1.5, 4+次×2.0
"""

import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor

class OptimalBettingStrategyV4Aggressive:
    """最优投泣策略 v4.2 - 稳健增强版连败感知策略"""
    
    def __init__(self, base_bet=15, win_reward=47, max_multiplier=10):
        """
        初始化策略
        
        参数:
            base_bet: 基础投注额（默认15元，15个号码）
            win_reward: 命中奖励（默认47元）
            max_multiplier: 最大倍数限制
        """
        self.base_bet = base_bet
        self.win_reward = win_reward
        self.max_multiplier = max_multiplier
        
        # Fibonacci序列
        self.fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        
        # 投注状态
        self.balance = 0
        self.min_balance = 0
        self.max_drawdown = 0
        self.total_bet = 0
        self.total_win = 0
        
        # 连败追踪
        self.consecutive_losses = 0
        self.fib_index = 0
        
        # 暂停控制
        self.pause_remaining = 0
        self.last_streak_before_hit = 0  # 记录命中前的连败次数
        
        # 统计
        self.hit_count = 0
        self.miss_count = 0
        self.period_count = 0
        self.pause_periods = 0
        self.paused_hit_count = 0
    
    def get_base_multiplier(self):
        """获取Fibonacci基础倍数"""
        if self.fib_index >= len(self.fib_sequence):
            return min(self.fib_sequence[-1], self.max_multiplier)
        return min(self.fib_sequence[self.fib_index], self.max_multiplier)
    
    def calculate_multiplier(self):
        """
        计算投注倍数（稳健增强版）
        
        核心逻辑：
        1. 基础倍数 = Fibonacci[连败次数]
        2. 连败调整（稳健增强策略）：
           - 连败0次（刚命中）：× 0.5（大幅降低，命中率仅28.9%）
           - 连败1次：× 2.0 ⭐黄金投注期（命中率40%）
           - 连败2-3次：× 1.0（标准倍投）
           - 连败4次+：× 1.5（中等追击）
        """
        # 1. 获取基础倍数
        base_mult = self.get_base_multiplier()
        
        # 2. 连败次数调整（稳健增强策略）
        if self.consecutive_losses == 0:
            # 刚命中，大幅降低倍投（命中率仅28.9%）
            streak_adj = 0.5
        elif self.consecutive_losses == 1:
            # 连败1次，黄金投注期，加大投注（命中率40.0%）⭐⭐⭐
            streak_adj = 2.0
        elif self.consecutive_losses in [2, 3]:
            # 连败2-3次，标准倍投
            streak_adj = 1.0
        else:
            # 连败4次+，中等追击
            streak_adj = 1.5
        
        # 3. 最终倍数（仅基于连败调整）
        multiplier = base_mult * streak_adj
        multiplier = max(0.5, multiplier)  # 最小0.5倍
        multiplier = min(multiplier, self.max_multiplier)
        
        return round(multiplier, 1)
    
    def should_pause_after_hit(self):
        """
        判断命中后是否暂停
        
        智能暂停逻辑：
        - 连胜后命中（连败0次）：暂停1期（避开28.9%低概率期）
        - 连败1次后命中：不暂停（可能继续高概率期）
        - 其他情况：暂停1期
        """
        if self.last_streak_before_hit == 0:
            # 连胜后命中，暂停
            return True
        elif self.last_streak_before_hit == 1:
            # 连败1次后命中，不暂停
            return False
        else:
            # 其他情况，暂停
            return True
    
    def process_period(self, hit):
        """
        处理一期投注
        
        参数:
            hit: 是否命中
            
        返回:
            dict: 包含倍数、投注额、盈亏等信息
        """
        self.period_count += 1
        
        # 检查是否在暂停期
        if self.pause_remaining > 0:
            self.pause_remaining -= 1
            self.pause_periods += 1
            if hit:
                self.paused_hit_count += 1
                # 暂停期命中，重置连败为0
                self.consecutive_losses = 0
            else:
                # 暂停期未中，设置连败为1
                self.consecutive_losses = 1
            return {
                'multiplier': 0,
                'bet': 0,
                'profit': 0,
                'paused': True
            }
        
        # 计算倍数
        multiplier = self.calculate_multiplier()
        bet = self.base_bet * multiplier
        
        # 更新投注总额
        self.total_bet += bet
        
        if hit:
            # 命中
            self.hit_count += 1
            win = self.win_reward * multiplier
            self.total_win += win
            profit = win - bet
            self.balance += profit
            
            # 记录命中前的连败次数
            self.last_streak_before_hit = self.consecutive_losses
            
            # 重置连败
            self.consecutive_losses = 0
            self.fib_index = 0
            
            # 判断是否暂停
            if self.should_pause_after_hit():
                self.pause_remaining = 1
        else:
            # 未中
            self.miss_count += 1
            profit = -bet
            self.balance += profit
            
            # 增加连败
            self.consecutive_losses += 1
            self.fib_index += 1
        
        # 更新最大回撤
        if self.balance < self.min_balance:
            self.min_balance = self.balance
            self.max_drawdown = abs(self.min_balance)
        
        return {
            'multiplier': multiplier,
            'bet': bet,
            'profit': profit,
            'paused': False
        }
    
    def get_stats(self):
        """获取统计数据"""
        actual_periods = self.period_count - self.pause_periods
        roi = (self.balance / self.total_bet * 100) if self.total_bet > 0 else 0
        hit_rate = (self.hit_count / actual_periods * 100) if actual_periods > 0 else 0
        
        return {
            'total_periods': self.period_count,
            'actual_periods': actual_periods,
            'pause_periods': self.pause_periods,
            'hits': self.hit_count,
            'misses': self.miss_count,
            'hit_rate': hit_rate,
            'total_bet': self.total_bet,
            'total_win': self.total_win,
            'balance': self.balance,
            'roi': roi,
            'max_drawdown': self.max_drawdown,
            'paused_hit_count': self.paused_hit_count
        }


def validate_strategy_v4_aggressive(test_periods=300):
    """验证v4.2稳健增强版策略效果，与v4.1对比"""
    print("="*70)
    print(" "*15 + "策略对比：v4.2稳健增强 vs v4.1稳健")
    print("="*70)
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    
    if len(df) < 50:
        print("数据不足50期，无法进行分析")
        return
    
    print(f"\n数据范围: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
    
    # 确定测试范围
    test_periods = min(test_periods, len(df) - 50)
    start_idx = len(df) - test_periods
    
    # 初始化预测器和策略
    predictor = PreciseTop15Predictor()
    strategy_aggressive = OptimalBettingStrategyV4Aggressive()
    
    # 从v4.1导入稳健版策略进行对比
    from optimal_betting_strategy_v4 import OptimalBettingStrategyV4
    strategy_stable = OptimalBettingStrategyV4()
    
    # 回测
    results_aggressive = []
    results_stable = []
    
    for i in range(start_idx, len(df)):
        # 预测
        train_data = df.iloc[:i]['number'].values
        predictions = predictor.predict(train_data)
        actual = df.iloc[i]['number']
        hit = actual in predictions
        
        # 更新预测器
        predictor.update_performance(predictions, actual)
        
        # 两个策略同时处理
        result_agg = strategy_aggressive.process_period(hit)
        result_stb = strategy_stable.process_period(hit)
        
        results_aggressive.append({
            'period': i - start_idx + 1,
            'date': df.iloc[i]['date'],
            'actual': actual,
            'hit': hit,
            **result_agg,
            'balance': strategy_aggressive.balance
        })
        
        results_stable.append({
            'period': i - start_idx + 1,
            'date': df.iloc[i]['date'],
            'actual': actual,
            'hit': hit,
            **result_stb,
            'balance': strategy_stable.balance
        })
    
    # 获取统计
    stats_agg = strategy_aggressive.get_stats()
    
    # 手动构建v4.1统计（因为它没有get_stats方法）
    actual_periods_stb = strategy_stable.period_count - strategy_stable.pause_periods
    roi_stb = (strategy_stable.balance / strategy_stable.total_bet * 100) if strategy_stable.total_bet > 0 else 0
    hit_rate_stb = (strategy_stable.hit_count / actual_periods_stb * 100) if actual_periods_stb > 0 else 0
    
    stats_stb = {
        'total_periods': strategy_stable.period_count,
        'actual_periods': actual_periods_stb,
        'pause_periods': strategy_stable.pause_periods,
        'hits': strategy_stable.hit_count,
        'misses': strategy_stable.miss_count,
        'hit_rate': hit_rate_stb,
        'total_bet': strategy_stable.total_bet,
        'total_win': strategy_stable.total_win,
        'balance': strategy_stable.balance,
        'roi': roi_stb,
        'max_drawdown': strategy_stable.max_drawdown,
        'paused_hit_count': strategy_stable.paused_hit_count
    }
    
    print("\n" + "="*70)
    print("【策略对比结果】")
    print("="*70)
    
    print(f"\nv4.2稳健增强版：")
    print(f"  总期数: {stats_agg['total_periods']}")
    print(f"  投注期数: {stats_agg['actual_periods']}")
    print(f"  暂停期数: {stats_agg['pause_periods']}")
    print(f"  命中次数: {stats_agg['hits']} ({stats_agg['hit_rate']:.1f}%)")
    print(f"  总投注: {stats_agg['total_bet']:.0f}元")
    print(f"  净利润: {stats_agg['balance']:+.0f}元")
    print(f"  ROI: {stats_agg['roi']:.2f}%")
    print(f"  最大回撤: {stats_agg['max_drawdown']:.0f}元")
    
    print(f"\nv4.1稳健版（低风险稳定）：")
    print(f"  总投注: {stats_stb['total_bet']:.0f}元")
    print(f"  净利润: {stats_stb['balance']:+.0f}元")
    print(f"  ROI: {stats_stb['roi']:.2f}%")
    print(f"  最大回撤: {stats_stb['max_drawdown']:.0f}元")
    
    # 对比差异
    profit_delta = stats_agg['balance'] - stats_stb['balance']
    roi_delta = stats_agg['roi'] - stats_stb['roi']
    drawdown_delta = stats_agg['max_drawdown'] - stats_stb['max_drawdown']
    
    print("\n" + "="*70)
    print("【改进对比】")
    print("="*70)
    print(f"净利润: {profit_delta:+.0f}元 ({profit_delta/stats_stb['balance']*100:+.1f}%)")
    print(f"ROI: {roi_delta:+.2f}个百分点")
    print(f"回撤: {drawdown_delta:+.0f}元 ({drawdown_delta/stats_stb['max_drawdown']*100:+.1f}%)")
    
    # 风险收益比
    risk_reward_agg = stats_agg['balance'] / stats_agg['max_drawdown'] if stats_agg['max_drawdown'] > 0 else 0
    risk_reward_stb = stats_stb['balance'] / stats_stb['max_drawdown'] if stats_stb['max_drawdown'] > 0 else 0
    
    print(f"\n风险收益比:")
    print(f"  v4.2稳健增强: {risk_reward_agg:.2f}")
    print(f"  v4.1稳健版: {risk_reward_stb:.2f}")
    
    # 显示前20期详情（稳健增强版）
    print("\n" + "="*70)
    print("【前20期详情（v4.2稳健增强版）】")
    print("="*70)
    print(f"{'期号':<8}{'日期':<12}{'倍数':<8}{'投注':<8}{'命中':<6}{'盈亏':<10}{'累计':<12}{'暂停':<6}")
    print("-"*80)
    
    for r in results_aggressive[:20]:
        period = r['period']
        date = r['date']
        multiplier = r['multiplier']
        bet = r['bet']
        hit_mark = '✅' if r['hit'] else '❌'
        profit = r['profit']
        balance = r['balance']
        paused = 'SKIP' if r['paused'] else ''
        
        print(f"{period:<8}{date:<12}{multiplier:<8.2f}{bet:<8.0f}{hit_mark:<6}"
              f"{profit:+10.0f}  {balance:+12.0f}  {paused:<6}")
    
    # 保存详细结果
    results_df = pd.DataFrame(results_aggressive)
    output_file = 'optimal_betting_v4.2_stable_enhanced_300periods.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 详细结果已保存到: {output_file}")
    
    print("\n" + "="*70)
    print("验证完成！")
    print("="*70)
    print("\n⚠️  注意事项：")
    print("  1. v4.2稳健增强版降低了极端情况的倍数")
    print("  2. 建议准备500-800元资金应对回撤")
    print("  3. 适合更多投资者，平衡风险和收益")
    print("  4. 如需更稳健，请使用v4.1版本")


if __name__ == '__main__':
    validate_strategy_v4_aggressive(test_periods=300)
