"""
TOP15最优智能投注策略 v4.1
基于命中率规律优化的新版本（纯连败感知）

核心改进：
1. 完全依据连败次数（连败1次是黄金投注期）
2. 智能暂停策略（连胜后暂停，连败1次后命中不暂停）
3. 移除12期命中率逻辑（分析显示效果不显著）
"""

import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor

class OptimalBettingStrategyV4:
    """最优投注策略 v4.1 - 纯连败感知策略"""
    
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
        
        # 历史记录
        self.recent_results = []  # 最近12期结果
        self.lookback = 12
        
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
    
    def get_recent_hit_rate(self):
        """获取最近12期命中率"""
        if len(self.recent_results) == 0:
            return 0.33
        return sum(self.recent_results) / len(self.recent_results)
    
    def calculate_multiplier(self):
        """
        计算投注倍数
        
        核心逻辑（纯连败感知）：
        1. 基础倍数 = Fibonacci[连败次数]
        2. 连败调整（唯一策略）：
           - 连败0次（刚命中）：× 0.8
           - 连败1次：× 1.5 ⭐黄金投注期
           - 连败2-3次：× 0.9
           - 连败4次+：× 1.0
        
        注：已移除12期命中率微调（分析显示高命中率期反而表现差）
        """
        # 1. 获取基础倍数
        base_mult = self.get_base_multiplier()
        
        # 2. 连败次数调整（唯一策略）
        if self.consecutive_losses == 0:
            # 刚命中，下期命中率低（28.9%）
            streak_adj = 0.8
        elif self.consecutive_losses == 1:
            # 连败1次，黄金投注期（命中率40.0%）⭐⭐⭐
            streak_adj = 1.5
        elif self.consecutive_losses in [2, 3]:
            # 连败2-3次，命中率回落（29-32%）
            streak_adj = 0.9
        else:
            # 连败4次+，按正常Fibonacci
            streak_adj = 1.0
        
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
            # 连败1次后命中，不暂停（可能继续）
            return False
        else:
            # 其他情况，暂停
            return True
    
    def process_period(self, hit):
        """
        处理一期投注
        
        返回: dict包含本期详情
        """
        self.period_count += 1
        
        # 检查是否在暂停期
        if self.pause_remaining > 0:
            self.pause_remaining -= 1
            self.pause_periods += 1
            if hit:
                self.paused_hit_count += 1
            
            return {
                'paused': True,
                'pause_remaining': self.pause_remaining,
                'multiplier': 0,
                'bet': 0,
                'profit': 0,
                'balance': self.balance,
                'consecutive_losses': self.consecutive_losses,
                'recent_rate': self.get_recent_hit_rate(),
                'hit': hit
            }
        
        # 记录投注前的连败次数
        betting_consecutive_losses = self.consecutive_losses
        
        # 计算倍数和投注
        multiplier = self.calculate_multiplier()
        bet = self.base_bet * multiplier
        self.total_bet += bet
        
        # 计算盈亏
        if hit:
            win = self.win_reward * multiplier
            self.total_win += win
            profit = win - bet
            self.balance += profit
            self.hit_count += 1
            
            # 记录命中前的连败次数（用于智能暂停）
            self.last_streak_before_hit = self.consecutive_losses
           
            # 重置状态
            self.consecutive_losses = 0
            self.fib_index = 0
            
            # 判断是否暂停
            if self.should_pause_after_hit():
                self.pause_remaining = 1
        else:
            profit = -bet
            self.balance += profit
            self.miss_count += 1
            self.consecutive_losses += 1
            self.fib_index += 1
        
        # 更新回撤
        if self.balance < self.min_balance:
            self.min_balance = self.balance
            self.max_drawdown = abs(self.min_balance)
        
        # 更新历史记录
        self.recent_results.append(1 if hit else 0)
        if len(self.recent_results) > self.lookback:
            self.recent_results.pop(0)
        
        return {
            'paused': False,
            'pause_remaining': self.pause_remaining,
            'multiplier': multiplier,
            'bet': bet,
            'profit': profit,
            'balance': self.balance,
            'consecutive_losses': betting_consecutive_losses,  # 投注时的连败
            'recent_rate': self.get_recent_hit_rate(),
            'hit': hit,
            'last_streak': self.last_streak_before_hit if hit else None
        }
    
    def get_statistics(self):
        """获取统计数据"""
        actual_bet_periods = self.period_count - self.pause_periods
        hit_rate = self.hit_count / actual_bet_periods if actual_bet_periods > 0 else 0
        roi = (self.balance / self.total_bet * 100) if self.total_bet > 0 else 0
        
        return {
            'total_periods': self.period_count,
            'bet_periods': actual_bet_periods,
            'pause_periods': self.pause_periods,
            'paused_hit_count': self.paused_hit_count,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'total_bet': self.total_bet,
            'total_win': self.total_win,
            'balance': self.balance,
            'max_drawdown': self.max_drawdown,
            'roi': roi
        }


def validate_strategy_v4(test_periods=300):
    """验证v4.1策略效果（纯连败感知），与v3.2对比"""
    print("="*70)
    print(" "*15 + "策略对比：v4.1 vs v3.2")
    print("="*70)
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])
    
    # 初始化预测器
    predictor = PreciseTop15Predictor()
    
    all_numbers = df['number'].tolist()
    all_dates = df['date'].dt.strftime('%Y/%m/%d').tolist()
    
    # 初始化两个策略
    strategy_v4 = OptimalBettingStrategyV4()
    
    # v3.2策略（基准）
    class StrategyV32:
        def __init__(self):
            self.fib_index = 0
            self.recent_results = []
            self.balance = 0
            self.min_balance = 0
            self.max_drawdown = 0
            self.total_bet = 0
            self.fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
            
        def process(self, hit):
            base_mult = min(self.fib_sequence[self.fib_index] if self.fib_index < len(self.fib_sequence) else self.fib_sequence[-1], 10)
            
            if len(self.recent_results) >= 12:
                rate = sum(self.recent_results) / len(self.recent_results)
                if rate >= 0.35:
                    mult = min(base_mult * 1.5, 10)
                elif rate <= 0.20:
                    mult = max(base_mult * 1.0, 1)
                else:
                    mult = base_mult
            else:
                mult = base_mult
            
            bet = 15 * mult
            self.total_bet += bet
            
            if hit:
                profit = 47 * mult - bet
                self.balance += profit
                self.fib_index = 0
            else:
                profit = -bet
                self.balance += profit
                self.fib_index += 1
            
            if self.balance < self.min_balance:
                self.min_balance = self.balance
                self.max_drawdown = abs(self.min_balance)
            
            self.recent_results.append(1 if hit else 0)
            if len(self.recent_results) > 12:
                self.recent_results.pop(0)
    
    strategy_v32 = StrategyV32()
    
    # 回测
    lookback = 50
    results_v4 = []
    
    print(f"\n正在回测最近 {test_periods} 期...")
    print(f"数据范围: {all_dates[-test_periods-lookback]} ~ {all_dates[-1]}\n")
    
    for i in range(len(all_numbers) - test_periods, len(all_numbers)):
        train_data = all_numbers[:i]
        actual = all_numbers[i]
        date = all_dates[i]
        
        # 预测
        predictions = predictor.predict(np.array(train_data))
        hit = actual in predictions
        predictor.update_performance(predictions, actual)
        
        # v4.0策略
        result_v4 = strategy_v4.process_period(hit)
        result_v4['date'] = date
        result_v4['actual'] = actual
        result_v4['predictions'] = predictions[:5]
        results_v4.append(result_v4)
        
        # v3.2策略
        strategy_v32.process(hit)
    
    # 统计对比
    stats_v4 = strategy_v4.get_statistics()
    
    print("="*70)
    print("【策略对比结果】")
    print("="*70)
    
    print(f"\nv4.1策略（纯连败感知+智能暂停）：")
    print(f"  总期数: {stats_v4['total_periods']}")
    print(f"  投注期数: {stats_v4['bet_periods']}")
    print(f"  暂停期数: {stats_v4['pause_periods']}")
    print(f"  命中次数: {stats_v4['hit_count']} ({stats_v4['hit_rate']:.1%})")
    print(f"  总投注: {stats_v4['total_bet']:.0f}元")
    print(f"  净利润: {stats_v4['balance']:+.0f}元")
    print(f"  ROI: {stats_v4['roi']:.2%}")
    print(f"  最大回撤: {stats_v4['max_drawdown']:.0f}元")
    print(f"  风险收益比: {stats_v4['balance']/stats_v4['max_drawdown']:.2f}" if stats_v4['max_drawdown'] > 0 else "  风险收益比: N/A")
    
    print(f"\nv3.2策略（12期命中率阈值）：")
    print(f"  总投注: {strategy_v32.total_bet:.0f}元")
    print(f"  净利润: {strategy_v32.balance:+.0f}元")
    roi_v32 = (strategy_v32.balance / strategy_v32.total_bet * 100) if strategy_v32.total_bet > 0 else 0
    print(f"  ROI: {roi_v32:.2%}")
    print(f"  最大回撤: {strategy_v32.max_drawdown:.0f}元")
    print(f"  风险收益比: {strategy_v32.balance/strategy_v32.max_drawdown:.2f}" if strategy_v32.max_drawdown > 0 else "  风险收益比: N/A")
    
    print("\n" + "="*70)
    print("【改进对比】")
    print("="*70)
    
    profit_diff = stats_v4['balance'] - strategy_v32.balance
    roi_diff = stats_v4['roi'] - roi_v32
    drawdown_diff = stats_v4['max_drawdown'] - strategy_v32.max_drawdown
    
    print(f"净利润: {profit_diff:+.0f}元 ({profit_diff/abs(strategy_v32.balance)*100:+.1f}%)" if strategy_v32.balance != 0 else f"净利润: {profit_diff:+.0f}元")
    print(f"ROI: {roi_diff:+.2f}个百分点")
    print(f"回撤: {drawdown_diff:+.0f}元 ({drawdown_diff/strategy_v32.max_drawdown*100:+.1f}%)" if strategy_v32.max_drawdown > 0 else f"回撤: {drawdown_diff:+.0f}元")
    
    # 显示前20期和后20期详情
    print("\n" + "="*70)
    print("【前20期详情（v4.1策略）】")
    print("="*70)
    print(f"{'日期':<12}{'开奖':<6}{'预测':<18}{'连败':<6}{'倍数':<8}{'投注':<8}{'命中':<6}{'盈亏':<10}{'累计':<10}{'暂停':<6}")
    print("-"*100)
    
    for r in results_v4[:20]:
        pred_str = str(r['predictions'])[:15] + "..."
        pause_mark = "SKIP" if r['paused'] else ""
        hit_mark = "✅" if r['hit'] else "❌"
        
        print(f"{r['date']:<12}{r['actual']:<6}{pred_str:<18}{r['consecutive_losses']:<6}"
              f"{r['multiplier']:<8.1f}{r['bet']:<8.0f}{hit_mark:<6}"
              f"{r['profit']:+10.0f}{r['balance']:+10.0f}{pause_mark:<6}")
    
    print("\n" + "."*70)
    print("\n【后20期详情（v4.1策略）】")
    print("="*70)
    print(f"{'日期':<12}{'开奖':<6}{'预测':<18}{'连败':<6}{'倍数':<8}{'投注':<8}{'命中':<6}{'盈亏':<10}{'累计':<10}{'暂停':<6}")
    print("-"*100)
    
    for r in results_v4[-20:]:
        pred_str = str(r['predictions'])[:15] + "..."
        pause_mark = "SKIP" if r['paused'] else ""
        hit_mark = "✅" if r['hit'] else "❌"
        
        print(f"{r['date']:<12}{r['actual']:<6}{pred_str:<18}{r['consecutive_losses']:<6}"
              f"{r['multiplier']:<8.1f}{r['bet']:<8.0f}{hit_mark:<6}"
              f"{r['profit']:+10.0f}{r['balance']:+10.0f}{pause_mark:<6}")
    
    print("\n" + "="*70)
    print("验证完成！")
    print("="*70)
    
    # 保存详细结果
    results_df = pd.DataFrame(results_v4)
    output_file = 'optimal_betting_v4.1_300periods.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 详细结果已保存到: {output_file}")


if __name__ == '__main__':
    validate_strategy_v4(test_periods=300)
