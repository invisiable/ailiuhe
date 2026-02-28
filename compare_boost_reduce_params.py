"""
对比不同boost和reduce参数的效果
测试：(1.2, 0.8) vs (1.5, 0.5)
"""

import pandas as pd
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
        """处理一期投注（修复后的正确时序）"""
        hit = actual in prediction
        
        # 获取基础倍数
        base_mult = self.get_base_multiplier()
        
        # 根据最近命中率计算动态倍数（使用投注前的历史数据）
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
        
        # 更新余额和统计
        bet, win, profit = self.update_balance(hit, multiplier)
        
        # 添加结果到历史（在投注和结算之后）
        self.recent_results.append(1 if hit else 0)
        if len(self.recent_results) > self.lookback:
            self.recent_results.pop(0)
        
        return {
            'multiplier': multiplier,
            'bet': bet,
            'win': win,
            'profit': profit,
            'hit': hit,
            'recent_rate': self.get_recent_rate()
        }


def backtest_strategy(boost_mult, reduce_mult):
    """回测指定参数的策略"""
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv')
    numbers = df['number'].tolist()
    
    total_periods = len(numbers)
    test_periods = 300
    start = total_periods - test_periods
    
    # 初始化预测器和策略
    predictor = PreciseTop15Predictor()
    strategy = SmartDynamicStrategy(
        base_bet=15,
        win_reward=47,
        lookback=12,
        good_thresh=0.35,
        bad_thresh=0.20,
        boost_mult=boost_mult,
        reduce_mult=reduce_mult,
        max_multiplier=10
    )
    
    # 回测
    hits = 0
    hit_10x_count = 0
    
    for i in range(start, total_periods):
        history = numbers[:i]
        actual_number = numbers[i]
        
        if len(history) >= 30:
            predicted_top15 = predictor.predict(history)
        else:
            predicted_top15 = list(range(1, 16))
        
        result = strategy.process_period(predicted_top15, actual_number)
        predictor.update_performance(predicted_top15, actual_number)
        
        if result['multiplier'] >= 10:
            hit_10x_count += 1
        
        if result['hit']:
            hits += 1
    
    # 返回结果
    return {
        'boost_mult': boost_mult,
        'reduce_mult': reduce_mult,
        'hits': hits,
        'hit_rate': hits / test_periods * 100,
        'total_bet': strategy.total_bet,
        'total_win': strategy.total_win,
        'profit': strategy.balance,
        'roi': (strategy.balance / strategy.total_bet * 100) if strategy.total_bet > 0 else 0,
        'drawdown': strategy.max_drawdown,
        'hit_10x': hit_10x_count
    }


if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"智能动态倍投策略 - boost和reduce参数对比测试")
    print(f"{'='*80}\n")
    
    # 测试两组参数
    configs = [
        (1.2, 0.8, "保守组合（原参数）"),
        (1.5, 0.5, "激进组合（新参数）")
    ]
    
    results = []
    for boost, reduce, desc in configs:
        print(f"测试 {desc}: boost={boost}x, reduce={reduce}x")
        result = backtest_strategy(boost, reduce)
        result['desc'] = desc
        results.append(result)
        print(f"  完成：ROI {result['roi']:.2f}%, 利润 {result['profit']:+.0f}元\n")
    
    # 详细对比
    print(f"\n{'='*80}")
    print(f"详细对比分析")
    print(f"{'='*80}\n")
    
    r1, r2 = results[0], results[1]
    
    print(f"【策略参数】")
    print(f"  保守组合: boost×{r1['boost_mult']}, reduce×{r1['reduce_mult']}")
    print(f"  激进组合: boost×{r2['boost_mult']}, reduce×{r2['reduce_mult']}\n")
    
    print(f"【性能对比】")
    print(f"{'指标':<15} {'保守组合':<20} {'激进组合':<20} {'差异':<20}")
    print(f"{'-'*80}")
    
    # 命中次数
    print(f"{'命中次数':<15} {r1['hits']}/300 ({r1['hit_rate']:.2f}%){'':<5} "
          f"{r2['hits']}/300 ({r2['hit_rate']:.2f}%){'':<5} "
          f"{'+' if r2['hits'] >= r1['hits'] else ''}{r2['hits'] - r1['hits']}次")
    
    # 总投入
    diff_bet = r2['total_bet'] - r1['total_bet']
    print(f"{'总投入':<15} {r1['total_bet']:.0f}元{'':<15} "
          f"{r2['total_bet']:.0f}元{'':<15} "
          f"{diff_bet:+.0f}元 ({diff_bet/r1['total_bet']*100:+.1f}%)")
    
    # 总收益
    diff_win = r2['total_win'] - r1['total_win']
    print(f"{'总收益':<15} {r1['total_win']:.0f}元{'':<15} "
          f"{r2['total_win']:.0f}元{'':<15} "
          f"{diff_win:+.0f}元 ({diff_win/r1['total_win']*100:+.1f}%)")
    
    # 净利润
    diff_profit = r2['profit'] - r1['profit']
    print(f"{'净利润':<15} {r1['profit']:+.0f}元{'':<14} "
          f"{r2['profit']:+.0f}元{'':<14} "
          f"{diff_profit:+.0f}元 ({diff_profit/abs(r1['profit'])*100:+.1f}%) {'✓' if diff_profit > 0 else '✗'}")
    
    # ROI
    diff_roi = r2['roi'] - r1['roi']
    print(f"{'ROI':<15} {r1['roi']:.2f}%{'':<16} "
          f"{r2['roi']:.2f}%{'':<16} "
          f"{diff_roi:+.2f}% ({diff_roi/r1['roi']*100:+.1f}%) {'✓' if diff_roi > 0 else '✗'}")
    
    # 最大回撤
    diff_dd = r2['drawdown'] - r1['drawdown']
    print(f"{'最大回撤':<15} {r1['drawdown']:.0f}元{'':<15} "
          f"{r2['drawdown']:.0f}元{'':<15} "
          f"{diff_dd:+.0f}元 ({diff_dd/r1['drawdown']*100:+.1f}%) {'✓' if diff_dd < 0 else '✗'}")
    
    # 触及10x
    diff_10x = r2['hit_10x'] - r1['hit_10x']
    print(f"{'触及10x次数':<15} {r1['hit_10x']}次{'':<17} "
          f"{r2['hit_10x']}次{'':<17} "
          f"{diff_10x:+}次")
    
    # 风险收益比
    risk_return_1 = r1['profit'] / r1['drawdown'] if r1['drawdown'] > 0 else 0
    risk_return_2 = r2['profit'] / r2['drawdown'] if r2['drawdown'] > 0 else 0
    print(f"{'风险收益比':<15} {risk_return_1:.2f}{'':<17} "
          f"{risk_return_2:.2f}{'':<17} "
          f"{risk_return_2 - risk_return_1:+.2f} {'✓' if risk_return_2 > risk_return_1 else '✗'}")
    
    print(f"\n{'='*80}")
    print(f"结论分析")
    print(f"{'='*80}\n")
    
    if r2['roi'] > r1['roi'] and r2['drawdown'] < r1['drawdown']:
        print(f"✅ 激进组合（boost×{r2['boost_mult']}, reduce×{r2['reduce_mult']}）全面优于保守组合！")
        print(f"   • ROI提升 {diff_roi:.2f}% ({diff_roi/r1['roi']*100:.1f}%)")
        print(f"   • 利润增加 {diff_profit:.0f}元 ({diff_profit/abs(r1['profit'])*100:.1f}%)")
        print(f"   • 回撤降低 {abs(diff_dd):.0f}元 ({abs(diff_dd)/r1['drawdown']*100:.1f}%)")
        print(f"   • 风险收益比提升 {risk_return_2 - risk_return_1:.2f}")
        print(f"\n💡 建议：采用激进组合参数作为新的最优策略！")
    elif r2['roi'] > r1['roi']:
        print(f"✓ 激进组合ROI更高，但回撤也更大")
        print(f"   • ROI提升 {diff_roi:.2f}%，但回撤增加 {diff_dd:.0f}元")
        print(f"   • 风险收益比: {risk_return_2:.2f} vs {risk_return_1:.2f}")
        if risk_return_2 > risk_return_1:
            print(f"\n💡 建议：激进组合风险收益比更优，可以采用")
        else:
            print(f"\n⚠️  建议：视风险偏好选择，保守投资者建议保持原参数")
    else:
        print(f"✗ 保守组合表现更好")
        print(f"\n💡 建议：保持原参数组合（boost×{r1['boost_mult']}, reduce×{r1['reduce_mult']}）")
