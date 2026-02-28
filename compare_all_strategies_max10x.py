"""
精准TOP15投注优化策略全面对比（含10倍限制）
测试4种优化策略 vs 基准策略（10倍限制）
"""

import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor

class BettingStrategyBase:
    """投注策略基类（含10倍限制）"""
    
    def __init__(self, predictor, base_bet=15, win_reward=47, max_multiplier=10):
        self.predictor = predictor
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


class BaselineStrategy(BettingStrategyBase):
    """基准策略：Fibonacci + 10倍限制"""
    
    def process_period(self, prediction, actual):
        multiplier = self.get_base_multiplier()
        hit = actual in prediction
        bet, win, profit = self.update_balance(hit, multiplier)
        return {'multiplier': multiplier, 'bet': bet, 'win': win, 'profit': profit, 'hit': hit}


class StopLossStrategy(BettingStrategyBase):
    """止损暂停策略"""
    
    def __init__(self, predictor, stop_after_n=5, pause_n=3, **kwargs):
        super().__init__(predictor, **kwargs)
        self.stop_after_n = stop_after_n
        self.pause_n = pause_n
        self.is_paused = False
        self.pause_remaining = 0
    
    def process_period(self, prediction, actual):
        hit = actual in prediction
        
        if self.is_paused:
            self.pause_remaining -= 1
            if hit:
                self.is_paused = False
                self.pause_remaining = 0
                self.consecutive_miss = 0
                self.fib_index = 0
            elif self.pause_remaining <= 0:
                self.is_paused = False
                self.consecutive_miss = 0
                self.fib_index = 0
            return {'multiplier': 0, 'bet': 0, 'win': 0, 'profit': 0, 'hit': hit, 'paused': True}
        
        multiplier = self.get_base_multiplier()
        bet, win, profit = self.update_balance(hit, multiplier)
        
        if not hit and self.consecutive_miss >= self.stop_after_n:
            self.is_paused = True
            self.pause_remaining = self.pause_n
        
        return {'multiplier': multiplier, 'bet': bet, 'win': win, 'profit': profit, 'hit': hit, 'paused': False}


class DowngradeStrategy(BettingStrategyBase):
    """降档倍投策略"""
    
    def __init__(self, predictor, trigger_after=5, downgrade_to=0, duration=3, **kwargs):
        super().__init__(predictor, **kwargs)
        self.trigger_after = trigger_after
        self.downgrade_to = downgrade_to
        self.duration = duration
        self.is_downgraded = False
        self.downgrade_remaining = 0
    
    def process_period(self, prediction, actual):
        hit = actual in prediction
        
        if self.is_downgraded:
            self.downgrade_remaining -= 1
            if self.downgrade_remaining <= 0:
                self.is_downgraded = False
            multiplier = self.fib_sequence[self.downgrade_to]
        else:
            multiplier = self.get_base_multiplier()
        
        bet, win, profit = self.update_balance(hit, multiplier)
        
        if not hit and self.consecutive_miss >= self.trigger_after and not self.is_downgraded:
            self.is_downgraded = True
            self.downgrade_remaining = self.duration
            self.fib_index = self.downgrade_to
        
        return {'multiplier': multiplier, 'bet': bet, 'win': win, 'profit': profit, 'hit': hit}


class DynamicStrategy(BettingStrategyBase):
    """智能动态倍投策略"""
    
    def __init__(self, predictor, lookback=12, good_thresh=0.35, bad_thresh=0.20, 
                 boost_mult=1.2, reduce_mult=0.8, **kwargs):
        super().__init__(predictor, **kwargs)
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
        return {'multiplier': multiplier, 'bet': bet, 'win': win, 'profit': profit, 'hit': hit}


class BoostOnlyStrategy(BettingStrategyBase):
    """纯增强倍投策略"""
    
    def __init__(self, predictor, lookback=12, boost_thresh=0.38, boost_mult=1.10, **kwargs):
        super().__init__(predictor, **kwargs)
        self.lookback = lookback
        self.boost_thresh = boost_thresh
        self.boost_mult = boost_mult
        self.recent_results = []
    
    def get_recent_rate(self):
        if len(self.recent_results) == 0:
            return 0.33
        return sum(self.recent_results) / len(self.recent_results)
    
    def process_period(self, prediction, actual):
        hit = actual in prediction
        
        self.recent_results.append(1 if hit else 0)
        if len(self.recent_results) > self.lookback:
            self.recent_results.pop(0)
        
        base_mult = self.get_base_multiplier()
        
        if len(self.recent_results) >= self.lookback:
            rate = self.get_recent_rate()
            if rate >= self.boost_thresh:
                multiplier = min(base_mult * self.boost_mult, self.max_multiplier)
            else:
                multiplier = base_mult
        else:
            multiplier = base_mult
        
        bet, win, profit = self.update_balance(hit, multiplier)
        return {'multiplier': multiplier, 'bet': bet, 'win': win, 'profit': profit, 'hit': hit}


def backtest_all_strategies(test_periods=300):
    """回测所有策略"""
    print("="*100)
    print(f"精准TOP15投注优化策略全面对比（10倍限制） - {test_periods}期回测")
    print("="*100)
    
    df = pd.read_csv('data/lucky_numbers.csv')
    numbers = df['number'].tolist()
    
    total = len(numbers)
    start = total - test_periods
    
    predictor = PreciseTop15Predictor()
    
    # 定义所有策略
    strategies = {
        '基准策略（10倍限制Fib）': BaselineStrategy(predictor),
        '止损暂停（7期暂停2期）': StopLossStrategy(predictor, stop_after_n=7, pause_n=2),
        '降档倍投（7期降Fib[2]持续2期）': DowngradeStrategy(predictor, trigger_after=7, downgrade_to=2, duration=2),
        '智能动态（回看12期）': DynamicStrategy(predictor, lookback=12, good_thresh=0.35, bad_thresh=0.20, 
                                          boost_mult=1.2, reduce_mult=0.8),
        '纯增强（回看12期>0.38增强1.1倍）': BoostOnlyStrategy(predictor, lookback=12, boost_thresh=0.38, boost_mult=1.10)
    }
    
    results = {name: [] for name in strategies.keys()}
    
    print(f"\n开始回测...")
    for i in range(start, total):
        history = numbers[:i]
        actual = numbers[i]
        predictions = predictor.predict(history)
        
        for name, strategy in strategies.items():
            result = strategy.process_period(predictions, actual)
            results[name].append(result)
        
        if (i - start + 1) % 50 == 0:
            print(f"已完成: {i - start + 1}/{test_periods}期")
    
    print("\n回测完成！")
    print("="*100)
    
    # 统计汇总
    summary = []
    
    for name, strategy in strategies.items():
        hit_count = sum([1 for r in results[name] if r['hit']])
        hit_rate = hit_count / test_periods * 100
        roi = (strategy.balance / strategy.total_bet * 100) if strategy.total_bet > 0 else 0
        
        summary.append({
            '策略': name,
            '命中次数': hit_count,
            '命中率': f"{hit_rate:.2f}%",
            '总投入': strategy.total_bet,
            '总收益': strategy.total_win,
            '净利润': strategy.balance,
            'ROI': roi,
            '最大回撤': strategy.max_drawdown
        })
    
    summary_df = pd.DataFrame(summary)
    
    # 显示结果
    print("\n【策略对比汇总】")
    print("-" * 100)
    print(summary_df.to_string(index=False))
    
    # 找出最优策略
    print("\n" + "="*100)
    print("【最优策略分析】")
    print("="*100)
    
    baseline = summary[0]
    print(f"\n基准策略表现：")
    print(f"  ROI: {baseline['ROI']:.2f}%")
    print(f"  净利润: {baseline['净利润']:.0f}元")
    print(f"  最大回撤: {baseline['最大回撤']:.0f}元")
    
    # 按ROI排序
    summary_df_sorted = summary_df.copy()
    summary_df_sorted['ROI_num'] = summary_df_sorted['ROI']
    summary_df_sorted = summary_df_sorted.sort_values('ROI_num', ascending=False)
    
    print(f"\n【ROI排名】")
    print("-" * 80)
    for idx, row in summary_df_sorted.iterrows():
        roi_diff = row['ROI_num'] - baseline['ROI']
        marker = "✅" if roi_diff > 0 else ("⚖️" if roi_diff == 0 else "❌")
        print(f"{marker} {row['策略']:<40} ROI: {row['ROI_num']:>7.2f}% ({roi_diff:+.2f}%)")
    
    # 按回撤排序
    summary_df_drawdown = summary_df.copy()
    summary_df_drawdown = summary_df_drawdown.sort_values('最大回撤', ascending=True)
    
    print(f"\n【回撤排名】（越低越好）")
    print("-" * 80)
    for idx, row in summary_df_drawdown.iterrows():
        drawdown_diff = row['最大回撤'] - baseline['最大回撤']
        marker = "✅" if drawdown_diff < 0 else ("⚖️" if drawdown_diff == 0 else "❌")
        print(f"{marker} {row['策略']:<40} 回撤: {row['最大回撤']:>7.0f}元 ({drawdown_diff:+.0f}元)")
    
    # 寻找双优策略
    print(f"\n【综合评价】")
    print("-" * 80)
    
    double_better = []
    for i, row in enumerate(summary):
        if i == 0:
            continue  # 跳过基准
        
        roi_improve = row['ROI'] - baseline['ROI']
        drawdown_reduce = baseline['最大回撤'] - row['最大回撤']
        
        if roi_improve > 0 and drawdown_reduce > 0:
            double_better.append((row['策略'], roi_improve, drawdown_reduce, row['ROI'], row['最大回撤']))
    
    if double_better:
        print(f"\n✅ 找到{len(double_better)}个双优策略（ROI提升且回撤降低）：")
        for name, roi_imp, draw_red, roi, drawdown in double_better:
            print(f"\n  【{name}】")
            print(f"    ROI提升: {roi_imp:+.2f}% → {roi:.2f}%")
            print(f"    回撤减少: {draw_red:.0f}元 → {drawdown:.0f}元")
    else:
        print(f"\n❌ 未找到双优策略")
        
        # 显示最接近的策略
        best_roi_idx = summary_df_sorted.index[0]
        best_roi = summary[best_roi_idx]
        
        best_drawdown_idx = summary_df_drawdown.index[0]
        best_drawdown = summary[best_drawdown_idx]
        
        print(f"\n备选方案：")
        print(f"\n  【方案A】最高ROI: {best_roi['策略']}")
        print(f"    ROI: {best_roi['ROI']:.2f}% ({best_roi['ROI'] - baseline['ROI']:+.2f}%)")
        print(f"    回撤: {best_roi['最大回撤']:.0f}元 ({best_roi['最大回撤'] - baseline['最大回撤']:+.0f}元)")
        
        if best_drawdown['策略'] != best_roi['策略']:
            print(f"\n  【方案B】最低回撤: {best_drawdown['策略']}")
            print(f"    ROI: {best_drawdown['ROI']:.2f}% ({best_drawdown['ROI'] - baseline['ROI']:+.2f}%)")
            print(f"    回撤: {best_drawdown['最大回撤']:.0f}元 ({best_drawdown['最大回撤'] - baseline['最大回撤']:+.0f}元)")
    
    # 保存结果
    summary_df.to_csv('top15_strategies_comparison_max10x.csv', index=False, encoding='utf-8-sig')
    print(f"\n详细对比数据已保存到: top15_strategies_comparison_max10x.csv")
    
    return summary_df


if __name__ == "__main__":
    result = backtest_all_strategies(test_periods=300)
    
    print("\n" + "="*100)
    print("✅ 全部策略对比完成！")
    print("="*100)
