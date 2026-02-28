"""
智能动态倍投策略 - 最近300期详细投注记录
参数：回看12期，命中率>=0.35增强1.2倍，<=0.20降低0.8倍，最大10倍限制
使用与compare_all_strategies_max10x.py完全相同的逻辑
"""

import pandas as pd
import numpy as np
from datetime import datetime
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
    """智能动态倍投策略（与compare脚本完全相同的实现）"""
    
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
        """处理一期投注（与compare脚本完全相同的逻辑）"""
        hit = actual in prediction
        
        # 先添加结果到历史
        self.recent_results.append(1 if hit else 0)
        if len(self.recent_results) > self.lookback:
            self.recent_results.pop(0)
        
        # 获取基础倍数
        base_mult = self.get_base_multiplier()
        
        # 根据最近命中率计算动态倍数
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
        
        return {
            'multiplier': multiplier,
            'bet': bet,
            'win': win,
            'profit': profit,
            'hit': hit,
            'recent_rate': self.get_recent_rate()
        }


def backtest_smart_dynamic_detail():
    """详细回测智能动态倍投策略"""
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    total_periods = len(df)
    test_periods = 300
    
    print(f"数据总期数: {total_periods}")
    print(f"回测期数: {test_periods}")
    print(f"回测范围: 第{total_periods - test_periods + 1}期 到 第{total_periods}期")
    print(f"="*100)
    
    # 初始化预测器和策略
    predictor = PreciseTop15Predictor()
    strategy = SmartDynamicBettingStrategy(
        base_bet=15,
        win_reward=47,
        lookback=12,
        boost_threshold=0.35,
        reduce_threshold=0.20,
        boost_multiplier=1.2,
        reduce_multiplier=0.8,
        max_multiplier=10
    )
    
    # 记录详细信息
    details = []
    
    # 统计数据
    hits = 0
    total_investment = 0
    total_return = 0
    max_drawdown = 0
    current_balance = 0
    hit_10x_limit_count = 0
    
    # 开始回测
    for i in range(total_periods - test_periods, total_periods):
        period_num = i + 1
        
        # 使用历史数据训练和预测
        train_data = df.iloc[:i]
        actual_number = df.iloc[i]['number']
        
        # 预测
        if len(train_data) >= 30:
            predicted_top15 = predictor.predict(train_data['number'].values)
        else:
            # 数据不足时使用简单预测
            predicted_top15 = list(range(1, 16))
        
        # 获取当前投注金额
        bet_amount, multiplier = strategy.get_current_bet()
        
        # 检查是否触及10倍上限
        hit_limit = (multiplier >= 10)
        if hit_limit:
            hit_10x_limit_count += 1
        
        # 检查是否命中
        is_hit = actual_number in predicted_top15
        
        # 计算收益（奖励金额也要乘以倍数）
        if is_hit:
            win_amount = strategy.win_reward * multiplier
            profit = win_amount - bet_amount
            hits += 1
        else:
            win_amount = 0
            profit = -bet_amount
        
        # 更新统计
        total_investment += bet_amount
        if is_hit:
            total_return += win_amount
        current_balance += profit
        max_drawdown = min(max_drawdown, current_balance)
        
        # 获取当前命中率
        recent_hit_rate = strategy.get_recent_hit_rate()
        
        # 记录详细信息
        details.append({
            '期号': period_num,
            '日期': df.iloc[i]['date'].strftime('%Y/%m/%d'),
            '开奖号码': int(actual_number),
            '预测TOP15': str(predicted_top15[:5]) + '...',  # 只显示前5个
            '投注倍数': f"{multiplier:.2f}",
            '投注金额': f"{bet_amount:.0f}",
            '是否命中': '✓' if is_hit else '✗',
            '当期盈亏': f"{profit:+.0f}",
            '累计盈亏': f"{current_balance:+.0f}",
            '最近12期命中率': f"{recent_hit_rate:.2%}",
            '触及10倍上限': '是' if hit_limit else '',
            'Fib索引': strategy.fib_index
        })
        
        # 更新策略状态
        strategy.update(is_hit)
    
    # 创建DataFrame
    detail_df = pd.DataFrame(details)
    
    # 保存到CSV
    output_file = 'smart_dynamic_300periods_detail.csv'
    detail_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n详细记录已保存到: {output_file}")
    
    # 打印统计摘要
    print(f"\n{'='*100}")
    print(f"【智能动态倍投策略 - 300期回测统计】")
    print(f"{'='*100}")
    print(f"策略参数:")
    print(f"  基础投注: {strategy.base_bet}元")
    print(f"  中奖奖励: {strategy.win_reward}元")
    print(f"  回看期数: {strategy.lookback}期")
    print(f"  增强阈值: 命中率>{strategy.boost_threshold:.0%} → 倍数×{strategy.boost_multiplier}")
    print(f"  降低阈值: 命中率<{strategy.reduce_threshold:.0%} → 倍数×{strategy.reduce_multiplier}")
    print(f"  最大倍数: {strategy.max_multiplier}倍")
    print(f"\n投注结果:")
    print(f"  命中次数: {hits}/{test_periods}")
    print(f"  命中率: {hits/test_periods*100:.2f}%")
    print(f"  总投入: {total_investment:.0f}元")
    print(f"  总收益: {total_return:.0f}元")
    print(f"  净利润: {total_return - total_investment:.0f}元")
    print(f"  ROI: {(total_return - total_investment)/total_investment*100:.2f}%")
    print(f"  最大回撤: {abs(max_drawdown):.0f}元")
    print(f"  触及10倍上限次数: {hit_10x_limit_count}次")
    print(f"{'='*100}")
    
    # 打印前20期详情
    print(f"\n前20期详细记录:")
    print(detail_df.head(20).to_string(index=False))
    
    # 打印最后20期详情
    print(f"\n最后20期详细记录:")
    print(detail_df.tail(20).to_string(index=False))
    
    # 分析触及上限的情况
    limit_periods = detail_df[detail_df['触及10倍上限'] == '是']
    if len(limit_periods) > 0:
        print(f"\n触及10倍上限的{len(limit_periods)}期详情:")
        print(limit_periods.to_string(index=False))
    
    return detail_df


if __name__ == "__main__":
    detail_df = backtest_smart_dynamic_detail()
