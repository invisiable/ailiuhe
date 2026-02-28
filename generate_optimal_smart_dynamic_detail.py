"""
最优智能动态倍投策略 - 300期详细投注记录
最优参数（排名#1）:
- 回看期数: 8期
- 增强阈值: 0.35 → 倍数×1.5
- 降低阈值: 0.20 → 倍数×0.6
- 最大倍数限制: 10倍
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


class OptimalSmartDynamicStrategy(BettingStrategyBase):
    """最优智能动态倍投策略"""
    
    def __init__(self, lookback=8, good_thresh=0.35, bad_thresh=0.20, 
                 boost_mult=1.5, reduce_mult=0.6, **kwargs):
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
            'hit': hit,
            'recent_rate': self.get_recent_rate()
        }


def generate_optimal_strategy_detail():
    """生成最优策略的详细投注记录"""
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv')
    df['date'] = pd.to_datetime(df['date'])
    numbers = df['number'].tolist()
    dates = df['date'].tolist()
    
    total_periods = len(numbers)
    test_periods = 300
    start = total_periods - test_periods
    
    print("="*100)
    print("最优智能动态倍投策略 - 300期详细回测")
    print("="*100)
    print(f"数据总期数: {total_periods}")
    print(f"回测期数: {test_periods}")
    print(f"回测范围: 第{start + 1}期 到 第{total_periods}期")
    print("\n【最优参数配置】")
    print("  回看期数: 8期")
    print("  增强阈值: 命中率>=35% → 倍数×1.5")
    print("  降低阈值: 命中率<=20% → 倍数×0.6")
    print("  最大倍数: 10倍")
    print("="*100)
    
    # 初始化预测器和策略
    predictor = PreciseTop15Predictor()
    strategy = OptimalSmartDynamicStrategy(
        lookback=8,
        good_thresh=0.35,
        bad_thresh=0.20,
        boost_mult=1.5,
        reduce_mult=0.6,
        base_bet=15,
        win_reward=47,
        max_multiplier=10
    )
    
    # 记录详细信息
    details = []
    hits = 0
    hit_10x_count = 0
    
    # 开始回测
    for i in range(start, total_periods):
        period_num = i + 1
        
        # 使用历史数据预测
        history = numbers[:i]
        actual = numbers[i]
        
        # 预测TOP15
        if len(history) >= 30:
            predictions = predictor.predict(history)
        else:
            predictions = list(range(1, 16))
        
        # 处理这一期
        result = strategy.process_period(predictions, actual)
        
        # 检查是否触及10倍上限
        hit_limit = (result['multiplier'] >= 10)
        if hit_limit:
            hit_10x_count += 1
        
        if result['hit']:
            hits += 1
        
        # 记录详细信息
        details.append({
            '期号': period_num,
            '日期': dates[i].strftime('%Y/%m/%d'),
            '开奖号码': int(actual),
            '预测TOP15': str(predictions[:5]) + '...',
            '投注倍数': f"{result['multiplier']:.2f}",
            '投注金额': f"{result['bet']:.0f}",
            '是否命中': '✓' if result['hit'] else '✗',
            '当期盈亏': f"{result['profit']:+.0f}",
            '累计盈亏': f"{strategy.balance:+.0f}",
            '最近8期命中率': f"{result['recent_rate']:.2%}",
            '触及10倍上限': '是' if hit_limit else '',
            'Fib索引': strategy.fib_index
        })
    
    # 创建DataFrame
    detail_df = pd.DataFrame(details)
    
    # 保存到CSV
    output_file = 'optimal_smart_dynamic_300periods_detail.csv'
    detail_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n详细记录已保存到: {output_file}")
    
    # 打印统计摘要
    print(f"\n{'='*100}")
    print(f"【最优策略 - 300期回测结果】")
    print(f"{'='*100}")
    print(f"参数配置:")
    print(f"  基础投注: 15元，中奖奖励: 47元")
    print(f"  回看期数: 8期")
    print(f"  增强阈值: 命中率>=35% → 倍数×1.5")
    print(f"  降低阈值: 命中率<=20% → 倍数×0.6")
    print(f"  最大倍数: 10倍")
    
    print(f"\n投注结果:")
    print(f"  命中次数: {hits}/{test_periods}")
    print(f"  命中率: {hits/test_periods*100:.2f}%")
    print(f"  总投入: {strategy.total_bet:.0f}元")
    print(f"  总收益: {strategy.total_win:.0f}元")
    print(f"  净利润: {strategy.balance:.0f}元")
    print(f"  ROI: {(strategy.balance/strategy.total_bet*100) if strategy.total_bet > 0 else 0:.2f}%")
    print(f"  最大回撤: {strategy.max_drawdown:.0f}元")
    print(f"  触及10倍上限: {hit_10x_count}次")
    
    # 与基准策略对比
    baseline_roi = 15.10
    baseline_profit = 1709
    baseline_drawdown = 775
    
    print(f"\n与基准策略对比:")
    roi_improve = (strategy.balance/strategy.total_bet*100) - baseline_roi
    profit_improve = strategy.balance - baseline_profit
    drawdown_reduce = baseline_drawdown - strategy.max_drawdown
    
    print(f"  ROI提升: {roi_improve:+.2f}% ({(roi_improve/baseline_roi*100):+.1f}%)")
    print(f"  净利润增加: {profit_improve:+.0f}元 ({(profit_improve/baseline_profit*100):+.1f}%)")
    print(f"  回撤降低: {drawdown_reduce:+.0f}元 ({(drawdown_reduce/baseline_drawdown*100):+.1f}%)")
    
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
    
    # 分析关键统计
    print(f"\n【策略效果分析】")
    
    # 连续盈利和亏损
    profits = [float(d['当期盈亏']) for d in details]
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_wins = 0
    current_losses = 0
    
    for p in profits:
        if p > 0:
            current_wins += 1
            current_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
    
    print(f"  最长连胜: {max_consecutive_wins}期")
    print(f"  最长连亏: {max_consecutive_losses}期")
    
    # 平均单期盈利
    avg_profit_per_period = strategy.balance / test_periods
    print(f"  平均单期盈利: {avg_profit_per_period:.2f}元")
    
    # 投入变化
    investments = [float(d['投注金额']) for d in details]
    avg_investment = sum(investments) / len(investments)
    max_investment = max(investments)
    min_investment = min(investments)
    
    print(f"  平均单期投入: {avg_investment:.2f}元")
    print(f"  单期最高投入: {max_investment:.0f}元")
    print(f"  单期最低投入: {min_investment:.0f}元")
    
    return detail_df


if __name__ == "__main__":
    detail_df = generate_optimal_strategy_detail()
    
    print("\n" + "="*100)
    print("✅ 最优策略详细记录生成完成！")
    print("="*100)
