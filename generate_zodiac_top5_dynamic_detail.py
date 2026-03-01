"""
生肖TOP5动态投注策略 - 详细300期回测
使用保守组合参数：boost×1.2, reduce×0.8

使用真正的ZodiacSimpleSmart预测器（v10.0 - 52%命中率）
"""

import pandas as pd
import numpy as np
from datetime import datetime
from zodiac_simple_smart import ZodiacSimpleSmart


class SmartDynamicZodiacBetting:
    """生肖TOP5智能动态投注策略"""
    
    def __init__(self, lookback=12, good_thresh=0.35, bad_thresh=0.20, 
                 boost_mult=1.2, reduce_mult=0.8, max_multiplier=10,
                 base_bet=20, win_reward=47):
        self.lookback = lookback
        self.good_thresh = good_thresh
        self.bad_thresh = bad_thresh
        self.boost_mult = boost_mult
        self.reduce_mult = reduce_mult
        self.max_multiplier = max_multiplier
        self.base_bet = base_bet
        self.win_reward = win_reward
        
        # Fibonacci序列
        self.fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        
        # 状态变量
        self.reset()
    
    def reset(self):
        """重置状态"""
        self.fib_index = 0
        self.recent_results = []
        self.total_bet = 0
        self.total_win = 0
        self.balance = 0
        self.min_balance = 0
        self.max_drawdown = 0
    
    def get_base_multiplier(self):
        """获取基础Fibonacci倍数（含限制）"""
        if self.fib_index >= len(self.fib_sequence):
            return min(self.fib_sequence[-1], self.max_multiplier)
        return min(self.fib_sequence[self.fib_index], self.max_multiplier)
    
    def get_recent_rate(self):
        """获取最近命中率"""
        if len(self.recent_results) == 0:
            return 0.42  # 生肖TOP5的理论命中率约42%
        return sum(self.recent_results) / len(self.recent_results)
    
    def process_period(self, hit):
        """处理一期投注（修复后的正确时序）"""
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
        
        # 记录投注前状态
        balance_before = self.balance
        recent_rate = self.get_recent_rate()
        
        # 计算投注和收益
        bet = self.base_bet * multiplier
        self.total_bet += bet
        
        if hit:
            win = self.win_reward * multiplier
            self.total_win += win
            profit = win - bet
            self.balance += profit
            self.fib_index = 0
        else:
            profit = -bet
            self.balance += profit
            self.fib_index += 1
            
            if self.balance < self.min_balance:
                self.min_balance = self.balance
                self.max_drawdown = abs(self.min_balance)
        
        # 添加结果到历史（在投注和结算之后）
        self.recent_results.append(1 if hit else 0)
        if len(self.recent_results) > self.lookback:
            self.recent_results.pop(0)
        
        return {
            'base_mult': base_mult,
            'multiplier': multiplier,
            'bet': bet,
            'profit': profit,
            'balance_before': balance_before,
            'recent_rate': recent_rate,
            'fib_index': self.fib_index
        }


def generate_zodiac_top5_dynamic_detail():
    """生成生肖TOP5动态投注详细回测"""
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    animals = df['animal'].tolist()
    
    total_periods = len(animals)
    test_periods = 300  # 测试300期
    start = total_periods - test_periods
    
    print(f"\n{'='*100}")
    print(f"生肖TOP5智能动态投注策略 - 详细300期回测")
    print(f"{'='*100}")
    print(f"策略参数：")
    print(f"  • boost×1.2, reduce×0.8（保守组合）")
    print(f"  • 回看期数: 12期")
    print(f"  • 增强阈值: 命中率≥35%")
    print(f"  • 降低阈值: 命中率≤20%")
    print(f"  • 最大倍数: 10倍")
    print(f"  • 基础投注: 20元/期（5个生肖×4元）")
    print(f"  • 中奖奖励: 47元×倍数\n")
    print(f"数据总期数: {total_periods}")
    print(f"回测期数: {test_periods}")
    print(f"回测范围: 第{start + 1}期 到 第{total_periods}期\n")
    
    # 初始化预测器和策略（使用真正的v10.0预测器）
    predictor = ZodiacSimpleSmart()  # v10.0 简化智能选择器 (52% 稳定)
    strategy = SmartDynamicZodiacBetting(
        base_bet=20,
        win_reward=47,
        lookback=12,
        good_thresh=0.35,
        bad_thresh=0.20,
        boost_mult=1.2,
        reduce_mult=0.8,
        max_multiplier=10
    )
    
    # 回测
    hits = 0
    hit_10x_count = 0
    details = []
    
    for i in range(start, total_periods):
        period_num = i + 1
        
        # 使用历史数据预测
        history = animals[:i]
        actual_animal = animals[i]
        
        # 预测TOP5生肖
        if len(history) >= 30:
            result = predictor.predict_from_history(history, top_n=5)
            predicted_top5 = result['top5']
        else:
            predicted_top5 = predictor.zodiac_list[:5]
        
        # 判断命中
        hit = actual_animal in predicted_top5
        
        # 处理这一期
        betting_result = strategy.process_period(hit)
        
        # 检查是否触及10倍上限
        hit_limit = betting_result['multiplier'] >= 10
        if hit_limit:
            hit_10x_count += 1
        
        if hit:
            hits += 1
        
        # 记录详情
        details.append({
            'period': period_num,
            'actual': actual_animal,
            'predicted_top5': ','.join(predicted_top5),
            'hit': '✓' if hit else '✗',
            'base_mult': betting_result['base_mult'],
            'final_mult': betting_result['multiplier'],
            'bet': betting_result['bet'],
            'profit': betting_result['profit'],
            'balance': strategy.balance,
            'recent_rate': f"{betting_result['recent_rate']:.1%}",
            'fib_index': betting_result['fib_index'],
            'hit_limit': '⚠️10x' if hit_limit else ''
        })
    
    # 保存到CSV
    df_details = pd.DataFrame(details)
    output_file = 'zodiac_top5_dynamic_300periods_detail.csv'
    df_details.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # 计算结果
    hit_rate = hits / test_periods * 100
    roi = (strategy.balance / strategy.total_bet * 100) if strategy.total_bet > 0 else 0
    risk_return = strategy.balance / strategy.max_drawdown if strategy.max_drawdown > 0 else 0
    
    # 输出结果
    print(f"\n{'='*100}")
    print(f"回测结果汇总")
    print(f"{'='*100}\n")
    print(f"投注结果:")
    print(f"  命中次数: {hits}/{test_periods}")
    print(f"  命中率: {hit_rate:.2f}%")
    print(f"  总投入: {strategy.total_bet:.0f}元")
    print(f"  总收益: {strategy.total_win:.0f}元")
    print(f"  净利润: {strategy.balance:+.0f}元")
    print(f"  ROI: {roi:.2f}%")
    print(f"  最大回撤: {strategy.max_drawdown:.0f}元")
    print(f"  触及10倍上限: {hit_10x_count}次")
    print(f"  风险收益比: {risk_return:.2f}\n")
    
    # 显示前15期和后15期详情
    print(f"\n{'='*100}")
    print(f"前15期详细记录")
    print(f"{'='*100}")
    print(f"{'期数':<6} {'实际':<6} {'命中':<6} {'基础倍':<8} {'最终倍':<8} "
          f"{'投注额':<10} {'盈亏':<10} {'累计':<10} {'最近12期':<10} {'Fib':<6}")
    print(f"{'-'*100}")
    for d in details[:15]:
        print(f"{d['period']:<6} {d['actual']:<6} {d['hit']:<6} "
              f"{d['base_mult']:<8.2f} {d['final_mult']:<8.2f} "
              f"{d['bet']:<10.0f} {d['profit']:+<10.0f} {d['balance']:+<10.0f} "
              f"{d['recent_rate']:<10} {d['fib_index']:<6} {d['hit_limit']}")
    
    print(f"\n{'='*100}")
    print(f"最后15期详细记录")
    print(f"{'='*100}")
    print(f"{'期数':<6} {'实际':<6} {'命中':<6} {'基础倍':<8} {'最终倍':<8} "
          f"{'投注额':<10} {'盈亏':<10} {'累计':<10} {'最近12期':<10} {'Fib':<6}")
    print(f"{'-'*100}")
    for d in details[-15:]:
        print(f"{d['period']:<6} {d['actual']:<6} {d['hit']:<6} "
              f"{d['base_mult']:<8.2f} {d['final_mult']:<8.2f} "
              f"{d['bet']:<10.0f} {d['profit']:+<10.0f} {d['balance']:+<10.0f} "
              f"{d['recent_rate']:<10} {d['fib_index']:<6} {d['hit_limit']}")
    
    print(f"\n✅ 详细数据已保存至: {output_file}")
    print(f"\n{'='*100}\n")
    
    return {
        'hits': hits,
        'hit_rate': hit_rate,
        'total_bet': strategy.total_bet,
        'total_win': strategy.total_win,
        'profit': strategy.balance,
        'roi': roi,
        'drawdown': strategy.max_drawdown,
        'hit_10x': hit_10x_count,
        'risk_return': risk_return
    }


if __name__ == "__main__":
    result = generate_zodiac_top5_dynamic_detail()
