"""
生肖TOP5动态投注策略参数优化测试
对比两组参数：
1. 保守组合：boost×1.2, reduce×0.8
2. 激进组合：boost×1.5, reduce×0.5

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
            'multiplier': multiplier,
            'bet': bet,
            'profit': profit,
            'recent_rate': self.get_recent_rate()
        }


def backtest_zodiac_top5_dynamic(boost_mult, reduce_mult, description):
    """回测生肖TOP5动态投注策略"""
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    animals = df['animal'].tolist()
    
    total_periods = len(animals)
    test_periods = 200  # 测试200期
    start = total_periods - test_periods
    
    print(f"\n{'='*80}")
    print(f"测试配置：{description}")
    print(f"  boost×{boost_mult}, reduce×{reduce_mult}")
    print(f"{'='*80}")
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
        boost_mult=boost_mult,
        reduce_mult=reduce_mult,
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
        
        # 记录详情（保存前20期和后20期）
        if (i - start) < 20 or (i - start) >= test_periods - 20:
            details.append({
                'period': period_num,
                'actual': actual_animal,
                'predicted': predicted_top5,
                'hit': hit,
                'multiplier': betting_result['multiplier'],
                'bet': betting_result['bet'],
                'profit': betting_result['profit'],
                'balance': strategy.balance,
                'recent_rate': betting_result['recent_rate'],
                'hit_limit': hit_limit
            })
    
    # 计算结果
    hit_rate = hits / test_periods * 100
    roi = (strategy.balance / strategy.total_bet * 100) if strategy.total_bet > 0 else 0
    risk_return = strategy.balance / strategy.max_drawdown if strategy.max_drawdown > 0 else 0
    
    # 输出结果
    print(f"投注结果:")
    print(f"  命中次数: {hits}/{test_periods}")
    print(f"  命中率: {hit_rate:.2f}%")
    print(f"  总投入: {strategy.total_bet:.0f}元")
    print(f"  总收益: {strategy.total_win:.0f}元")
    print(f"  净利润: {strategy.balance:+.0f}元")
    print(f"  ROI: {roi:.2f}%")
    print(f"  最大回撤: {strategy.max_drawdown:.0f}元")
    print(f"  触及10倍上限: {hit_10x_count}次")
    print(f"  风险收益比: {risk_return:.2f}")
    
    # 显示前5期和后5期详情
    if details:
        print(f"\n前5期详情:")
        for d in details[:5]:
            status = "✓" if d['hit'] else "✗"
            print(f"  第{d['period']}期: {d['actual']} {status} | "
                  f"倍数{d['multiplier']:.2f} | 投注{d['bet']:.0f}元 | "
                  f"盈亏{d['profit']:+.0f}元 | 累计{d['balance']:+.0f}元")
        
        print(f"\n最后5期详情:")
        for d in details[-5:]:
            status = "✓" if d['hit'] else "✗"
            print(f"  第{d['period']}期: {d['actual']} {status} | "
                  f"倍数{d['multiplier']:.2f} | 投注{d['bet']:.0f}元 | "
                  f"盈亏{d['profit']:+.0f}元 | 累计{d['balance']:+.0f}元")
    
    return {
        'boost_mult': boost_mult,
        'reduce_mult': reduce_mult,
        'description': description,
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
    print(f"\n{'='*80}")
    print(f"生肖TOP5动态投注策略 - boost和reduce参数对比测试")
    print(f"{'='*80}\n")
    print(f"测试说明:")
    print(f"  • 回看期数: 12期")
    print(f"  • 增强阈值: 命中率≥35%")
    print(f"  • 降低阈值: 命中率≤20%")
    print(f"  • 最大倍数: 10倍")
    print(f"  • 基础投注: 20元/期（5个生肖×4元）")
    print(f"  • 中奖奖励: 47元×倍数\n")
    
    # 测试两组参数
    configs = [
        (1.2, 0.8, "保守组合（原参数）"),
        (1.5, 0.5, "激进组合（新参数）")
    ]
    
    results = []
    for boost, reduce, desc in configs:
        result = backtest_zodiac_top5_dynamic(boost, reduce, desc)
        results.append(result)
    
    # 详细对比
    print(f"\n\n{'='*80}")
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
    print(f"{'命中次数':<15} {r1['hits']}/200 ({r1['hit_rate']:.2f}%){'':<5} "
          f"{r2['hits']}/200 ({r2['hit_rate']:.2f}%){'':<5} "
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
    print(f"{'风险收益比':<15} {r1['risk_return']:.2f}{'':<17} "
          f"{r2['risk_return']:.2f}{'':<17} "
          f"{r2['risk_return'] - r1['risk_return']:+.2f} {'✓' if r2['risk_return'] > r1['risk_return'] else '✗'}")
    
    print(f"\n{'='*80}")
    print(f"结论分析")
    print(f"{'='*80}\n")
    
    if r2['roi'] > r1['roi'] and r2['drawdown'] < r1['drawdown']:
        print(f"✅ 激进组合（boost×{r2['boost_mult']}, reduce×{r2['reduce_mult']}）全面优于保守组合！")
        print(f"   • ROI提升 {diff_roi:.2f}% ({diff_roi/r1['roi']*100:.1f}%)")
        print(f"   • 利润增加 {diff_profit:.0f}元 ({diff_profit/abs(r1['profit'])*100:.1f}%)")
        print(f"   • 回撤降低 {abs(diff_dd):.0f}元 ({abs(diff_dd)/r1['drawdown']*100:.1f}%)")
        print(f"   • 风险收益比提升 {r2['risk_return'] - r1['risk_return']:.2f}")
        print(f"\n💡 建议：采用激进组合参数作为生肖TOP5最优策略！")
    elif r2['roi'] > r1['roi']:
        print(f"✓ 激进组合ROI更高")
        print(f"   • ROI提升 {diff_roi:.2f}%")
        print(f"   • 利润增加 {diff_profit:.0f}元")
        if diff_dd > 0:
            print(f"   • 但回撤增加 {diff_dd:.0f}元")
        print(f"   • 风险收益比: {r2['risk_return']:.2f} vs {r1['risk_return']:.2f}")
        if r2['risk_return'] > r1['risk_return']:
            print(f"\n💡 建议：激进组合风险收益比更优，可以采用")
        else:
            print(f"\n⚠️  建议：视风险偏好选择")
    else:
        print(f"✗ 保守组合表现更好")
        print(f"\n💡 建议：保持原参数组合（boost×{r1['boost_mult']}, reduce×{r1['reduce_mult']}）")
