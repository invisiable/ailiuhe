"""
生肖TOP5智能动态投注v3.1 - 参数优化对比测试
测试不同的lookback期数和boost/reduce参数组合

对比配置：
1. 保守-12期：lookback=12, boost×1.2, reduce×0.8
2. 激进-12期：lookback=12, boost×1.5, reduce×0.5
3. 保守-8期：lookback=8, boost×1.2, reduce×0.8
4. 激进-8期：lookback=8, boost×1.5, reduce×0.5
"""

import pandas as pd
import numpy as np
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


def test_parameter_combination(boost_mult, reduce_mult, lookback, description):
    """测试特定参数组合"""
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    animals = df['animal'].tolist()
    
    total_periods = len(animals)
    test_periods = 200  # 测试200期
    start = total_periods - test_periods
    
    print(f"\n{'='*80}")
    print(f"测试配置：{description}")
    print(f"{'='*80}")
    print(f"参数设置：")
    print(f"  • 回看期数: {lookback}期")
    print(f"  • 增强阈值: 命中率≥35% → boost×{boost_mult}")
    print(f"  • 降低阈值: 命中率≤20% → reduce×{reduce_mult}")
    print(f"  • 最大倍数: 10倍")
    print(f"  • 测试期数: {test_periods}期\n")
    
    # 初始化预测器和策略
    predictor = ZodiacSimpleSmart()
    strategy = SmartDynamicZodiacBetting(
        base_bet=20,
        win_reward=47,
        lookback=lookback,
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
        if betting_result['multiplier'] >= 10:
            hit_10x_count += 1
        
        if hit:
            hits += 1
    
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
    print(f"  风险收益比: {risk_return:.2f}\n")
    
    return {
        'description': description,
        'lookback': lookback,
        'boost_mult': boost_mult,
        'reduce_mult': reduce_mult,
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
    print(f"生肖TOP5智能动态投注v3.1 - 参数优化对比测试")
    print(f"{'='*80}\n")
    print(f"测试目标：")
    print(f"  1. 对比不同lookback期数（8期 vs 12期）")
    print(f"  2. 对比不同boost/reduce组合（保守 vs 激进）")
    print(f"  3. 找出最优参数组合\n")
    
    # 测试四组配置
    configs = [
        (1.2, 0.8, 12, "保守-12期（当前GUI默认）"),
        (1.5, 0.5, 12, "激进-12期"),
        (1.2, 0.8, 8, "保守-8期"),
        (1.5, 0.5, 8, "激进-8期")
    ]
    
    results = []
    for boost, reduce, lookback, desc in configs:
        result = test_parameter_combination(boost, reduce, lookback, desc)
        results.append(result)
    
    # 详细对比分析
    print(f"\n{'='*80}")
    print(f"详细对比分析")
    print(f"{'='*80}\n")
    
    print(f"{'配置':<20} {'命中':<12} {'ROI':<12} {'利润':<12} {'回撤':<12} {'10x':<8} {'风险比':<8}")
    print(f"{'-'*100}")
    
    for r in results:
        print(f"{r['description']:<20} {r['hits']}/200 {r['hit_rate']:.1f}% "
              f"{r['roi']:>8.2f}%  {r['profit']:>8.0f}元  {r['drawdown']:>8.0f}元  "
              f"{r['hit_10x']:>4}次  {r['risk_return']:>6.2f}")
    
    # 找出最优配置
    print(f"\n{'='*80}")
    print(f"多维度排名")
    print(f"{'='*80}\n")
    
    # 按ROI排序
    sorted_by_roi = sorted(results, key=lambda x: x['roi'], reverse=True)
    print(f"【ROI排名】")
    for i, r in enumerate(sorted_by_roi, 1):
        marker = "🏆" if i == 1 else f"{i}."
        print(f"  {marker} {r['description']:<20} ROI {r['roi']:>7.2f}%")
    
    print()
    
    # 按利润排序
    sorted_by_profit = sorted(results, key=lambda x: x['profit'], reverse=True)
    print(f"【利润排名】")
    for i, r in enumerate(sorted_by_profit, 1):
        marker = "🏆" if i == 1 else f"{i}."
        print(f"  {marker} {r['description']:<20} 利润 {r['profit']:>7.0f}元")
    
    print()
    
    # 按回撤排序（越小越好）
    sorted_by_drawdown = sorted(results, key=lambda x: x['drawdown'])
    print(f"【风险控制排名】（回撤越小越好）")
    for i, r in enumerate(sorted_by_drawdown, 1):
        marker = "🏆" if i == 1 else f"{i}."
        print(f"  {marker} {r['description']:<20} 回撤 {r['drawdown']:>7.0f}元")
    
    print()
    
    # 按风险收益比排序
    sorted_by_risk_return = sorted(results, key=lambda x: x['risk_return'], reverse=True)
    print(f"【风险收益比排名】（越高越好）")
    for i, r in enumerate(sorted_by_risk_return, 1):
        marker = "🏆" if i == 1 else f"{i}."
        print(f"  {marker} {r['description']:<20} 风险收益比 {r['risk_return']:>6.2f}")
    
    # 综合评分
    print(f"\n{'='*80}")
    print(f"综合评分分析")
    print(f"{'='*80}\n")
    
    # 计算综合得分（ROI权重40%，利润30%，风险控制30%）
    for r in results:
        roi_score = r['roi'] / max(res['roi'] for res in results) * 40
        profit_score = r['profit'] / max(res['profit'] for res in results) * 30
        risk_score = (1 - r['drawdown'] / max(res['drawdown'] for res in results)) * 30
        r['total_score'] = roi_score + profit_score + risk_score
    
    sorted_by_score = sorted(results, key=lambda x: x['total_score'], reverse=True)
    
    print(f"综合评分（ROI 40% + 利润 30% + 风险控制 30%）:\n")
    for i, r in enumerate(sorted_by_score, 1):
        marker = "⭐" if i == 1 else "✓" if i == 2 else f"{i}."
        print(f"  {marker} {r['description']:<20} 综合得分 {r['total_score']:.2f}")
        print(f"      ROI {r['roi']:.2f}% | 利润 {r['profit']:+.0f}元 | 回撤 {r['drawdown']:.0f}元 | 风险比 {r['risk_return']:.2f}")
        print()
    
    # 最终推荐
    best = sorted_by_score[0]
    print(f"{'='*80}")
    print(f"💡 最终推荐")
    print(f"{'='*80}\n")
    print(f"⭐ 最优配置：{best['description']}")
    print(f"   • 回看期数: {best['lookback']}期")
    print(f"   • boost×{best['boost_mult']}, reduce×{best['reduce_mult']}")
    print(f"   • ROI: {best['roi']:.2f}%")
    print(f"   • 利润: {best['profit']:+.0f}元")
    print(f"   • 回撤: {best['drawdown']:.0f}元")
    print(f"   • 风险收益比: {best['risk_return']:.2f}")
    print(f"   • 触及10x: {best['hit_10x']}次\n")
    
    # 与当前GUI默认配置对比
    current = results[0]  # 保守-12期（当前GUI默认）
    if best != current:
        improvement_roi = best['roi'] - current['roi']
        improvement_profit = best['profit'] - current['profit']
        improvement_drawdown = current['drawdown'] - best['drawdown']
        
        print(f"对比当前GUI默认配置（保守-12期）:")
        print(f"  • ROI提升: {improvement_roi:+.2f}% ({improvement_roi/current['roi']*100:+.1f}%)")
        print(f"  • 利润提升: {improvement_profit:+.0f}元 ({improvement_profit/abs(current['profit'])*100:+.1f}%)")
        print(f"  • 回撤变化: {improvement_drawdown:+.0f}元 ({improvement_drawdown/current['drawdown']*100:+.1f}%)")
        print(f"\n💡 建议：更新GUI配置为最优参数组合！")
    else:
        print(f"✅ 当前GUI默认配置已经是最优选择！")
    
    print(f"\n{'='*80}\n")
