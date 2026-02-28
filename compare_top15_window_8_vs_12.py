"""
测试TOP15最优智能动态倍投策略 - 对比12期窗口 vs 8期窗口

目标：测试窗口期改为8期后，回撤金额是否会更小
"""

import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor


class SmartDynamicStrategy:
    """智能动态倍投策略"""
    
    def __init__(self, config):
        self.cfg = config
        self.fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        self.fib_index = 0
        self.recent_results = []
        self.total_bet = 0
        self.total_win = 0
        self.balance = 0
        self.min_balance = 0
        self.max_drawdown = 0
        self.hit_10x_count = 0
        self.period_details = []
    
    def get_base_multiplier(self):
        """获取基础Fibonacci倍数"""
        if self.fib_index >= len(self.fib_sequence):
            return min(self.fib_sequence[-1], self.cfg['max_multiplier'])
        return min(self.fib_sequence[self.fib_index], self.cfg['max_multiplier'])
    
    def get_recent_rate(self):
        """获取最近命中率"""
        if len(self.recent_results) == 0:
            return 0.63  # TOP15理论命中率
        return sum(self.recent_results) / len(self.recent_results)
    
    def process_period(self, hit):
        """处理一期投注"""
        # 获取基础倍数
        base_mult = self.get_base_multiplier()
        
        # 计算动态倍数（使用投注前的历史数据）
        if len(self.recent_results) >= self.cfg['lookback']:
            rate = self.get_recent_rate()
            if rate >= self.cfg['good_thresh']:
                multiplier = min(base_mult * self.cfg['boost_mult'], self.cfg['max_multiplier'])
            elif rate <= self.cfg['bad_thresh']:
                multiplier = max(base_mult * self.cfg['reduce_mult'], 1)
            else:
                multiplier = base_mult
        else:
            multiplier = base_mult
        
        # 计算投注和收益
        bet = self.cfg['base_bet'] * multiplier
        self.total_bet += bet
        
        if multiplier >= 10:
            self.hit_10x_count += 1
        
        if hit:
            win = self.cfg['win_reward'] * multiplier
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
        
        # 添加结果到历史（在投注后）
        self.recent_results.append(1 if hit else 0)
        if len(self.recent_results) > self.cfg['lookback']:
            self.recent_results.pop(0)
        
        return {
            'multiplier': multiplier,
            'bet': bet,
            'profit': profit,
            'recent_rate': self.get_recent_rate()
        }


def test_optimal_strategy_window_comparison():
    """对比测试12期窗口 vs 8期窗口"""
    
    print("="*80)
    print("TOP15最优智能动态倍投策略 - 窗口期对比测试")
    print("="*80)
    print()
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].tolist()
    
    total_periods = len(numbers)
    test_periods = 300
    start_idx = total_periods - test_periods
    
    print(f"数据总期数: {total_periods}")
    print(f"测试期数: {test_periods}")
    print(f"起始索引: {start_idx}")
    print()
    
    # 初始化预测器
    predictor = PreciseTop15Predictor()
    
    # 生成预测和实际命中记录
    print("生成TOP15预测...")
    predictions = []
    actuals = []
    hit_records = []
    
    for i in range(start_idx, total_periods):
        history = numbers[:i]
        actual = numbers[i]
        
        top15 = predictor.predict(history)  # 直接返回TOP15列表
        
        predictions.append(top15)
        actuals.append(actual)
        hit_records.append(actual in top15)
    
    hit_count = sum(hit_records)
    hit_rate = hit_count / len(hit_records) * 100
    print(f"预测完成: 命中{hit_count}/{len(hit_records)}期 (命中率{hit_rate:.2f}%)")
    print()
    
    # 测试配置
    configs = [
        {
            'name': '激进组合-12期窗口（当前）',
            'lookback': 12,
            'good_thresh': 0.35,
            'bad_thresh': 0.20,
            'boost_mult': 1.5,
            'reduce_mult': 0.5,
            'max_multiplier': 10,
            'base_bet': 15,
            'win_reward': 47
        },
        {
            'name': '激进组合-8期窗口（测试）',
            'lookback': 8,
            'good_thresh': 0.35,
            'bad_thresh': 0.20,
            'boost_mult': 1.5,
            'reduce_mult': 0.5,
            'max_multiplier': 10,
            'base_bet': 15,
            'win_reward': 47
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"{'='*80}")
        print(f"测试配置：{config['name']}")
        print(f"{'='*80}")
        print(f"参数设置：")
        print(f"  • 窗口期数: {config['lookback']}期")
        print(f"  • 增强阈值: 命中率≥{config['good_thresh']*100:.0f}% → boost×{config['boost_mult']}")
        print(f"  • 降低阈值: 命中率≤{config['bad_thresh']*100:.0f}% → reduce×{config['reduce_mult']}")
        print(f"  • 最大倍数: {config['max_multiplier']}倍")
        print()
        
        # 执行回测
        strategy = SmartDynamicStrategy(config)
        
        for hit in hit_records:
            strategy.process_period(hit)
        
        # 计算结果
        roi = (strategy.balance / strategy.total_bet * 100) if strategy.total_bet > 0 else 0
        risk_return = strategy.balance / strategy.max_drawdown if strategy.max_drawdown > 0 else 0
        
        # 输出结果
        print(f"回测结果:")
        print(f"  命中次数: {hit_count}/{test_periods}")
        print(f"  命中率: {hit_rate:.2f}%")
        print(f"  总投入: {strategy.total_bet:.0f}元")
        print(f"  总收益: {strategy.total_win:.0f}元")
        print(f"  净利润: {strategy.balance:+.0f}元")
        print(f"  ROI: {roi:.2f}%")
        print(f"  最大回撤: {strategy.max_drawdown:.0f}元")
        print(f"  触及10x: {strategy.hit_10x_count}次")
        print(f"  风险收益比: {risk_return:.2f}")
        print()
        
        results.append({
            'name': config['name'],
            'lookback': config['lookback'],
            'profit': strategy.balance,
            'roi': roi,
            'drawdown': strategy.max_drawdown,
            'hit_10x': strategy.hit_10x_count,
            'risk_return': risk_return,
            'total_bet': strategy.total_bet
        })
    
    # 对比分析
    print(f"{'='*80}")
    print(f"对比分析")
    print(f"{'='*80}")
    print()
    
    # 表格对比
    print(f"{'配置':<25} {'窗口':<8} {'ROI':<12} {'利润':<12} {'回撤':<12} {'触10x':<8} {'风险比':<8}")
    print("-"*90)
    
    for r in results:
        print(f"{r['name']:<25} {r['lookback']:<8} {r['roi']:>8.2f}%  {r['profit']:>8.0f}元  {r['drawdown']:>8.0f}元  {r['hit_10x']:>4}次  {r['risk_return']:>6.2f}")
    
    print()
    
    # 计算改进
    r1 = results[0]  # 12期窗口
    r2 = results[1]  # 8期窗口
    
    print(f"详细对比（8期 vs 12期）:")
    print()
    
    # ROI对比
    roi_diff = r2['roi'] - r1['roi']
    roi_pct = (roi_diff / r1['roi'] * 100) if r1['roi'] != 0 else 0
    print(f"ROI变化: {r2['roi']:.2f}% vs {r1['roi']:.2f}% ({roi_diff:+.2f}%, {roi_pct:+.1f}%)")
    
    # 利润对比
    profit_diff = r2['profit'] - r1['profit']
    profit_pct = (profit_diff / r1['profit'] * 100) if r1['profit'] != 0 else 0
    print(f"利润变化: {r2['profit']:+.0f}元 vs {r1['profit']:+.0f}元 ({profit_diff:+.0f}元, {profit_pct:+.1f}%)")
    
    # 回撤对比 - 关键指标
    drawdown_diff = r2['drawdown'] - r1['drawdown']
    drawdown_pct = (drawdown_diff / r1['drawdown'] * 100) if r1['drawdown'] != 0 else 0
    print(f"回撤变化: {r2['drawdown']:.0f}元 vs {r1['drawdown']:.0f}元 ({drawdown_diff:+.0f}元, {drawdown_pct:+.1f}%) {'✅' if drawdown_diff < 0 else '⚠️'}")
    
    # 触10x对比
    hit_10x_diff = r2['hit_10x'] - r1['hit_10x']
    print(f"触10x变化: {r2['hit_10x']}次 vs {r1['hit_10x']}次 ({hit_10x_diff:+}次)")
    
    # 风险收益比对比
    risk_return_diff = r2['risk_return'] - r1['risk_return']
    risk_return_pct = (risk_return_diff / r1['risk_return'] * 100) if r1['risk_return'] != 0 else 0
    print(f"风险收益比: {r2['risk_return']:.2f} vs {r1['risk_return']:.2f} ({risk_return_diff:+.2f}, {risk_return_pct:+.1f}%)")
    
    print()
    print("="*80)
    print()
    
    # 结论
    better_config = r2 if r2['drawdown'] < r1['drawdown'] else r1
    
    print("💡 分析结论:")
    print()
    
    if r2['drawdown'] < r1['drawdown']:
        print(f"✅ 8期窗口回撤更小！")
        print(f"   • 回撤降低: {abs(drawdown_diff):.0f}元 ({abs(drawdown_pct):.1f}%)")
        if r2['profit'] >= r1['profit']:
            print(f"   • 利润提升: {profit_diff:.0f}元 ({profit_pct:+.1f}%)")
            print(f"   • 综合表现: 回撤降低的同时利润提升，8期窗口全面优于12期！⭐")
        else:
            print(f"   • 利润下降: {profit_diff:.0f}元 ({profit_pct:.1f}%)")
            print(f"   • 权衡考虑: 回撤降低{abs(drawdown_pct):.1f}%，利润下降{abs(profit_pct):.1f}%")
            if abs(drawdown_pct) > abs(profit_pct):
                print(f"   • 建议: 回撤降低幅度更大，建议采用8期窗口（更好的风险控制）")
            else:
                print(f"   • 建议: 利润下降幅度更大，是否改用8期需权衡风险偏好")
    else:
        print(f"⚠️ 12期窗口回撤更小")
        print(f"   • 回撤增加: {drawdown_diff:.0f}元 ({drawdown_pct:+.1f}%)")
        if r2['profit'] > r1['profit']:
            print(f"   • 利润提升: {profit_diff:.0f}元 ({profit_pct:+.1f}%)")
            print(f"   • 权衡考虑: 8期窗口利润更高但回撤增加")
        else:
            print(f"   • 利润下降: {profit_diff:.0f}元 ({profit_pct:.1f}%)")
            print(f"   • 建议: 保持12期窗口配置（更优风控和收益）")
    
    print()
    
    # 最优推荐
    print("🏆 最优配置推荐:")
    print(f"   配置: {better_config['name']}")
    print(f"   窗口: {better_config['lookback']}期")
    print(f"   ROI: {better_config['roi']:.2f}%")
    print(f"   利润: {better_config['profit']:+.0f}元")
    print(f"   回撤: {better_config['drawdown']:.0f}元")
    print(f"   风险收益比: {better_config['risk_return']:.2f}")
    print()
    print("="*80)
    print()


if __name__ == "__main__":
    test_optimal_strategy_window_comparison()
