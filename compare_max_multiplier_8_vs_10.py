"""
对比分析：最大倍数8倍 vs 10倍的影响
测试纯斐波那契策略在不同最大倍数限制下的收益率和回撤差异
"""

import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor

# Fibonacci数列
fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

class FibonacciStrategy:
    """纯斐波那契倍投策略"""
    def __init__(self, base_bet=15, win_reward=47, max_multiplier=10):
        self.base_bet = base_bet
        self.win_reward = win_reward
        self.max_multiplier = max_multiplier
        self.fib_index = 0
        self.total_bet = 0
        self.total_win = 0
        self.balance = 0
        self.min_balance = 0
        self.max_drawdown = 0
        self.hit_limit_count = 0  # 达到最大倍数限制的次数
        
    def get_multiplier(self):
        """获取当前倍数"""
        if self.fib_index >= len(fib_sequence):
            return min(fib_sequence[-1], self.max_multiplier)
        return min(fib_sequence[self.fib_index], self.max_multiplier)
    
    def process_period(self, hit):
        """处理一期投注"""
        multiplier = self.get_multiplier()
        bet = self.base_bet * multiplier
        self.total_bet += bet
        
        # 记录是否达到限制
        if multiplier >= self.max_multiplier:
            self.hit_limit_count += 1
        
        if hit:
            win = self.win_reward * multiplier
            self.total_win += win
            profit = win - bet
            self.balance += profit
            self.fib_index = 0  # 重置
        else:
            profit = -bet
            self.balance += profit
            self.fib_index += 1  # 递增
        
        # 更新最大回撤
        if self.balance < self.min_balance:
            self.min_balance = self.balance
            self.max_drawdown = abs(self.min_balance)
        
        return {
            'multiplier': multiplier,
            'bet': bet,
            'profit': profit
        }
    
    def get_stats(self):
        """获取统计数据"""
        roi = (self.balance / self.total_bet * 100) if self.total_bet > 0 else 0
        risk_reward = self.balance / self.max_drawdown if self.max_drawdown > 0 else 0
        return {
            'total_bet': self.total_bet,
            'total_win': self.total_win,
            'balance': self.balance,
            'roi': roi,
            'max_drawdown': self.max_drawdown,
            'risk_reward': risk_reward,
            'hit_limit_count': self.hit_limit_count
        }


def simulate_with_pause(predictor, df, start_idx, test_periods, max_multiplier, pause_length=1):
    """模拟带暂停策略的投注"""
    strategy = FibonacciStrategy(max_multiplier=max_multiplier)
    pause_remaining = 0
    pause_periods = 0
    actual_periods = 0
    hits = 0
    losses = 0
    consecutive_losses = 0
    max_consecutive_losses = 0
    
    results = []
    
    for i in range(start_idx, start_idx + test_periods):
        # 预测
        train_data = df.iloc[:i]
        predicted_numbers = predictor.predict(train_data['number'].tolist())
        actual = df.iloc[i]['number']
        hit = actual in predicted_numbers
        
        # 暂停期
        if pause_remaining > 0:
            pause_remaining -= 1
            pause_periods += 1
            results.append({
                'period': i - start_idx + 1,
                'hit': hit,
                'paused': True,
                'multiplier': 0,
                'bet': 0,
                'profit': 0,
                'balance': strategy.balance
            })
            continue
        
        # 正常投注
        actual_periods += 1
        result = strategy.process_period(hit)
        
        if hit:
            hits += 1
            pause_remaining = pause_length
            consecutive_losses = 0
        else:
            losses += 1
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        results.append({
            'period': i - start_idx + 1,
            'hit': hit,
            'paused': False,
            'multiplier': result['multiplier'],
            'bet': result['bet'],
            'profit': result['profit'],
            'balance': strategy.balance
        })
    
    stats = strategy.get_stats()
    stats['total_periods'] = test_periods
    stats['actual_periods'] = actual_periods
    stats['pause_periods'] = pause_periods
    stats['hits'] = hits
    stats['losses'] = losses
    stats['hit_rate'] = hits / actual_periods * 100 if actual_periods > 0 else 0
    stats['max_consecutive_losses'] = max_consecutive_losses
    
    return stats, results


def main():
    print("="*80)
    print(" "*20 + "最大倍数对比分析：8倍 vs 10倍")
    print("="*80)
    print()
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    
    if len(df) < 50:
        print("❌ 数据不足50期，无法进行分析")
        return
    
    print(f"✅ 数据加载完成: {len(df)}期")
    print(f"数据范围: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
    print()
    
    # 初始化预测器
    predictor = PreciseTop15Predictor()
    
    # 测试配置
    test_periods = min(300, len(df) - 50)
    start_idx = len(df) - test_periods
    
    print(f"回测期数: {test_periods}期")
    print()
    print("="*80)
    print("开始对比测试...")
    print("="*80)
    print()
    
    # 测试1: max_multiplier = 10
    print("【测试1】最大倍数 = 10倍")
    print("-"*80)
    stats_10, results_10 = simulate_with_pause(predictor, df, start_idx, test_periods, max_multiplier=10)
    
    print(f"总期数: {stats_10['total_periods']}")
    print(f"投注期数: {stats_10['actual_periods']}")
    print(f"暂停期数: {stats_10['pause_periods']}")
    print(f"命中次数: {stats_10['hits']} ({stats_10['hit_rate']:.1f}%)")
    print(f"总投注: {stats_10['total_bet']:.0f}元")
    print(f"总回报: {stats_10['total_win']:.0f}元")
    print(f"净利润: {stats_10['balance']:+.0f}元")
    print(f"ROI: {stats_10['roi']:.2f}%")
    print(f"最大回撤: {stats_10['max_drawdown']:.0f}元")
    print(f"风险收益比: {stats_10['risk_reward']:.2f}")
    print(f"最大连败: {stats_10['max_consecutive_losses']}次")
    print(f"达到10倍限制: {stats_10['hit_limit_count']}次")
    print()
    
    # 测试2: max_multiplier = 8
    print("【测试2】最大倍数 = 8倍")
    print("-"*80)
    stats_8, results_8 = simulate_with_pause(predictor, df, start_idx, test_periods, max_multiplier=8)
    
    print(f"总期数: {stats_8['total_periods']}")
    print(f"投注期数: {stats_8['actual_periods']}")
    print(f"暂停期数: {stats_8['pause_periods']}")
    print(f"命中次数: {stats_8['hits']} ({stats_8['hit_rate']:.1f}%)")
    print(f"总投注: {stats_8['total_bet']:.0f}元")
    print(f"总回报: {stats_8['total_win']:.0f}元")
    print(f"净利润: {stats_8['balance']:+.0f}元")
    print(f"ROI: {stats_8['roi']:.2f}%")
    print(f"最大回撤: {stats_8['max_drawdown']:.0f}元")
    print(f"风险收益比: {stats_8['risk_reward']:.2f}")
    print(f"最大连败: {stats_8['max_consecutive_losses']}次")
    print(f"达到8倍限制: {stats_8['hit_limit_count']}次")
    print()
    
    # 对比分析
    print("="*80)
    print(" "*30 + "【对比分析】")
    print("="*80)
    print()
    
    profit_diff = stats_8['balance'] - stats_10['balance']
    roi_diff = stats_8['roi'] - stats_10['roi']
    drawdown_diff = stats_8['max_drawdown'] - stats_10['max_drawdown']
    risk_reward_diff = stats_8['risk_reward'] - stats_10['risk_reward']
    bet_diff = stats_8['total_bet'] - stats_10['total_bet']
    
    print(f"{'指标':<20} {'10倍':<15} {'8倍':<15} {'差异':<20}")
    print("-"*80)
    print(f"{'净利润':<20} {stats_10['balance']:>14.0f} {stats_8['balance']:>14.0f} {profit_diff:>+19.0f}")
    print(f"{'ROI':<20} {stats_10['roi']:>13.2f}% {stats_8['roi']:>13.2f}% {roi_diff:>+18.2f}%")
    print(f"{'最大回撤':<20} {stats_10['max_drawdown']:>14.0f} {stats_8['max_drawdown']:>14.0f} {drawdown_diff:>+19.0f}")
    print(f"{'风险收益比':<20} {stats_10['risk_reward']:>14.2f} {stats_8['risk_reward']:>14.2f} {risk_reward_diff:>+19.2f}")
    print(f"{'总投注':<20} {stats_10['total_bet']:>14.0f} {stats_8['total_bet']:>14.0f} {bet_diff:>+19.0f}")
    print(f"{'达到限制次数':<20} {stats_10['hit_limit_count']:>14} {stats_8['hit_limit_count']:>14} {stats_8['hit_limit_count']-stats_10['hit_limit_count']:>+19}")
    print()
    
    # 百分比变化
    print("="*80)
    print(" "*30 + "【8倍相对10倍的变化】")
    print("="*80)
    print()
    
    profit_change = (profit_diff / abs(stats_10['balance']) * 100) if stats_10['balance'] != 0 else 0
    drawdown_change = (drawdown_diff / stats_10['max_drawdown'] * 100) if stats_10['max_drawdown'] > 0 else 0
    bet_change = (bet_diff / stats_10['total_bet'] * 100) if stats_10['total_bet'] > 0 else 0
    
    print(f"净利润变化: {profit_diff:+.0f}元 ({profit_change:+.1f}%)")
    print(f"ROI变化: {roi_diff:+.2f}个百分点")
    print(f"回撤变化: {drawdown_diff:+.0f}元 ({drawdown_change:+.1f}%)")
    print(f"总投注变化: {bet_diff:+.0f}元 ({bet_change:+.1f}%)")
    print()
    
    # 结论
    print("="*80)
    print(" "*35 + "【结论】")
    print("="*80)
    print()
    
    if drawdown_diff < 0:
        print(f"✅ 回撤降低: 8倍比10倍减少{abs(drawdown_diff):.0f}元回撤 ({abs(drawdown_change):.1f}%)")
    else:
        print(f"⚠️ 回撤增加: 8倍比10倍增加{drawdown_diff:.0f}元回撤 ({drawdown_change:.1f}%)")
    
    if profit_diff > 0:
        print(f"✅ 收益提升: 8倍比10倍多赚{profit_diff:.0f}元 ({profit_change:.1f}%)")
    else:
        print(f"⚠️ 收益降低: 8倍比10倍少赚{abs(profit_diff):.0f}元 ({abs(profit_change):.1f}%)")
    
    if roi_diff > 0:
        print(f"✅ ROI提升: 8倍比10倍高{roi_diff:.2f}个百分点")
    else:
        print(f"⚠️ ROI降低: 8倍比10倍低{abs(roi_diff):.2f}个百分点")
    
    if risk_reward_diff > 0:
        print(f"✅ 风险收益比提升: 8倍({stats_8['risk_reward']:.2f}) > 10倍({stats_10['risk_reward']:.2f})")
    else:
        print(f"⚠️ 风险收益比降低: 8倍({stats_8['risk_reward']:.2f}) < 10倍({stats_10['risk_reward']:.2f})")
    
    print()
    print("📊 关键发现:")
    print(f"  - 斐波那契数列：[1,1,2,3,5,8,13,21,34,55,89,144]")
    print(f"  - 8倍限制会在Fib索引≥5时生效 (fib[5]=8)")
    print(f"  - 10倍限制会在Fib索引≥6时生效 (fib[6]=13)")
    print(f"  - 差异主要体现在连败6-9次的情况")
    print()
    
    if abs(drawdown_change) > 10:
        print("💡 建议：")
        if drawdown_diff < 0:
            print("  8倍限制能显著降低回撤，适合风险厌恶型投资者")
            if profit_change > -5:
                print("  收益损失较小，风险收益比更优")
        else:
            print("  10倍限制虽然回撤较大，但可能带来更高收益")
    else:
        print("💡 建议：两种限制差异不大，选择任一配置均可")
    
    print()
    print("="*80)
    
    # 保存结果
    df_10 = pd.DataFrame(results_10)
    df_10.to_csv('max_mult_10_results.csv', index=False, encoding='utf-8-sig')
    
    df_8 = pd.DataFrame(results_8)
    df_8.to_csv('max_mult_8_results.csv', index=False, encoding='utf-8-sig')
    
    print(f"✅ 详细结果已保存:")
    print(f"  - max_mult_10_results.csv (10倍限制)")
    print(f"  - max_mult_8_results.csv (8倍限制)")
    print()


if __name__ == '__main__':
    main()
