"""
当前精准TOP15方案回测 - 最近300期详情
精准TOP15预测器 + 原始Fibonacci倍投
"""

import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor

def backtest_current_strategy(test_periods=300):
    """回测当前精准TOP15方案"""
    print("="*100)
    print(f"当前方案回测：精准TOP15预测器 + Fibonacci倍投 - 最近{test_periods}期")
    print("="*100)
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv')
    numbers = df['number'].tolist()
    
    total = len(numbers)
    start = total - test_periods
    
    print(f"\n数据范围:")
    print(f"  总期数: {total}期")
    print(f"  测试期数: {test_periods}期 (第{start+1}期 到 第{total}期)")
    
    # 初始化预测器
    predictor = PreciseTop15Predictor()
    
    # Fibonacci序列
    fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    
    # 状态变量
    consecutive_miss = 0
    fib_index = 0
    total_bet = 0
    total_win = 0
    balance = 0
    max_drawdown = 0
    min_balance = 0
    
    base_bet = 15
    win_reward = 47
    
    # 详细记录
    details = []
    hit_count = 0
    
    print("\n开始回测...")
    print("-" * 100)
    
    for i in range(start, total):
        period_num = i - start + 1
        history = numbers[:i]
        actual = numbers[i]
        
        # 预测
        predictions = predictor.predict(history)
        hit = actual in predictions
        
        # 获取当前倍数
        if fib_index >= len(fib_sequence):
            multiplier = fib_sequence[-1]
        else:
            multiplier = fib_sequence[fib_index]
        
        # 计算投注和收益
        bet = base_bet * multiplier
        total_bet += bet
        
        if hit:
            win = win_reward * multiplier
            total_win += win
            balance += (win - bet)
            profit_this_period = win - bet
            consecutive_miss = 0
            fib_index = 0
            hit_count += 1
        else:
            win = 0
            balance -= bet
            profit_this_period = -bet
            consecutive_miss += 1
            fib_index += 1
        
        # 更新最大回撤
        if balance < min_balance:
            min_balance = balance
            max_drawdown = abs(min_balance)
        
        # 记录详情
        details.append({
            '期数': period_num,
            '实际号码': actual,
            '命中': '✓' if hit else '✗',
            '连续失败': consecutive_miss if not hit else 0,
            'Fib档位': fib_index if not hit else 0,
            '倍数': multiplier,
            '投注额': bet,
            '奖金': win,
            '本期盈亏': profit_this_period,
            '累计余额': balance,
            '最大回撤': max_drawdown
        })
        
        # 每50期显示一次进度
        if period_num % 50 == 0:
            print(f"已完成: {period_num}/{test_periods}期")
    
    print("\n回测完成！")
    print("="*100)
    
    # 统计汇总
    hit_rate = hit_count / test_periods * 100
    roi = (balance / total_bet * 100) if total_bet > 0 else 0
    
    print("\n【统计汇总】")
    print("-" * 100)
    print(f"测试期数: {test_periods}期")
    print(f"命中次数: {hit_count}次")
    print(f"命中率: {hit_rate:.2f}%")
    print(f"总投入: {total_bet:.0f}元")
    print(f"总收益: {total_win:.0f}元")
    print(f"净利润: {balance:.0f}元")
    print(f"ROI: {roi:+.2f}%")
    print(f"最大回撤: {max_drawdown:.0f}元")
    
    # 连续失败分析
    miss_segments = []
    current_miss = 0
    for detail in details:
        if detail['命中'] == '✗':
            current_miss += 1
        else:
            if current_miss > 0:
                miss_segments.append(current_miss)
            current_miss = 0
    if current_miss > 0:
        miss_segments.append(current_miss)
    
    if miss_segments:
        print(f"\n【连续失败分析】")
        print(f"最大连续失败: {max(miss_segments)}期")
        print(f"平均连续失败: {np.mean(miss_segments):.2f}期")
        print(f"≥5期长连败: {len([s for s in miss_segments if s >= 5])}次")
        print(f"≥7期长连败: {len([s for s in miss_segments if s >= 7])}次")
        print(f"≥9期长连败: {len([s for s in miss_segments if s >= 9])}次")
    
    # 盈亏分布
    profits = [d['本期盈亏'] for d in details]
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p < 0]
    
    print(f"\n【盈亏分布】")
    print(f"盈利期数: {len(wins)}期，平均盈利: {np.mean(wins):.0f}元")
    print(f"亏损期数: {len(losses)}期，平均亏损: {np.mean(losses):.0f}元")
    print(f"盈亏比: {abs(np.mean(wins)/np.mean(losses)):.2f}" if losses else "N/A")
    
    # 最大单期
    print(f"\n【极值记录】")
    max_profit_detail = max(details, key=lambda x: x['本期盈亏'])
    max_loss_detail = min(details, key=lambda x: x['本期盈亏'])
    max_bet_detail = max(details, key=lambda x: x['投注额'])
    
    print(f"最大单期盈利: {max_profit_detail['本期盈亏']:.0f}元 (第{max_profit_detail['期数']}期, 倍数{max_profit_detail['倍数']})")
    print(f"最大单期亏损: {max_loss_detail['本期盈亏']:.0f}元 (第{max_loss_detail['期数']}期, 倍数{max_loss_detail['倍数']})")
    print(f"最高投注额: {max_bet_detail['投注额']:.0f}元 (第{max_bet_detail['期数']}期, 倍数{max_bet_detail['倍数']})")
    
    # 保存详细数据
    details_df = pd.DataFrame(details)
    filename = f'current_top15_strategy_{test_periods}periods_detail.csv'
    details_df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"\n详细数据已保存到: {filename}")
    print("="*100)
    
    # 显示前10期和后10期
    print("\n【前10期详情】")
    print("-" * 100)
    print(details_df.head(10).to_string(index=False))
    
    print("\n【后10期详情】")
    print("-" * 100)
    print(details_df.tail(10).to_string(index=False))
    
    return details_df


if __name__ == "__main__":
    result_df = backtest_current_strategy(test_periods=300)
    
    print("\n" + "="*100)
    print("回测完成！详细数据已保存。")
    print("="*100)
