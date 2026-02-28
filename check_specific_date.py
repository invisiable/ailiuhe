"""
检查特定日期的投注详情
"""

import pandas as pd
from precise_top15_predictor import PreciseTop15Predictor


def check_date_details(target_date):
    """检查指定日期的投注详情"""
    
    print("=" * 100)
    print(f"检查 {target_date} 的投注详情")
    print("=" * 100)
    print()
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    
    # 找到目标日期
    target_idx = df[df['date'] == target_date].index
    
    if len(target_idx) == 0:
        print(f"❌ 未找到日期 {target_date}")
        return
    
    target_idx = target_idx[0]
    
    # 计算该期是300期内的第几期
    test_periods = 300
    start_idx = len(df) - test_periods
    
    if target_idx < start_idx:
        print(f"❌ {target_date} 不在最近300期范围内")
        print(f"   300期范围: {df.iloc[start_idx]['date']} ~ {df.iloc[-1]['date']}")
        return
    
    period_num = target_idx - start_idx + 1
    
    print(f"✅ 找到目标期数")
    print(f"   日期: {target_date}")
    print(f"   期号: 第{period_num}期 (300期内)")
    print(f"   数组索引: {target_idx}")
    print()
    
    # 初始化预测器
    predictor = PreciseTop15Predictor()
    
    # 智能动态配置
    config = {
        'base_bet': 15,
        'win_reward': 45,
        'lookback': 8,
        'good_thresh': 0.35,
        'bad_thresh': 0.20,
        'boost_mult': 1.5,
        'reduce_mult': 0.6,
        'max_multiplier': 10
    }
    
    fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    
    # 回测到目标期
    recent_results = []
    balance = 0
    fib_index = 0
    
    for i in range(start_idx, target_idx + 1):
        period_num_current = i - start_idx + 1
        train_data = df.iloc[:i]['number'].values
        predictions = predictor.predict(train_data)
        actual = df.iloc[i]['number']
        date = df.iloc[i]['date']
        hit = actual in predictions
        
        predictor.update_performance(predictions, actual)
        
        # 计算倍数
        if fib_index >= len(fib_sequence):
            base_mult = min(fib_sequence[-1], config['max_multiplier'])
        else:
            base_mult = min(fib_sequence[fib_index], config['max_multiplier'])
        
        # 动态调整
        adjustment_type = "无调整"
        multiplier_before_cap = base_mult
        
        if len(recent_results) >= config['lookback']:
            recent_hits = sum(recent_results[-config['lookback']:])
            rate = recent_hits / config['lookback']
            
            if rate >= config['good_thresh']:
                multiplier_before_cap = base_mult * config['boost_mult']
                adjustment_type = f"增强×{config['boost_mult']}"
                multiplier = min(multiplier_before_cap, config['max_multiplier'])
            elif rate <= config['bad_thresh']:
                multiplier_before_cap = base_mult * config['reduce_mult']
                adjustment_type = f"降低×{config['reduce_mult']}"
                multiplier = max(multiplier_before_cap, 1)
            else:
                multiplier = base_mult
                adjustment_type = "保持基础"
        else:
            multiplier = base_mult
            adjustment_type = "数据不足"
        
        bet = config['base_bet'] * multiplier
        
        if hit:
            win = config['win_reward'] * multiplier
            profit = win - bet
            balance += profit
            fib_index = 0
            recent_results.append(1)
        else:
            profit = -bet
            balance += profit
            fib_index += 1
            recent_results.append(0)
        
        # 如果是目标期，输出详情
        if i == target_idx:
            print(f"【{date} 第{period_num_current}期投注详情】")
            print()
            
            # 连败情况
            consecutive_losses = 0
            for j in range(len(recent_results) - 1, -1, -1):
                if recent_results[j] == 0:
                    consecutive_losses += 1
                else:
                    break
            
            print(f"开奖号码: {actual}")
            print(f"预测号码: {predictions[:10]}")
            print(f"是否命中: {'✅ 命中' if hit else '❌ 未中'}")
            print()
            
            print(f"【倍数计算】")
            print(f"连续未中: {consecutive_losses}期")
            print(f"Fibonacci索引: {fib_index if not hit else 0} → 基础倍数: {base_mult:.2f}x")
            print()
            
            if len(recent_results) >= config['lookback']:
                recent_hits = sum(recent_results[-config['lookback']:])
                rate = recent_hits / config['lookback']
                print(f"最近{config['lookback']}期命中率: {rate*100:.1f}% ({recent_hits}/{config['lookback']})")
                print(f"最近{config['lookback']}期结果: {recent_results[-config['lookback']:]}")
                print(f"  (1=命中, 0=未中)")
                print()
                
                if rate >= config['good_thresh']:
                    print(f"✅ 命中率{rate*100:.1f}% ≥ {config['good_thresh']*100}% → 增强×{config['boost_mult']}")
                    print(f"   调整前倍数: {multiplier_before_cap:.2f}x")
                    print(f"   上限限制: 10x")
                elif rate <= config['bad_thresh']:
                    print(f"⚠️  命中率{rate*100:.1f}% ≤ {config['bad_thresh']*100}% → 降低×{config['reduce_mult']}")
                    print(f"   调整前倍数: {multiplier_before_cap:.2f}x")
                else:
                    print(f"➡️  命中率{rate*100:.1f}% 在{config['bad_thresh']*100}%-{config['good_thresh']*100}%之间 → 保持基础")
            else:
                print(f"⚠️  数据不足{config['lookback']}期，使用基础倍数")
            
            print()
            print(f"【最终倍数】: {multiplier:.2f}x")
            print(f"【投注金额】: {bet:.0f}元")
            print()
            
            print(f"【本期结果】")
            if hit:
                print(f"✅ 命中!")
                print(f"   奖金: {win:.0f}元")
                print(f"   盈利: +{profit:.0f}元")
            else:
                print(f"❌ 未中")
                print(f"   损失: {profit:.0f}元")
            
            print()
            print(f"【累计余额】: {balance:+.0f}元")
            print()
            
            # 判断是否触发10倍
            if multiplier >= 9.99:
                print("🔴 【触发10倍上限】")
            elif multiplier >= 8:
                print(f"🟡 【接近上限】距离10倍还有 {10-multiplier:.2f}x")
            else:
                print(f"🟢 【正常倍数】距离10倍还有 {10-multiplier:.2f}x")
            
            print()
            
            # 前后10期对比
            print("=" * 100)
            print("【前后10期对比】")
            print("=" * 100)
            print()
            
            print(f"{'期号':<6} {'日期':<12} {'开奖':<6} {'倍数':<8} {'结果':<6} {'盈亏':<10} {'余额':<10}")
            print("-" * 100)
            
            # 重新计算前后10期
            predictor2 = PreciseTop15Predictor()
            recent_results2 = []
            balance2 = 0
            fib_index2 = 0
            
            display_start = max(start_idx, target_idx - 10)
            display_end = min(len(df), target_idx + 11)
            
            for i in range(start_idx, display_end):
                period_num_i = i - start_idx + 1
                train_data2 = df.iloc[:i]['number'].values
                predictions2 = predictor2.predict(train_data2)
                actual2 = df.iloc[i]['number']
                date2 = df.iloc[i]['date']
                hit2 = actual2 in predictions2
                
                predictor2.update_performance(predictions2, actual2)
                
                # 计算倍数
                if fib_index2 >= len(fib_sequence):
                    base_mult2 = min(fib_sequence[-1], config['max_multiplier'])
                else:
                    base_mult2 = min(fib_sequence[fib_index2], config['max_multiplier'])
                
                if len(recent_results2) >= config['lookback']:
                    recent_hits2 = sum(recent_results2[-config['lookback']:])
                    rate2 = recent_hits2 / config['lookback']
                    
                    if rate2 >= config['good_thresh']:
                        multiplier2 = min(base_mult2 * config['boost_mult'], config['max_multiplier'])
                    elif rate2 <= config['bad_thresh']:
                        multiplier2 = max(base_mult2 * config['reduce_mult'], 1)
                    else:
                        multiplier2 = base_mult2
                else:
                    multiplier2 = base_mult2
                
                bet2 = config['base_bet'] * multiplier2
                
                if hit2:
                    win2 = config['win_reward'] * multiplier2
                    profit2 = win2 - bet2
                    balance2 += profit2
                    fib_index2 = 0
                    recent_results2.append(1)
                else:
                    profit2 = -bet2
                    balance2 += profit2
                    fib_index2 += 1
                    recent_results2.append(0)
                
                if i >= display_start:
                    marker = "👉" if i == target_idx else "  "
                    result_str = "✅命中" if hit2 else "❌未中"
                    print(f"{marker} {period_num_i:<6} {date2:<12} {actual2:<6} {multiplier2:<8.2f} "
                          f"{result_str:<6} {profit2:>+10.0f} {balance2:>+10.0f}")


if __name__ == '__main__':
    # 检查用户提到的日期
    check_date_details('2026/1/20')
    
    # 也可以检查其他可疑日期
    print("\n\n")
    print("=" * 100)
    print("也可以检查其他日期，例如:")
    print("- 2026/1/20")
    print("- 2025/12/25")
    print("等等...")
