"""
验证时序修复：对比修复前后2026/1/20等关键期数的倍数变化
"""

import pandas as pd
from precise_top15_predictor import PreciseTop15Predictor


def compare_timing():
    """对比正确时序和错误时序的差异"""
    
    print("=" * 100)
    print("验证时序修复：对比修复前后的倍数计算")
    print("=" * 100)
    print()
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    test_periods = 300
    start_idx = len(df) - test_periods
    
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
    
    # ===== 方法1：正确时序（先计算再更新）=====
    print("【方法1：正确时序】先计算倍数（基于投注前历史），再更新历史")
    print("-" * 100)
    
    predictor1 = PreciseTop15Predictor()
    recent_results1 = []
    fib_index1 = 0
    balance1 = 0
    total_bet1 = 0
    results_correct = []
    
    for i in range(start_idx, len(df)):
        period_num = i - start_idx + 1
        train_data = df.iloc[:i]['number'].values
        predictions = predictor1.predict(train_data)
        actual = df.iloc[i]['number']
        date = df.iloc[i]['date']
        hit = actual in predictions
        
        predictor1.update_performance(predictions, actual)
        
        # 先计算倍数（基于投注前的历史）
        if fib_index1 >= len(fib_sequence):
            base_mult = min(fib_sequence[-1], config['max_multiplier'])
        else:
            base_mult = min(fib_sequence[fib_index1], config['max_multiplier'])
        
        if len(recent_results1) >= config['lookback']:
            recent_hits = sum(recent_results1[-config['lookback']:])
            rate = recent_hits / config['lookback']
            
            if rate >= config['good_thresh']:
                multiplier = min(base_mult * config['boost_mult'], config['max_multiplier'])
                adj_type = "增强"
            elif rate <= config['bad_thresh']:
                multiplier = max(base_mult * config['reduce_mult'], 1)
                adj_type = "降低"
            else:
                multiplier = base_mult
                adj_type = "保持"
        else:
            multiplier = base_mult
            rate = sum(recent_results1) / len(recent_results1) if recent_results1 else 0
            adj_type = "不足"
        
        bet = config['base_bet'] * multiplier
        total_bet1 += bet
        
        if hit:
            profit = config['win_reward'] * multiplier - bet
            balance1 += profit
            fib_index1 = 0
        else:
            profit = -bet
            balance1 += profit
            fib_index1 += 1
        
        results_correct.append({
            'period': period_num,
            'date': date,
            'hit': hit,
            'rate_before': rate,
            'multiplier': multiplier,
            'adj_type': adj_type,
            'profit': profit
        })
        
        # 再更新历史
        recent_results1.append(1 if hit else 0)
    
    # ===== 方法2：错误时序（先更新再计算）=====
    print("【方法2：错误时序（GUI原bug）】先更新历史，再计算倍数（包含未来信息）")
    print("-" * 100)
    
    predictor2 = PreciseTop15Predictor()
    recent_results2 = []
    fib_index2 = 0
    balance2 = 0
    total_bet2 = 0
    results_wrong = []
    
    for i in range(start_idx, len(df)):
        period_num = i - start_idx + 1
        train_data = df.iloc[:i]['number'].values
        predictions = predictor2.predict(train_data)
        actual = df.iloc[i]['number']
        date = df.iloc[i]['date']
        hit = actual in predictions
        
        predictor2.update_performance(predictions, actual)
        
        # 先更新历史（错误：包含了当期结果）
        recent_results2.append(1 if hit else 0)
        
        # 再计算倍数（使用了未来信息）
        if fib_index2 >= len(fib_sequence):
            base_mult = min(fib_sequence[-1], config['max_multiplier'])
        else:
            base_mult = min(fib_sequence[fib_index2], config['max_multiplier'])
        
        if len(recent_results2) >= config['lookback']:
            recent_hits = sum(recent_results2[-config['lookback']:])
            rate = recent_hits / config['lookback']
            
            if rate >= config['good_thresh']:
                multiplier = min(base_mult * config['boost_mult'], config['max_multiplier'])
                adj_type = "增强"
            elif rate <= config['bad_thresh']:
                multiplier = max(base_mult * config['reduce_mult'], 1)
                adj_type = "降低"
            else:
                multiplier = base_mult
                adj_type = "保持"
        else:
            multiplier = base_mult
            rate = sum(recent_results2) / len(recent_results2) if recent_results2 else 0
            adj_type = "不足"
        
        bet = config['base_bet'] * multiplier
        total_bet2 += bet
        
        if hit:
            profit = config['win_reward'] * multiplier - bet
            balance2 += profit
            fib_index2 = 0
        else:
            profit = -bet
            balance2 += profit
            fib_index2 += 1
        
        results_wrong.append({
            'period': period_num,
            'date': date,
            'hit': hit,
            'rate_after': rate,
            'multiplier': multiplier,
            'adj_type': adj_type,
            'profit': profit
        })
    
    # ===== 对比差异 =====
    print()
    print("=" * 100)
    print("【关键差异对比】查找倍数不一致的期数")
    print("=" * 100)
    print()
    
    diff_count = 0
    target_found = False
    
    for i in range(len(results_correct)):
        correct = results_correct[i]
        wrong = results_wrong[i]
        
        if abs(correct['multiplier'] - wrong['multiplier']) > 0.01:
            diff_count += 1
            
            # 重点关注2026/1/20和高倍数差异
            is_target = correct['date'] == '2026/1/20'
            is_high_mult = correct['multiplier'] >= 8 or wrong['multiplier'] >= 8
            
            if is_target or is_high_mult or diff_count <= 20:
                marker = "🎯" if is_target else "⚠️"
                hit_str = "✅" if correct['hit'] else "❌"
                
                print(f"\n{marker} 期号{correct['period']} ({correct['date']}) {hit_str}")
                print(f"   正确时序: 倍数{correct['multiplier']:.2f}x (投注前命中率{correct['rate_before']*100:.1f}%, {correct['adj_type']})")
                print(f"   错误时序: 倍数{wrong['multiplier']:.2f}x (投注后命中率{wrong['rate_after']*100:.1f}%, {wrong['adj_type']})")
                print(f"   差异: {wrong['multiplier'] - correct['multiplier']:+.2f}x")
                print(f"   影响: 盈亏差{wrong['profit'] - correct['profit']:+.0f}元")
                
                if is_target:
                    target_found = True
    
    if not target_found:
        print("\n⚠️ 未找到2026/1/20，检查数据范围...")
    
    print()
    print("=" * 100)
    print("【整体影响统计】")
    print("=" * 100)
    print()
    
    print(f"不一致期数: {diff_count}/{len(results_correct)} ({diff_count/len(results_correct)*100:.1f}%)")
    print()
    
    print(f"正确时序: ROI {balance1/total_bet1*100:.2f}%, 净收益{balance1:+.0f}元, 总投注{total_bet1:.0f}元")
    print(f"错误时序: ROI {balance2/total_bet2*100:.2f}%, 净收益{balance2:+.0f}元, 总投注{total_bet2:.0f}元")
    print()
    
    print(f"差异影响:")
    print(f"  ROI差异: {(balance2/total_bet2 - balance1/total_bet1)*100:+.2f}%")
    print(f"  收益差异: {balance2 - balance1:+.0f}元")
    print(f"  投注差异: {total_bet2 - total_bet1:+.0f}元")
    print()
    
    # 找出所有≥8倍的期数
    print("=" * 100)
    print("【所有≥8倍投注对比】")
    print("=" * 100)
    print()
    
    high_mult_correct = [(r, w) for r, w in zip(results_correct, results_wrong) if r['multiplier'] >= 8 or w['multiplier'] >= 8]
    
    if high_mult_correct:
        print(f"{'期号':<6} {'日期':<12} {'结果':<6} {'正确倍数':<10} {'错误倍数':<10} {'差异':<8}")
        print("-" * 70)
        
        for correct, wrong in high_mult_correct:
            hit_str = "✅命中" if correct['hit'] else "❌未中"
            marker = "🎯" if correct['date'] == '2026/1/20' else "  "
            
            print(f"{marker} {correct['period']:<6} {correct['date']:<12} {hit_str:<6} "
                  f"{correct['multiplier']:<10.2f} {wrong['multiplier']:<10.2f} "
                  f"{wrong['multiplier'] - correct['multiplier']:>+8.2f}")
    
    print()


if __name__ == '__main__':
    compare_timing()
