"""
精准TOP15投注策略优化测试
对比三种方案的收益：
1. 原方案：纯斐波那契倍投
2. 方案1：连续5期未中 → 暂停2期（第6、7期） → 第8期恢复
3. 方案2：连续5期未中 → 倍数重置为1倍重新开始
"""

import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor


def fibonacci_multiplier(losses):
    """斐波那契数列倍数"""
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    if losses < len(fib):
        return fib[losses]
    return fib[-1]


def test_original_strategy(hit_records, base_bet=15, win_amount=47):
    """原方案：纯斐波那契倍投"""
    total_profit = 0
    total_investment = 0
    consecutive_losses = 0
    max_consecutive_losses = 0
    max_bet = base_bet
    max_drawdown = 0
    peak_balance = 0
    hits = 0
    
    balance_history = []
    period_details = []
    
    for i, hit in enumerate(hit_records):
        multiplier = fibonacci_multiplier(consecutive_losses)
        current_bet = base_bet * multiplier
        total_investment += current_bet
        max_bet = max(max_bet, current_bet)
        
        if hit:
            profit = win_amount * multiplier - current_bet
            total_profit += profit
            consecutive_losses = 0
            hits += 1
            status = '✓中'
        else:
            profit = -current_bet
            total_profit += profit
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            status = '✗失'
        
        balance_history.append(total_profit)
        peak_balance = max(peak_balance, total_profit)
        drawdown = peak_balance - total_profit
        max_drawdown = max(max_drawdown, drawdown)
        
        period_details.append({
            'period': i + 1,
            'multiplier': multiplier,
            'bet': current_bet,
            'status': status,
            'profit': profit,
            'cumulative': total_profit,
            'consecutive_losses': consecutive_losses
        })
    
    roi = (total_profit / total_investment * 100) if total_investment > 0 else 0
    hit_rate = hits / len(hit_records) if len(hit_records) > 0 else 0
    
    return {
        'total_profit': total_profit,
        'total_investment': total_investment,
        'roi': roi,
        'hit_rate': hit_rate * 100,
        'hits': hits,
        'max_consecutive_losses': max_consecutive_losses,
        'max_bet': max_bet,
        'max_drawdown': max_drawdown,
        'balance_history': balance_history,
        'period_details': period_details
    }


def test_pause_strategy(hit_records, base_bet=15, win_amount=47):
    """方案1：连续5期未中 → 暂停2期 → 第8期恢复"""
    total_profit = 0
    total_investment = 0
    consecutive_losses = 0
    max_consecutive_losses = 0
    max_bet = base_bet
    max_drawdown = 0
    peak_balance = 0
    hits = 0
    
    is_paused = False
    pause_remaining = 0
    paused_periods = 0
    actual_betting_periods = 0
    
    balance_history = []
    period_details = []
    
    for i, hit in enumerate(hit_records):
        if is_paused:
            # 暂停期间不投注
            pause_remaining -= 1
            paused_periods += 1
            
            if pause_remaining <= 0:
                # 暂停期满，恢复投注，倍数重置
                is_paused = False
                consecutive_losses = 0
            
            balance_history.append(total_profit)
            drawdown = peak_balance - total_profit
            max_drawdown = max(max_drawdown, drawdown)
            
            period_details.append({
                'period': i + 1,
                'multiplier': 0,
                'bet': 0,
                'status': f'⏸暂停({2-pause_remaining}/2)',
                'profit': 0,
                'cumulative': total_profit,
                'consecutive_losses': consecutive_losses
            })
            
            continue
        
        # 正常投注期
        multiplier = fibonacci_multiplier(consecutive_losses)
        current_bet = base_bet * multiplier
        total_investment += current_bet
        actual_betting_periods += 1
        max_bet = max(max_bet, current_bet)
        
        if hit:
            profit = win_amount * multiplier - current_bet
            total_profit += profit
            consecutive_losses = 0
            hits += 1
            status = '✓中'
        else:
            profit = -current_bet
            total_profit += profit
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            status = '✗失'
            
            # 检查是否触发暂停
            if consecutive_losses >= 5:
                is_paused = True
                pause_remaining = 2  # 暂停2期
                status = '✗失(触发暂停)'
        
        balance_history.append(total_profit)
        peak_balance = max(peak_balance, total_profit)
        drawdown = peak_balance - total_profit
        max_drawdown = max(max_drawdown, drawdown)
        
        period_details.append({
            'period': i + 1,
            'multiplier': multiplier,
            'bet': current_bet,
            'status': status,
            'profit': profit,
            'cumulative': total_profit,
            'consecutive_losses': consecutive_losses
        })
    
    roi = (total_profit / total_investment * 100) if total_investment > 0 else 0
    hit_rate = hits / actual_betting_periods if actual_betting_periods > 0 else 0
    
    return {
        'total_profit': total_profit,
        'total_investment': total_investment,
        'roi': roi,
        'hit_rate': hit_rate * 100,
        'hits': hits,
        'actual_betting_periods': actual_betting_periods,
        'paused_periods': paused_periods,
        'max_consecutive_losses': max_consecutive_losses,
        'max_bet': max_bet,
        'max_drawdown': max_drawdown,
        'balance_history': balance_history,
        'period_details': period_details
    }


def test_reset_strategy(hit_records, base_bet=15, win_amount=47):
    """方案2：连续5期未中 → 倍数重置为1倍"""
    total_profit = 0
    total_investment = 0
    consecutive_losses = 0
    max_consecutive_losses = 0
    max_bet = base_bet
    max_drawdown = 0
    peak_balance = 0
    hits = 0
    reset_count = 0
    
    balance_history = []
    period_details = []
    
    for i, hit in enumerate(hit_records):
        multiplier = fibonacci_multiplier(consecutive_losses)
        current_bet = base_bet * multiplier
        total_investment += current_bet
        max_bet = max(max_bet, current_bet)
        
        if hit:
            profit = win_amount * multiplier - current_bet
            total_profit += profit
            consecutive_losses = 0
            hits += 1
            status = '✓中'
        else:
            profit = -current_bet
            total_profit += profit
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            
            # 检查是否触发重置
            if consecutive_losses >= 5:
                consecutive_losses = 0  # 重置倍数
                reset_count += 1
                status = '✗失(倍数重置)'
            else:
                status = '✗失'
        
        balance_history.append(total_profit)
        peak_balance = max(peak_balance, total_profit)
        drawdown = peak_balance - total_profit
        max_drawdown = max(max_drawdown, drawdown)
        
        period_details.append({
            'period': i + 1,
            'multiplier': multiplier,
            'bet': current_bet,
            'status': status,
            'profit': profit,
            'cumulative': total_profit,
            'consecutive_losses': consecutive_losses,
            'reset_count': reset_count
        })
    
    roi = (total_profit / total_investment * 100) if total_investment > 0 else 0
    hit_rate = hits / len(hit_records) if len(hit_records) > 0 else 0
    
    return {
        'total_profit': total_profit,
        'total_investment': total_investment,
        'roi': roi,
        'hit_rate': hit_rate * 100,
        'hits': hits,
        'reset_count': reset_count,
        'max_consecutive_losses': max_consecutive_losses,
        'max_bet': max_bet,
        'max_drawdown': max_drawdown,
        'balance_history': balance_history,
        'period_details': period_details
    }


def main():
    print("="*100)
    print("💎 精准TOP15投注策略优化测试")
    print("="*100)
    print("对比三种方案：")
    print("  1. 原方案：纯斐波那契倍投")
    print("  2. 方案1：连续5期未中 → 暂停2期（第6、7期） → 第8期恢复")
    print("  3. 方案2：连续5期未中 → 倍数重置为1倍重新开始")
    print()
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"✅ 数据加载完成: {len(df)}期\n")
    
    # 测试最近300期
    test_periods = min(300, len(df))
    start_idx = len(df) - test_periods
    
    print(f"测试期数: {test_periods}期 (第{start_idx+1}期 ~ 第{len(df)}期)")
    print(f"日期范围: {df.iloc[start_idx]['date']} ~ {df.iloc[-1]['date']}\n")
    
    # 使用精准TOP15预测器生成预测
    print("🔄 使用精准TOP15预测器生成预测...")
    predictor = PreciseTop15Predictor()
    
    predictions_top15 = []
    hit_records = []
    dates = []
    actuals = []
    
    for i in range(start_idx, len(df)):
        train_data = df.iloc[:i]['number'].values
        top15 = predictor.predict(train_data)
        predictions_top15.append(top15)
        
        actual = df.iloc[i]['number']
        actuals.append(actual)
        hit = actual in top15
        hit_records.append(hit)
        
        predictor.update_performance(top15, actual)
        dates.append(df.iloc[i]['date'])
        
        if (i - start_idx + 1) % 50 == 0:
            print(f"  已处理 {i - start_idx + 1}/{test_periods} 期...")
    
    hits = sum(hit_records)
    hit_rate = hits / len(hit_records)
    print(f"✅ 预测完成！命中率: {hit_rate*100:.2f}% ({hits}/{len(hit_records)})\n")
    
    # 测试三种方案
    print("="*100)
    print("【方案对比测试】")
    print("="*100)
    print()
    
    # 原方案
    print("【原方案】纯斐波那契倍投")
    print("-"*100)
    original_result = test_original_strategy(hit_records)
    print(f"总投入: {original_result['total_investment']:.2f}元")
    print(f"总收益: {original_result['total_profit']:+.2f}元")
    print(f"ROI: {original_result['roi']:+.2f}%")
    print(f"命中率: {original_result['hit_rate']:.2f}%")
    print(f"最大连败: {original_result['max_consecutive_losses']}期")
    print(f"最大单期投入: {original_result['max_bet']:.2f}元")
    print(f"最大回撤: {original_result['max_drawdown']:.2f}元")
    print()
    
    # 方案1：暂停策略
    print("【方案1】连续5期未中 → 暂停2期 → 第8期恢复")
    print("-"*100)
    pause_result = test_pause_strategy(hit_records)
    print(f"总投入: {pause_result['total_investment']:.2f}元")
    print(f"总收益: {pause_result['total_profit']:+.2f}元")
    print(f"ROI: {pause_result['roi']:+.2f}%")
    print(f"命中率: {pause_result['hit_rate']:.2f}%")
    print(f"实际投注: {pause_result['actual_betting_periods']}期")
    print(f"暂停期数: {pause_result['paused_periods']}期 ({pause_result['paused_periods']/test_periods*100:.1f}%)")
    print(f"最大连败: {pause_result['max_consecutive_losses']}期")
    print(f"最大单期投入: {pause_result['max_bet']:.2f}元")
    print(f"最大回撤: {pause_result['max_drawdown']:.2f}元")
    print(f"vs原方案: ROI {pause_result['roi'] - original_result['roi']:+.2f}%  |  收益 {pause_result['total_profit'] - original_result['total_profit']:+.2f}元")
    print()
    
    # 方案2：重置策略
    print("【方案2】连续5期未中 → 倍数重置为1倍")
    print("-"*100)
    reset_result = test_reset_strategy(hit_records)
    print(f"总投入: {reset_result['total_investment']:.2f}元")
    print(f"总收益: {reset_result['total_profit']:+.2f}元")
    print(f"ROI: {reset_result['roi']:+.2f}%")
    print(f"命中率: {reset_result['hit_rate']:.2f}%")
    print(f"重置次数: {reset_result['reset_count']}次")
    print(f"最大连败: {reset_result['max_consecutive_losses']}期")
    print(f"最大单期投入: {reset_result['max_bet']:.2f}元")
    print(f"最大回撤: {reset_result['max_drawdown']:.2f}元")
    print(f"vs原方案: ROI {reset_result['roi'] - original_result['roi']:+.2f}%  |  收益 {reset_result['total_profit'] - original_result['total_profit']:+.2f}元")
    print()
    
    # 按ROI排序
    print("="*100)
    print("【结果排名】按ROI排序")
    print("="*100)
    
    results = [
        ('原方案（纯倍投）', original_result),
        ('方案1（暂停2期）', pause_result),
        ('方案2（倍数重置）', reset_result)
    ]
    
    sorted_results = sorted(results, key=lambda x: x[1]['roi'], reverse=True)
    
    print(f"{'排名':<8} {'方案':<25} {'ROI':<15} {'收益':<15} {'投入':<15} {'回撤':<15}")
    print("-" * 100)
    
    for i, (name, result) in enumerate(sorted_results):
        rank = f"#{i+1}"
        marker = " ⭐⭐⭐" if i == 0 else " ⭐⭐" if i == 1 else " ⭐"
        
        print(f"{rank:<8} {name:<25} {result['roi']:>+7.2f}%{marker:<6} {result['total_profit']:>+10.2f}元  "
              f"{result['total_investment']:>10.2f}元  {result['max_drawdown']:>10.2f}元")
    
    print()
    
    # 详细对比分析
    print("="*100)
    print("【详细对比分析】")
    print("="*100)
    print()
    
    best_strategy = sorted_results[0]
    best_name = best_strategy[0]
    best_result = best_strategy[1]
    
    print(f"🏆 最优方案: {best_name}")
    print(f"   ROI: {best_result['roi']:+.2f}%")
    print(f"   总收益: {best_result['total_profit']:+.2f}元")
    print()
    
    # 对比表格
    print("详细指标对比：")
    print()
    print(f"{'指标':<20} {'原方案':<20} {'方案1(暂停)':<20} {'方案2(重置)':<20}")
    print("-" * 85)
    
    metrics = [
        ('ROI', 'roi', '%', True),
        ('总收益', 'total_profit', '元', True),
        ('总投入', 'total_investment', '元', False),
        ('命中率', 'hit_rate', '%', False),
        ('最大连败', 'max_consecutive_losses', '期', False),
        ('最大单期投入', 'max_bet', '元', False),
        ('最大回撤', 'max_drawdown', '元', False),
    ]
    
    for metric_name, key, unit, show_plus in metrics:
        orig_val = original_result[key]
        pause_val = pause_result[key]
        reset_val = reset_result[key]
        
        if show_plus:
            orig_str = f"{orig_val:+.2f}{unit}"
            pause_str = f"{pause_val:+.2f}{unit}"
            reset_str = f"{reset_val:+.2f}{unit}"
        else:
            orig_str = f"{orig_val:.2f}{unit}"
            pause_str = f"{pause_val:.2f}{unit}"
            reset_str = f"{reset_val:.2f}{unit}"
        
        print(f"{metric_name:<20} {orig_str:<20} {pause_str:<20} {reset_str:<20}")
    
    # 额外指标
    if 'paused_periods' in pause_result:
        print(f"{'暂停期数':<20} {'-':<20} {pause_result['paused_periods']}期 ({pause_result['paused_periods']/test_periods*100:.1f}%){'':>5} {'-':<20}")
    
    if 'reset_count' in reset_result:
        print(f"{'重置次数':<20} {'-':<20} {'-':<20} {reset_result['reset_count']}次{'':>15}")
    
    print()
    
    # 结论
    print("="*100)
    print("【分析结论】")
    print("="*100)
    print()
    
    if pause_result['roi'] > original_result['roi'] or reset_result['roi'] > original_result['roi']:
        print("✅ 优化策略有效！")
        if pause_result['roi'] > reset_result['roi']:
            print(f"   推荐使用：方案1（暂停2期策略）")
            print(f"   ROI提升: {pause_result['roi'] - original_result['roi']:+.2f}%")
            print(f"   优势: 通过暂停避免继续在不利期投注，有效降低风险")
        else:
            print(f"   推荐使用：方案2（倍数重置策略）")
            print(f"   ROI提升: {reset_result['roi'] - original_result['roi']:+.2f}%")
            print(f"   优势: 通过重置倍数避免高倍投注的大额亏损")
    else:
        print("⚠️ 优化策略未能提升ROI")
        print(f"   原方案仍然是最优选择")
        print(f"   原因分析：")
        if pause_result['roi'] < original_result['roi']:
            print(f"     - 暂停策略：暂停期间错过了{pause_result['paused_periods']}期，可能错过恢复机会")
        if reset_result['roi'] < original_result['roi']:
            print(f"     - 重置策略：频繁重置倍数导致命中时收益不足以弥补亏损")
    
    print()
    print("💡 策略建议：")
    print(f"   • 当前命中率: {hit_rate*100:.2f}%")
    print(f"   • 连续5期未中是重要的风险信号")
    if best_result['roi'] > 30:
        print(f"   • {best_name} 能够有效控制风险，推荐使用")
    else:
        print(f"   • 即使采用最优策略，ROI仍然有限")
        print(f"   • 根本提升需要提高预测准确率")
    
    print()
    print("="*100)
    print("测试完成！")
    print("="*100)


if __name__ == '__main__':
    main()
