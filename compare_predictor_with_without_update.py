"""
对比修复前后的预测器性能差异
修复前：不调用 update_performance（无记忆模式）
修复后：调用 update_performance（带学习功能）
"""

import pandas as pd
import numpy as np
from precise_top15_predictor import PreciseTop15Predictor

def test_without_update(df, start_idx):
    """模拟修复前的逻辑（不调用update_performance）"""
    predictor = PreciseTop15Predictor()
    
    results = []
    consecutive_misses = 0
    max_consecutive_misses = 0
    
    for i in range(start_idx, len(df)):
        train_data = df.iloc[:i]['number'].values
        predictions = predictor.predict(train_data)
        actual = df.iloc[i]['number']
        hit = actual in predictions
        
        # 不调用 update_performance！
        
        if hit:
            consecutive_misses = 0
        else:
            consecutive_misses += 1
            max_consecutive_misses = max(max_consecutive_misses, consecutive_misses)
        
        results.append({
            'period': i - start_idx + 1,
            'date': df.iloc[i]['date'],
            'actual': actual,
            'predictions': predictions,
            'hit': hit
        })
    
    hits = sum(1 for r in results if r['hit'])
    hit_rate = hits / len(results) if results else 0
    
    return {
        'results': results,
        'hits': hits,
        'hit_rate': hit_rate,
        'max_consecutive_misses': max_consecutive_misses
    }


def test_with_update(df, start_idx):
    """模拟修复后的逻辑（调用update_performance）"""
    predictor = PreciseTop15Predictor()
    
    results = []
    max_consecutive_misses = 0
    
    for i in range(start_idx, len(df)):
        train_data = df.iloc[:i]['number'].values
        predictions = predictor.predict(train_data)
        actual = df.iloc[i]['number']
        hit = actual in predictions
        
        # 调用 update_performance（关键差异）
        predictor.update_performance(predictions, actual)
        
        results.append({
            'period': i - start_idx + 1,
            'date': df.iloc[i]['date'],
            'actual': actual,
            'predictions': predictions,
            'hit': hit
        })
    
    hits = sum(1 for r in results if r['hit'])
    hit_rate = hits / len(results) if results else 0
    max_consecutive_misses = predictor.consecutive_misses
    
    # 找最大连续未中
    consecutive = 0
    for r in results:
        if not r['hit']:
            consecutive += 1
            max_consecutive_misses = max(max_consecutive_misses, consecutive)
        else:
            consecutive = 0
    
    return {
        'results': results,
        'hits': hits,
        'hit_rate': hit_rate,
        'max_consecutive_misses': max_consecutive_misses
    }


def calculate_smart_betting(results):
    """使用智能动态倍投策略计算收益"""
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
    
    recent_results = []
    balance = 0
    total_bet = 0
    total_win = 0
    min_balance = 0
    max_drawdown = 0
    fib_index = 0
    
    betting_results = []
    
    for r in results:
        hit = r['hit']
        
        # 获取基础倍数
        if fib_index >= len(fib_sequence):
            base_mult = min(fib_sequence[-1], config['max_multiplier'])
        else:
            base_mult = min(fib_sequence[fib_index], config['max_multiplier'])
        
        # 动态调整
        if len(recent_results) >= config['lookback']:
            recent_hits = sum(recent_results[-config['lookback']:])
            rate = recent_hits / config['lookback']
            
            if rate >= config['good_thresh']:
                multiplier = min(base_mult * config['boost_mult'], config['max_multiplier'])
            elif rate <= config['bad_thresh']:
                multiplier = max(base_mult * config['reduce_mult'], 1)
            else:
                multiplier = base_mult
        else:
            multiplier = base_mult
        
        # 计算投注和收益
        bet = config['base_bet'] * multiplier
        total_bet += bet
        
        if hit:
            win = config['win_reward'] * multiplier
            total_win += win
            profit = win - bet
            balance += profit
            fib_index = 0
            recent_results.append(1)
        else:
            profit = -bet
            balance += profit
            fib_index += 1
            recent_results.append(0)
        
        if balance < min_balance:
            min_balance = balance
            max_drawdown = abs(min_balance)
        
        betting_results.append({
            'period': r['period'],
            'date': r['date'],
            'hit': hit,
            'multiplier': multiplier,
            'bet': bet,
            'profit': profit,
            'balance': balance
        })
    
    roi = (balance / total_bet * 100) if total_bet > 0 else 0
    
    return {
        'betting_results': betting_results,
        'total_bet': total_bet,
        'total_win': total_win,
        'balance': balance,
        'roi': roi,
        'max_drawdown': max_drawdown
    }


def main():
    print("=" * 80)
    print("修复前后性能对比分析")
    print("=" * 80)
    print()
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"总数据: {len(df)}期")
    
    # 300期回测
    test_periods = 300
    start_idx = len(df) - test_periods
    print(f"回测期数: {test_periods}期")
    print(f"起始日期: {df.iloc[start_idx]['date']}")
    print(f"结束日期: {df.iloc[-1]['date']}")
    print()
    
    # 测试修复前的逻辑（不调用update_performance）
    print("正在测试修复前的逻辑（无记忆模式）...")
    result_before = test_without_update(df, start_idx)
    betting_before = calculate_smart_betting(result_before['results'])
    
    # 测试修复后的逻辑（调用update_performance）
    print("正在测试修复后的逻辑（带学习功能）...")
    result_after = test_with_update(df, start_idx)
    betting_after = calculate_smart_betting(result_after['results'])
    
    print()
    print("=" * 80)
    print("【预测性能对比】")
    print("=" * 80)
    print()
    
    print(f"{'指标':<30} {'修复前（无记忆）':<20} {'修复后（带学习）':<20} {'变化':<15}")
    print("-" * 85)
    
    # 命中率
    before_rate = result_before['hit_rate'] * 100
    after_rate = result_after['hit_rate'] * 100
    rate_diff = after_rate - before_rate
    rate_diff_str = f"{rate_diff:+.2f}%"
    print(f"{'命中率':<30} {before_rate:<20.2f}% {after_rate:<20.2f}% {rate_diff_str:<15}")
    
    # 命中次数
    print(f"{'命中次数':<30} {result_before['hits']:<20} {result_after['hits']:<20} {result_after['hits']-result_before['hits']:+}")
    
    # 最大连续未中
    before_max_miss = result_before['max_consecutive_misses']
    after_max_miss = result_after['max_consecutive_misses']
    print(f"{'最大连续未中':<30} {before_max_miss:<20}期 {after_max_miss:<20}期 {after_max_miss-before_max_miss:+}期")
    
    print()
    print("=" * 80)
    print("【投注收益对比】（智能动态倍投策略）")
    print("=" * 80)
    print()
    
    print(f"{'指标':<30} {'修复前':<20} {'修复后':<20} {'变化':<15}")
    print("-" * 85)
    
    # ROI
    roi_diff = betting_after['roi'] - betting_before['roi']
    roi_diff_str = f"{roi_diff:+.2f}%"
    print(f"{'ROI (投资回报率)':<30} {betting_before['roi']:<20.2f}% {betting_after['roi']:<20.2f}% {roi_diff_str:<15}")
    
    # 净收益
    balance_diff = betting_after['balance'] - betting_before['balance']
    balance_diff_str = f"{balance_diff:+.0f}元"
    print(f"{'净收益':<30} {betting_before['balance']:<20.0f}元 {betting_after['balance']:<20.0f}元 {balance_diff_str:<15}")
    
    # 最大回撤
    drawdown_diff = betting_after['max_drawdown'] - betting_before['max_drawdown']
    drawdown_diff_str = f"{drawdown_diff:+.0f}元"
    print(f"{'最大回撤':<30} {betting_before['max_drawdown']:<20.0f}元 {betting_after['max_drawdown']:<20.0f}元 {drawdown_diff_str:<15}")
    
    # 总投注
    print(f"{'总投注额':<30} {betting_before['total_bet']:<20.0f}元 {betting_after['total_bet']:<20.0f}元 {betting_after['total_bet']-betting_before['total_bet']:+.0f}元")
    
    print()
    print("=" * 80)
    print("【预测差异分析】")
    print("=" * 80)
    print()
    
    # 找出预测不同的期数
    different_predictions = 0
    different_outcomes = 0
    
    for i in range(len(result_before['results'])):
        before_pred = set(result_before['results'][i]['predictions'])
        after_pred = set(result_after['results'][i]['predictions'])
        
        if before_pred != after_pred:
            different_predictions += 1
            
            # 检查结果是否不同
            if result_before['results'][i]['hit'] != result_after['results'][i]['hit']:
                different_outcomes += 1
    
    print(f"预测内容不同的期数: {different_predictions}/{test_periods} ({different_predictions/test_periods*100:.1f}%)")
    print(f"导致命中结果改变的期数: {different_outcomes}/{test_periods} ({different_outcomes/test_periods*100:.1f}%)")
    
    print()
    print("=" * 80)
    print("【结论】")
    print("=" * 80)
    print()
    
    if after_rate > before_rate:
        print(f"✅ 修复后命中率提升 {rate_diff:.2f}%")
    elif after_rate < before_rate:
        print(f"⚠️ 修复后命中率下降 {abs(rate_diff):.2f}%")
    else:
        print(f"➖ 修复前后命中率相同")
    
    if betting_after['roi'] > betting_before['roi']:
        print(f"✅ 修复后ROI提升 {roi_diff:.2f}%")
    elif betting_after['roi'] < betting_before['roi']:
        print(f"⚠️ 修复后ROI下降 {abs(roi_diff):.2f}%")
    else:
        print(f"➖ 修复前后ROI相同")
    
    if betting_after['max_drawdown'] < betting_before['max_drawdown']:
        print(f"✅ 修复后回撤减少 {abs(drawdown_diff):.0f}元 (-{abs(drawdown_diff)/betting_before['max_drawdown']*100:.1f}%)")
    elif betting_after['max_drawdown'] > betting_before['max_drawdown']:
        print(f"⚠️ 修复后回撤增加 {drawdown_diff:.0f}元 (+{drawdown_diff/betting_before['max_drawdown']*100:.1f}%)")
    else:
        print(f"➖ 修复前后回撤相同")
    
    print()
    print("💡 说明:")
    print("  修复前: 预测器每次独立预测，不记录历史错误")
    print("  修复后: 预测器学习历史错误，避免频繁预测错误的号码")
    print("  差异来源: 'method_avoid_recent_misses' 方法只在有历史记录时才生效")
    print()


if __name__ == '__main__':
    main()
