#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生肖TOP5概率预测投注策略验证

对比：
1. 固定倍投（斐波那契）
2. 智能动态v3.2
3. 概率预测动态倍投（新）
"""

import pandas as pd
import sys
from zodiac_simple_smart import ZodiacSimpleSmart
from zodiac_top5_probability_betting import validate_zodiac_probability_strategy


def fibonacci_multiplier(consecutive_losses):
    """斐波那契序列"""
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    if consecutive_losses >= len(fib):
        return min(fib[-1], 10)
    return min(fib[consecutive_losses], 10)


def simulate_fibonacci_betting(hit_records, base_bet=20, win_reward=47):
    """斐波那契倍投策略"""
    total_bet = 0
    total_win = 0
    balance = 0
    min_balance = 0
    consecutive_losses = 0
    history = []
    
    for i, hit in enumerate(hit_records, 1):
        mult = fibonacci_multiplier(consecutive_losses)
        bet = base_bet * mult
        total_bet += bet
        
        if hit:
            win = win_reward * mult
            total_win += win
            profit = win - bet
            balance += profit
            consecutive_losses = 0
        else:
            profit = -bet
            balance += profit
            consecutive_losses += 1
            if balance < min_balance:
                min_balance = balance
        
        history.append({
            'period': i,
            'multiplier': mult,
            'bet': bet,
            'hit': hit,
            'profit': profit,
            'balance': balance
        })
    
    return {
        'total_bet': total_bet,
        'total_win': total_win,
        'balance': balance,
        'max_drawdown': abs(min_balance),
        'history': history
    }


def simulate_smart_dynamic_v32(hit_records, base_bet=20, win_reward=47):
    """智能动态v3.2策略（激进组合）"""
    lookback = 8
    good_thresh = 0.35
    bad_thresh = 0.20
    boost_mult = 1.5
    reduce_mult = 0.5
    max_mult = 10
    
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    fib_index = 0
    
    total_bet = 0
    total_win = 0
    balance = 0
    min_balance = 0
    recent_results = []
    history = []
    
    for i, hit in enumerate(hit_records, 1):
        # 计算近期命中率
        if len(recent_results) >= lookback:
            recent_rate = sum(recent_results[-lookback:]) / lookback
        else:
            recent_rate = sum(recent_results) / len(recent_results) if recent_results else 0.42
        
        # 基础倍数
        base_mult = min(fib[fib_index] if fib_index < len(fib) else fib[-1], max_mult)
        
        # 动态调整
        if recent_rate >= good_thresh:
            mult = min(base_mult * boost_mult, max_mult)
        elif recent_rate <= bad_thresh:
            mult = max(base_mult * reduce_mult, 1)
        else:
            mult = base_mult
        
        bet = base_bet * mult
        total_bet += bet
        
        if hit:
            win = win_reward * mult
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
        
        history.append({
            'period': i,
            'multiplier': mult,
            'bet': bet,
            'hit': hit,
            'profit': profit,
            'balance': balance,
            'recent_rate': recent_rate
        })
    
    return {
        'total_bet': total_bet,
        'total_win': total_win,
        'balance': balance,
        'max_drawdown': abs(min_balance),
        'history': history
    }


def main():
    print("=" * 80)
    print("生肖TOP5概率预测投注策略验证")
    print("=" * 80)
    print()
    
    # 加载数据
    file_path = 'data/lucky_numbers.csv'
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    print(f"数据加载完成: {len(df)}期")
    
    # 验证期数
    test_periods = min(300, len(df))
    print(f"验证期数: 最近{test_periods}期")
    print()
    
    animals = df['animal'].values
    
    # 创建预测器
    predictor = ZodiacSimpleSmart()
    
    print("=" * 80)
    print("第一步：生成生肖TOP5预测")
    print("=" * 80)
    print()
    
    # 生成预测
    start_idx = len(df) - test_periods
    predictions = []
    hit_records = []
    
    print("正在生成预测...")
    for i in range(start_idx, len(df)):
        train_data = animals[:i].tolist()
        result = predictor.predict_from_history(train_data, top_n=5, debug=False)
        top5 = result['top5']
        predictions.append(top5)
        
        actual = animals[i]
        hit = actual in top5
        hit_records.append(hit)
    
    print(f"预测生成完成: {len(predictions)}期")
    print()
    
    # 计算基础命中率
    hits = sum(hit_records)
    hit_rate = hits / len(hit_records)
    
    print(f"基础命中率: {hit_rate*100:.2f}% ({hits}/{len(hit_records)})")
    print()
    
    # ========== 策略对比 ==========
    print("=" * 80)
    print("第二步：三种策略对比验证")
    print("=" * 80)
    print()
    
    # 策略1: 固定倍投（斐波那契）
    print("【策略1：斐波那契固定倍投】")
    fib_result = simulate_fibonacci_betting(hit_records)
    fib_roi = (fib_result['balance'] / fib_result['total_bet'] * 100) if fib_result['total_bet'] > 0 else 0
    
    print(f"  总投注: {fib_result['total_bet']:.0f}元")
    print(f"  总收益: {fib_result['total_win']:.0f}元")
    print(f"  净利润: {fib_result['balance']:+.0f}元")
    print(f"  ROI: {fib_roi:+.2f}%")
    print(f"  最大回撤: {fib_result['max_drawdown']:.0f}元")
    print()
    
    # 策略2: 智能动态v3.2
    print("【策略2：智能动态v3.2（激进组合）】")
    smart_result = simulate_smart_dynamic_v32(hit_records)
    smart_roi = (smart_result['balance'] / smart_result['total_bet'] * 100) if smart_result['total_bet'] > 0 else 0
    
    print(f"  总投注: {smart_result['total_bet']:.0f}元")
    print(f"  总收益: {smart_result['total_win']:.0f}元")
    print(f"  净利润: {smart_result['balance']:+.0f}元")
    print(f"  ROI: {smart_roi:+.2f}%")
    print(f"  最大回撤: {smart_result['max_drawdown']:.0f}元")
    print()
    
    # 策略3: 概率预测动态倍投
    print("【策略3：概率预测动态倍投（新）】")
    prob_result = validate_zodiac_probability_strategy(predictor, animals, test_periods=test_periods)
    
    print(f"  总投注: {prob_result['total_bet']:.0f}元")
    print(f"  总收益: {prob_result['total_win']:.0f}元")
    print(f"  净利润: {prob_result['total_profit']:+.0f}元")
    print(f"  ROI: {prob_result['roi']:+.2f}%")
    print(f"  最大回撤: {prob_result['max_drawdown']:.0f}元")
    
    if prob_result['prediction_accuracy']:
        acc = prob_result['prediction_accuracy']
        print(f"  预测MAE: {acc['mae']:.4f}")
        print(f"  预测RMSE: {acc['rmse']:.4f}")
    print()
    
    # ========== 综合对比 ==========
    print("=" * 80)
    print("第三步：综合对比分析")
    print("=" * 80)
    print()
    
    strategies = [
        {
            'name': '斐波那契',
            'roi': fib_roi,
            'profit': fib_result['balance'],
            'drawdown': fib_result['max_drawdown'],
            'cost': fib_result['total_bet']
        },
        {
            'name': '<智能动态v3.2',
            'roi': smart_roi,
            'profit': smart_result['balance'],
            'drawdown': smart_result['max_drawdown'],
            'cost': smart_result['total_bet']
        },
        {
            'name': '概率预测🔮',
            'roi': prob_result['roi'],
            'profit': prob_result['total_profit'],
            'drawdown': prob_result['max_drawdown'],
            'cost': prob_result['total_bet']
        }
    ]
    
    print(f"{'策略名称':<20} | {'ROI':>8} | {'净利润':>9} | {'回撤':>8} | {'总投注':>9}")
    print("-" * 80)
    
    for s in strategies:
        print(
            f"{s['name']:<20} | {s['roi']:>7.2f}% | {s['profit']:>+8.0f}元 | "
            f"{s['drawdown']:>7.0f}元 | {s['cost']:>8.0f}元"
        )
    
    print()
    
    # 排名
    print("【各项指标排名】")
    
    roi_sorted = sorted(strategies, key=lambda x: x['roi'], reverse=True)
    print(f"  ROI最高: {roi_sorted[0]['name']} ({roi_sorted[0]['roi']:+.2f}%)")
    
    profit_sorted = sorted(strategies, key=lambda x: x['profit'], reverse=True)
    print(f"  利润最高: {profit_sorted[0]['name']} ({profit_sorted[0]['profit']:+.0f}元)")
    
    drawdown_sorted = sorted(strategies, key=lambda x: x['drawdown'])
    print(f"  回撤最低: {drawdown_sorted[0]['name']} ({drawdown_sorted[0]['drawdown']:.0f}元) ⭐")
    
    cost_sorted = sorted(strategies, key=lambda x: x['cost'])
    print(f"  成本最低: {cost_sorted[0]['name']} ({cost_sorted[0]['cost']:.0f}元)")
    
    print()
    
    # 风险收益比
    print("【风险收益比】（利润/回撤）")
    for s in strategies:
        if s['drawdown'] > 0:
            ratio = s['profit'] / s['drawdown']
            print(f"  {s['name']:<20}: {ratio:>6.2f}")
    print()
    
    # 对比优势分析
    print("=" * 80)
    print("第四步：概率预测策略优势分析")
    print("=" * 80)
    print()
    
    # vs 斐波那契
    profit_vs_fib = prob_result['total_profit'] - fib_result['balance']
    roi_vs_fib = prob_result['roi'] - fib_roi
    drawdown_vs_fib = fib_result['max_drawdown'] - prob_result['max_drawdown']
    
    print(f"【vs 斐波那契固定倍投】")
    print(f"  净利润差异: {profit_vs_fib:+.0f}元")
    print(f"  ROI差异: {roi_vs_fib:+.2f}%")
    print(f"  回撤差异: {drawdown_vs_fib:+.0f}元")
    
    if profit_vs_fib > 0 and drawdown_vs_fib > 0:
        print(f"  ✅ 概率预测策略双优：收益更高+回撤更低")
    elif profit_vs_fib > 0:
        print(f"  🟡 收益提升: +{profit_vs_fib:.0f}元 ({profit_vs_fib/abs(fib_result['balance'])*100:+.1f}%)")
    elif drawdown_vs_fib > 0:
        print(f"  🟡 风险降低: 回撤减少{drawdown_vs_fib:.0f}元 ({drawdown_vs_fib/fib_result['max_drawdown']*100:.1f}%)")
    else:
        print(f"  ⚠️  斐波那契策略更优")
    print()
    
    # vs 智能动态v3.2
    profit_vs_smart = prob_result['total_profit'] - smart_result['balance']
    roi_vs_smart = prob_result['roi'] - smart_roi
    drawdown_vs_smart = smart_result['max_drawdown'] - prob_result['max_drawdown']
    
    print(f"【vs 智能动态v3.2】")
    print(f"  净利润差异: {profit_vs_smart:+.0f}元")
    print(f"  ROI差异: {roi_vs_smart:+.2f}%")
    print(f"  回撤差异: {drawdown_vs_smart:+.0f}元")
    
    if profit_vs_smart > 0 and drawdown_vs_smart > 0:
        print(f"  ✅ 概率预测策略双优：收益更高+回撤更低")
    elif profit_vs_smart > 0:
        print(f"  🟡 收益提升: +{profit_vs_smart:.0f}元 ({profit_vs_smart/abs(smart_result['balance'])*100:+.1f}%)")
    elif drawdown_vs_smart > 0:
        print(f"  🟡 风险降低: 回撤减少{drawdown_vs_smart:.0f}元 ({drawdown_vs_smart/smart_result['max_drawdown']*100:.1f}%)")
    else:
        print(f"  ⚠️  智能动态v3.2更优")
    print()
    
    # 总结
    print("=" * 80)
    print("总结与建议")
    print("=" * 80)
    print()
    
    if prob_result['roi'] > max(fib_roi, smart_roi):
        print("✅ 概率预测策略ROI最高，推荐使用！")
    elif prob_result['max_drawdown'] < min(fib_result['max_drawdown'], smart_result['max_drawdown']):
        print("✅ 概率预测策略风险最低，适合稳健投资！")
    elif prob_result['total_profit'] > max(fib_result['balance'], smart_result['balance']):
        print("✅ 概率预测策略收益最高，推荐使用！")
    else:
        print("🟡 概率预测策略在某些方面有优势，可作为补充策略")
    
    print()
    print(f"最优策略推荐:")
    if roi_sorted[0]['name'] == profit_sorted[0]['name']:
        print(f"  🏆 {roi_sorted[0]['name']} - ROI和利润双第一")
    else:
        print(f"  🏆 追求收益: {profit_sorted[0]['name']} (利润{profit_sorted[0]['profit']:+.0f}元)")
        print(f"  🏆 追求ROI: {roi_sorted[0]['name']} (ROI {roi_sorted[0]['roi']:+.2f}%)")
    print(f"  🛡️  追求稳健: {drawdown_sorted[0]['name']} (回撤{drawdown_sorted[0]['drawdown']:.0f}元)")
    print()
    
    # 详细概率预测数据
    if prob_result['prediction_accuracy'] and 'calibration' in prob_result['prediction_accuracy']:
        print("=" * 80)
        print("附录：概率预测校准度分析")
        print("=" * 80)
        print()
        
        acc = prob_result['prediction_accuracy']
        if acc['calibration']:
            print(f"  概率范围     | 预测次数 | 平均预测概率 | 实际命中率 |   偏差")
            print(f"  {'-'*70}")
            for cal in acc['calibration']:
                print(
                    f"  {cal['range']:>12} | {cal['count']:>8} | "
                    f"{cal['avg_predicted']:>12.1%} | {cal['avg_actual']:>10.1%} | "
                    f"{cal['bias']:>+7.1%}"
                )
        print()
    
    print("=" * 80)
    print("验证完成！")
    print("=" * 80)
    print()


if __name__ == '__main__':
    main()
