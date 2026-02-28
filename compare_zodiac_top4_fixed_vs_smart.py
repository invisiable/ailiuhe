"""
对比生肖TOP4投注策略：固定1倍 vs 智能动态倍投
测试智能动态策略在生肖TOP4投注中的效果
"""

import pandas as pd
import numpy as np
from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy


def test_fixed_betting(df, start_idx):
    """固定1倍投注策略（当前策略）"""
    strategy = RecommendedZodiacTop4Strategy(use_emergency_backup=True)
    
    results = []
    balance = 0
    total_bet = 0
    total_win = 0
    min_balance = 0
    max_drawdown = 0
    
    base_bet = 16  # 4个生肖×4元
    win_reward = 46  # 命中奖励
    
    for i in range(start_idx, len(df)):
        train_animals = [str(a).strip() for a in df['animal'].iloc[:i].tolist()]
        prediction = strategy.predict_top4(train_animals)
        top4 = prediction['top4']
        
        actual = str(df.iloc[i]['animal']).strip()
        hit = actual in top4
        
        # 固定1倍投注
        bet = base_bet
        total_bet += bet
        
        if hit:
            win = win_reward
            total_win += win
            profit = win - bet
        else:
            profit = -bet
        
        balance += profit
        
        if balance < min_balance:
            min_balance = balance
            max_drawdown = abs(min_balance)
        
        # 更新策略性能
        strategy.update_performance(hit)
        
        if (i - start_idx + 1) % 10 == 0:
            strategy.check_and_switch_model()
        
        results.append({
            'period': i - start_idx + 1,
            'date': df.iloc[i]['date'],
            'actual': actual,
            'top4': top4,
            'hit': hit,
            'multiplier': 1,
            'bet': bet,
            'profit': profit,
            'balance': balance
        })
    
    roi = (balance / total_bet * 100) if total_bet > 0 else 0
    hits = sum(1 for r in results if r['hit'])
    hit_rate = hits / len(results) if results else 0
    
    return {
        'results': results,
        'total_bet': total_bet,
        'total_win': total_win,
        'balance': balance,
        'roi': roi,
        'max_drawdown': max_drawdown,
        'hits': hits,
        'hit_rate': hit_rate
    }


def test_smart_dynamic_betting(df, start_idx):
    """智能动态倍投策略"""
    strategy = RecommendedZodiacTop4Strategy(use_emergency_backup=True)
    
    # 智能动态配置
    config = {
        'base_bet': 16,       # 基础投注
        'win_reward': 46,     # 命中奖励
        'lookback': 8,        # 回看期数
        'good_thresh': 0.35,  # 增强阈值
        'bad_thresh': 0.20,   # 降低阈值
        'boost_mult': 1.5,    # 增强倍数
        'reduce_mult': 0.6,   # 降低倍数
        'max_multiplier': 10  # 最大倍数
    }
    
    fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    
    results = []
    recent_results = []
    balance = 0
    total_bet = 0
    total_win = 0
    min_balance = 0
    max_drawdown = 0
    fib_index = 0
    
    for i in range(start_idx, len(df)):
        train_animals = [str(a).strip() for a in df['animal'].iloc[:i].tolist()]
        prediction = strategy.predict_top4(train_animals)
        top4 = prediction['top4']
        
        actual = str(df.iloc[i]['animal']).strip()
        hit = actual in top4
        
        # 智能动态倍数计算
        # 1. 获取Fibonacci基础倍数
        if fib_index >= len(fib_sequence):
            base_mult = min(fib_sequence[-1], config['max_multiplier'])
        else:
            base_mult = min(fib_sequence[fib_index], config['max_multiplier'])
        
        # 2. 根据最近表现动态调整
        if len(recent_results) >= config['lookback']:
            recent_hits = sum(recent_results[-config['lookback']:])
            rate = recent_hits / config['lookback']
            
            if rate >= config['good_thresh']:
                # 热门状态：增强倍投
                multiplier = min(base_mult * config['boost_mult'], config['max_multiplier'])
            elif rate <= config['bad_thresh']:
                # 冷门状态：降低倍投
                multiplier = max(base_mult * config['reduce_mult'], 1)
            else:
                # 正常状态：保持基础倍数
                multiplier = base_mult
        else:
            multiplier = base_mult
        
        # 投注
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
        
        # 更新策略性能
        strategy.update_performance(hit)
        
        if (i - start_idx + 1) % 10 == 0:
            strategy.check_and_switch_model()
        
        # 计算最近命中率
        if len(recent_results) >= config['lookback']:
            recent_rate = sum(recent_results[-config['lookback']:]) / config['lookback']
        else:
            recent_rate = sum(recent_results) / len(recent_results) if recent_results else 0
        
        results.append({
            'period': i - start_idx + 1,
            'date': df.iloc[i]['date'],
            'actual': actual,
            'top4': top4,
            'hit': hit,
            'multiplier': multiplier,
            'bet': bet,
            'profit': profit,
            'balance': balance,
            'recent_rate': recent_rate,
            'fib_index': fib_index
        })
    
    roi = (balance / total_bet * 100) if total_bet > 0 else 0
    hits = sum(1 for r in results if r['hit'])
    hit_rate = hits / len(results) if results else 0
    
    return {
        'results': results,
        'total_bet': total_bet,
        'total_win': total_win,
        'balance': balance,
        'roi': roi,
        'max_drawdown': max_drawdown,
        'hits': hits,
        'hit_rate': hit_rate
    }


def main():
    print("=" * 100)
    print("生肖TOP4投注策略对比：固定1倍 vs 智能动态倍投")
    print("=" * 100)
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
    
    # 测试固定倍投
    print("正在测试固定1倍投注策略...")
    result_fixed = test_fixed_betting(df, start_idx)
    
    # 测试智能动态倍投
    print("正在测试智能动态倍投策略...")
    result_smart = test_smart_dynamic_betting(df, start_idx)
    
    print()
    print("=" * 100)
    print("【策略对比报告】")
    print("=" * 100)
    print()
    
    print(f"{'指标':<30} {'固定1倍':<25} {'智能动态':<25} {'变化':<20}")
    print("-" * 100)
    
    # 命中率（应该相同）
    fixed_rate = result_fixed['hit_rate'] * 100
    smart_rate = result_smart['hit_rate'] * 100
    print(f"{'命中率':<30} {fixed_rate:<25.2f}% {smart_rate:<25.2f}% {smart_rate-fixed_rate:+.2f}%")
    
    # 命中次数
    print(f"{'命中次数':<30} {result_fixed['hits']:<25} {result_smart['hits']:<25} {result_smart['hits']-result_fixed['hits']:+}")
    
    print()
    
    # ROI
    roi_diff = result_smart['roi'] - result_fixed['roi']
    roi_improve_pct = (roi_diff / result_fixed['roi'] * 100) if result_fixed['roi'] != 0 else 0
    print(f"{'ROI (投资回报率)':<30} {result_fixed['roi']:<25.2f}% {result_smart['roi']:<25.2f}% {roi_diff:+.2f}% ({roi_improve_pct:+.1f}%)")
    
    # 净收益
    balance_diff = result_smart['balance'] - result_fixed['balance']
    balance_improve_pct = (balance_diff / result_fixed['balance'] * 100) if result_fixed['balance'] != 0 else 0
    print(f"{'净收益':<30} {result_fixed['balance']:<25.0f}元 {result_smart['balance']:<25.0f}元 {balance_diff:+.0f}元 ({balance_improve_pct:+.1f}%)")
    
    # 最大回撤
    drawdown_diff = result_smart['max_drawdown'] - result_fixed['max_drawdown']
    drawdown_change_pct = (drawdown_diff / result_fixed['max_drawdown'] * 100) if result_fixed['max_drawdown'] != 0 else 0
    print(f"{'最大回撤':<30} {result_fixed['max_drawdown']:<25.0f}元 {result_smart['max_drawdown']:<25.0f}元 {drawdown_diff:+.0f}元 ({drawdown_change_pct:+.1f}%)")
    
    # 总投注
    bet_diff = result_smart['total_bet'] - result_fixed['total_bet']
    bet_change_pct = (bet_diff / result_fixed['total_bet'] * 100) if result_fixed['total_bet'] != 0 else 0
    print(f"{'总投注额':<30} {result_fixed['total_bet']:<25.0f}元 {result_smart['total_bet']:<25.0f}元 {bet_diff:+.0f}元 ({bet_change_pct:+.1f}%)")
    
    print()
    print("=" * 100)
    print("【智能动态策略详细分析】")
    print("=" * 100)
    print()
    
    # 统计倍数分布
    multipliers = [r['multiplier'] for r in result_smart['results']]
    print(f"倍数统计:")
    print(f"  平均倍数: {np.mean(multipliers):.2f}x")
    print(f"  最大倍数: {max(multipliers):.2f}x")
    print(f"  最小倍数: {min(multipliers):.2f}x")
    print(f"  触及10倍上限: {sum(1 for m in multipliers if m >= 10)}次")
    print()
    
    # 统计调整效果
    boost_count = sum(1 for r in result_smart['results'] 
                      if len([x for x in result_smart['results'] if x['period'] < r['period']]) >= 8 
                      and r['recent_rate'] >= 0.35 and r['multiplier'] > 1)
    reduce_count = sum(1 for r in result_smart['results'] 
                       if len([x for x in result_smart['results'] if x['period'] < r['period']]) >= 8 
                       and r['recent_rate'] <= 0.20)
    
    print(f"动态调整统计:")
    print(f"  增强倍投次数: {boost_count}次 (命中率≥35%)")
    print(f"  降低倍投次数: {reduce_count}次 (命中率≤20%)")
    print()
    
    # 连续统计
    max_consecutive_wins_fixed = 0
    max_consecutive_losses_fixed = 0
    current_wins = 0
    current_losses = 0
    
    for r in result_fixed['results']:
        if r['hit']:
            current_wins += 1
            current_losses = 0
            max_consecutive_wins_fixed = max(max_consecutive_wins_fixed, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consecutive_losses_fixed = max(max_consecutive_losses_fixed, current_losses)
    
    max_consecutive_wins_smart = 0
    max_consecutive_losses_smart = 0
    current_wins = 0
    current_losses = 0
    
    for r in result_smart['results']:
        if r['hit']:
            current_wins += 1
            current_losses = 0
            max_consecutive_wins_smart = max(max_consecutive_wins_smart, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consecutive_losses_smart = max(max_consecutive_losses_smart, current_losses)
    
    print(f"连续表现:")
    print(f"  {'策略':<20} {'最长连胜':<15} {'最长连败':<15}")
    print(f"  {'固定1倍':<20} {max_consecutive_wins_fixed:<15}期 {max_consecutive_losses_fixed:<15}期")
    print(f"  {'智能动态':<20} {max_consecutive_wins_smart:<15}期 {max_consecutive_losses_smart:<15}期")
    print()
    
    print("=" * 100)
    print("【结论与建议】")
    print("=" * 100)
    print()
    
    if result_smart['roi'] > result_fixed['roi']:
        print(f"✅ 智能动态策略ROI提升 {roi_diff:.2f}% (相对提升{roi_improve_pct:.1f}%)")
    else:
        print(f"⚠️ 智能动态策略ROI下降 {abs(roi_diff):.2f}%")
    
    if result_smart['balance'] > result_fixed['balance']:
        print(f"✅ 智能动态策略净收益增加 {balance_diff:.0f}元 (提升{balance_improve_pct:.1f}%)")
    else:
        print(f"⚠️ 智能动态策略净收益减少 {abs(balance_diff):.0f}元")
    
    if result_smart['max_drawdown'] < result_fixed['max_drawdown']:
        print(f"✅ 智能动态策略回撤减少 {abs(drawdown_diff):.0f}元 (降低{abs(drawdown_change_pct):.1f}%)")
    else:
        print(f"⚠️ 智能动态策略回撤增加 {drawdown_diff:.0f}元 (增加{drawdown_change_pct:.1f}%)")
    
    print()
    
    # 综合评分
    roi_score = 1 if result_smart['roi'] > result_fixed['roi'] else 0
    balance_score = 1 if result_smart['balance'] > result_fixed['balance'] else 0
    drawdown_score = 1 if result_smart['max_drawdown'] < result_fixed['max_drawdown'] else 0
    total_score = roi_score + balance_score + drawdown_score
    
    stars = "⭐" * total_score
    
    print(f"综合评分: {total_score}/3 {stars}")
    
    if total_score >= 2:
        print("💡 推荐: 智能动态策略表现更优，建议替换当前固定1倍策略")
    elif total_score == 1:
        print("💡 建议: 智能动态策略有优势也有劣势，可根据风险偏好选择")
    else:
        print("💡 建议: 保持当前固定1倍策略")
    
    print()


if __name__ == '__main__':
    main()
