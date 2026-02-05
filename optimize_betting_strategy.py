"""
基于实际预测成功分布优化投注策略
分析100期数据，设计既降低风险又提升收益的方案
"""

import pandas as pd
import numpy as np
from collections import Counter

def analyze_pattern(csv_file):
    """深入分析成功失败模式"""
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    
    print("="*80)
    print("预测成功分布深度分析")
    print("="*80)
    
    # 连胜/连败统计
    consecutive_wins = []
    consecutive_losses = []
    current_wins = 0
    current_losses = 0
    
    for hit in df['is_hit']:
        if hit:
            if current_losses > 0:
                consecutive_losses.append(current_losses)
                current_losses = 0
            current_wins += 1
        else:
            if current_wins > 0:
                consecutive_wins.append(current_wins)
                current_wins = 0
            current_losses += 1
    
    if current_wins > 0:
        consecutive_wins.append(current_wins)
    if current_losses > 0:
        consecutive_losses.append(current_losses)
    
    print(f"\n【核心统计】")
    print(f"总期数: {len(df)}")
    print(f"命中率: {df['is_hit'].sum() / len(df) * 100:.1f}%")
    print(f"\n连胜分布: {Counter(consecutive_wins)}")
    print(f"连败分布: {Counter(consecutive_losses)}")
    
    # 关键洞察
    wins_1_2 = sum(1 for x in consecutive_wins if x <= 2)
    wins_3_plus = sum(1 for x in consecutive_wins if x >= 3)
    losses_1 = sum(1 for x in consecutive_losses if x == 1)
    losses_2 = sum(1 for x in consecutive_losses if x == 2)
    losses_3_plus = sum(1 for x in consecutive_losses if x >= 3)
    
    print(f"\n【关键洞察】")
    print(f"1. 短连胜(1-2期): {wins_1_2}/{len(consecutive_wins)} = {wins_1_2/len(consecutive_wins)*100:.1f}%")
    print(f"2. 长连胜(3+期): {wins_3_plus}/{len(consecutive_wins)} = {wins_3_plus/len(consecutive_wins)*100:.1f}%")
    print(f"3. 单次连败: {losses_1}/{len(consecutive_losses)} = {losses_1/len(consecutive_losses)*100:.1f}%")
    print(f"4. 2次连败: {losses_2}/{len(consecutive_losses)} = {losses_2/len(consecutive_losses)*100:.1f}%")
    print(f"5. 长连败(3+期): {losses_3_plus}/{len(consecutive_losses)} = {losses_3_plus/len(consecutive_losses)*100:.1f}%")
    
    return consecutive_wins, consecutive_losses

def simulate_strategy(csv_file, strategy_name, mult_func, max_mult=10):
    """模拟投注策略"""
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    
    base_bet = 17
    win_reward = 47
    total_investment = 0
    total_reward = 0
    consecutive_wins = 0
    consecutive_losses = 0
    max_drawdown = 0
    current_profit = 0
    
    mult_usage = {}
    
    for idx, row in df.iterrows():
        # 注意：计算倍数时使用的是上一期的连胜/连败数
        multiplier = mult_func(consecutive_wins, consecutive_losses, max_mult)
        bet = base_bet * multiplier
        total_investment += bet
        
        mult_usage[multiplier] = mult_usage.get(multiplier, 0) + 1
        
        if row['is_hit']:
            total_reward += win_reward
            current_profit = total_reward - total_investment
            consecutive_wins += 1
            consecutive_losses = 0
        else:
            current_profit = total_reward - total_investment
            max_drawdown = min(max_drawdown, current_profit)
            consecutive_wins = 0
            consecutive_losses += 1
    
    profit = total_reward - total_investment
    roi = (profit / total_investment) * 100 if total_investment > 0 else 0
    
    return {
        'name': strategy_name,
        'investment': total_investment,
        'reward': total_reward,
        'profit': profit,
        'roi': roi,
        'drawdown': max_drawdown,
        'mult_usage': mult_usage
    }

# 定义各种策略的倍数函数
def fixed_1x(w, l, max_mult):
    return 1.0

def stable_dynamic(w, l, max_mult):
    """稳健动态：连胜保持1倍"""
    if w > 0:
        return 1.0
    elif l == 1:
        return 2.0
    elif l == 2:
        return 4.0
    else:
        return min(4.0 + (l - 2) * 2, max_mult)

def selective_dynamic(w, l, max_mult):
    """选择性动态：需要2期触发"""
    if w >= 3:
        return 0.5
    elif w == 2:
        return 0.8
    elif l >= 2:
        if l == 2:
            return 2.0
        elif l == 3:
            return 4.0
        else:
            return min(4.0 + (l - 3) * 2, max_mult)
    return 1.0

def optimized_v1(w, l, max_mult):
    """优化方案V1：温和加倍，适度保护"""
    if w > 0:
        if w <= 2:
            return 1.0  # 前2次胜利保持1倍
        elif w == 3:
            return 0.8  # 第3次适度保守
        else:
            return 0.6  # 4+次保护利润
    else:
        if l == 1:
            return 1.5  # 首败温和
        elif l == 2:
            return 3.0  # 连败2期
        elif l == 3:
            return 5.0  # 连败3期
        else:
            return min(5.0 + (l - 3) * 2, max_mult)

def optimized_v2(w, l, max_mult):
    """优化方案V2：激进追回，保持盈利"""
    if w > 0:
        if w <= 3:
            return 1.0  # 前3次胜利保持1倍
        else:
            return 0.7  # 4+次适度保护
    else:
        if l == 1:
            return 2.0  # 首败立即加倍
        elif l == 2:
            return 3.5  # 连败2期
        elif l == 3:
            return 5.0  # 连败3期
        else:
            return min(5.0 + (l - 3) * 2.5, max_mult)

def optimized_v3(w, l, max_mult):
    """优化方案V3：平衡型，结合稳健和选择性优点"""
    if w > 0:
        if w <= 2:
            return 1.0  # 前2次保持
        elif w == 3:
            return 0.9  # 第3次轻微保守
        else:
            return 0.7  # 4+次保护
    else:
        if l == 1:
            return 1.8  # 首败温和加倍
        elif l == 2:
            return 3.5  # 连败2期
        elif l == 3:
            return 5.5  # 连败3期
        else:
            return min(5.5 + (l - 3) * 2, max_mult)

def compare_all_strategies(csv_file):
    """对比所有策略"""
    print("\n" + "="*80)
    print("六策略完整对比（基于实际100期数据）")
    print("="*80)
    
    strategies = [
        ('固定1倍', fixed_1x, 1),
        ('稳健动态', stable_dynamic, 10),
        ('选择性动态', selective_dynamic, 10),
        ('优化V1-温和型', optimized_v1, 8),
        ('优化V2-激进型', optimized_v2, 10),
        ('优化V3-平衡型', optimized_v3, 9),
    ]
    
    results = []
    for name, func, max_mult in strategies:
        result = simulate_strategy(csv_file, name, func, max_mult)
        results.append(result)
    
    # 打印表格
    print(f"\n{'策略':<15} {'总投注':<12} {'净收益':<12} {'ROI':<10} {'回撤':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<15} {r['investment']:>10.0f}元 {r['profit']:>+10.0f}元 "
              f"{r['roi']:>+8.2f}% {r['drawdown']:>+8.0f}元")
    
    # 详细分析每个策略
    print("\n" + "="*80)
    print("策略详细分析")
    print("="*80)
    
    for r in results:
        print(f"\n【{r['name']}】")
        print(f"总投注: {r['investment']:.0f}元 | 净收益: {r['profit']:+.0f}元 | ROI: {r['roi']:+.2f}% | 回撤: {r['drawdown']:+.0f}元")
        print(f"倍数分布: ", end="")
        for mult in sorted(r['mult_usage'].keys()):
            count = r['mult_usage'][mult]
            pct = count / 100 * 100
            print(f"{mult:.1f}x({pct:.0f}%) ", end="")
        print()
    
    # 综合评分
    print("\n" + "="*80)
    print("综合评分（收益35% + ROI35% + 风险控制30%）")
    print("="*80)
    
    best_profit = max(r['profit'] for r in results)
    best_roi = max(r['roi'] for r in results)
    best_risk = max(r['drawdown'] for r in results)  # 回撤越接近0越好
    
    scores = {}
    for r in results:
        profit_score = (r['profit'] / best_profit) * 35 if best_profit > 0 else 0
        roi_score = (r['roi'] / best_roi) * 35 if best_roi > 0 else 0
        risk_score = (r['drawdown'] / best_risk) * 30 if best_risk != 0 else 30
        total = profit_score + roi_score + risk_score
        scores[r['name']] = total
        print(f"{r['name']:<15}: {total:>6.1f}分 (收益{profit_score:.1f} + ROI{roi_score:.1f} + 风控{risk_score:.1f})")
    
    best_strategy = max(scores.items(), key=lambda x: x[1])
    print(f"\n🌟 综合最佳: {best_strategy[0]} ({best_strategy[1]:.1f}分)")
    
    return results

if __name__ == '__main__':
    csv_file = 'zodiac_top4_stable_betting_100periods.csv'
    
    # 1. 分析模式
    print("\n第一步：分析预测成功分布模式")
    wins, losses = analyze_pattern(csv_file)
    
    # 2. 对比策略
    print("\n第二步：模拟六种投注策略")
    results = compare_all_strategies(csv_file)
    
    # 3. 最终推荐
    print("\n" + "="*80)
    print("最终推荐")
    print("="*80)
    
    # 按不同目标推荐
    best_roi = max(results, key=lambda x: x['roi'])
    best_profit = max(results, key=lambda x: x['profit'])
    best_risk = max(results, key=lambda x: x['drawdown'])
    
    print(f"\n🎯 追求最高ROI: {best_roi['name']} (ROI {best_roi['roi']:+.2f}%)")
    print(f"💰 追求最高收益: {best_profit['name']} (收益 {best_profit['profit']:+.0f}元)")
    print(f"🛡️ 追求最低风险: {best_risk['name']} (回撤 {best_risk['drawdown']:+.0f}元)")
    
    # 找出综合最优
    balanced_scores = []
    for r in results:
        # 综合评分：收益/成本 + ROI/100 - 风险/收益
        score = (r['profit'] / r['investment']) + (r['roi'] / 100) - (abs(r['drawdown']) / max(abs(r['profit']), 1))
        balanced_scores.append((r['name'], score, r))
    
    balanced_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n" + "="*80)
    print("策略推荐矩阵")
    print("="*80)
    print(f"\n{'投资目标':<20} {'推荐策略':<20} {'关键指标'}")
    print("-" * 80)
    print(f"{'最高投资回报率':<20} {best_roi['name']:<20} ROI {best_roi['roi']:+.2f}%")
    print(f"{'最高绝对收益':<20} {best_profit['name']:<20} 收益 {best_profit['profit']:+.0f}元")
    print(f"{'最低风险控制':<20} {best_risk['name']:<20} 回撤 {best_risk['drawdown']:+.0f}元")
    print(f"{'综合平衡推荐':<20} {balanced_scores[0][0]:<20} 综合分 {balanced_scores[0][1]:.3f}")
