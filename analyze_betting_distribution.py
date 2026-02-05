"""
分析预测成功分布，找出最佳投注方案
基于实际100期数据的统计分析
"""

import pandas as pd
import numpy as np
from collections import Counter

def analyze_success_distribution(csv_file):
    """分析成功分布特征"""
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    
    print("="*80)
    print("预测成功分布统计分析")
    print("="*80)
    
    # 基础统计
    total = len(df)
    hits = df['is_hit'].sum()
    hit_rate = hits / total
    
    print(f"\n【基础统计】")
    print(f"总期数: {total}")
    print(f"命中期数: {hits}")
    print(f"命中率: {hit_rate:.2%}")
    
    # 连胜/连败统计
    consecutive_wins = []
    consecutive_losses = []
    current_wins = 0
    current_losses = 0
    
    win_loss_pattern = []
    
    for hit in df['is_hit']:
        if hit == 1 or hit == True:
            if current_losses > 0:
                consecutive_losses.append(current_losses)
                current_losses = 0
            current_wins += 1
            win_loss_pattern.append('W')
        else:
            if current_wins > 0:
                consecutive_wins.append(current_wins)
                current_wins = 0
            current_losses += 1
            win_loss_pattern.append('L')
    
    # 添加最后一段
    if current_wins > 0:
        consecutive_wins.append(current_wins)
    if current_losses > 0:
        consecutive_losses.append(current_losses)
    
    print(f"\n【连胜统计】")
    print(f"连胜次数分布: {Counter(consecutive_wins)}")
    print(f"最长连胜: {max(consecutive_wins) if consecutive_wins else 0}期")
    print(f"平均连胜长度: {np.mean(consecutive_wins):.2f}期")
    print(f"中位数连胜: {np.median(consecutive_wins):.1f}期")
    
    print(f"\n【连败统计】")
    print(f"连败次数分布: {Counter(consecutive_losses)}")
    print(f"最长连败: {max(consecutive_losses) if consecutive_losses else 0}期")
    print(f"平均连败长度: {np.mean(consecutive_losses):.2f}期")
    print(f"中位数连败: {np.median(consecutive_losses):.1f}期")
    
    # 转折点分析
    print(f"\n【转折点分析】")
    win_to_loss = 0  # 从胜转败
    loss_to_win = 0  # 从败转胜
    
    for i in range(1, len(win_loss_pattern)):
        if win_loss_pattern[i-1] == 'W' and win_loss_pattern[i] == 'L':
            win_to_loss += 1
        elif win_loss_pattern[i-1] == 'L' and win_loss_pattern[i] == 'W':
            loss_to_win += 1
    
    print(f"从胜转败次数: {win_to_loss}")
    print(f"从败转胜次数: {loss_to_win}")
    
    # 连胜后的失败概率
    wins_1 = sum(1 for x in consecutive_wins if x == 1)
    wins_2 = sum(1 for x in consecutive_wins if x == 2)
    wins_3 = sum(1 for x in consecutive_wins if x == 3)
    wins_4_plus = sum(1 for x in consecutive_wins if x >= 4)
    
    print(f"\n【连胜长度分布详细】")
    print(f"1连胜: {wins_1}次 ({wins_1/len(consecutive_wins)*100:.1f}%)")
    print(f"2连胜: {wins_2}次 ({wins_2/len(consecutive_wins)*100:.1f}%)")
    print(f"3连胜: {wins_3}次 ({wins_3/len(consecutive_wins)*100:.1f}%)")
    print(f"4+连胜: {wins_4_plus}次 ({wins_4_plus/len(consecutive_wins)*100:.1f}%)")
    
    # 连败长度分布详细
    losses_1 = sum(1 for x in consecutive_losses if x == 1)
    losses_2 = sum(1 for x in consecutive_losses if x == 2)
    losses_3 = sum(1 for x in consecutive_losses if x == 3)
    losses_4_plus = sum(1 for x in consecutive_losses if x >= 4)
    
    print(f"\n【连败长度分布详细】")
    print(f"1连败: {losses_1}次 ({losses_1/len(consecutive_losses)*100:.1f}%)")
    print(f"2连败: {losses_2}次 ({losses_2/len(consecutive_losses)*100:.1f}%)")
    print(f"3连败: {losses_3}次 ({losses_3/len(consecutive_losses)*100:.1f}%)")
    print(f"4+连败: {losses_4_plus}次 ({losses_4_plus/len(consecutive_losses)*100:.1f}%)")
    
    # 关键发现
    print(f"\n" + "="*80)
    print("【关键发现与投注建议】")
    print("="*80)
    
    # 发现1：短连胜占比
    short_wins_pct = (wins_1 + wins_2) / len(consecutive_wins) * 100
    print(f"\n1. 短连胜(1-2期)占比高达 {short_wins_pct:.1f}%")
    print(f"   建议：前2次胜利保持标准1倍投注，充分享受短连胜")
    
    # 发现2：长连胜不多
    long_wins_pct = wins_4_plus / len(consecutive_wins) * 100
    print(f"\n2. 长连胜(4+期)仅占 {long_wins_pct:.1f}%")
    print(f"   建议：连胜3期后适度降低投注保护利润")
    
    # 发现3：短连败占比
    short_losses_pct = losses_1 / len(consecutive_losses) * 100
    print(f"\n3. 单次连败占 {short_losses_pct:.1f}%")
    print(f"   建议：首次失败保持冷静，仅轻微加倍（1.5倍）")
    
    # 发现4：长连败风险
    long_losses_pct = losses_4_plus / len(consecutive_losses) * 100
    print(f"\n4. 长连败(4+期)占 {long_losses_pct:.1f}%")
    print(f"   建议：连败3期后加速追回，但控制最大倍数")
    
    return {
        'consecutive_wins': consecutive_wins,
        'consecutive_losses': consecutive_losses,
        'win_loss_pattern': win_loss_pattern,
        'hit_rate': hit_rate
    }

def simulate_optimized_strategy(csv_file):
    """模拟优化投注策略"""
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    
    print(f"\n" + "="*80)
    print("【优化投注策略模拟】")
    print("="*80)
    
    # 策略规则
    print("\n策略规则：")
    print("• 初始/胜1-2期：1.0倍（充分享受短连胜）")
    print("• 胜3期：0.8倍（适度保护）")
    print("• 胜4+期：0.6倍（锁定利润）")
    print("• 败1期：1.5倍（温和加倍）")
    print("• 败2期：3倍（加速追回）")
    print("• 败3期：5倍（强力回本）")
    print("• 败4+期：+2倍/期（最大8倍）")
    
    base_bet = 17  # 基础投注
    win_reward = 47  # 中奖奖励
    
    total_investment = 0
    total_reward = 0
    consecutive_wins = 0
    consecutive_losses = 0
    
    multiplier_usage = {}
    
    for idx, row in df.iterrows():
        hit = row['is_hit']
        
        # 计算倍数
        if consecutive_wins > 0:
            if consecutive_wins <= 2:
                multiplier = 1.0  # 前2次胜利保持标准
            elif consecutive_wins == 3:
                multiplier = 0.8  # 第3次适度保守
            else:
                multiplier = 0.6  # 4+次大幅保守
        elif consecutive_losses == 1:
            multiplier = 1.5  # 首败温和
        elif consecutive_losses == 2:
            multiplier = 3.0  # 连败2期加速
        elif consecutive_losses == 3:
            multiplier = 5.0  # 连败3期强力
        else:
            multiplier = min(5.0 + (consecutive_losses - 3) * 2, 8.0)  # 最大8倍
        
        bet = base_bet * multiplier
        total_investment += bet
        
        # 记录倍数使用
        multiplier_usage[multiplier] = multiplier_usage.get(multiplier, 0) + 1
        
        if hit:
            total_reward += win_reward
            consecutive_wins += 1
            consecutive_losses = 0
        else:
            consecutive_wins = 0
            consecutive_losses += 1
    
    profit = total_reward - total_investment
    roi = (profit / total_investment) * 100
    
    print(f"\n【优化策略回测结果】")
    print(f"总投注: {total_investment:.2f}元")
    print(f"总奖励: {total_reward:.2f}元")
    print(f"净收益: {profit:+.2f}元")
    print(f"ROI: {roi:+.2f}%")
    
    print(f"\n【倍数使用分布】")
    for mult in sorted(multiplier_usage.keys()):
        count = multiplier_usage[mult]
        pct = count / len(df) * 100
        print(f"{mult}倍: {count}期 ({pct:.1f}%)")
    
    return {
        'investment': total_investment,
        'profit': profit,
        'roi': roi,
        'multiplier_usage': multiplier_usage
    }

def compare_all_strategies(csv_file):
    """对比所有策略"""
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    
    strategies = {
        '固定1倍': {'mult_func': lambda w, l: 1.0, 'max_mult': 1.0},
        '稳健动态': {
            'mult_func': lambda w, l: 1.0 if w > 0 else (2.0 if l == 1 else (4.0 if l == 2 else min(4.0 + (l-2)*2, 10.0))),
            'max_mult': 10.0
        },
        '选择性动态': {
            'mult_func': lambda w, l: (0.5 if w >= 3 else (0.8 if w == 2 else (2.0 if l >= 2 and l < 3 else (4.0 if l == 3 else min(4.0 + (l-3)*2, 10.0) if l > 3 else 1.0)))),
            'max_mult': 10.0
        },
        '优化策略': {
            'mult_func': lambda w, l: (
                1.0 if w <= 2 else (0.8 if w == 3 else 0.6)
            ) if w > 0 else (
                1.5 if l == 1 else (3.0 if l == 2 else (5.0 if l == 3 else min(5.0 + (l-3)*2, 8.0)))
            ),
            'max_mult': 8.0
        }
    }
    
    print(f"\n" + "="*80)
    print("【六策略完整对比】")
    print("="*80)
    
    results = {}
    
    for strategy_name, strategy_config in strategies.items():
        base_bet = 17
        win_reward = 47
        total_investment = 0
        total_reward = 0
        consecutive_wins = 0
        consecutive_losses = 0
        max_drawdown = 0
        current_profit = 0
        
        for idx, row in df.iterrows():
            hit = row['is_hit']
            
            multiplier = strategy_config['mult_func'](consecutive_wins, consecutive_losses)
            bet = base_bet * multiplier
            total_investment += bet
            
            if hit:
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
        roi = (profit / total_investment) * 100
        
        results[strategy_name] = {
            'investment': total_investment,
            'profit': profit,
            'roi': roi,
            'drawdown': max_drawdown
        }
    
    # 打印对比表格
    print(f"\n{'策略':<12} {'总投注':<12} {'净收益':<12} {'ROI':<10} {'最大回撤':<10}")
    print("-" * 80)
    
    for name, result in results.items():
        print(f"{name:<12} {result['investment']:>10.0f}元 {result['profit']:>+10.0f}元 "
              f"{result['roi']:>+8.2f}% {result['drawdown']:>+8.0f}元")
    
    return results

if __name__ == '__main__':
    csv_file = 'zodiac_top4_stable_betting_100periods.csv'
    
    # 1. 分析成功分布
    distribution = analyze_success_distribution(csv_file)
    
    # 2. 模拟优化策略
    optimized_result = simulate_optimized_strategy(csv_file)
    
    # 3. 对比所有策略
    comparison = compare_all_strategies(csv_file)
    
    print(f"\n" + "="*80)
    print("【最终推荐】")
    print("="*80)
    
    # 找出最佳策略
    best_roi = max(comparison.items(), key=lambda x: x[1]['roi'])
    best_profit = max(comparison.items(), key=lambda x: x[1]['profit'])
    best_risk = max(comparison.items(), key=lambda x: x[1]['drawdown'])
    
    print(f"\n🏆 最高ROI: {best_roi[0]} ({best_roi[1]['roi']:+.2f}%)")
    print(f"💰 最高收益: {best_profit[0]} ({best_profit[1]['profit']:+.0f}元)")
    print(f"🛡️ 最低风险: {best_risk[0]} ({best_risk[1]['drawdown']:+.0f}元)")
    
    # 综合评分
    print(f"\n【综合评分】(收益40% + ROI30% + 风险控制30%)")
    scores = {}
    for name, result in comparison.items():
        # 标准化评分
        profit_score = (result['profit'] / best_profit[1]['profit']) * 40
        roi_score = (result['roi'] / best_roi[1]['roi']) * 30
        risk_score = (result['drawdown'] / best_risk[1]['drawdown']) * 30
        total_score = profit_score + roi_score + risk_score
        scores[name] = total_score
        print(f"{name}: {total_score:.1f}分")
    
    best_overall = max(scores.items(), key=lambda x: x[1])
    print(f"\n🌟 综合最佳: {best_overall[0]} ({best_overall[1]:.1f}分)")
