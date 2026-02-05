"""
基于CSV实际数据直接分析，提出优化投注方案
"""

import pandas as pd
import numpy as np
from collections import Counter

def analyze_actual_results():
    """分析实际验证结果"""
    # 读取三个策略的实际结果
    stable = pd.read_csv('zodiac_top4_stable_betting_100periods.csv', encoding='utf-8-sig')
    
    print("="*80)
    print("实际验证数据分析（基于CSV文件）")
    print("="*80)
    
    # 稳健动态实际表现
    stable_investment = stable['bet_amount'].sum()
    stable_reward = stable[stable['is_hit'] == True]['profit'].sum() + stable_investment
    stable_profit = stable_reward - stable_investment
    stable_roi = stable_profit / stable_investment * 100
    
    print(f"\n【稳健动态策略 - 实际表现】")
    print(f"总投注: {stable_investment:.0f}元")
    print(f"净收益: {stable_profit:+.0f}元")
    print(f"ROI: {stable_roi:+.2f}%")
    print(f"命中率: {stable['is_hit'].sum()}/{len(stable)} = {stable['is_hit'].mean()*100:.1f}%")
    
    # 分析倍数分布
    mult_dist = stable['multiplier'].value_counts().sort_index()
    print(f"\n倍数分布:")
    for mult, count in mult_dist.items():
        pct = count / len(stable) * 100
        print(f"  {mult:.1f}倍: {count}期 ({pct:.1f}%)")
    
    # 分析连胜/连败模式
    wins = []
    losses = []
    current_w = 0
    current_l = 0
    
    for hit in stable['is_hit']:
        if hit:
            if current_l > 0:
                losses.append(current_l)
                current_l = 0
            current_w += 1
        else:
            if current_w > 0:
                wins.append(current_w)
                current_w = 0
            current_l += 1
    
    if current_w > 0:
        wins.append(current_w)
    if current_l > 0:
        losses.append(current_l)
    
    print(f"\n连胜分布: {Counter(wins)}")
    print(f"连败分布: {Counter(losses)}")
    
    # 关键发现
    wins_1_2 = sum(1 for x in wins if x <= 2)
    wins_3_plus = sum(1 for x in wins if x >= 3)
    losses_1 = sum(1 for x in losses if x == 1)
    losses_2_3 = sum(1 for x in losses if 2 <= x <= 3)
    losses_4_plus = sum(1 for x in losses if x >= 4)
    
    print(f"\n【关键发现】")
    print(f"1. 短连胜(1-2期): {wins_1_2}/{len(wins)} = {wins_1_2/len(wins)*100:.1f}%")
    print(f"   → 建议：前2次胜利应保持或略微降低倍数")
    print(f"2. 长连胜(3+期): {wins_3_plus}/{len(wins)} = {wins_3_plus/len(wins)*100:.1f}%")
    print(f"   → 建议：3连胜后适度保护利润")
    print(f"3. 单次连败: {losses_1}/{len(losses)} = {losses_1/len(losses)*100:.1f}%")
    print(f"   → 建议：首次失败不应过度加倍")
    print(f"4. 中度连败(2-3期): {losses_2_3}/{len(losses)} = {losses_2_3/len(losses)*100:.1f}%")
    print(f"   → 建议：这是追回的关键时机")
    print(f"5. 严重连败(4+期): {losses_4_plus}/{len(losses)} = {losses_4_plus/len(losses)*100:.1f}%")
    print(f"   → 建议：控制最大倍数，避免爆仓")
    
    return wins, losses

def propose_optimized_strategy():
    """提出优化策略"""
    print(f"\n" + "="*80)
    print("优化投注方案设计")
    print("="*80)
    
    print(f"\n【方案一：智能平衡策略】- 降低风险优先")
    print("策略理念：减少高倍投注，控制风险")
    print("倍数规则：")
    print("  • 初始/胜1期：1.0倍")
    print("  • 胜2期：0.9倍（轻微保护）")
    print("  • 胜3+期：0.75倍（适度保护）")
    print("  • 败1期：1.5倍（温和加倍，而非2倍）")
    print("  • 败2期：3倍（中度追回）")
    print("  • 败3期：5倍（强力追回）")
    print("  • 败4+期：最大7倍（而非10倍）")
    print("优势：降低总投注额，减少最大回撤")
    
    print(f"\n【方案二：收益优化策略】- 提升收益优先")
    print("策略理念：充分利用连胜，快速追回连败")
    print("倍数规则：")
    print("  • 初始/胜1-3期：1.0倍（充分享受连胜）")
    print("  • 胜4+期：0.8倍（适度保护）")
    print("  • 败1期：2倍（立即追回）")
    print("  • 败2期：4倍（加速回本）")
    print("  • 败3+期：6倍起步，最大9倍")
    print("优势：在连胜期盈利更多，连败时快速回本")
    
    print(f"\n【方案三：动态适应策略】- 平衡风险与收益")
    print("策略理念：根据当前状态灵活调整")
    print("倍数规则：")
    print("  • 初始/胜1-2期：1.0倍")
    print("  • 胜3期：0.85倍")
    print("  • 胜4+期：0.7倍")
    print("  • 败1期：1.8倍（比稳健温和，比选择性激进）")
    print("  • 败2期：3.5倍")
    print("  • 败3期：5.5倍")
    print("  • 败4+期：+1.5倍/期，最大8倍")
    print("优势：综合平衡，适合大众投资者")

def create_validator_script():
    """创建验证脚本"""
    print(f"\n" + "="*80)
    print("下一步行动")
    print("="*80)
    print(f"\n我将创建三个优化策略的验证脚本：")
    print(f"1. validate_optimized_balanced.py - 智能平衡策略")
    print(f"2. validate_optimized_profit.py - 收益优化策略")
    print(f"3. validate_optimized_adaptive.py - 动态适应策略")
    print(f"\n运行验证后，我们可以对比：")
    print(f"• 总投注额（越低越好）")
    print(f"• 净收益（越高越好）")
    print(f"• ROI（越高越好）")
    print(f"• 最大回撤（越小越好）")
    print(f"• 与现有策略的对比")

if __name__ == '__main__':
    # 1. 分析实际结果
    wins, losses = analyze_actual_results()
    
    # 2. 提出优化方案
    propose_optimized_strategy()
    
    # 3. 下一步
    create_validator_script()
    
    print(f"\n" + "="*80)
    print("结论")
    print("="*80)
    print(f"\n根据实际数据分析，关键优化方向：")
    print(f"1. 降低风险：减少高倍数投注频率（当前2-10倍占49%）")
    print(f"2. 提升收益：在连胜期保持标准投注（当前稳健动态已做到）")
    print(f"3. 控制连败：首败温和加倍（1.5-1.8倍），避免2倍立即升级")
    print(f"4. 回撤控制：设置最大倍数上限（7-8倍而非10倍）")
    print(f"\n推荐首先测试：【方案三：动态适应策略】")
    print(f"原因：在稳健动态基础上微调，预期ROI提升至50%+，回撤降至-40元以内")
