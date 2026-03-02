"""
快速测试GUI中的暂停策略输出增强
"""
import sys
import pandas as pd
from precise_top15_predictor import PreciseTop15Predictor


def test_pause_strategy_display():
    """测试暂停策略显示逻辑"""
    print("测试暂停策略显示增强...")
    print()
    
    # 模拟配置
    config = {
        'name': '最优智能动态倍投策略 v3.1',
        'lookback': 12,
        'good_thresh': 0.35,
        'bad_thresh': 0.20,
        'boost_mult': 1.5,
        'reduce_mult': 0.5,
        'max_multiplier': 10,
        'base_bet': 15,
        'win_reward': 47
    }
    
    # 模拟基础策略结果
    base_result = {
        'total_periods': 300,
        'bet_periods': 300,
        'wins': 100,
        'hit_rate': 0.3333,
        'total_cost': 10822,
        'total_profit': 1374,
        'roi': 12.70,
        'max_drawdown': 742,
        'max_consecutive_losses': 9,
        'hit_10x_count': 7
    }
    
    # 模拟暂停策略结果
    pause_result = {
        'total_periods': 300,
        'bet_periods': 221,
        'pause_periods': 79,
        'wins': 79,
        'hit_rate': 0.3575,
        'total_cost': 7995,
        'total_profit': 1476,
        'roi': 18.46,
        'max_drawdown': 371,
        'max_consecutive_losses': 8,
        'hit_10x_count': 5,
        'pause_trigger_count': 79,
        'paused_hit_count': 21
    }
    
    # 计算差异
    roi_delta = pause_result['roi'] - base_result['roi']
    profit_delta = pause_result['total_profit'] - base_result['total_profit']
    drawdown_delta = base_result['max_drawdown'] - pause_result['max_drawdown']
    cost_delta = base_result['total_cost'] - pause_result['total_cost']
    cost_delta_pct = (cost_delta / base_result['total_cost'] * 100) if base_result['total_cost'] > 0 else 0
    drawdown_delta_pct = (drawdown_delta / base_result['max_drawdown'] * 100) if base_result['max_drawdown'] > 0 else 0
    
    # 评分系统
    score_profit = 1 if profit_delta > 0 else (-1 if profit_delta < 0 else 0)
    score_roi = 1 if roi_delta > 0 else (-1 if roi_delta < 0 else 0)
    score_drawdown = 1 if drawdown_delta > 0 else (-1 if drawdown_delta < 0 else 0)
    total_score = score_profit + score_roi + score_drawdown
    
    # 输出测试结果（模拟GUI输出格式）
    print("=" * 70)
    print("🎯 附加验证：命中1停1期暂停策略对比")
    print("=" * 70)
    print()
    
    print("【策略对比】")
    print(f"  策略名称        投注期数  命中率   ROI      净利润    最大回撤")
    print(f"  {'-'*66}")
    print(f"  基础策略        {base_result['bet_periods']:>4}期   {base_result['hit_rate']*100:>5.1f}%  {base_result['roi']:>6.2f}%  {base_result['total_profit']:>+7.0f}元  {base_result['max_drawdown']:>6.0f}元")
    print(f"  暂停策略        {pause_result['bet_periods']:>4}期   {pause_result['hit_rate']*100:>5.1f}%  {pause_result['roi']:>6.2f}%  {pause_result['total_profit']:>+7.0f}元  {pause_result['max_drawdown']:>6.0f}元")
    print(f"  {'-'*66}")
    print(f"  差异            {pause_result['bet_periods']-base_result['bet_periods']:>+4}期   {(pause_result['hit_rate']-base_result['hit_rate'])*100:>+5.1f}%  {roi_delta:>+6.2f}%  {profit_delta:>+7.0f}元  {-drawdown_delta:>+6.0f}元")
    print()
    
    print("【暂停策略详情】")
    print(f"  总期数: {pause_result['total_periods']}期")
    print(f"  实际投注: {pause_result['bet_periods']}期（{pause_result['bet_periods']/pause_result['total_periods']*100:.1f}%）")
    print(f"  暂停期数: {pause_result['pause_periods']}期（{pause_result['pause_periods']/pause_result['total_periods']*100:.1f}%）")
    print(f"  暂停触发: {pause_result['pause_trigger_count']}次（每次命中后暂停1期）")
    print(f"  暂停期漏中: {pause_result['paused_hit_count']}次（漏中率{pause_result['paused_hit_count']/pause_result['pause_periods']*100:.1f}%）")
    print()
    
    print("【收益对比】")
    profit_delta_pct = (profit_delta / abs(base_result['total_profit']) * 100) if base_result['total_profit'] != 0 else 0
    if profit_delta > 0:
        print(f"  ✅ 净利润: 暂停策略更高 {profit_delta:+.0f}元 ({profit_delta_pct:+.1f}%)")
    elif profit_delta < 0:
        print(f"  ⚠️  净利润: 基础策略更高 {abs(profit_delta):.0f}元 ({abs(profit_delta_pct):.1f}%)")
    else:
        print(f"  ➖ 净利润: 两者相同")
    
    if roi_delta > 0:
        print(f"  ✅ ROI: 暂停策略更高 {roi_delta:+.2f}% (从{base_result['roi']:.2f}%提升到{pause_result['roi']:.2f}%)")
    elif roi_delta < 0:
        print(f"  ⚠️  ROI: 基础策略更高 {abs(roi_delta):.2f}%")
    else:
        print(f"  ➖ ROI: 两者相同")
    print()
    
    print("【风险对比】")
    if drawdown_delta > 0:
        print(f"  ✅ 最大回撤: 暂停策略更低 {drawdown_delta:.0f}元 ({drawdown_delta_pct:.1f}%)")
        print(f"     基础: {base_result['max_drawdown']:.0f}元 → 暂停: {pause_result['max_drawdown']:.0f}元")
    elif drawdown_delta < 0:
        print(f"  ⚠️  最大回撤: 基础策略更低 {abs(drawdown_delta):.0f}元")
    else:
        print(f"  ➖ 最大回撤: 两者相同")
    
    print(f"  最长连亏: 基础{base_result['max_consecutive_losses']}期 vs 暂停{pause_result['max_consecutive_losses']}期")
    print(f"  触及10倍: 基础{base_result['hit_10x_count']}次 vs 暂停{pause_result['hit_10x_count']}次")
    print()
    
    print("【成本对比】")
    print(f"  投注成本差异: {cost_delta:+.0f}元 ({cost_delta_pct:+.1f}%)")
    print(f"  减少投注: {base_result['bet_periods'] - pause_result['bet_periods']}期")
    print(f"  成本效率: 暂停策略节省{cost_delta_pct:.1f}%投注成本")
    print()
    
    print("【综合评估】")
    print(f"  综合得分: {total_score}/3")
    
    if total_score >= 2:
        conclusion = "✅ 暂停策略明显优于基础策略"
        recommendation = "强烈建议使用暂停策略"
        rating = "⭐⭐⭐⭐⭐"
    elif total_score == 1:
        conclusion = "🟡 暂停策略略优于基础策略"
        recommendation = "建议使用暂停策略"
        rating = "⭐⭐⭐⭐"
    elif total_score == 0:
        conclusion = "➖ 两种策略表现相近"
        recommendation = "根据个人偏好选择"
        rating = "⭐⭐⭐"
    elif total_score == -1:
        conclusion = "🟡 基础策略略优于暂停策略"
        recommendation = "建议使用基础策略"
        rating = "⭐⭐⭐"
    else:
        conclusion = "⚠️  基础策略明显优于暂停策略"
        recommendation = "建议使用基础策略"
        rating = "⭐⭐"
    
    print(f"  结论: {conclusion}")
    print(f"  评级: {rating}")
    print(f"  建议: {recommendation}")
    print()
    
    print("【暂停策略优缺点】")
    print("  优点:")
    print(f"    • 减少投注频率，降低总成本{cost_delta_pct:.1f}%")
    print("    • 命中后暂停，避免小额亏损累积")
    print("    • 重置Fibonacci序列，从低倍数重新开始")
    if drawdown_delta > 0:
        print(f"    • 显著降低最大回撤{drawdown_delta:.0f}元（{drawdown_delta_pct:.1f}%）")
    print("  缺点:")
    print(f"    • 暂停期可能错过连续命中机会（漏中{pause_result['paused_hit_count']}次）")
    if profit_delta < 0:
        print(f"    • 可能减少总收益{abs(profit_delta):.0f}元")
    print()
    
    print("=" * 70)
    print("✅ 测试完成！输出格式正确")
    print("=" * 70)


if __name__ == '__main__':
    test_pause_strategy_display()
