"""
生肖TOP4投注问题分析
分析最近50期的预测与实际开奖的差异，找出成功率低的根本原因
"""

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

def analyze_zodiac_top4_issues():
    print("="*80)
    print("生肖TOP4投注 - 最近50期深度分析")
    print("="*80)
    
    # 读取回测数据
    df_backtest = pd.read_csv('zodiac_top4_backtest_300periods.csv', encoding='utf-8-sig')
    df_source = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    
    # 获取最近50期
    last_50_backtest = df_backtest.tail(50)
    last_50_source = df_source.tail(50)
    
    # 1. 统计预测生肖的频率分布
    print("\n【1. 预测生肖频率分布】")
    predicted_zodiacs = []
    for zodiacs_str in last_50_backtest['top4_zodiacs']:
        zodiacs = [z.strip() for z in zodiacs_str.split(',')]
        predicted_zodiacs.extend(zodiacs)
    
    pred_counter = Counter(predicted_zodiacs)
    print(f"总共预测: {len(predicted_zodiacs)}次生肖（50期 × 4 = 200次应该）")
    print(f"\n高频预测生肖（预测次数/50期）:")
    for zodiac, count in pred_counter.most_common():
        pct = count / 50 * 100
        print(f"  {zodiac}: {count}次 ({pct:.1f}%)")
    
    # 2. 统计实际开奖生肖的频率分布
    print(f"\n【2. 实际开奖生肖频率分布】")
    actual_zodiacs = [str(z).strip() for z in last_50_source['animal']]
    actual_counter = Counter(actual_zodiacs)
    print(f"最近50期实际开奖:")
    for zodiac, count in actual_counter.most_common():
        pct = count / 50 * 100
        print(f"  {zodiac}: {count}次 ({pct:.1f}%)")
    
    # 3. 对比分析：哪些生肖被低估，哪些被高估
    print(f"\n【3. 预测偏差分析】")
    print(f"{'生肖':<6} {'实际次数':<10} {'预测次数':<10} {'预测覆盖率':<12} {'偏差'}")
    print("-" * 60)
    
    all_zodiacs = set(pred_counter.keys()) | set(actual_counter.keys())
    underestimated = []
    overestimated = []
    
    for zodiac in sorted(all_zodiacs):
        actual = actual_counter.get(zodiac, 0)
        predicted = pred_counter.get(zodiac, 0)
        coverage = predicted / 50 * 100  # 预测覆盖率
        bias = predicted / 50 - actual / 50
        
        bias_str = "高估" if bias > 0.2 else "低估" if bias < -0.05 else "合理"
        print(f"{zodiac:<6} {actual:<10} {predicted:<10} {coverage:<10.1f}% {bias_str}")
        
        if bias < -0.05 and actual >= 3:
            underestimated.append((zodiac, actual, predicted))
        elif bias > 0.2:
            overestimated.append((zodiac, actual, predicted))
    
    # 4. 命中率分析
    print(f"\n【4. 命中率详细分析】")
    hits = (last_50_backtest['is_hit'] == '是').sum()
    hit_rate = hits / 50 * 100
    print(f"命中次数: {hits}/50")
    print(f"命中率: {hit_rate:.1f}%")
    print(f"理论命中率: 33.3% (4/12生肖)")
    print(f"提升幅度: +{hit_rate - 33.3:.1f}%")
    
    # 5. 连续不中分析
    print(f"\n【5. 连续不中问题分析】")
    max_loss = last_50_backtest['consecutive_losses'].max()
    loss_4_plus = (last_50_backtest['consecutive_losses'] >= 4).sum()
    loss_7_plus = (last_50_backtest['consecutive_losses'] >= 7).sum()
    
    print(f"最长连续不中: {max_loss}期")
    print(f"连续不中≥4期的次数: {loss_4_plus}次")
    print(f"连续不中≥7期的次数: {loss_7_plus}次")
    print(f"\n⚠️ 连续不中导致的问题:")
    print(f"  - 4连不中 → 投注倍数翻至6倍，单期96元")
    print(f"  - 7连不中 → 投注倍数达10倍上限，单期160元")
    print(f"  - 严重侵蚀盈利，甚至导致大额亏损")
    
    # 6. 关键问题总结
    print(f"\n{'='*80}")
    print("【核心问题诊断】")
    print(f"{'='*80}")
    
    print(f"\n❌ 问题1：预测生肖过度集中")
    print(f"  • 龙、马、蛇三大生肖占据{(pred_counter['龙']+pred_counter['马']+pred_counter['蛇'])/200*100:.1f}%预测")
    print(f"  • 导致模型缺乏灵活性，无法适应短期波动")
    
    if underestimated:
        print(f"\n❌ 问题2：低估高频开奖生肖")
        for zodiac, actual, pred in underestimated[:3]:
            print(f"  • {zodiac}实际出现{actual}次，但预测仅{pred}次覆盖")
    
    if overestimated:
        print(f"\n❌ 问题3：高估低频开奖生肖")
        for zodiac, actual, pred in overestimated[:3]:
            print(f"  • {zodiac}预测{pred}次覆盖，实际仅出现{actual}次")
    
    print(f"\n❌ 问题4：马丁格尔倍投风险")
    print(f"  • 连续不中导致投注额指数增长")
    print(f"  • 50期内有{loss_4_plus}次需要高额投注（≥96元/期）")
    print(f"  • 即使命中率46%，资金管理不当仍会亏损")
    
    # 7. 计算盈亏情况
    print(f"\n【6. 盈亏详情】")
    cum_profit = last_50_backtest['cumulative_profit'].iloc[-1] - last_50_backtest['cumulative_profit'].iloc[0]
    total_bet = last_50_backtest['bet_amount'].sum()
    total_profit = last_50_backtest['profit'].sum()
    
    print(f"50期累计投注: {total_bet:.0f}元")
    print(f"50期累计盈利: {total_profit:.0f}元")
    print(f"投入产出比: {(total_profit/total_bet*100):.1f}%")
    
    return {
        'hit_rate': hit_rate,
        'max_consecutive_loss': max_loss,
        'pred_counter': pred_counter,
        'actual_counter': actual_counter,
        'underestimated': underestimated,
        'overestimated': overestimated
    }

if __name__ == '__main__':
    result = analyze_zodiac_top4_issues()
