"""
集成多生肖预测器 - 300期回测验证
验证集成预测器能否将最大连续不中降低到6期以内
"""

import pandas as pd
import numpy as np
from ensemble_multi_zodiac_predictor import EnsembleMultiZodiacPredictor


def backtest_ensemble_predictor(test_periods=300):
    """
    回测集成预测器
    
    Args:
        test_periods: 回测期数（默认300期）
    """
    print("="*80)
    print(f"集成多生肖预测器 - {test_periods}期回测")
    print("="*80 + "\n")
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total_periods = len(df)
    
    if test_periods > total_periods - 20:
        test_periods = total_periods - 20
        print(f"调整回测期数为: {test_periods}期")
    
    # 初始化预测器
    predictor = EnsembleMultiZodiacPredictor()
    
    # 回测
    start_idx = total_periods - test_periods
    results = []
    
    print(f"开始回测第{start_idx+1}期到第{total_periods}期...\n")
    
    hits = 0
    current_misses = 0
    max_consecutive_misses = 0
    total_bets = 0
    total_profit = 0
    
    # 投注成本和奖金
    BET_PER_ZODIAC = 15  # 每个生肖15元
    WIN_REWARD = 45      # 中奖45元
    
    for i in range(start_idx, total_periods):
        period = i + 1
        history = df['animal'].iloc[:i].tolist()
        
        # 预测TOP4
        result = predictor.predict_top4(history)
        top4 = result['top4']
        
        # 实际开奖
        actual = df.iloc[i]['animal']
        is_hit = actual in top4
        
        # 计算投注成本和收益
        cost = len(top4) * BET_PER_ZODIAC  # 4个生肖，每个15元
        reward = WIN_REWARD if is_hit else 0
        profit = reward - cost
        
        total_bets += cost
        total_profit += profit
        
        # 更新命中统计
        if is_hit:
            hits += 1
            current_misses = 0
        else:
            current_misses += 1
            max_consecutive_misses = max(max_consecutive_misses, current_misses)
        
        # 更新预测器性能
        if 'details' in result:
            predictor.update_performance(result['details'], actual)
        
        # 记录结果
        results.append({
            'period': period,
            'predicted': ','.join(top4),
            'actual': actual,
            'is_hit': '是' if is_hit else '否',
            'consecutive_misses': current_misses if not is_hit else 0,
            'cost': cost,
            'reward': reward,
            'profit': profit,
            'cumulative_profit': total_profit
        })
        
        # 每50期显示进度
        if (i - start_idx + 1) % 50 == 0:
            current_period = i - start_idx + 1
            current_rate = hits / current_period * 100
            print(f"进度: {current_period}/{test_periods}期, "
                  f"命中率: {current_rate:.2f}%, "
                  f"当前连续不中: {current_misses}期, "
                  f"最大连续不中: {max_consecutive_misses}期")
    
    # 计算最终统计
    hit_rate = hits / test_periods * 100
    roi = (total_profit / total_bets) * 100
    
    # 计算平均连续不中
    miss_streaks = []
    current_streak = 0
    for r in results:
        if r['is_hit'] == '否':
            current_streak += 1
        else:
            if current_streak > 0:
                miss_streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        miss_streaks.append(current_streak)
    
    avg_consecutive_misses = np.mean(miss_streaks) if miss_streaks else 0
    
    # 显示结果
    print("\n" + "="*80)
    print("回测结果汇总")
    print("="*80)
    print(f"总期数: {test_periods}期")
    print(f"命中数: {hits}期")
    print(f"命中率: {hit_rate:.2f}%")
    print(f"最大连续不中: {max_consecutive_misses}期  ← 【目标: ≤6期】")
    print(f"平均连续不中: {avg_consecutive_misses:.2f}期")
    print(f"连续不中次数: {len(miss_streaks)}次")
    print(f"\n投注总额: {total_bets}元")
    print(f"总奖金: {total_bets + total_profit}元")
    print(f"净利润: {total_profit:+.0f}元")
    print(f"投资回报率(ROI): {roi:+.2f}%")
    
    # 显示各预测器表现
    print(f"\n各预测器表现:")
    stats = predictor.get_stats()
    for key, data in sorted(stats.items()):
        name_map = {
            'predictor1': '重训练v2.0',
            'predictor2': '自适应预测器',
            'predictor3': '最终版预测器'
        }
        name = name_map.get(key, key)
        print(f"  {name}: {data['hits']}/{data['total']} = {data['rate']:.1f}%, "
              f"权重={data['weight']:.3f}")
    
    # 评估是否达标
    print("\n" + "="*80)
    if max_consecutive_misses <= 6:
        print("✓ 成功! 最大连续不中 ≤ 6期")
        improvement = ((12 - max_consecutive_misses) / 12) * 100
        print(f"  改进幅度: {improvement:.1f}% (从12期降至{max_consecutive_misses}期)")
    else:
        print(f"✗ 未达标! 最大连续不中 = {max_consecutive_misses}期 > 6期")
        print(f"  需要继续优化")
    
    if hit_rate >= 38.33:
        print(f"✓ 命中率保持: {hit_rate:.2f}% ≥ 38.33% (基准)")
    else:
        print(f"✗ 命中率下降: {hit_rate:.2f}% < 38.33% (基准)")
    
    if roi > 0:
        print(f"✓ 投资回报为正: ROI = {roi:+.2f}%")
    else:
        print(f"✗ 投资回报为负: ROI = {roi:+.2f}%")
    
    print("="*80)
    
    # 保存详细结果
    df_results = pd.DataFrame(results)
    output_file = f'ensemble_multi_zodiac_{test_periods}periods.csv'
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存至: {output_file}")
    
    # 显示最大连续不中的具体期数
    print(f"\n最大连续不中详情:")
    max_streak_periods = []
    current_streak = []
    for r in results:
        if r['is_hit'] == '否':
            current_streak.append(r['period'])
        else:
            if len(current_streak) == max_consecutive_misses:
                max_streak_periods = current_streak.copy()
            current_streak = []
    
    if len(current_streak) == max_consecutive_misses:
        max_streak_periods = current_streak.copy()
    
    if max_streak_periods:
        print(f"  期数范围: 第{max_streak_periods[0]}期 至 第{max_streak_periods[-1]}期")
        print(f"  共{len(max_streak_periods)}期连续不中")
    
    return {
        'hit_rate': hit_rate,
        'max_consecutive_misses': max_consecutive_misses,
        'avg_consecutive_misses': avg_consecutive_misses,
        'roi': roi,
        'total_profit': total_profit
    }


if __name__ == '__main__':
    # 300期回测
    result = backtest_ensemble_predictor(test_periods=300)
    
    print(f"\n" + "="*80)
    print("关键指标总结")
    print("="*80)
    print(f"命中率: {result['hit_rate']:.2f}%")
    print(f"最大连续不中: {result['max_consecutive_misses']}期 (目标: ≤6期)")
    print(f"平均连续不中: {result['avg_consecutive_misses']:.2f}期")
    print(f"ROI: {result['roi']:+.2f}%")
    print(f"净利润: {result['total_profit']:+.0f}元")
