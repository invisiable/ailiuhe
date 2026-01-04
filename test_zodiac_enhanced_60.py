"""
测试增强版生肖预测器 - 详细分析
"""

import pandas as pd
from zodiac_enhanced_60_predictor import ZodiacEnhanced60Predictor


def detailed_test():
    """详细测试并显示失败案例"""
    print("=" * 80)
    print("增强版生肖预测器 - 详细验证分析")
    print("=" * 80)
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    
    predictor = ZodiacEnhanced60Predictor()
    
    # 100期回测
    test_periods = 100
    hits = 0
    misses = []
    hit_details = []
    
    print(f"\n开始{test_periods}期回测验证...\n")
    print(f"{'期数':<8} {'实际':<6} {'生肖':<6} {'命中':<6} {'Top5预测':<40}")
    print("-" * 90)
    
    for i in range(len(numbers) - test_periods, len(numbers)):
        actual_num = numbers[i]
        actual_zodiac = predictor.number_to_zodiac.get(actual_num, '未知')
        history = numbers[:i]
        
        if len(history) < 30:
            continue
        
        # 预测Top5生肖
        top5_zodiacs = predictor.predict_top5(history, recent_periods=100)
        
        # 检查命中
        hit = actual_zodiac in top5_zodiacs
        
        if hit:
            hits += 1
            hit_rank = top5_zodiacs.index(actual_zodiac) + 1
            hit_details.append({
                'period': i + 1,
                'number': actual_num,
                'zodiac': actual_zodiac,
                'rank': hit_rank,
                'top5': top5_zodiacs
            })
        else:
            misses.append({
                'period': i + 1,
                'number': actual_num,
                'zodiac': actual_zodiac,
                'top5': top5_zodiacs
            })
        
        status = "Y" if hit else "N"
        print(f"第{i+1:<5}期 {actual_num:<6} {actual_zodiac:<6} {status:<6} {str(top5_zodiacs):<40}")
    
    total = test_periods
    success_rate = hits / total * 100
    
    # 统计结果
    print("\n" + "=" * 90)
    print(f"验证完成!")
    print(f"测试期数: {total} 期")
    print(f"命中次数: {hits} 期")
    print(f"未命中: {len(misses)} 期")
    print(f"成功率: {success_rate:.1f}%")
    print("=" * 90)
    
    # 命中详情分析
    print(f"\n【命中分布分析】")
    rank_dist = {}
    for detail in hit_details:
        rank = detail['rank']
        rank_dist[rank] = rank_dist.get(rank, 0) + 1
    
    for rank in sorted(rank_dist.keys()):
        count = rank_dist[rank]
        pct = count / hits * 100
        print(f"  第{rank}位命中: {count}次 ({pct:.1f}%)")
    
    # 未命中案例分析
    if misses:
        print(f"\n【未命中案例分析】（共{len(misses)}期）")
        print(f"{'期数':<8} {'实际':<6} {'生肖':<6} {'预测Top5':<40}")
        print("-" * 80)
        for miss in misses[:20]:  # 只显示前20个
            print(f"第{miss['period']:<5}期 {miss['number']:<6} {miss['zodiac']:<6} {str(miss['top5']):<40}")
        
        if len(misses) > 20:
            print(f"... 还有{len(misses)-20}期未显示")
    
    # 生肖命中统计
    print(f"\n【各生肖命中统计】")
    zodiac_hits = {}
    zodiac_total = {}
    
    for i in range(len(numbers) - test_periods, len(numbers)):
        actual_zodiac = predictor.number_to_zodiac.get(numbers[i], '未知')
        zodiac_total[actual_zodiac] = zodiac_total.get(actual_zodiac, 0) + 1
    
    for detail in hit_details:
        zodiac = detail['zodiac']
        zodiac_hits[zodiac] = zodiac_hits.get(zodiac, 0) + 1
    
    for zodiac in sorted(predictor.zodiac_numbers.keys()):
        total_count = zodiac_total.get(zodiac, 0)
        hit_count = zodiac_hits.get(zodiac, 0)
        if total_count > 0:
            zodiac_rate = hit_count / total_count * 100
            print(f"  {zodiac}: {hit_count}/{total_count} ({zodiac_rate:.0f}%)")
    
    # 预测下一期
    print(f"\n{'='*80}")
    print("【下一期预测】")
    top5 = predictor.predict_top5(numbers)
    top20 = predictor.predict_numbers(numbers, top_n=20)
    
    print(f"\nTop5生肖: {top5}")
    print(f"\n生肖对应数字:")
    for i, zodiac in enumerate(top5, 1):
        nums = predictor.zodiac_numbers[zodiac]
        print(f"  {i}. {zodiac}: {nums}")
    
    print(f"\nTop20推荐数字:")
    for i in range(0, 20, 10):
        print(f"  {top20[i:i+10]}")
    
    print(f"{'='*80}")


if __name__ == '__main__':
    detailed_test()
