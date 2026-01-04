"""
检查最近10期的预测表现
分析为什么成功率低
"""

import pandas as pd
import numpy as np
from zodiac_super_predictor import ZodiacSuperPredictor

def analyze_recent_10_periods():
    """详细分析最近10期"""
    predictor = ZodiacSuperPredictor()
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total = len(df)
    
    print('='*100)
    print('最近10期详细分析')
    print('='*100)
    print(f'总期数: {total}')
    print(f'分析期数: 第{total-10+1}期 到 第{total}期')
    print()
    
    hits = {'TOP1': 0, 'TOP2': 0, 'TOP3': 0, 'TOP5': 0}
    
    for i in range(total - 10, total):
        period_num = i + 1
        animals = [str(a).strip() for a in df['animal'].values[:i]]
        
        # 获取预测（直接计算各策略得分）
        strategies_scores = {
            'ultra_cold': predictor._ultra_cold_strategy(animals),
            'anti_hot': predictor._anti_hot_strategy(animals),
            'gap': predictor._gap_analysis(animals),
            'rotation': predictor._rotation_advanced(animals),
            'absence_penalty': predictor._continuous_absence_penalty(animals),
            'diversity': predictor._diversity_boost(animals),
            'similarity': predictor._historical_similarity(animals)
        }
        
        # 加权融合
        final_scores = {}
        for zodiac in predictor.zodiacs:
            score = 0.0
            score += strategies_scores['ultra_cold'].get(zodiac, 0) * 0.35
            score += strategies_scores['anti_hot'].get(zodiac, 0) * 0.20
            score += strategies_scores['gap'].get(zodiac, 0) * 0.18
            score += strategies_scores['rotation'].get(zodiac, 0) * 0.12
            score += strategies_scores['absence_penalty'].get(zodiac, 0) * 0.08
            score += strategies_scores['diversity'].get(zodiac, 0) * 0.04
            score += strategies_scores['similarity'].get(zodiac, 0) * 0.03
            final_scores[zodiac] = score
        
        # 排序获取TOP5
        sorted_zodiacs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top5 = [z for z, s in sorted_zodiacs[:5]]
        
        # 实际结果
        actual = str(df['animal'].values[i]).strip()
        
        # 判断命中
        hit_rank = None
        if actual in top5:
            rank = top5.index(actual) + 1
            hit_rank = rank
            if rank == 1:
                hits['TOP1'] += 1
                hits['TOP2'] += 1
                hits['TOP3'] += 1
                hits['TOP5'] += 1
            elif rank == 2:
                hits['TOP2'] += 1
                hits['TOP3'] += 1
                hits['TOP5'] += 1
            elif rank == 3:
                hits['TOP3'] += 1
                hits['TOP5'] += 1
            else:
                hits['TOP5'] += 1
        
        # 显示详情
        print(f"{'='*100}")
        print(f"第{period_num}期 (验证数据第{i-total+10+1}/10期)")
        print(f"{'='*100}")
        print(f"  实际开奖: {actual}")
        print(f"  预测TOP5: {', '.join(top5)}")
        
        if hit_rank:
            print(f"  [OK] 命中! 排名第{hit_rank} {'[1]' if hit_rank == 1 else '[*]' if hit_rank <= 3 else '[+]'}")
        else:
            print(f"  [X] 未命中")
        
        # 显示各策略评分（针对实际开奖生肖）
        print(f"\n  实际生肖'{actual}'的各策略得分:")
        weights = {
            'ultra_cold': 0.35,
            'anti_hot': 0.20,
            'gap': 0.18,
            'rotation': 0.12,
            'absence_penalty': 0.08,
            'diversity': 0.04,
            'similarity': 0.03
        }
        actual_score = 0
        for strategy_name, scores in strategies_scores.items():
            score = scores.get(actual, 0)
            weight = weights.get(strategy_name, 0)
            weighted = score * weight
            actual_score += weighted
            print(f"    {strategy_name:20s}: {score:6.2f} × {weight:5.2f} = {weighted:6.2f}")
        
        print(f"  总分: {actual_score:.2f}")
        
        # 显示TOP5的总分
        print(f"\n  预测TOP5的总分:")
        for rank, zodiac in enumerate(top5, 1):
            zodiac_score = final_scores[zodiac]
            symbol = '[OK]' if zodiac == actual else '    '
            print(f"  {symbol} {rank}. {zodiac:4s}: {zodiac_score:6.2f}")
        
        # 分析为什么没命中
        if not hit_rank:
            print(f"\n  [?] 未命中原因分析:")
            print(f"    实际生肖总分: {actual_score:.2f}")
            top5_min_score = min([final_scores[z] for z in top5])
            print(f"    TOP5最低分:   {top5_min_score:.2f}")
            print(f"    差距: {top5_min_score - actual_score:.2f}")
            
            # 看看实际生肖在哪些策略上得分低
            print(f"\n    实际生肖'{actual}'得分最低的策略:")
            strategy_details = []
            for strategy_name, scores in strategies_scores.items():
                score = scores.get(actual, 0)
                weight = weights.get(strategy_name, 0)
                weighted = score * weight
                strategy_details.append((strategy_name, score, weight, weighted))
            
            strategy_details.sort(key=lambda x: x[3])
            for name, score, weight, weighted in strategy_details[:3]:
                print(f"      {name:20s}: {score:6.2f} × {weight:5.2f} = {weighted:6.2f}")
        
        # 显示最近历史
        recent_20 = animals[-20:] if len(animals) >= 20 else animals
        actual_count = recent_20.count(actual)
        print(f"\n  历史信息:")
        print(f"    最近20期出现次数: {actual_count}")
        print(f"    最近20期: {', '.join(recent_20[-20:])}")
        
        print()
    
    # 汇总统计
    print(f"{'='*100}")
    print('最近10期汇总统计')
    print(f"{'='*100}")
    print(f"  TOP1命中率: {hits['TOP1']}/10 = {hits['TOP1']/10*100:5.1f}% (理论8.3%,  超出{(hits['TOP1']/10-0.083)*100:+5.1f}%)")
    print(f"  TOP2命中率: {hits['TOP2']}/10 = {hits['TOP2']/10*100:5.1f}% (理论16.7%, 超出{(hits['TOP2']/10-0.167)*100:+5.1f}%)")
    print(f"  TOP3命中率: {hits['TOP3']}/10 = {hits['TOP3']/10*100:5.1f}% (理论25.0%, 超出{(hits['TOP3']/10-0.250)*100:+5.1f}%)")
    print(f"  TOP5命中率: {hits['TOP5']}/10 = {hits['TOP5']/10*100:5.1f}% (理论41.7%, 超出{(hits['TOP5']/10-0.417)*100:+5.1f}%)")
    print()
    
    if hits['TOP5'] / 10 < 0.40:
        print("  [!] 警告: TOP5命中率低于理论值！")
    elif hits['TOP5'] / 10 < 0.50:
        print("  [!] 注意: TOP5命中率接近理论值，低于预期50%")
    else:
        print("  [OK] TOP5命中率达到预期水平")
    
    print(f"{'='*100}")
    
    # 对比100期的表现
    print()
    print('与100期整体表现对比')
    print('='*100)
    
    # 计算100期整体
    hits_100 = {'TOP1': 0, 'TOP2': 0, 'TOP3': 0, 'TOP5': 0}
    for i in range(total - 100, total):
        animals = [str(a).strip() for a in df['animal'].values[:i]]
        
        # 计算预测
        strategies_scores = {
            'ultra_cold': predictor._ultra_cold_strategy(animals),
            'anti_hot': predictor._anti_hot_strategy(animals),
            'gap': predictor._gap_analysis(animals),
            'rotation': predictor._rotation_advanced(animals),
            'absence_penalty': predictor._continuous_absence_penalty(animals),
            'diversity': predictor._diversity_boost(animals),
            'similarity': predictor._historical_similarity(animals)
        }
        
        final_scores = {}
        for zodiac in predictor.zodiacs:
            score = 0.0
            score += strategies_scores['ultra_cold'].get(zodiac, 0) * 0.35
            score += strategies_scores['anti_hot'].get(zodiac, 0) * 0.20
            score += strategies_scores['gap'].get(zodiac, 0) * 0.18
            score += strategies_scores['rotation'].get(zodiac, 0) * 0.12
            score += strategies_scores['absence_penalty'].get(zodiac, 0) * 0.08
            score += strategies_scores['diversity'].get(zodiac, 0) * 0.04
            score += strategies_scores['similarity'].get(zodiac, 0) * 0.03
            final_scores[zodiac] = score
        
        sorted_zodiacs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top5 = [z for z, s in sorted_zodiacs[:5]]
        actual = str(df['animal'].values[i]).strip()
        
        if actual in top5:
            rank = top5.index(actual) + 1
            if rank == 1:
                hits_100['TOP1'] += 1
                hits_100['TOP2'] += 1
                hits_100['TOP3'] += 1
                hits_100['TOP5'] += 1
            elif rank == 2:
                hits_100['TOP2'] += 1
                hits_100['TOP3'] += 1
                hits_100['TOP5'] += 1
            elif rank == 3:
                hits_100['TOP3'] += 1
                hits_100['TOP5'] += 1
            else:
                hits_100['TOP5'] += 1
    
    print(f"  指标          最近10期    100期整体    差异")
    print('-'*100)
    print(f"  TOP1命中率    {hits['TOP1']/10*100:5.1f}%      {hits_100['TOP1']/100*100:5.1f}%      {(hits['TOP1']/10-hits_100['TOP1']/100)*100:+5.1f}%")
    print(f"  TOP2命中率    {hits['TOP2']/10*100:5.1f}%      {hits_100['TOP2']/100*100:5.1f}%      {(hits['TOP2']/10-hits_100['TOP2']/100)*100:+5.1f}%")
    print(f"  TOP3命中率    {hits['TOP3']/10*100:5.1f}%      {hits_100['TOP3']/100*100:5.1f}%      {(hits['TOP3']/10-hits_100['TOP3']/100)*100:+5.1f}%")
    print(f"  TOP5命中率    {hits['TOP5']/10*100:5.1f}%      {hits_100['TOP5']/100*100:5.1f}%      {(hits['TOP5']/10-hits_100['TOP5']/100)*100:+5.1f}%")
    print('='*100)
    
    return hits

if __name__ == '__main__':
    analyze_recent_10_periods()
