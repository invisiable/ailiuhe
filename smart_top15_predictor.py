"""
基于数据分析的智能Top15预测器
策略：分析历史命中模式，针对性优化
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict


class SmartTop15Predictor:
    """智能Top15预测器 - 数据驱动"""
    
    def __init__(self):
        pass
    
    def predict(self, numbers):
        """核心预测逻辑"""
        numbers_list = numbers.tolist() if hasattr(numbers, 'tolist') else list(numbers)
        
        # 确保有足够数据
        if len(numbers_list) < 20:
            return list(range(1, 16))
        
        recent_100 = numbers_list[-100:] if len(numbers_list) >= 100 else numbers_list
        recent_50 = numbers_list[-50:]
        recent_30 = numbers_list[-30:]
        recent_20 = numbers_list[-20:]
        recent_10 = numbers_list[-10:]
        recent_5 = set(numbers_list[-5:])
        
        # 综合评分
        scores = defaultdict(float)
        
        # ========== 策略1：多时间窗口频率 (30%) ==========
        # 100期频率
        freq_100 = Counter(recent_100)
        for n, count in freq_100.items():
            scores[n] += count * 0.5
        
        # 50期频率
        freq_50 = Counter(recent_50)
        for n, count in freq_50.items():
            scores[n] += count * 0.8
        
        # 30期频率
        freq_30 = Counter(recent_30)
        for n, count in freq_30.items():
            scores[n] += count * 1.2
        
        # 20期频率
        freq_20 = Counter(recent_20)
        for n, count in freq_20.items():
            scores[n] += count * 1.5
        
        # 10期频率（但权重适中，避免过拟合）
        freq_10 = Counter(recent_10)
        for n, count in freq_10.items():
            scores[n] += count * 1.0
        
        # ========== 策略2：冷号回补 (25%) ==========
        # 计算间隔
        last_seen = {}
        for i, n in enumerate(recent_50):
            last_seen[n] = len(recent_50) - i
        
        # 间隔评分
        for n in range(1, 50):
            gap = last_seen.get(n, 60)
            
            # 关键：8-30期间隔的号码容易出现
            if 8 <= gap <= 30:
                gap_score = (gap - 8) * 0.4 + 3.0
                scores[n] += gap_score
            elif gap > 30:
                scores[n] += 2.5
            elif 4 <= gap < 8:
                scores[n] += 1.5
        
        # ========== 策略3：周期性 (20%) ==========
        # 检查多个周期
        for period in [3, 5, 7, 10]:
            if len(numbers_list) > period:
                periodic_num = numbers_list[-period]
                scores[periodic_num] += 2.0
            
            if len(numbers_list) > period * 2:
                periodic_num2 = numbers_list[-period * 2]
                scores[periodic_num2] += 1.5
        
        # ========== 策略4：相邻号码 (10%) ==========
        # 最近3个号码的相邻号码
        for n in recent_10[-3:]:
            for offset in [-3, -2, -1, 1, 2, 3]:
                neighbor = n + offset
                if 1 <= neighbor <= 49:
                    scores[neighbor] += 1.0
        
        # ========== 策略5：尾数模式 (10%) ==========
        # 统计尾数分布
        tail_dist = Counter([n % 10 for n in recent_20])
        cold_tails = [t for t in range(10) if tail_dist.get(t, 0) <= 1]
        
        for n in range(1, 50):
            if n % 10 in cold_tails:
                scores[n] += 1.5
        
        # ========== 策略6：区域平衡 (5%) ==========
        # 确保每个区域都有代表
        zones = [(1,10), (11,20), (21,30), (31,40), (41,49)]
        for start, end in zones:
            zone_nums = [n for n in range(start, end+1)]
            zone_scores = {n: scores.get(n, 0) for n in zone_nums}
            top_zone = max(zone_scores, key=zone_scores.get) if zone_scores else start
            scores[top_zone] += 1.0
        
        # ========== 最终调整 ==========
        # 最近5期出现的号码降权（但不完全排除）
        for n in recent_5:
            if n in scores:
                scores[n] *= 0.6
        
        # 确保所有号码都有基础分
        for n in range(1, 50):
            if n not in scores:
                scores[n] = 0.1
        
        # 排序并返回Top15
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:15]]


def validate_100_periods():
    """100期回测"""
    print("=" * 80)
    print("Smart Top 15 Predictor - 100期回测")
    print("=" * 80)
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    
    total_periods = len(numbers)
    test_periods = min(100, total_periods - 50)
    
    print(f"\n数据: {total_periods}期, 回测: {test_periods}期")
    
    predictor = SmartTop15Predictor()
    
    results = {'top5': 0, 'top10': 0, 'top15': 0, 'details': []}
    
    print("\n期数\t实际\tTop5\tTop10\tTop15")
    print("-" * 50)
    
    for i in range(total_periods - test_periods, total_periods):
        period_num = i + 1
        actual = numbers[i]
        history = numbers[:i]
        
        if len(history) < 30:
            continue
        
        try:
            top15_pred = predictor.predict(history)
            
            top5_hit = actual in top15_pred[:5]
            top10_hit = actual in top15_pred[:10]
            top15_hit = actual in top15_pred
            
            if top5_hit:
                results['top5'] += 1
            if top10_hit:
                results['top10'] += 1
            if top15_hit:
                results['top15'] += 1
            
            mark = 'V' if top15_hit else ''
            print(f"{period_num}\t{actual}\t{'V' if top5_hit else ''}\t{'V' if top10_hit else ''}\t{mark}")
            
            results['details'].append({
                'period': period_num,
                'actual': actual,
                'top15_hit': top15_hit
            })
            
        except Exception as e:
            print(f"{period_num}\t{actual}\tERR")
    
    total = len(results['details'])
    if total == 0:
        return 0
    
    top5_rate = results['top5'] / total * 100
    top10_rate = results['top10'] / total * 100
    top15_rate = results['top15'] / total * 100
    
    print("\n" + "=" * 80)
    print("统计结果")
    print("=" * 80)
    print(f"\nTop 5:  {results['top5']}/{total} = {top5_rate:.1f}%")
    print(f"Top 10: {results['top10']}/{total} = {top10_rate:.1f}%")
    print(f"Top 15: {results['top15']}/{total} = {top15_rate:.1f}%")
    
    if top15_rate >= 60:
        print(f"\n[成功] 达成目标! {top15_rate:.1f}% >= 60%")
    else:
        print(f"\n[待优化] 还差 {60 - top15_rate:.1f}%")
    
    # 最近表现
    recent_20 = results['details'][-20:]
    recent_hits = sum(1 for d in recent_20 if d['top15_hit'])
    print(f"最近20期: {recent_hits}/20 = {recent_hits/20*100:.1f}%")
    
    return top15_rate


if __name__ == '__main__':
    rate = validate_100_periods()
    print(f"\n最终成功率: {rate:.1f}%")
