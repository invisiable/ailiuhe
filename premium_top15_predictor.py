"""
Top15 Premium Predictor - 精品版
策略：Top25候选池 + 智能筛选 = Top15精准预测
目标：通过扩大候选池和精准筛选达到60%
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict


class PremiumTop15Predictor:
    """精品Top15预测器"""
    
    def predict(self, numbers):
        """核心预测逻辑"""
        numbers_list = numbers.tolist() if hasattr(numbers, 'tolist') else list(numbers)
        
        if len(numbers_list) < 20:
            return list(range(1, 16))
        
        # 数据切片
        recent_100 = numbers_list[-100:] if len(numbers_list) >= 100 else numbers_list
        recent_60 = numbers_list[-60:]
        recent_40 = numbers_list[-40:]
        recent_30 = numbers_list[-30:]
        recent_25 = numbers_list[-25:]
        recent_20 = numbers_list[-20:]
        recent_15 = numbers_list[-15:]
        recent_12 = numbers_list[-12:]
        recent_10 = numbers_list[-10:]
        recent_8 = numbers_list[-8:]
        recent_5 = set(numbers_list[-5:])
        recent_3 = set(numbers_list[-3:])
        
        scores = defaultdict(float)
        
        # ========== 超级频率分析 ==========
        freq_100 = Counter(recent_100)
        freq_60 = Counter(recent_60)
        freq_40 = Counter(recent_40)
        freq_30 = Counter(recent_30)
        freq_25 = Counter(recent_25)
        freq_20 = Counter(recent_20)
        freq_15 = Counter(recent_15)
        freq_12 = Counter(recent_12)
        freq_10 = Counter(recent_10)
        freq_8 = Counter(recent_8)
        
        # 多窗口加权
        weights = {
            100: 0.4,
            60: 0.6,
            40: 0.9,
            30: 1.3,
            25: 1.6,
            20: 2.0,
            15: 2.3,
            12: 2.5,
            10: 2.2,
            8: 1.8
        }
        
        freq_maps = {
            100: freq_100, 60: freq_60, 40: freq_40,
            30: freq_30, 25: freq_25, 20: freq_20,
            15: freq_15, 12: freq_12, 10: freq_10, 8: freq_8
        }
        
        for window, freq_map in freq_maps.items():
            weight = weights[window]
            for n, count in freq_map.items():
                scores[n] += count * weight
        
        # ========== 超级间隔分析 ==========
        last_seen = {}
        for i, n in enumerate(recent_100):
            last_seen[n] = len(recent_100) - i
        
        for n in range(1, 50):
            gap = last_seen.get(n, 100)
            
            # 精细化间隔评分
            if 4 <= gap <= 8:
                gap_score = 15.0  # 最佳间隔
            elif 9 <= gap <= 12:
                gap_score = 13.0
            elif 13 <= gap <= 16:
                gap_score = 11.0
            elif 17 <= gap <= 22:
                gap_score = 9.0
            elif 23 <= gap <= 28:
                gap_score = 7.0
            elif 29 <= gap <= 35:
                gap_score = 6.0
            elif gap > 35:
                gap_score = 8.0 + (gap - 35) * 0.15  # 超长间隔回补
            elif 2 <= gap <= 3:
                gap_score = 5.0
            else:
                gap_score = 1.0
            
            scores[n] += gap_score
        
        # ========== 超级周期共振 ==========
        cycle_scores = defaultdict(float)
        
        # 检测3-20期的所有周期
        for period in range(3, 21):
            if len(numbers_list) <= period:
                continue
                
            # 当前周期
            num1 = numbers_list[-period]
            cycle_scores[num1] += 2.5
            
            # 双周期
            if len(numbers_list) > period * 2:
                num2 = numbers_list[-period * 2]
                cycle_scores[num2] += 2.0
            
            # 三周期
            if len(numbers_list) > period * 3:
                num3 = numbers_list[-period * 3]
                cycle_scores[num3] += 1.5
        
        for n, cycle_score in cycle_scores.items():
            scores[n] += cycle_score
        
        # ========== 号码群聚效应 ==========
        # 相邻号
        for base in recent_12:
            for offset in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                neighbor = base + offset
                if 1 <= neighbor <= 49:
                    scores[neighbor] += 1.2
        
        # 同尾数
        tail_freq = Counter([n % 10 for n in recent_20])
        avg_tail = sum(tail_freq.values()) / 10
        for n in range(1, 50):
            if tail_freq.get(n % 10, 0) < avg_tail * 0.6:
                scores[n] += 1.5
        
        # 同区域
        zones = [(1,10), (11,20), (21,30), (31,40), (41,49)]
        zone_freq = defaultdict(int)
        for num in recent_15:
            for idx, (start, end) in enumerate(zones):
                if start <= num <= end:
                    zone_freq[idx] += 1
        
        # 冷区域补强
        avg_zone = sum(zone_freq.values()) / 5
        for idx, (start, end) in enumerate(zones):
            if zone_freq[idx] < avg_zone * 0.7:
                for n in range(start, end+1):
                    scores[n] += 0.8
        
        # ========== 热号必选策略 ==========
        # 100期超热号，但10期未出现
        top_热_100 = [n for n, _ in freq_100.most_common(18)]
        for n in top_热_100:
            if n not in recent_10:
                scores[n] += 5.0
        
        # 30期热号，但8期未出现
        top_热_30 = [n for n, _ in freq_30.most_common(12)]
        for n in top_热_30:
            if n not in recent_8:
                scores[n] += 3.5
        
        # ========== 最终调整 ==========
        # 最近5期降权
        for n in recent_5:
            if n in scores:
                scores[n] *= 0.45
        
        # 最近3期进一步降权
        for n in recent_3:
            if n in scores:
                scores[n] *= 0.6
        
        # 基础分
        for n in range(1, 50):
            if n not in scores:
                scores[n] = 0.01
        
        # 排序并返回Top15
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:15]]


def validate():
    """验证"""
    print("=" * 80)
    print("Premium Top 15 Predictor - 100期回测")
    print("=" * 80)
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    
    predictor = PremiumTop15Predictor()
    
    test_periods = min(100, len(numbers) - 50)
    hits = 0
    total = 0
    
    details = []
    
    for i in range(len(numbers) - test_periods, len(numbers)):
        actual = numbers[i]
        history = numbers[:i]
        
        if len(history) < 30:
            continue
        
        top15 = predictor.predict(history)
        hit = actual in top15
        
        if hit:
            hits += 1
        
        total += 1
        details.append({'period': i+1, 'actual': actual, 'hit': hit, 'top15': top15})
    
    rate = hits / total * 100 if total > 0 else 0
    
    print(f"\n总期数: {total}")
    print(f"命中数: {hits}")
    print(f"成功率: {rate:.1f}%")
    
    if rate >= 60:
        print(f"\n*** 成功! 达到60%目标 ***")
    else:
        print(f"\n还差: {60-rate:.1f}%")
    
    # 最近20期详情
    print("\n最近20期详情:")
    for d in details[-20:]:
        mark = 'V' if d['hit'] else 'X'
        print(f"第{d['period']}期: {d['actual']} {mark}")
    
    recent_20_hits = sum(1 for d in details[-20:] if d['hit'])
    print(f"\n最近20期: {recent_20_hits}/20 = {recent_20_hits/20*100:.1f}%")
    
    # 保存
    df_result = pd.DataFrame(details)
    df_result.to_csv('premium_top15_validation.csv', index=False, encoding='utf-8-sig')
    
    return rate


if __name__ == '__main__':
    final_rate = validate()
    print(f"\n{'='*80}")
    print(f"最终成功率: {final_rate:.1f}%")
    print(f"{'='*80}")
