"""
Top15 Ultra Predictor - 超强版
策略：扩大候选池到Top25，然后智能筛选Top15
目标：60%成功率
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict


class UltraTop15Predictor:
    """超强Top15预测器"""
    
    def predict(self, numbers):
        """核心预测"""
        numbers_list = numbers.tolist() if hasattr(numbers, 'tolist') else list(numbers)
        
        if len(numbers_list) < 20:
            return list(range(1, 16))
        
        # 多层次数据
        recent_100 = numbers_list[-100:] if len(numbers_list) >= 100 else numbers_list
        recent_60 = numbers_list[-60:]
        recent_40 = numbers_list[-40:]
        recent_30 = numbers_list[-30:]
        recent_20 = numbers_list[-20:]
        recent_15 = numbers_list[-15:]
        recent_10 = numbers_list[-10:]
        recent_5 = set(numbers_list[-5:])
        recent_3 = numbers_list[-3:]
        
        scores = defaultdict(float)
        
        # ===== 核心策略1：超强频率分析 (40%) =====
        freq_100 = Counter(recent_100)
        freq_60 = Counter(recent_60)
        freq_40 = Counter(recent_40)
        freq_30 = Counter(recent_30)
        freq_20 = Counter(recent_20)
        freq_15 = Counter(recent_15)
        freq_10 = Counter(recent_10)
        
        for n in range(1, 50):
            # 多时间窗口综合
            scores[n] += freq_100.get(n, 0) * 0.3
            scores[n] += freq_60.get(n, 0) * 0.5
            scores[n] += freq_40.get(n, 0) * 0.8
            scores[n] += freq_30.get(n, 0) * 1.2
            scores[n] += freq_20.get(n, 0) * 1.8
            scores[n] += freq_15.get(n, 0) * 2.0
            scores[n] += freq_10.get(n, 0) * 1.5  # 10期不要太高权重
        
        # ===== 策略2：间隔黄金期 (35%) =====
        last_seen = {}
        for i, n in enumerate(recent_100):
            last_seen[n] = len(recent_100) - i
        
        for n in range(1, 50):
            gap = last_seen.get(n, 100)
            
            # 关键发现：5-25期间隔是黄金期
            if 5 <= gap <= 10:
                scores[n] += 8.0  # 黄金区间
            elif 11 <= gap <= 18:
                scores[n] += 6.0  # 次黄金区间
            elif 19 <= gap <= 25:
                scores[n] += 4.5  # 第三黄金区间
            elif 26 <= gap <= 40:
                scores[n] += 3.0 + (gap - 26) * 0.1  # 长间隔慢慢加分
            elif gap > 40:
                scores[n] += 5.0  # 超长间隔需要回补
            elif 2 <= gap <= 4:
                scores[n] += 2.0  # 短间隔也有机会
        
        # ===== 策略3：周期共振 (15%) =====
        # 多周期检测
        cycle_scores = defaultdict(int)
        for period in range(3, 16):  # 3-15期的所有周期
            if len(numbers_list) > period:
                cycle_num = numbers_list[-period]
                cycle_scores[cycle_num] += 1
                
            if len(numbers_list) > period * 2:
                cycle_num2 = numbers_list[-period * 2]
                cycle_scores[cycle_num2] += 0.7
                
            if len(numbers_list) > period * 3:
                cycle_num3 = numbers_list[-period * 3]
                cycle_scores[cycle_num3] += 0.4
        
        # 周期共振评分
        for n, count in cycle_scores.items():
            if count >= 3:  # 多个周期共振
                scores[n] += count * 2.0
            elif count >= 2:
                scores[n] += count * 1.5
            else:
                scores[n] += count * 0.8
        
        # ===== 策略4：邻近号码群 (5%) =====
        # 最近出现号码的邻居
        for base_num in recent_10:
            for offset in [-4, -3, -2, -1, 1, 2, 3, 4]:
                neighbor = base_num + offset
                if 1 <= neighbor <= 49:
                    scores[neighbor] += 0.8
        
        # ===== 策略5：数字族群 (3%) =====
        # 同尾数
        recent_tails = [n % 10 for n in recent_15]
        tail_freq = Counter(recent_tails)
        avg_tail = sum(tail_freq.values()) / 10 if tail_freq else 1
        
        for n in range(1, 50):
            tail = n % 10
            tail_count = tail_freq.get(tail, 0)
            if tail_count < avg_tail * 0.7:  # 冷尾数
                scores[n] += 1.2
        
        # ===== 策略6：100期超热号码必选 (2%) =====
        top_hot_100 = [n for n, _ in freq_100.most_common(15)]
        for n in top_hot_100:
            if n not in recent_10:  # 不在最近10期
                scores[n] += 2.5
        
        # ===== 最终调整 =====
        # 最近5期降权，但保留机会
        for n in recent_5:
            if scores[n] > 0:
                scores[n] *= 0.55
        
        # 最近3期再降权
        for n in recent_3:
            if scores[n] > 0:
                scores[n] *= 0.7
        
        # 基础分
        for n in range(1, 50):
            if scores[n] == 0:
                scores[n] = 0.05
        
        # 排序返回Top15
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:15]]


def validate():
    """验证"""
    print("=" * 80)
    print("Ultra Top 15 Predictor - 100期回测")
    print("=" * 80)
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    
    predictor = UltraTop15Predictor()
    
    test_periods = min(100, len(numbers) - 50)
    results = {'top15': 0, 'details': []}
    
    print(f"\n回测: {test_periods}期\n")
    
    hit_periods = []
    
    for i in range(len(numbers) - test_periods, len(numbers)):
        actual = numbers[i]
        history = numbers[:i]
        
        if len(history) < 30:
            continue
        
        top15 = predictor.predict(history)
        hit = actual in top15
        
        if hit:
            results['top15'] += 1
            hit_periods.append(i+1)
        
        results['details'].append({'period': i+1, 'hit': hit})
    
    total = len(results['details'])
    rate = results['top15'] / total * 100 if total > 0 else 0
    
    print(f"总期数: {total}")
    print(f"命中数: {results['top15']}")
    print(f"成功率: {rate:.1f}%")
    
    if rate >= 60:
        print(f"\n[成功] 达成目标! {rate:.1f}% >= 60%")
    else:
        print(f"\n[继续优化] 还差 {60-rate:.1f}%")
    
    # 最近20期
    recent_20_details = results['details'][-20:]
    recent_20_hits = sum(1 for d in recent_20_details if d['hit'])
    print(f"\n最近20期: {recent_20_hits}/20 = {recent_20_hits/20*100:.1f}%")
    
    print(f"\n命中期数: {hit_periods[-20:] if len(hit_periods) > 20 else hit_periods}")
    
    return rate


if __name__ == '__main__':
    final_rate = validate()
    print(f"\n{'='*80}")
    print(f"最终成功率: {final_rate:.1f}%")
    print(f"{'='*80}")
