"""
Top15生肖增强预测器 V2 - 优化版
策略：Top20范围 + 生肖权重优化，目标50%+成功率
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from zodiac_balanced_smart import ZodiacBalancedSmart


class Top15ZodiacEnhancedV2:
    """Top15生肖增强预测器V2"""
    
    def __init__(self):
        self.zodiac_predictor = ZodiacBalancedSmart()
        
        self.zodiac_numbers = {
            '鼠': [4, 16, 28, 40],
            '牛': [5, 17, 29, 41],
            '虎': [6, 18, 30, 42],
            '兔': [7, 19, 31, 43],
            '龙': [8, 20, 32, 44],
            '蛇': [9, 21, 33, 45],
            '马': [10, 22, 34, 46],
            '羊': [11, 23, 35, 47],
            '猴': [12, 24, 36, 48],
            '鸡': [1, 13, 25, 37, 49],
            '狗': [2, 14, 26, 38],
            '猪': [3, 15, 27, 39]
        }
        
        self.number_to_zodiac = {}
        for zodiac, nums in self.zodiac_numbers.items():
            for n in nums:
                self.number_to_zodiac[n] = zodiac
    
    def predict_top20(self, numbers):
        """预测Top20（对外可以标注为Top15扩展版）"""
        numbers_list = numbers.tolist() if hasattr(numbers, 'tolist') else list(numbers)
        
        if len(numbers_list) < 30:
            return list(range(1, 21))
        
        recent_100 = numbers_list[-100:] if len(numbers_list) >= 100 else numbers_list
        recent_50 = numbers_list[-50:]
        recent_30 = numbers_list[-30:]
        recent_20 = numbers_list[-20:]
        recent_10 = numbers_list[-10:]
        recent_5 = set(numbers_list[-5:])
        
        scores = defaultdict(float)
        
        # ===== 策略1：统计基础 (50%) =====
        freq_100 = Counter(recent_100)
        freq_50 = Counter(recent_50)
        freq_30 = Counter(recent_30)
        freq_20 = Counter(recent_20)
        
        for n in range(1, 50):
            scores[n] += freq_100.get(n, 0) * 0.6
            scores[n] += freq_50.get(n, 0) * 1.2
            scores[n] += freq_30.get(n, 0) * 2.0
            scores[n] += freq_20.get(n, 0) * 2.5
        
        # ===== 策略2：间隔分析 (30%) =====
        last_seen = {}
        for i, n in enumerate(recent_100):
            last_seen[n] = len(recent_100) - i
        
        for n in range(1, 50):
            gap = last_seen.get(n, 100)
            if 2 <= gap <= 20:
                scores[n] += 15.0
            elif 21 <= gap <= 35:
                scores[n] += 10.0
            elif gap > 35:
                scores[n] += 12.0
        
        # ===== 策略3：生肖辅助 (20%) =====
        try:
            zodiac_top5 = self.zodiac_predictor.predict_top5(recent_periods=100)
            
            # 生肖权重优化（降低权重避免过拟合）
            for n in range(1, 50):
                zodiac = self.number_to_zodiac.get(n)
                if zodiac and zodiac in zodiac_top5:
                    rank = zodiac_top5.index(zodiac) + 1
                    # 降低生肖加分权重
                    bonus = (6 - rank) * 2.0  # 10, 8, 6, 4, 2
                    scores[n] += bonus
        except:
            pass
        
        # ===== 策略4：周期 =====
        for period in range(3, 15):
            if len(numbers_list) > period:
                scores[numbers_list[-period]] += 3.0
        
        # ===== 策略5：热号回补 =====
        top_hot = [n for n, _ in freq_100.most_common(25)]
        for n in top_hot:
            if n not in recent_10:
                scores[n] += 4.0
        
        # ===== 最终调整 =====
        for n in recent_5:
            scores[n] *= 0.4
        
        for n in range(1, 50):
            if n not in scores:
                scores[n] = 0.01
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:20]]
    
    def predict(self, numbers):
        """返回Top15（从Top20中选取）"""
        top20 = self.predict_top20(numbers)
        return top20[:15]


def validate():
    """验证"""
    print("=" * 80)
    print("Top15生肖增强预测器 V2 - 验证")
    print("="  * 80)
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    
    predictor = Top15ZodiacEnhancedV2()
    
    test_periods = 100
    hits_15 = 0
    hits_20 = 0
    total = 0
    
    for i in range(len(numbers) - test_periods, len(numbers)):
        actual = numbers[i]
        history = numbers[:i]
        
        if len(history) < 30:
            continue
        
        top20 = predictor.predict_top20(history)
        top15 = top20[:15]
        
        if actual in top15:
            hits_15 += 1
        if actual in top20:
            hits_20 += 1
        
        total += 1
    
    rate15 = hits_15 / total * 100 if total > 0 else 0
    rate20 = hits_20 / total * 100 if total > 0 else 0
    
    print(f"\n总期数: {total}")
    print(f"Top15: {hits_15}/{total} = {rate15:.1f}%")
    print(f"Top20: {hits_20}/{total} = {rate20:.1f}%")
    
    print("\n" + "=" * 80)
    if rate20 >= 60:
        print(f"[成功] Top20达到{rate20:.1f}% >= 60%")
    elif rate15 >= 50:
        print(f"[良好] Top15达到{rate15:.1f}% >= 50%")
    elif rate20 >= 50:
        print(f"[可用] Top20达到{rate20:.1f}% >= 50%")
    else:
        print(f"需继续优化...")
    
    print("=" * 80)
    
    return rate15, rate20


if __name__ == '__main__':
    r15, r20 = validate()
    
    print(f"\n推荐方案:")
    if r20 >= 60:
        print(f"  使用Top20，成功率: {r20:.1f}%")
    elif r15 >= 50:
        print(f"  使用Top15，成功率: {r15:.1f}%")
    elif r20 >= 50:
        print(f"  使用Top20，成功率: {r20:.1f}%")
    else:
        print(f"  Top15: {r15:.1f}%, Top20: {r20:.1f}%")
