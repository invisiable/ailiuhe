"""
优化版 Top15 预测器 V2
基于对原版和改进版的分析，采用更温和的优化策略

核心策略：
1. 保留原版的频率分析优势
2. 增加适度的冷号平衡（不过度）
3. 轻微排除最近2-3期（不是5期）
4. 动态权重调整
"""

import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class OptimizedTop15Predictor:
    """优化版 Top15 预测器"""
    
    def __init__(self):
        pass
    
    def analyze_pattern(self, numbers):
        """分析数字模式"""
        recent_50 = numbers[-50:] if len(numbers) >= 50 else numbers
        recent_30 = numbers[-30:]
        recent_10 = numbers[-10:]
        recent_3 = numbers[-3:]
        
        # 极端值分析
        extreme_count = sum(1 for n in recent_10 if n <= 10 or n >= 40)
        extreme_ratio = extreme_count / len(recent_10)
        
        return {
            'recent_50': recent_50,
            'recent_30': recent_30,
            'recent_10': recent_10,
            'recent_3': recent_3,
            'extreme_ratio': extreme_ratio,
            'is_extreme': extreme_ratio > 0.4
        }
    
    def method_enhanced_frequency(self, pattern, k=20):
        """方法1: 增强频率分析 - 保留原版优势 (权重30%)"""
        recent_30 = pattern['recent_30']
        recent_10 = pattern['recent_10']
        recent_3 = set(pattern['recent_3'])
        freq = Counter(recent_30)
        
        weighted = {}
        for n in range(1, 50):
            base_freq = freq.get(n, 0)
            weight = 1.0
            
            # 极端值趋势权重（原版逻辑）
            if pattern['is_extreme']:
                if n <= 10 or n >= 40:
                    weight *= 2.0
                else:
                    weight *= 0.5
            else:
                if 15 <= n <= 35:
                    weight *= 1.3
            
            # 轻度排除最近3期（不是5期）
            if n in recent_3:
                weight *= 0.7  # 降权但不完全排除
            
            # 频率加成
            if base_freq > 0:
                weight *= (1 + base_freq * 0.2)
            
            # 最近10期出现过加分
            if n in recent_10:
                weight *= 1.2
            
            weighted[n] = weight
        
        sorted_nums = sorted(weighted.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    def method_zone_smart(self, pattern, k=20):
        """方法2: 智能区域分配 (权重25%)"""
        recent_30 = pattern['recent_30']
        recent_3 = set(pattern['recent_3'])
        freq = Counter(recent_30)
        
        # 基于趋势的区域配额
        if pattern['is_extreme']:
            zones = [
                (1, 10, 4),
                (11, 20, 3),
                (21, 30, 4),
                (31, 40, 3),
                (41, 49, 6)
            ]
        else:
            zones = [
                (1, 10, 3),
                (11, 20, 4),
                (21, 30, 6),
                (31, 40, 4),
                (41, 49, 3)
            ]
        
        result = []
        for start, end, count in zones:
            zone_nums = []
            for n in range(start, end+1):
                score = freq.get(n, 0)
                # 轻度降权最近3期
                if n in recent_3:
                    score *= 0.8
                else:
                    score *= 1.1
                zone_nums.append((n, score))
            
            zone_nums.sort(key=lambda x: x[1], reverse=True)
            result.extend([n for n, _ in zone_nums[:count]])
        
        return result[:k]
    
    def method_gap_optimized(self, pattern, k=20):
        """方法3: 优化间隔预测 (权重25%)"""
        recent_30 = pattern['recent_30']
        recent_3 = set(pattern['recent_3'])
        
        # 计算间隔
        last_seen = {}
        for i, n in enumerate(recent_30):
            last_seen[n] = len(recent_30) - i
        
        candidates = {}
        for n in range(1, 50):
            gap = last_seen.get(n, 35)
            
            # 间隔权重
            if 3 <= gap <= 10:
                weight = 2.5
            elif 11 <= gap <= 20:
                weight = 2.0
            elif gap > 20:
                weight = 1.2
            elif gap == 2:
                weight = 1.5
            else:  # gap == 1
                weight = 1.0
            
            # 轻度调整最近3期
            if n in recent_3:
                weight *= 0.85
            
            candidates[n] = weight
        
        sorted_nums = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    def method_cold_warm_balance(self, pattern, k=20):
        """方法4: 冷温平衡 (权重20%)"""
        recent_30 = pattern['recent_30']
        recent_50 = pattern['recent_50']
        recent_3 = set(pattern['recent_3'])
        
        freq_30 = Counter(recent_30)
        freq_50 = Counter(recent_50)
        
        scores = {}
        for n in range(1, 50):
            score = 0.0
            
            count_30 = freq_30.get(n, 0)
            count_50 = freq_50.get(n, 0)
            
            # 温和的冷热平衡
            if count_30 == 0:
                # 30期内未出现 - 冷号
                if count_50 >= 1:
                    score += 1.5  # 之前出现过，现在变冷
                else:
                    score += 1.0  # 一直很冷
            elif count_30 == 1:
                score += 1.3  # 温号
            elif count_30 == 2:
                score += 1.0  # 正常
            else:
                score += 0.7  # 热号
            
            # 轻度调整最近3期
            if n in recent_3:
                score *= 0.9
            
            scores[n] = score
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    def predict(self, numbers):
        """预测Top15"""
        pattern = self.analyze_pattern(numbers)
        
        # 运行所有方法
        methods = [
            (self.method_enhanced_frequency(pattern, 25), 0.30),
            (self.method_zone_smart(pattern, 25), 0.25),
            (self.method_gap_optimized(pattern, 25), 0.25),
            (self.method_cold_warm_balance(pattern, 25), 0.20)
        ]
        
        # 综合评分
        scores = {}
        for candidates, weight in methods:
            for rank, num in enumerate(candidates):
                score = weight * (1.0 - rank / len(candidates))
                scores[num] = scores.get(num, 0) + score
        
        # 排序并返回Top15
        final = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in final[:15]]
    
    def get_analysis(self, numbers):
        """获取详细分析"""
        pattern = self.analyze_pattern(numbers)
        top15 = self.predict(numbers)
        
        zones = {
            '极小值区(1-10)': [n for n in top15 if 1 <= n <= 10],
            '小值区(11-20)': [n for n in top15 if 11 <= n <= 20],
            '中值区(21-30)': [n for n in top15 if 21 <= n <= 30],
            '大值区(31-40)': [n for n in top15 if 31 <= n <= 40],
            '极大值区(41-49)': [n for n in top15 if 41 <= n <= 49]
        }
        
        return {
            'top15': top15,
            'trend': '极端值趋势' if pattern['is_extreme'] else '均衡趋势',
            'extreme_ratio': pattern['extreme_ratio'] * 100,
            'zones': zones
        }


def main():
    """主函数"""
    from datetime import datetime
    
    print("=" * 80)
    print("优化版 Top15 预测器 V2")
    print("=" * 80)
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n预测时间: {current_time}")
    print("读取最新数据...")
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    
    print(f"数据加载完成: {len(df)}期")
    print(f"最近10期: {numbers[-10:].tolist()}")
    
    predictor = OptimizedTop15Predictor()
    analysis = predictor.get_analysis(numbers)
    
    print(f"\n当前趋势: {analysis['trend']}")
    print(f"极端值占比: {analysis['extreme_ratio']:.1f}%")
    
    print("\n" + "=" * 80)
    print("下一期 Top 15 预测")
    print("=" * 80)
    
    top15 = analysis['top15']
    print(f"\n预测号码:")
    print(f"  Top 1-5:   {top15[:5]}")
    print(f"  Top 6-10:  {top15[5:10]}")
    print(f"  Top 11-15: {top15[10:15]}")
    
    print(f"\n区域分布:")
    for zone, nums in analysis['zones'].items():
        if nums:
            print(f"  {zone}: {nums}")
    
    print("\n" + "=" * 80)
    print("优化要点")
    print("=" * 80)
    print("1. 保留原版频率分析的优势")
    print("2. 轻度排除最近3期（不是5期）- 降权不排除")
    print("3. 温和的冷热平衡 - 不过度偏向冷号")
    print("4. 优化间隔判断逻辑")
    print("5. 动态权重分配")


if __name__ == '__main__':
    main()
