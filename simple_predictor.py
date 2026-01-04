"""
简化训练脚本 - 避免复杂依赖和错误
"""

import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class SimplePredictor:
    """简化预测器 - 基于统计方法，无需复杂训练"""
    
    def __init__(self):
        self.element_numbers = {
            '金': [3, 4, 11, 12, 25, 26, 33, 34, 41, 42],
            '木': [7, 8, 15, 16, 23, 24, 37, 38, 45, 46],
            '水': [13, 14, 21, 22, 29, 30, 43, 44],
            '火': [1, 2, 9, 10, 17, 18, 31, 32, 39, 40, 47, 48],
            '土': [5, 6, 19, 20, 27, 28, 35, 36, 49]
        }
    
    def analyze_pattern(self, numbers):
        """分析数字模式"""
        recent_30 = numbers[-30:]
        recent_10 = numbers[-10:]
        
        # 极端值分析
        extreme_count = sum(1 for n in recent_10 if n <= 10 or n >= 40)
        extreme_ratio = extreme_count / len(recent_10)
        
        # 区域分布
        zones = {
            'extreme_small': [n for n in recent_10 if 1 <= n <= 10],
            'small': [n for n in recent_10 if 11 <= n <= 20],
            'mid': [n for n in recent_10 if 21 <= n <= 30],
            'large': [n for n in recent_10 if 31 <= n <= 40],
            'extreme_large': [n for n in recent_10 if 41 <= n <= 49]
        }
        
        return {
            'recent_30': recent_30,
            'recent_10': recent_10,
            'extreme_ratio': extreme_ratio,
            'zones': zones,
            'is_extreme': extreme_ratio > 0.4
        }
    
    def method_frequency(self, pattern, k=15):
        """方法1: 频率分析"""
        recent_30 = pattern['recent_30']
        freq = Counter(recent_30)
        
        # 根据趋势调整权重
        if pattern['is_extreme']:
            # 极端值趋势 - 增加极端值权重
            weighted = {}
            for n in range(1, 50):
                base_freq = freq.get(n, 0)
                if n <= 10 or n >= 40:
                    weighted[n] = base_freq * 2.0  # 极端值加权
                else:
                    weighted[n] = base_freq * 0.5
        else:
            weighted = {n: freq.get(n, 0) for n in range(1, 50)}
        
        sorted_nums = sorted(weighted.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    def method_extreme_focus(self, pattern, k=15):
        """方法2: 极端值重点"""
        recent_30 = pattern['recent_30']
        recent_3 = set(recent_30[-3:])
        freq = Counter(recent_30)
        
        # 极小值
        small = [(n, freq.get(n, 0)) for n in range(1, 11) if n not in recent_3]
        small.sort(key=lambda x: x[1], reverse=True)
        
        # 极大值
        large = [(n, freq.get(n, 0)) for n in range(40, 50) if n not in recent_3]
        large.sort(key=lambda x: x[1], reverse=True)
        
        # 中间值
        mid = [(n, freq.get(n, 0)) for n in range(11, 40) if n not in recent_3]
        mid.sort(key=lambda x: x[1], reverse=True)
        
        # 分配
        if pattern['is_extreme']:
            small_k = k // 3 + 1
            large_k = k // 3
            mid_k = k - small_k - large_k
        else:
            small_k = k // 5
            large_k = k // 5
            mid_k = k - small_k - large_k
        
        result = []
        result.extend([n for n, _ in small[:small_k]])
        result.extend([n for n, _ in large[:large_k]])
        result.extend([n for n, _ in mid[:mid_k]])
        
        return result[:k]
    
    def method_zone_balance(self, pattern, k=15):
        """方法3: 区域平衡"""
        recent_30 = pattern['recent_30']
        freq = Counter(recent_30)
        
        zones = [
            (1, 10, 3),    # 极小
            (11, 20, 2),   # 小
            (21, 30, 4),   # 中
            (31, 40, 3),   # 大
            (41, 49, 3)    # 极大
        ]
        
        result = []
        for start, end, count in zones:
            zone_nums = [(n, freq.get(n, 0)) for n in range(start, end+1)]
            zone_nums.sort(key=lambda x: x[1], reverse=True)
            result.extend([n for n, _ in zone_nums[:count]])
        
        return result[:k]
    
    def predict(self, numbers, top_k=15):
        """综合预测"""
        # 分析模式
        pattern = self.analyze_pattern(numbers)
        
        # 运行三种方法
        methods = [
            (self.method_frequency(pattern, top_k), 0.40),
            (self.method_extreme_focus(pattern, top_k), 0.35),
            (self.method_zone_balance(pattern, top_k), 0.25)
        ]
        
        # 综合评分
        scores = {}
        for candidates, weight in methods:
            for rank, num in enumerate(candidates):
                score = weight * (1.0 - rank / len(candidates))
                scores[num] = scores.get(num, 0) + score
        
        # 排序
        final = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in final[:top_k]]


def test_simple_predictor():
    """测试简化预测器"""
    print("=" * 80)
    print("简化预测器测试 - 无需复杂训练")
    print("=" * 80)
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    
    print(f"\n数据集: {len(df)}期")
    print(f"最近10期: {numbers[-10:].tolist()}")
    
    # 创建预测器
    predictor = SimplePredictor()
    
    # 测试最近10期
    print("\n" + "=" * 80)
    print("测试最近10期")
    print("=" * 80)
    
    results = {'top5': 0, 'top10': 0, 'top15': 0, 'top20': 0, 'details': []}
    
    for i in range(len(numbers) - 10, len(numbers)):
        actual = numbers[i]
        history = numbers[:i]
        
        print(f"\n第{i+1}期: 实际 = {actual}")
        
        # 预测
        pattern = predictor.analyze_pattern(history)
        print(f"  趋势: {'极端' if pattern['is_extreme'] else '正常'} (极端值{pattern['extreme_ratio']*100:.0f}%)")
        
        predictions = predictor.predict(history, top_k=20)
        
        # 检查命中
        if actual in predictions:
            rank = predictions.index(actual) + 1
            
            if rank <= 5:
                level = "[*] Top 5"
                results['top5'] += 1
                results['top10'] += 1
                results['top15'] += 1
                results['top20'] += 1
            elif rank <= 10:
                level = "[v] Top 10"
                results['top10'] += 1
                results['top15'] += 1
                results['top20'] += 1
            elif rank <= 15:
                level = "[o] Top 15"
                results['top15'] += 1
                results['top20'] += 1
            else:
                level = "[+] Top 20"
                results['top20'] += 1
            
            print(f"  [HIT] 命中! 排名: {rank} {level}")
        else:
            print(f"  [MISS] 未命中")
        
        print(f"  预测Top15: {predictions[:15]}")
        
        results['details'].append({
            'period': i + 1,
            'actual': actual,
            'hit': actual in predictions[:15]
        })
    
    # 统计
    print("\n" + "=" * 80)
    print("统计结果")
    print("=" * 80)
    
    total = len(results['details'])
    
    print(f"\n命中统计 (最近{total}期):")
    print(f"  Top 5:  {results['top5']}/{total} = {results['top5']/total*100:.1f}%")
    print(f"  Top 10: {results['top10']}/{total} = {results['top10']/total*100:.1f}%")
    print(f"  Top 15: {results['top15']}/{total} = {results['top15']/total*100:.1f}%")
    print(f"  Top 20: {results['top20']}/{total} = {results['top20']/total*100:.1f}%")
    
    # 对比
    for k, name in [(5, 'top5'), (10, 'top10'), (15, 'top15'), (20, 'top20')]:
        actual_rate = results[name] / total * 100
        random_rate = k / 49 * 100
        improvement = actual_rate / random_rate if random_rate > 0 else 0
        status = "[OK]" if improvement > 1.2 else "[WARN]"
        
        print(f"\n{name.upper()}:")
        print(f"  实际: {actual_rate:.1f}%")
        print(f"  随机: {random_rate:.1f}%")
        print(f"  提升: {improvement:.2f}x {status}")
    
    # 评估
    top15_rate = results['top15'] / total * 100
    top20_rate = results['top20'] / total * 100
    
    print("\n" + "=" * 80)
    print("评估")
    print("=" * 80)
    
    if top15_rate >= 60:
        print(f"\n[SUCCESS] Top 15: {top15_rate:.1f}% - 达到60%目标!")
    elif top15_rate >= 50:
        print(f"\n[GOOD] Top 15: {top15_rate:.1f}% - 超过50%")
    elif top15_rate >= 40:
        print(f"\n[OK] Top 15: {top15_rate:.1f}% - 超过随机")
    else:
        print(f"\n[WARN] Top 15: {top15_rate:.1f}%")
    
    if top20_rate >= 60:
        print(f"[SUCCESS] Top 20: {top20_rate:.1f}% - 达到60%目标!")
    elif top20_rate >= 50:
        print(f"[GOOD] Top 20: {top20_rate:.1f}% - 超过50%")
    
    print("\n推荐策略:")
    if top20_rate >= 60:
        print("  使用 Top 20 策略 (已达60%)")
    elif top15_rate >= 50:
        print("  使用 Top 15 策略 (已达50%)")
    else:
        print("  建议使用 Top 20 策略提升成功率")
    
    print("\n" + "=" * 80)
    
    return results


if __name__ == '__main__':
    test_simple_predictor()
