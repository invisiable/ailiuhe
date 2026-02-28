"""
TOP15预测器优化版 - 降低连续不中期数
新增特性:
1. 自适应预测范围（根据历史表现动态调整）
2. 冷热号码分析（避开过热号码）
3. 连续不中时的策略调整
4. 多时间窗口融合
"""

import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class EnhancedTop15Predictor:
    """增强版TOP15预测器 - 降低连续不中风险"""
    
    def __init__(self):
        self.element_numbers = {
            '金': [3, 4, 11, 12, 25, 26, 33, 34, 41, 42],
            '木': [7, 8, 15, 16, 23, 24, 37, 38, 45, 46],
            '水': [13, 14, 21, 22, 29, 30, 43, 44],
            '火': [1, 2, 9, 10, 17, 18, 31, 32, 39, 40, 47, 48],
            '土': [5, 6, 19, 20, 27, 28, 35, 36, 49]
        }
        
        # 新增：追踪连续不中情况
        self.consecutive_misses = 0
        self.recent_hits = []  # 最近20期的命中记录
        
    def update_performance(self, hit):
        """更新预测性能（用于自适应）"""
        if hit:
            self.consecutive_misses = 0
        else:
            self.consecutive_misses += 1
        
        # 保留最近20期记录
        self.recent_hits.append(hit)
        if len(self.recent_hits) > 20:
            self.recent_hits.pop(0)
    
    def get_recent_hit_rate(self):
        """获取最近命中率"""
        if not self.recent_hits:
            return 0.5
        return sum(self.recent_hits) / len(self.recent_hits)
    
    def analyze_hot_cold_numbers(self, numbers, window=20):
        """分析冷热号码"""
        recent = numbers[-window:] if len(numbers) >= window else numbers
        freq = Counter(recent)
        
        # 计算每个号码的出现频率
        hot_threshold = len(recent) / 49 * 2  # 高于平均2倍为过热
        cold_threshold = len(recent) / 49 * 0.5  # 低于平均一半为冷门
        
        hot_numbers = [n for n in range(1, 50) if freq.get(n, 0) > hot_threshold]
        cold_numbers = [n for n in range(1, 50) if freq.get(n, 0) < cold_threshold]
        normal_numbers = [n for n in range(1, 50) if n not in hot_numbers and n not in cold_numbers]
        
        return {
            'hot': set(hot_numbers),
            'cold': set(cold_numbers),
            'normal': set(normal_numbers),
            'freq': freq
        }
    
    def analyze_pattern(self, numbers):
        """增强的模式分析"""
        recent_50 = numbers[-50:] if len(numbers) >= 50 else numbers
        recent_30 = numbers[-30:] if len(numbers) >= 30 else numbers
        recent_10 = numbers[-10:]
        recent_5 = numbers[-5:]
        
        # 极端值分析
        extreme_count = sum(1 for n in recent_10 if n <= 10 or n >= 40)
        extreme_ratio = extreme_count / len(recent_10) if len(recent_10) > 0 else 0
        
        # 连续性分析
        gaps = np.diff(recent_10) if len(recent_10) > 1 else np.array([0])
        avg_gap = np.mean(np.abs(gaps))
        
        # 冷热分析
        hot_cold = self.analyze_hot_cold_numbers(numbers, window=30)
        
        # 区域分布分析
        zone_dist = {
            'zone1_10': sum(1 for n in recent_10 if 1 <= n <= 10),
            'zone11_20': sum(1 for n in recent_10 if 11 <= n <= 20),
            'zone21_30': sum(1 for n in recent_10 if 21 <= n <= 30),
            'zone31_40': sum(1 for n in recent_10 if 31 <= n <= 40),
            'zone41_49': sum(1 for n in recent_10 if 41 <= n <= 49)
        }
        
        return {
            'recent_50': recent_50,
            'recent_30': recent_30,
            'recent_10': recent_10,
            'recent_5': recent_5,
            'extreme_ratio': extreme_ratio,
            'is_extreme': extreme_ratio > 0.4,
            'avg_gap': avg_gap,
            'hot_cold': hot_cold,
            'zone_dist': zone_dist
        }
    
    def method_adaptive_frequency(self, pattern, k=25):
        """自适应频率方法 - 根据连续不中调整"""
        recent_30 = pattern['recent_30']
        recent_5 = set(pattern['recent_5'])
        freq = Counter(recent_30)
        hot_cold = pattern['hot_cold']
        
        # 连续不中时的调整系数（降低影响）
        miss_factor = min(1.2, 1.0 + self.consecutive_misses * 0.05)
        
        weighted = {}
        for n in range(1, 50):
            base_freq = freq.get(n, 0)
            weight = 1.0
            
            # 冷热号码策略（调整权重）
            if n in hot_cold['hot']:
                # 过热号码降权，但不要太激进
                weight *= 0.7
            elif n in hot_cold['cold']:
                # 冷门号码适度提权
                weight *= 1.2
            
            # 最近5期出现过的降权（但不要太低）
            if n in recent_5:
                weight *= 0.5
            
            # 频率加成（提高基础频率的重要性）
            if base_freq > 0:
                weight *= (1 + base_freq * 0.4)
            
            # 连续不中时的调整：扩大选择范围
            if self.consecutive_misses >= 5:
                # 适度提升边缘号码权重
                if n <= 8 or n >= 42:
                    weight *= miss_factor
            
            weighted[n] = weight
        
        sorted_nums = sorted(weighted.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    def method_zone_balanced(self, pattern, k=25):
        """区域平衡方法 - 确保各区域都有覆盖"""
        recent_30 = pattern['recent_30']
        recent_5 = set(pattern['recent_5'])
        zone_dist = pattern['zone_dist']
        
        # 根据最近区域分布调整配额
        zones = [
            (1, 10, 5),
            (11, 20, 5),
            (21, 30, 5),
            (31, 40, 5),
            (41, 49, 5)
        ]
        
        # 连续不中时增加每个区域的配额
        if self.consecutive_misses >= 5:
            zones = [(start, end, quota + 1) for start, end, quota in zones]
        
        candidates = []
        freq = Counter(recent_30)
        
        for start, end, quota in zones:
            zone_nums = []
            for n in range(start, end + 1):
                if n in recent_5:
                    score = 0.1
                else:
                    score = 1.0 + freq.get(n, 0) * 0.2
                zone_nums.append((n, score))
            
            zone_nums.sort(key=lambda x: x[1], reverse=True)
            candidates.extend([n for n, _ in zone_nums[:quota]])
        
        return candidates[:k]
    
    def method_gap_prediction(self, pattern, k=25):
        """间隔预测方法"""
        recent_numbers = pattern['recent_30']
        
        # 计算每个号码距离上次出现的间隔
        last_seen = {}
        for i, n in enumerate(recent_numbers):
            last_seen[n] = i
        
        current_pos = len(recent_numbers)
        gaps = {}
        
        for n in range(1, 50):
            if n in last_seen:
                gap = current_pos - last_seen[n]
            else:
                gap = 50  # 未出现过，设置大间隔
            
            # 间隔越大，越可能再次出现
            score = gap
            
            # 但如果在最近5期出现过，降权
            if n in pattern['recent_5']:
                score *= 0.2
            
            gaps[n] = score
        
        sorted_nums = sorted(gaps.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    def method_trend_following(self, pattern, k=25):
        """趋势跟踪方法"""
        recent_10 = pattern['recent_10']
        recent_30 = pattern['recent_30']
        
        # 检测上升/下降趋势
        if len(recent_10) >= 3:
            last_3_avg = np.mean(recent_10[-3:])
            prev_3_avg = np.mean(recent_10[-6:-3]) if len(recent_10) >= 6 else np.mean(recent_10[:3])
            
            trend = 'up' if last_3_avg > prev_3_avg else 'down'
        else:
            trend = 'neutral'
        
        weighted = {}
        for n in range(1, 50):
            score = 1.0
            
            if trend == 'up':
                # 上升趋势，偏好较大数字
                if n >= 25:
                    score *= 1.5
            elif trend == 'down':
                # 下降趋势，偏好较小数字
                if n <= 25:
                    score *= 1.5
            
            # 检查在30期内的出现次数
            freq = np.sum(recent_30 == n)
            score *= (1 + freq * 0.1)
            
            # 避开最近5期
            if n in pattern['recent_5']:
                score *= 0.3
            
            weighted[n] = score
        
        sorted_nums = sorted(weighted.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:k]]
    
    def predict(self, numbers, adaptive=True):
        """
        增强预测方法
        adaptive: 是否启用自适应（根据连续不中情况调整）
        """
        pattern = self.analyze_pattern(numbers)
        
        # 根据连续不中情况动态调整候选数量（更保守）
        base_k = 25
        if adaptive and self.consecutive_misses >= 7:
            base_k = 28  # 稍微扩大候选池
        elif adaptive and self.consecutive_misses >= 12:
            base_k = 30  # 进一步扩大
        
        # 运行多个方法（调整权重，更重视频率）
        methods = [
            (self.method_adaptive_frequency(pattern, base_k), 0.35),  # 提高频率法权重
            (self.method_zone_balanced(pattern, base_k), 0.25),
            (self.method_gap_prediction(pattern, base_k), 0.2),
            (self.method_trend_following(pattern, base_k), 0.2)
        ]
        
        # 综合评分
        scores = {}
        for candidates, weight in methods:
            for rank, num in enumerate(candidates):
                score = weight * (1.0 - rank / len(candidates))
                scores[num] = scores.get(num, 0) + score
        
        # 排序并返回Top N
        final = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 根据连续不中情况调整返回数量（更保守）
        if adaptive and self.consecutive_misses >= 8:
            return_count = 17  # 略微提升到17个号码
        elif adaptive and self.consecutive_misses >= 12:
            return_count = 18  # 进一步提升到18个
        else:
            return_count = 15  # 默认15个
        
        return [num for num, _ in final[:return_count]]
    
    def get_analysis(self, numbers):
        """获取详细分析（兼容原接口）"""
        pattern = self.analyze_pattern(numbers)
        prediction = self.predict(numbers)
        
        # 分析预测结果
        zones = {
            '极小值区(1-10)': [n for n in prediction if 1 <= n <= 10],
            '小值区(11-20)': [n for n in prediction if 11 <= n <= 20],
            '中值区(21-30)': [n for n in prediction if 21 <= n <= 30],
            '大值区(31-40)': [n for n in prediction if 31 <= n <= 40],
            '极大值区(41-49)': [n for n in prediction if 41 <= n <= 49]
        }
        
        elements = {'金': [], '木': [], '水': [], '火': [], '土': []}
        for n in prediction:
            for element, nums in self.element_numbers.items():
                if n in nums:
                    elements[element].append(n)
                    break
        
        # 冷热分析
        hot_cold = pattern['hot_cold']
        hot_in_pred = [n for n in prediction if n in hot_cold['hot']]
        cold_in_pred = [n for n in prediction if n in hot_cold['cold']]
        
        return {
            'top15': prediction,  # 实际可能是15-20个
            'trend': '极端值趋势' if pattern['is_extreme'] else '正常趋势',
            'extreme_ratio': pattern['extreme_ratio'] * 100,
            'zones': zones,
            'elements': {k: v for k, v in elements.items() if v},
            'consecutive_misses': self.consecutive_misses,
            'recent_hit_rate': self.get_recent_hit_rate() * 100,
            'hot_numbers': len(hot_in_pred),
            'cold_numbers': len(cold_in_pred),
            'prediction_count': len(prediction)
        }


def main():
    """测试增强预测器"""
    from datetime import datetime
    
    print("=" * 80)
    print("增强版 Top 15 预测器 - 降低连续不中风险")
    print("=" * 80)
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n预测时间: {current_time}")
    print("🔄 读取最新数据...")
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    
    print(f"✅ 数据加载完成")
    print(f"基于历史数据: {len(df)}期")
    print(f"最近10期: {numbers[-10:].tolist()}")
    
    # 创建预测器
    predictor = EnhancedTop15Predictor()
    
    # 获取分析
    analysis = predictor.get_analysis(numbers)
    
    print(f"\n当前趋势分析:")
    print(f"  趋势判断: {analysis['trend']}")
    print(f"  极端值占比: {analysis['extreme_ratio']:.0f}% (最近10期)")
    print(f"  连续不中: {analysis['consecutive_misses']}期")
    print(f"  最近命中率: {analysis['recent_hit_rate']:.1f}%")
    
    print("\n" + "=" * 80)
    print("🎯 下一期预测")
    print("=" * 80)
    
    prediction = analysis['top15']
    print(f"\n预测号码 (共{analysis['prediction_count']}个):")
    print(f"  {prediction}")
    
    # 分组显示
    print(f"\n按区域分布:")
    for zone, nums in analysis['zones'].items():
        if nums:
            print(f"  {zone}: {nums}")
    
    print(f"\n按五行分布:")
    for element, nums in analysis['elements'].items():
        print(f"  {element}: {nums}")
    
    print(f"\n冷热分析:")
    print(f"  包含热号: {analysis['hot_numbers']}个")
    print(f"  包含冷号: {analysis['cold_numbers']}个")
    
    print("\n" + "=" * 80)
    print("✅ 预测完成")
    print("=" * 80)
    print("\n优化特性:")
    print("  • 自适应预测数量（15-20个，根据连续不中情况调整）")
    print("  • 冷热号码分析（避开过热号码）")
    print("  • 区域平衡策略（确保各区域覆盖）")
    print("  • 间隔预测（优先选择久未出现的号码）")
    print("  • 趋势跟踪（识别上升/下降趋势）")


if __name__ == '__main__':
    main()
