"""
Advanced Top 15 Predictor - 新一代预测模型
目标：最近100期成功率达到60%

核心创新：
1. 多维度统计模型（9种方法）
2. 智能权重自适应
3. 人工规则辅助
4. 趋势动态感知
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')


class AdvancedTop15Predictor:
    """高级Top 15预测器 - 目标60%成功率"""
    
    def __init__(self):
        # 五行映射
        self.element_numbers = {
            '金': [3, 4, 11, 12, 25, 26, 33, 34, 41, 42],
            '木': [7, 8, 15, 16, 23, 24, 37, 38, 45, 46],
            '水': [13, 14, 21, 22, 29, 30, 43, 44],
            '火': [1, 2, 9, 10, 17, 18, 31, 32, 39, 40, 47, 48],
            '土': [5, 6, 19, 20, 27, 28, 35, 36, 49]
        }
        
        # 生肖映射
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
        
        # 区域定义
        self.zones = {
            '极小': (1, 10),
            '小': (11, 20),
            '中': (21, 30),
            '大': (31, 40),
            '极大': (41, 49)
        }
    
    def analyze_deep_pattern(self, numbers):
        """深度模式分析"""
        recent_100 = numbers[-100:] if len(numbers) >= 100 else numbers
        recent_50 = numbers[-50:]
        recent_30 = numbers[-30:]
        recent_10 = numbers[-10:]
        recent_5 = numbers[-5:]
        recent_3 = numbers[-3:]
        
        # 转换为列表（避免numpy数组问题）
        if hasattr(recent_100, 'tolist'):
            recent_100 = recent_100.tolist()
            recent_50 = recent_50.tolist()
            recent_30 = recent_30.tolist()
            recent_10 = recent_10.tolist()
            recent_5 = recent_5.tolist()
            recent_3 = recent_3.tolist()
        
        # 1. 趋势分析
        extreme_count_10 = sum(1 for n in recent_10 if n <= 10 or n >= 40)
        extreme_ratio = extreme_count_10 / len(recent_10)
        
        # 2. 波动分析
        volatility = np.std(recent_10)
        
        # 3. 连续性分析
        gaps = np.diff(recent_10)
        avg_gap = np.mean(np.abs(gaps))
        
        # 4. 周期性分析（多周期）
        periods = {
            'p3': recent_30[-6:-3] if len(recent_30) >= 6 else recent_30[:3],
            'p5': recent_30[-10:-5] if len(recent_30) >= 10 else recent_30[:5],
            'p7': recent_30[-14:-7] if len(recent_30) >= 14 else recent_30[:7],
            'p10': recent_30[-20:-10] if len(recent_30) >= 20 else recent_30[:10]
        }
        
        # 5. 五行平衡分析
        element_dist = self._analyze_element_distribution(recent_10)
        
        # 6. 奇偶平衡分析
        odd_count = sum(1 for n in recent_10 if n % 2 == 1)
        odd_ratio = odd_count / len(recent_10)
        
        # 7. 尾数分析
        tail_dist = Counter([n % 10 for n in recent_10])
        
        # 8. 区域跳转模式
        zone_jumps = self._analyze_zone_jumps(recent_10)
        
        return {
            'recent_100': recent_100,
            'recent_50': recent_50,
            'recent_30': recent_30,
            'recent_10': recent_10,
            'recent_5': recent_5,
            'recent_3': recent_3,
            'extreme_ratio': extreme_ratio,
            'is_extreme': extreme_ratio >= 0.4,
            'volatility': volatility,
            'avg_gap': avg_gap,
            'periods': periods,
            'element_dist': element_dist,
            'odd_ratio': odd_ratio,
            'tail_dist': tail_dist,
            'zone_jumps': zone_jumps
        }
    
    def _analyze_element_distribution(self, numbers):
        """分析五行分布"""
        dist = {element: 0 for element in self.element_numbers}
        for n in numbers:
            for element, nums in self.element_numbers.items():
                if n in nums:
                    dist[element] += 1
                    break
        return dist
    
    def _analyze_zone_jumps(self, numbers):
        """分析区域跳转模式"""
        def get_zone(n):
            for zone, (start, end) in self.zones.items():
                if start <= n <= end:
                    return zone
            return None
        
        jumps = []
        for i in range(1, len(numbers)):
            prev_zone = get_zone(numbers[i-1])
            curr_zone = get_zone(numbers[i])
            jumps.append((prev_zone, curr_zone))
        return jumps
    
    # ==================== 9种预测方法 ====================
    
    def method1_weighted_frequency(self, pattern, k=25):
        """方法1：加权频率分析 - 多时间窗口"""
        scores = defaultdict(float)
        recent_5 = set(pattern['recent_5'])
        
        # 多时间窗口权重
        time_windows = [
            (pattern['recent_100'], 0.15),
            (pattern['recent_50'], 0.20),
            (pattern['recent_30'], 0.25),
            (pattern['recent_10'], 0.40)
        ]
        
        for window, weight in time_windows:
            freq = Counter(window)
            for n, count in freq.items():
                # 最近5期出现过的降权
                penalty = 0.3 if n in recent_5 else 1.0
                scores[n] += count * weight * penalty
        
        return self._get_top_k(scores, k)
    
    def method2_adaptive_zone(self, pattern, k=25):
        """方法2：自适应区域分配"""
        recent_30 = pattern['recent_30']
        recent_5 = set(pattern['recent_5'])
        freq = Counter(recent_30)
        
        # 根据趋势动态调整区域配额
        if pattern['is_extreme']:
            # 极端值趋势
            zone_quotas = {
                '极小': 6, '小': 3, '中': 5, '大': 3, '极大': 8
            }
        elif pattern['volatility'] > 15:
            # 高波动
            zone_quotas = {
                '极小': 4, '小': 5, '中': 6, '大': 5, '极大': 5
            }
        else:
            # 正常趋势
            zone_quotas = {
                '极小': 4, '小': 5, '中': 7, '大': 5, '极大': 4
            }
        
        result = []
        for zone, quota in zone_quotas.items():
            start, end = self.zones[zone]
            zone_nums = [
                (n, freq.get(n, 0) + np.random.random() * 0.1)
                for n in range(start, end + 1)
                if n not in recent_5
            ]
            zone_nums.sort(key=lambda x: x[1], reverse=True)
            result.extend([n for n, _ in zone_nums[:quota]])
        
        return result[:k]
    
    def method3_cyclic_multi_period(self, pattern, k=25):
        """方法3：多周期循环模式"""
        scores = defaultdict(float)
        
        # 多个周期权重
        period_weights = {
            'p3': 0.35,   # 3期周期
            'p5': 0.30,   # 5期周期
            'p7': 0.20,   # 7期周期
            'p10': 0.15   # 10期周期
        }
        
        for period_name, weight in period_weights.items():
            period_data = pattern['periods'][period_name]
            freq = Counter(period_data)
            for n, count in freq.items():
                scores[n] += count * weight * 2.0
        
        # 补充最近趋势
        recent_freq = Counter(pattern['recent_30'][-20:])
        for n, count in recent_freq.items():
            scores[n] += count * 0.3
        
        return self._get_top_k(scores, k)
    
    def method4_gap_intelligent(self, pattern, k=25):
        """方法4：智能间隔预测"""
        recent_50 = pattern['recent_50']
        recent_5 = set(pattern['recent_5'])
        
        # 计算间隔
        last_seen = {}
        for i, n in enumerate(recent_50):
            last_seen[n] = len(recent_50) - i
        
        scores = {}
        for n in range(1, 50):
            gap = last_seen.get(n, 50)
            
            # 智能间隔评分
            if 4 <= gap <= 12:
                score = 2.5
            elif 2 <= gap <= 18:
                score = 2.0
            elif 19 <= gap <= 30:
                score = 1.5 + (gap - 19) * 0.05
            elif gap > 30:
                score = 2.0 + (gap - 30) * 0.08
            else:
                score = 0.5
            
            # 最近5期出现过的降权
            if n in recent_5:
                score *= 0.2
            
            scores[n] = score
        
        return self._get_top_k(scores, k)
    
    def method5_element_balance(self, pattern, k=25):
        """方法5：五行平衡策略"""
        element_dist = pattern['element_dist']
        recent_10 = pattern['recent_10']
        
        # 找出缺失的五行
        min_count = min(element_dist.values())
        lacking_elements = [e for e, c in element_dist.items() if c <= min_count + 1]
        
        scores = defaultdict(float)
        
        # 倾向于补充缺失的五行
        for element in lacking_elements:
            for n in self.element_numbers[element]:
                if n not in recent_10:
                    scores[n] += 2.0
        
        # 加入频率因素
        freq = Counter(pattern['recent_30'])
        for n, count in freq.items():
            scores[n] += count * 0.5
        
        return self._get_top_k(scores, k)
    
    def method6_odd_even_balance(self, pattern, k=25):
        """方法6：奇偶平衡策略"""
        odd_ratio = pattern['odd_ratio']
        recent_5 = set(pattern['recent_5'])
        freq = Counter(pattern['recent_30'])
        
        scores = {}
        for n in range(1, 50):
            score = freq.get(n, 0) * 0.5
            
            # 奇偶平衡调整
            if odd_ratio > 0.6 and n % 2 == 0:
                score *= 1.8
            elif odd_ratio < 0.4 and n % 2 == 1:
                score *= 1.8
            
            # 最近5期降权
            if n in recent_5:
                score *= 0.3
            
            scores[n] = score + 0.5
        
        return self._get_top_k(scores, k)
    
    def method7_tail_pattern(self, pattern, k=25):
        """方法7：尾数模式预测"""
        tail_dist = pattern['tail_dist']
        recent_5 = set(pattern['recent_5'])
        
        # 找出冷门尾数
        avg_count = sum(tail_dist.values()) / len(tail_dist) if tail_dist else 1
        hot_tails = [t for t, c in tail_dist.items() if c > avg_count]
        cold_tails = [t for t in range(10) if tail_dist.get(t, 0) < avg_count]
        
        scores = defaultdict(float)
        freq = Counter(pattern['recent_30'])
        
        for n in range(1, 50):
            score = freq.get(n, 0) * 0.3
            
            # 尾数调整
            tail = n % 10
            if tail in cold_tails:
                score += 1.5
            elif tail in hot_tails:
                score += 0.5
            
            # 最近5期降权
            if n in recent_5:
                score *= 0.2
            
            scores[n] = score + 0.3
        
        return self._get_top_k(scores, k)
    
    def method8_zone_jump_prediction(self, pattern, k=25):
        """方法8：区域跳转预测"""
        jumps = pattern['zone_jumps']
        recent_3 = pattern['recent_3']
        
        # 获取最近的区域
        def get_zone(n):
            for zone, (start, end) in self.zones.items():
                if start <= n <= end:
                    return zone
            return None
        
        last_zone = get_zone(recent_3[-1]) if recent_3 else '中'
        
        # 统计跳转模式
        jump_patterns = defaultdict(int)
        for prev_zone, curr_zone in jumps:
            jump_patterns[(prev_zone, curr_zone)] += 1
        
        # 预测最可能的目标区域
        target_zones = defaultdict(int)
        for (prev, curr), count in jump_patterns.items():
            if prev == last_zone:
                target_zones[curr] += count
        
        # 如果没有历史跳转，使用默认
        if not target_zones:
            target_zones = {'极小': 1, '小': 1, '中': 2, '大': 1, '极大': 1}
        
        # 从目标区域选择号码
        scores = defaultdict(float)
        freq = Counter(pattern['recent_30'])
        
        for zone, weight in target_zones.items():
            start, end = self.zones[zone]
            for n in range(start, end + 1):
                scores[n] = freq.get(n, 0) * 0.5 + weight
        
        return self._get_top_k(scores, k)
    
    def method9_artificial_rules(self, pattern, k=25):
        """方法9：人工经验规则"""
        recent_10 = pattern['recent_10']
        recent_5 = set(pattern['recent_5'])
        recent_3 = pattern['recent_3']
        
        scores = defaultdict(float)
        freq = Counter(pattern['recent_30'])
        
        # 规则1：连续3期都在某个区间，下期可能跳出
        zone_3 = [self._get_number_zone(n) for n in recent_3]
        if len(set(zone_3)) == 1:
            # 偏好其他区域
            avoid_zone = zone_3[0]
            for n in range(1, 50):
                if self._get_number_zone(n) != avoid_zone:
                    scores[n] += 1.5
        
        # 规则2：最近10期未出现的热门号码
        all_freq = Counter(pattern['recent_100'])
        hot_numbers = [n for n, _ in all_freq.most_common(20)]
        for n in hot_numbers:
            if n not in recent_10:
                scores[n] += 2.0
        
        # 规则3：极端值反弹
        extreme_recent = [n for n in recent_3 if n <= 10 or n >= 40]
        if len(extreme_recent) >= 2:
            # 偏好中间值
            for n in range(15, 36):
                scores[n] += 1.2
        
        # 规则4：波动调整
        if pattern['volatility'] < 10:
            # 低波动，可能出现跳跃
            for n in range(1, 50):
                if abs(n - recent_3[-1]) > 15:
                    scores[n] += 1.0
        
        # 基础频率
        for n, count in freq.items():
            scores[n] += count * 0.3
        
        # 最近5期降权
        for n in recent_5:
            scores[n] *= 0.2
        
        return self._get_top_k(scores, k)
    
    def _get_number_zone(self, n):
        """获取数字所在区域"""
        for zone, (start, end) in self.zones.items():
            if start <= n <= end:
                return zone
        return '中'
    
    def _get_top_k(self, scores, k):
        """从评分字典中获取TopK"""
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_items[:k]]
    
    # ==================== 智能融合 ====================
    
    def predict(self, numbers):
        """智能预测Top 15"""
        # 深度分析
        pattern = self.analyze_deep_pattern(numbers)
        
        # 执行9种方法
        methods = [
            (self.method1_weighted_frequency(pattern, 25), 0.15),
            (self.method2_adaptive_zone(pattern, 25), 0.12),
            (self.method3_cyclic_multi_period(pattern, 25), 0.13),
            (self.method4_gap_intelligent(pattern, 25), 0.12),
            (self.method5_element_balance(pattern, 25), 0.10),
            (self.method6_odd_even_balance(pattern, 25), 0.10),
            (self.method7_tail_pattern(pattern, 25), 0.08),
            (self.method8_zone_jump_prediction(pattern, 25), 0.10),
            (self.method9_artificial_rules(pattern, 25), 0.10)
        ]
        
        # 自适应权重调整
        if pattern['is_extreme']:
            # 极端值趋势，调整权重
            methods[1] = (methods[1][0], 0.18)  # 增强区域方法
            methods[8] = (methods[8][0], 0.15)  # 增强人工规则
        
        if pattern['volatility'] > 15:
            # 高波动，增强间隔和跳转预测
            methods[3] = (methods[3][0], 0.15)
            methods[7] = (methods[7][0], 0.13)
        
        # 重新归一化权重
        total_weight = sum(w for _, w in methods)
        methods = [(candidates, w / total_weight) for candidates, w in methods]
        
        # 综合评分
        final_scores = defaultdict(float)
        for candidates, weight in methods:
            for rank, num in enumerate(candidates):
                # 位置衰减评分
                position_score = 1.0 - (rank / len(candidates)) * 0.5
                final_scores[num] += weight * position_score
        
        # 排序并返回Top 15
        sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_results[:15]]
    
    def get_analysis(self, numbers):
        """获取详细分析结果"""
        pattern = self.analyze_deep_pattern(numbers)
        top15 = self.predict(numbers)
        
        # 区域分布
        zones = {}
        for zone_name, (start, end) in self.zones.items():
            zone_nums = [n for n in top15 if start <= n <= end]
            if zone_nums:
                zones[f'{zone_name}区({start}-{end})'] = zone_nums
        
        # 五行分布
        elements = defaultdict(list)
        for n in top15:
            for element, nums in self.element_numbers.items():
                if n in nums:
                    elements[element].append(n)
                    break
        
        # 奇偶分布
        odd_nums = [n for n in top15 if n % 2 == 1]
        even_nums = [n for n in top15 if n % 2 == 0]
        
        return {
            'top15': top15,
            'trend': '极端值趋势' if pattern['is_extreme'] else '正常趋势',
            'extreme_ratio': pattern['extreme_ratio'] * 100,
            'volatility': pattern['volatility'],
            'odd_ratio': pattern['odd_ratio'] * 100,
            'zones': zones,
            'elements': dict(elements),
            'odd_nums': odd_nums,
            'even_nums': even_nums
        }


def main():
    """测试函数"""
    print("=" * 80)
    print("Advanced Top 15 Predictor - 新一代预测模型")
    print("=" * 80)
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    
    print(f"\n✅ 数据加载: {len(numbers)}期")
    print(f"   最近10期: {numbers[-10:].tolist()}")
    
    # 创建预测器
    predictor = AdvancedTop15Predictor()
    
    # 获取分析
    analysis = predictor.get_analysis(numbers)
    
    print("\n" + "=" * 80)
    print("📊 趋势分析")
    print("=" * 80)
    print(f"  趋势类型: {analysis['trend']}")
    print(f"  极端值占比: {analysis['extreme_ratio']:.1f}%")
    print(f"  波动率: {analysis['volatility']:.2f}")
    print(f"  奇数占比: {analysis['odd_ratio']:.1f}%")
    
    print("\n" + "=" * 80)
    print("🎯 Top 15 预测")
    print("=" * 80)
    print(f"\n预测号码: {analysis['top15']}")
    
    print(f"\n区域分布:")
    for zone, nums in analysis['zones'].items():
        print(f"  {zone}: {nums}")
    
    print(f"\n五行分布:")
    for element, nums in analysis['elements'].items():
        print(f"  {element}: {nums}")
    
    print(f"\n奇偶分布:")
    print(f"  奇数({len(analysis['odd_nums'])}): {analysis['odd_nums']}")
    print(f"  偶数({len(analysis['even_nums'])}): {analysis['even_nums']}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
