"""
深度特征分析TOP15预测器
利用发现的统计偏差和深度模式分析
"""

import numpy as np
import pandas as pd
from collections import Counter, deque
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


class DeepFeatureTop15Predictor:
    """
    深度特征分析TOP15预测器
    
    核心发现：
    - 区间41-49显著高于期望(+28%)
    - 区间31-40显著低于期望(-24%)
    
    策略：
    1. 利用区间偏差
    2. 深度序列模式分析
    3. 多维特征融合
    4. 自适应区间动态调整
    """
    
    def __init__(self):
        self.all_numbers = list(range(1, 50))
        
        # 区间定义
        self.zones = {
            'extreme_low': (1, 10),
            'low': (11, 20),
            'mid': (21, 30),
            'high': (31, 40),
            'extreme_high': (41, 49)
        }
        
        # 历史表现追踪
        self.zone_performance = deque(maxlen=50)
    
    def analyze_zone_bias(self, numbers, window=100):
        """分析各区间的统计偏差"""
        recent = numbers[-window:] if len(numbers) >= window else numbers
        
        bias = {}
        for zone_name, (z_start, z_end) in self.zones.items():
            actual = sum(1 for n in recent if z_start <= n <= z_end)
            expected = len(recent) * (z_end - z_start + 1) / 49
            bias[zone_name] = {
                'actual': actual,
                'expected': expected,
                'deviation': actual - expected,
                'ratio': actual / expected if expected > 0 else 1
            }
        
        return bias
    
    def method_bias_exploitation(self, numbers, k=20):
        """
        偏差利用法
        根据统计偏差调整区间选择
        """
        if len(numbers) < 30:
            return self._simple_freq(numbers, k)
        
        # 分析偏差
        bias = self.analyze_zone_bias(numbers, window=100)
        
        # 计算各区间的选择权重
        zone_weights = {}
        for zone_name, data in bias.items():
            # 原则：高频区间多选，但考虑回归均值
            ratio = data['ratio']
            
            # 如果比率>1.1，该区间"热"，适当增加
            # 如果比率<0.9，该区间"冷"，考虑回归
            if ratio > 1.1:
                weight = ratio * 0.8  # 热区适度追涨
            elif ratio < 0.9:
                weight = 1 / ratio * 0.5  # 冷区考虑回归
            else:
                weight = 1.0
            
            zone_weights[zone_name] = weight
        
        # 归一化权重
        total_weight = sum(zone_weights.values())
        for zone_name in zone_weights:
            zone_weights[zone_name] /= total_weight
        
        # 分配每个区间的选择数量
        zone_selection = {}
        remaining = k
        for zone_name, weight in sorted(zone_weights.items(), key=lambda x: x[1], reverse=True):
            count = max(1, int(k * weight))
            zone_selection[zone_name] = min(count, remaining)
            remaining -= zone_selection[zone_name]
            if remaining <= 0:
                break
        
        # 在每个区间内选择号码
        recent_30 = numbers[-30:]
        recent_5 = numbers[-5:]
        freq = Counter(recent_30)
        
        selected = []
        for zone_name, count in zone_selection.items():
            z_start, z_end = self.zones[zone_name]
            zone_nums = list(range(z_start, z_end + 1))
            
            # 评分
            zone_scores = {}
            for n in zone_nums:
                freq_score = freq.get(n, 0) / 30
                
                # 间隔得分
                if n in recent_30:
                    last_pos = max([i for i, x in enumerate(recent_30) if x == n])
                    gap = len(recent_30) - 1 - last_pos
                    gap_score = min(1.0, gap / 15)
                else:
                    gap_score = 0.7
                
                # 避开最近5期
                if n in recent_5:
                    penalty = 0.15
                else:
                    penalty = 1.0
                
                zone_scores[n] = (freq_score * 0.3 + gap_score * 0.7) * penalty
            
            # 选择
            sorted_zone = sorted(zone_scores.items(), key=lambda x: x[1], reverse=True)
            for n, _ in sorted_zone[:count]:
                if n not in selected:
                    selected.append(n)
        
        return selected[:k]
    
    def method_sequence_pattern(self, numbers, k=20):
        """
        序列模式分析
        识别号码序列中的隐藏模式
        """
        if len(numbers) < 50:
            return self._simple_freq(numbers, k)
        
        recent_50 = numbers[-50:]
        recent_30 = numbers[-30:]
        recent_5 = numbers[-5:]
        
        # 模式1: 差值模式
        diffs = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
        recent_diffs = diffs[-20:]
        avg_diff = np.mean(recent_diffs)
        
        # 模式2: 周期性分析
        # 检查是否有号码在固定间隔出现
        periodicity_scores = {}
        for n in self.all_numbers:
            positions = [i for i, x in enumerate(recent_50) if x == n]
            if len(positions) >= 2:
                gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                if len(gaps) >= 2:
                    gap_std = np.std(gaps)
                    gap_mean = np.mean(gaps)
                    # 间隔稳定性得分（标准差越小越稳定）
                    periodicity_scores[n] = 1 / (1 + gap_std)
                else:
                    periodicity_scores[n] = 0.3
            else:
                periodicity_scores[n] = 0.2
        
        # 模式3: 相邻相关性
        # 分析当前号码与下一期的关系
        transition_scores = {}
        last_num = numbers[-1]
        
        # 统计历史上last_num之后出现的号码
        next_counts = Counter()
        for i in range(len(numbers) - 1):
            if numbers[i] == last_num:
                next_counts[numbers[i+1]] += 1
        
        for n in self.all_numbers:
            transition_scores[n] = next_counts.get(n, 0) + 0.1
        
        # 归一化
        max_trans = max(transition_scores.values())
        for n in transition_scores:
            transition_scores[n] /= max_trans
        
        # 综合评分
        freq = Counter(recent_30)
        scores = {}
        
        for n in self.all_numbers:
            freq_score = freq.get(n, 0) / 30
            period_score = periodicity_scores.get(n, 0)
            trans_score = transition_scores.get(n, 0)
            
            # 差值预测
            predicted_from_diff = last_num + int(avg_diff)
            if 1 <= predicted_from_diff <= 49:
                diff_bonus = 1 - abs(n - predicted_from_diff) / 49
            else:
                diff_bonus = 0.3
            
            scores[n] = (freq_score * 0.25 + 
                        period_score * 0.25 + 
                        trans_score * 0.30 + 
                        diff_bonus * 0.20)
            
            # 避开最近5期
            if n in recent_5:
                scores[n] *= 0.2
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in sorted_nums[:k]]
    
    def method_hot_cold_balance(self, numbers, k=20):
        """
        冷热平衡法
        平衡热号追涨和冷号回归
        """
        if len(numbers) < 50:
            return self._simple_freq(numbers, k)
        
        recent_100 = numbers[-100:] if len(numbers) >= 100 else numbers
        recent_30 = numbers[-30:]
        recent_10 = numbers[-10:]
        recent_5 = numbers[-5:]
        
        freq_100 = Counter(recent_100)
        freq_30 = Counter(recent_30)
        freq_10 = Counter(recent_10)
        
        scores = {}
        
        for n in self.all_numbers:
            # 长期频率
            long_freq = freq_100.get(n, 0) / len(recent_100)
            
            # 中期频率
            mid_freq = freq_30.get(n, 0) / 30
            
            # 短期频率
            short_freq = freq_10.get(n, 0) / 10
            
            # 趋势判断
            if short_freq > mid_freq > long_freq:
                # 上升趋势（热号）
                trend_bonus = 0.3
            elif short_freq < mid_freq < long_freq:
                # 下降趋势（变冷）- 可能回归
                trend_bonus = 0.4
            else:
                trend_bonus = 0.2
            
            # 间隔分析
            if n in recent_30:
                last_pos = max([i for i, x in enumerate(recent_30) if x == n])
                gap = len(recent_30) - 1 - last_pos
                if 5 <= gap <= 15:
                    gap_bonus = 0.4  # 最佳间隔
                elif gap > 15:
                    gap_bonus = 0.3  # 可能要出
                else:
                    gap_bonus = 0.1  # 刚出过
            else:
                gap_bonus = 0.25  # 久未出现
            
            scores[n] = (long_freq * 0.15 + 
                        mid_freq * 0.25 + 
                        trend_bonus + 
                        gap_bonus)
            
            # 避开最近5期
            if n in recent_5:
                scores[n] *= 0.15
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in sorted_nums[:k]]
    
    def method_number_genetics(self, numbers, k=20):
        """
        号码基因法
        分析号码的"基因特征"（奇偶、大小、质合等）
        """
        if len(numbers) < 30:
            return self._simple_freq(numbers, k)
        
        recent_20 = numbers[-20:]
        recent_5 = numbers[-5:]
        
        # 分析最近的基因分布
        odd_ratio = sum(1 for n in recent_20 if n % 2 == 1) / 20
        big_ratio = sum(1 for n in recent_20 if n > 25) / 20
        prime_ratio = sum(1 for n in recent_20 if self._is_prime(n)) / 20
        
        # 期望值（平衡状态）
        expected_odd = 0.5
        expected_big = 0.49  # 24个大数/49
        expected_prime = 15/49  # 1-49中有15个质数
        
        # 计算偏差，预测回归方向
        need_odd = expected_odd > odd_ratio
        need_big = expected_big > big_ratio
        need_prime = expected_prime > prime_ratio
        
        # 评分
        scores = {}
        freq = Counter(recent_20)
        
        for n in self.all_numbers:
            base_score = freq.get(n, 0) / 20 + 0.1
            
            gene_bonus = 0
            
            # 奇偶匹配
            if (n % 2 == 1) == need_odd:
                gene_bonus += 0.15
            
            # 大小匹配
            if (n > 25) == need_big:
                gene_bonus += 0.15
            
            # 质合匹配
            if self._is_prime(n) == need_prime:
                gene_bonus += 0.10
            
            scores[n] = base_score + gene_bonus
            
            # 避开最近5期
            if n in recent_5:
                scores[n] *= 0.2
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in sorted_nums[:k]]
    
    def _is_prime(self, n):
        """判断质数"""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def _simple_freq(self, numbers, k):
        """简单频率法"""
        freq = Counter(numbers[-30:] if len(numbers) >= 30 else numbers)
        sorted_nums = sorted(self.all_numbers, key=lambda x: freq.get(x, 0), reverse=True)
        return sorted_nums[:k]
    
    def predict(self, numbers):
        """
        综合预测
        融合所有深度特征方法
        """
        if len(numbers) < 30:
            return self._simple_freq(numbers, 15)
        
        # 获取各方法预测
        methods = {
            'bias': self.method_bias_exploitation(numbers, 20),
            'sequence': self.method_sequence_pattern(numbers, 20),
            'hot_cold': self.method_hot_cold_balance(numbers, 20),
            'genetics': self.method_number_genetics(numbers, 20)
        }
        
        # 权重
        weights = {
            'bias': 0.30,
            'sequence': 0.25,
            'hot_cold': 0.25,
            'genetics': 0.20
        }
        
        # 综合评分
        scores = {}
        for method_name, preds in methods.items():
            weight = weights[method_name]
            for rank, num in enumerate(preds):
                score = weight * (1.0 - rank / len(preds))
                scores[num] = scores.get(num, 0) + score
        
        # 返回TOP15
        final = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in final[:15]]


def validate_deep_feature_predictor(test_periods=200):
    """验证深度特征预测器"""
    print("="*80)
    print("深度特征分析TOP15预测器验证")
    print("="*80)
    
    df = pd.read_csv('data/lucky_numbers.csv')
    numbers_all = df['number'].tolist()
    
    total_periods = len(numbers_all)
    start_idx = total_periods - test_periods
    
    predictor = DeepFeatureTop15Predictor()
    
    # 测试各个方法
    methods = {
        '偏差利用': predictor.method_bias_exploitation,
        '序列模式': predictor.method_sequence_pattern,
        '冷热平衡': predictor.method_hot_cold_balance,
        '号码基因': predictor.method_number_genetics,
    }
    
    results = {}
    
    for name, method in methods.items():
        print(f"\n测试: {name}")
        hits = 0
        max_miss = 0
        miss = 0
        
        for i in range(start_idx, total_periods):
            history = numbers_all[:i]
            actual = numbers_all[i]
            
            preds = method(history, k=15)
            if actual in preds:
                hits += 1
                miss = 0
            else:
                miss += 1
                max_miss = max(max_miss, miss)
        
        hit_rate = hits / test_periods * 100
        results[name] = {
            'hits': hits,
            'hit_rate': hit_rate,
            'max_miss': max_miss
        }
        print(f"  命中: {hits}/{test_periods} ({hit_rate:.2f}%)")
        print(f"  最大连不中: {max_miss}期")
    
    # 测试综合预测
    print(f"\n测试: 深度特征综合")
    hits = 0
    max_miss = 0
    miss = 0
    
    for i in range(start_idx, total_periods):
        history = numbers_all[:i]
        actual = numbers_all[i]
        
        preds = predictor.predict(history)
        if actual in preds:
            hits += 1
            miss = 0
        else:
            miss += 1
            max_miss = max(max_miss, miss)
    
    hit_rate = hits / test_periods * 100
    results['深度特征综合'] = {
        'hits': hits,
        'hit_rate': hit_rate,
        'max_miss': max_miss
    }
    print(f"  命中: {hits}/{test_periods} ({hit_rate:.2f}%)")
    print(f"  最大连不中: {max_miss}期")
    
    # 与原始对比
    from precise_top15_predictor import PreciseTop15Predictor
    orig_predictor = PreciseTop15Predictor()
    
    print(f"\n测试: 原精准TOP15")
    hits = 0
    max_miss = 0
    miss = 0
    
    for i in range(start_idx, total_periods):
        history = numbers_all[:i]
        actual = numbers_all[i]
        
        preds = orig_predictor.predict(history)
        if actual in preds:
            hits += 1
            miss = 0
        else:
            miss += 1
            max_miss = max(max_miss, miss)
    
    hit_rate = hits / test_periods * 100
    results['原精准TOP15'] = {
        'hits': hits,
        'hit_rate': hit_rate,
        'max_miss': max_miss
    }
    print(f"  命中: {hits}/{test_periods} ({hit_rate:.2f}%)")
    print(f"  最大连不中: {max_miss}期")
    
    # 排名
    print("\n" + "="*80)
    print("命中率排名")
    print("="*80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['hit_rate'], reverse=True)
    
    print(f"\n{'排名':<5} {'方法':<15} {'命中率':<12} {'命中次数':<10} {'最大连不中':<10}")
    print("-" * 60)
    
    for rank, (name, data) in enumerate(sorted_results, 1):
        print(f"{rank:<5} {name:<15} {data['hit_rate']:<11.2f}% {data['hits']:<10}次 {data['max_miss']:<10}期")
    
    best = sorted_results[0]
    print(f"\n最佳方法: {best[0]} ({best[1]['hit_rate']:.2f}%)")
    
    if best[1]['hit_rate'] >= 50:
        print("🎉 成功达到50%目标!")
    else:
        print(f"距离50%目标: {50 - best[1]['hit_rate']:.2f}个百分点")
    
    return results


if __name__ == "__main__":
    validate_deep_feature_predictor(test_periods=200)
    
    print("\n" + "="*80)
    print("验证完成!")
    print("="*80)
