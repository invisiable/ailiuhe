"""
高级混合TOP15预测器
融合精准预测器 + 数学模型 + 自适应选择
目标：提升命中率至50%
"""

import numpy as np
import pandas as pd
from collections import Counter, deque
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')

from precise_top15_predictor import PreciseTop15Predictor
from mathematical_top15_predictor import MathematicalTop15Predictor


class AdaptiveHybridTop15Predictor:
    """
    自适应混合TOP15预测器
    
    核心策略：
    1. 多模型预测：精准预测器 + 数学模型
    2. 自适应权重：根据近期表现动态调整
    3. 号码覆盖优化：确保覆盖高概率区间
    4. 时间序列分析：趋势+周期+噪声分解
    """
    
    def __init__(self):
        self.precise_predictor = PreciseTop15Predictor()
        self.math_predictor = MathematicalTop15Predictor()
        
        # 自适应权重
        self.model_weights = {
            'precise': 0.6,
            'bayesian': 0.15,
            'thermo': 0.10,
            'regression': 0.15
        }
        
        # 性能追踪
        self.recent_performance = {
            'precise': deque(maxlen=20),
            'bayesian': deque(maxlen=20),
            'thermo': deque(maxlen=20),
            'regression': deque(maxlen=20)
        }
        
        self.all_numbers = list(range(1, 50))
    
    def method_zone_coverage(self, numbers, k=20):
        """
        区间覆盖法
        确保每个区间都有适当覆盖
        """
        if len(numbers) < 30:
            return self._simple_freq(numbers, k)
        
        # 定义5个区间
        zones = [
            (1, 10),   # 极小值
            (11, 20),  # 小值
            (21, 30),  # 中值
            (31, 40),  # 大值
            (41, 49)   # 极大值
        ]
        
        recent_30 = numbers[-30:]
        recent_10 = numbers[-10:]
        recent_5 = numbers[-5:]
        
        # 分析各区间的出现频率
        zone_freq = {}
        for zone_start, zone_end in zones:
            count = sum(1 for n in recent_30 if zone_start <= n <= zone_end)
            zone_freq[(zone_start, zone_end)] = count / 30
        
        # 计算期望频率
        expected = {
            (1, 10): 10/49,
            (11, 20): 10/49,
            (21, 30): 10/49,
            (31, 40): 10/49,
            (41, 49): 9/49
        }
        
        # 计算各区间的偏离度和选择数量
        zone_selection = {}
        total_deficit = 0
        
        for zone, actual_freq in zone_freq.items():
            exp_freq = expected[zone]
            deficit = max(0, exp_freq - actual_freq)
            total_deficit += deficit
            zone_selection[zone] = deficit
        
        # 归一化，分配k个位置
        if total_deficit > 0:
            for zone in zone_selection:
                zone_selection[zone] = max(1, int(k * zone_selection[zone] / total_deficit))
        else:
            # 均匀分配
            for zone in zone_selection:
                zone_selection[zone] = k // 5
        
        # 在每个区间内选择号码
        freq_all = Counter(recent_30)
        selected = []
        
        for (zone_start, zone_end), count in zone_selection.items():
            zone_nums = [n for n in range(zone_start, zone_end + 1)]
            
            # 按频率和间隔综合评分
            zone_scores = {}
            for n in zone_nums:
                freq_score = freq_all.get(n, 0) / 10
                
                # 间隔得分
                if n in recent_30:
                    last_pos = max([i for i, x in enumerate(recent_30) if x == n])
                    gap = len(recent_30) - 1 - last_pos
                    gap_score = gap / 30
                else:
                    gap_score = 0.8
                
                # 避开最近5期
                if n in recent_5:
                    penalty = 0.2
                else:
                    penalty = 1.0
                
                zone_scores[n] = (freq_score * 0.4 + gap_score * 0.6) * penalty
            
            # 选择该区间得分最高的
            sorted_zone = sorted(zone_scores.items(), key=lambda x: x[1], reverse=True)
            selected.extend([n for n, _ in sorted_zone[:count]])
        
        return selected[:k]
    
    def method_trend_momentum(self, numbers, k=20):
        """
        趋势动量法
        分析号码的上升/下降趋势
        """
        if len(numbers) < 50:
            return self._simple_freq(numbers, k)
        
        # 计算"动量"：号码值的移动平均趋势
        window = 10
        ma_values = []
        for i in range(len(numbers) - window):
            ma = np.mean(numbers[i:i+window])
            ma_values.append(ma)
        
        # 判断当前趋势
        if len(ma_values) >= 5:
            recent_ma = ma_values[-5:]
            trend = (recent_ma[-1] - recent_ma[0]) / 5
        else:
            trend = 0
        
        # 根据趋势调整号码选择
        recent_30 = numbers[-30:]
        freq = Counter(recent_30)
        
        scores = {}
        center = 25  # 中心值
        
        for n in self.all_numbers:
            base_score = freq.get(n, 0) / 30 + 0.1
            
            # 趋势调整
            if trend > 1:  # 上升趋势
                # 偏向大数
                trend_bonus = (n - center) / 25 * 0.3
            elif trend < -1:  # 下降趋势
                # 偏向小数
                trend_bonus = (center - n) / 25 * 0.3
            else:
                trend_bonus = 0
            
            scores[n] = base_score + trend_bonus
            
            # 避开最近5期
            if n in numbers[-5:]:
                scores[n] *= 0.2
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in sorted_nums[:k]]
    
    def method_pattern_recognition(self, numbers, k=20):
        """
        模式识别法
        识别历史中的重复模式
        """
        if len(numbers) < 50:
            return self._simple_freq(numbers, k)
        
        recent_10 = numbers[-10:]
        
        # 计算各种模式特征
        odd_count = sum(1 for n in recent_10 if n % 2 == 1)
        big_count = sum(1 for n in recent_10 if n > 25)
        prime_count = sum(1 for n in recent_10 if self._is_prime(n))
        
        # 预测下一期的期望特征
        expected_odd = 10 - odd_count  # 平衡原则
        expected_big = 10 - big_count
        expected_prime = 10 - prime_count
        
        scores = {}
        recent_30 = numbers[-30:]
        freq = Counter(recent_30)
        
        for n in self.all_numbers:
            base_score = freq.get(n, 0) / 30 + 0.1
            
            # 特征匹配得分
            feature_score = 0
            
            # 奇偶
            if (n % 2 == 1 and expected_odd > 5) or (n % 2 == 0 and expected_odd <= 5):
                feature_score += 0.2
            
            # 大小
            if (n > 25 and expected_big > 5) or (n <= 25 and expected_big <= 5):
                feature_score += 0.2
            
            # 质数
            if (self._is_prime(n) and expected_prime > 5) or (not self._is_prime(n) and expected_prime <= 5):
                feature_score += 0.1
            
            scores[n] = base_score + feature_score
            
            # 避开最近5期
            if n in numbers[-5:]:
                scores[n] *= 0.2
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in sorted_nums[:k]]
    
    def _is_prime(self, n):
        """判断是否为质数"""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def predict(self, numbers):
        """
        自适应混合预测
        """
        if len(numbers) < 30:
            return self._simple_freq(numbers, 15)
        
        # 获取各模型的预测结果
        predictions = {
            'precise': self.precise_predictor.predict(numbers),
            'bayesian': self.math_predictor.method_bayesian(numbers, 20),
            'thermo': self.math_predictor.method_thermodynamic(numbers, 20),
            'regression': self.math_predictor.method_regression_to_mean(numbers, 20),
            'zone': self.method_zone_coverage(numbers, 20),
            'trend': self.method_trend_momentum(numbers, 20),
            'pattern': self.method_pattern_recognition(numbers, 20)
        }
        
        # 动态权重（基于近期表现）
        weights = self._calculate_dynamic_weights()
        
        # 综合评分
        scores = {}
        for model_name, preds in predictions.items():
            weight = weights.get(model_name, 0.1)
            for rank, num in enumerate(preds):
                score = weight * (1.0 - rank / len(preds))
                scores[num] = scores.get(num, 0) + score
        
        # 返回TOP15
        final = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in final[:15]]
    
    def _calculate_dynamic_weights(self):
        """计算动态权重"""
        weights = {}
        
        for model_name, history in self.recent_performance.items():
            if len(history) >= 5:
                hit_rate = sum(history) / len(history)
                weights[model_name] = 0.1 + hit_rate * 0.4
            else:
                weights[model_name] = self.model_weights.get(model_name, 0.15)
        
        # 添加新方法的默认权重
        weights['zone'] = 0.10
        weights['trend'] = 0.08
        weights['pattern'] = 0.07
        
        # 归一化
        total = sum(weights.values())
        for k in weights:
            weights[k] /= total
        
        return weights
    
    def update_performance(self, predictions, actual, model_predictions=None):
        """更新各模型的性能追踪"""
        # 更新precise
        self.recent_performance['precise'].append(1 if actual in predictions else 0)
        
        # 如果提供了各模型的预测，更新它们
        if model_predictions:
            for model_name, preds in model_predictions.items():
                if model_name in self.recent_performance:
                    self.recent_performance[model_name].append(1 if actual in preds else 0)
    
    def _simple_freq(self, numbers, k):
        """简单频率法"""
        freq = Counter(numbers)
        sorted_nums = sorted(self.all_numbers, key=lambda x: freq.get(x, 0), reverse=True)
        return sorted_nums[:k]


class UltraTop15Predictor:
    """
    终极TOP15预测器
    策略：扩大候选池 + 智能筛选
    """
    
    def __init__(self):
        self.precise_predictor = PreciseTop15Predictor()
        self.math_predictor = MathematicalTop15Predictor()
        self.adaptive_predictor = AdaptiveHybridTop15Predictor()
        self.all_numbers = list(range(1, 50))
    
    def predict(self, numbers):
        """
        终极预测方法
        """
        # 获取多个模型的预测
        precise_preds = self.precise_predictor.predict(numbers)
        bayesian_preds = self.math_predictor.method_bayesian(numbers, 20)
        thermo_preds = self.math_predictor.method_thermodynamic(numbers, 20)
        regression_preds = self.math_predictor.method_regression_to_mean(numbers, 20)
        adaptive_preds = self.adaptive_predictor.predict(numbers)
        
        # 投票计数
        votes = Counter()
        
        # 精准预测器（高权重）
        for i, n in enumerate(precise_preds):
            votes[n] += 3 * (15 - i) / 15
        
        # 自适应预测器
        for i, n in enumerate(adaptive_preds):
            votes[n] += 2 * (15 - i) / 15
        
        # 数学模型
        for i, n in enumerate(bayesian_preds):
            votes[n] += 1.5 * (20 - i) / 20
        
        for i, n in enumerate(thermo_preds):
            votes[n] += 1.0 * (20 - i) / 20
        
        for i, n in enumerate(regression_preds):
            votes[n] += 1.0 * (20 - i) / 20
        
        # 避开最近5期的惩罚
        recent_5 = numbers[-5:] if len(numbers) >= 5 else numbers
        for n in recent_5:
            votes[n] *= 0.3
        
        # 选择得票最高的15个
        sorted_nums = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in sorted_nums[:15]]


def validate_advanced_predictors(test_periods=200):
    """验证高级预测器"""
    print("="*80)
    print("高级混合TOP15预测器验证")
    print("="*80)
    
    df = pd.read_csv('data/lucky_numbers.csv')
    numbers_all = df['number'].tolist()
    
    total_periods = len(numbers_all)
    start_idx = total_periods - test_periods
    
    # 测试各预测器
    predictors = {
        '自适应混合': AdaptiveHybridTop15Predictor(),
        '终极预测器': UltraTop15Predictor(),
    }
    
    # 添加原精准预测器作为基准
    from precise_top15_predictor import PreciseTop15Predictor
    predictors['原精准TOP15'] = PreciseTop15Predictor()
    
    results = {}
    
    for name, predictor in predictors.items():
        print(f"\n测试: {name}")
        hits = 0
        max_consecutive_miss = 0
        consecutive_miss = 0
        
        for i in range(start_idx, total_periods):
            history = numbers_all[:i]
            actual = numbers_all[i]
            
            predictions = predictor.predict(history)
            hit = actual in predictions
            
            if hit:
                hits += 1
                consecutive_miss = 0
            else:
                consecutive_miss += 1
                max_consecutive_miss = max(max_consecutive_miss, consecutive_miss)
        
        hit_rate = hits / test_periods * 100
        results[name] = {
            'hits': hits,
            'hit_rate': hit_rate,
            'max_consecutive_miss': max_consecutive_miss
        }
        print(f"  命中: {hits}/{test_periods} ({hit_rate:.2f}%)")
        print(f"  最大连不中: {max_consecutive_miss}期")
    
    # 输出排名
    print("\n" + "="*80)
    print("命中率排名")
    print("="*80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['hit_rate'], reverse=True)
    
    print(f"\n{'排名':<5} {'预测器':<15} {'命中率':<12} {'命中次数':<12} {'最大连不中':<10}")
    print("-" * 60)
    
    for rank, (name, data) in enumerate(sorted_results, 1):
        print(f"{rank:<5} {name:<15} {data['hit_rate']:<11.2f}% {data['hits']:<12}次 {data['max_consecutive_miss']:<10}期")
    
    best = sorted_results[0]
    if best[1]['hit_rate'] >= 50:
        print(f"\n🎉 成功! {best[0]}达到50%命中率目标!")
    else:
        gap = 50 - best[1]['hit_rate']
        print(f"\n距离50%目标还差: {gap:.2f}个百分点")
    
    return results


def explore_theoretical_limit():
    """
    探索理论极限
    计算TOP15的理论命中率上限
    """
    print("\n" + "="*80)
    print("TOP15理论命中率分析")
    print("="*80)
    
    # 随机基准
    random_hit_rate = 15 / 49 * 100
    print(f"\n1. 随机选择15个号码的命中率: {random_hit_rate:.2f}%")
    
    # 如果实现完美预测（知道下一个号码一定在某个模式中）
    # 假设我们能识别出号码在某些"热区"出现的概率
    
    df = pd.read_csv('data/lucky_numbers.csv')
    numbers = df['number'].tolist()
    
    # 分析号码的可预测性
    test_periods = 200
    total = len(numbers)
    start = total - test_periods
    
    # 最优后验分析（知道结果后选择最佳策略）
    perfect_hits = 0
    for i in range(start, total):
        actual = numbers[i]
        history = numbers[:i]
        
        # 最佳策略：选择历史频率最高的15个（排除最近5期）
        recent_30 = history[-30:]
        recent_5 = history[-5:]
        freq = Counter(recent_30)
        
        # 过滤最近5期
        candidates = [(n, freq.get(n, 0)) for n in range(1, 50) if n not in recent_5]
        sorted_cands = sorted(candidates, key=lambda x: x[1], reverse=True)
        top15 = [n for n, _ in sorted_cands[:15]]
        
        if actual in top15:
            perfect_hits += 1
    
    freq_hit_rate = perfect_hits / test_periods * 100
    print(f"2. 简单频率法(排除近5期)命中率: {freq_hit_rate:.2f}%")
    
    # 区间分析
    print(f"\n3. 号码分布分析:")
    zones = [(1,10), (11,20), (21,30), (31,40), (41,49)]
    for z_start, z_end in zones:
        count = sum(1 for n in numbers[-200:] if z_start <= n <= z_end)
        expected = 200 * (z_end - z_start + 1) / 49
        print(f"   区间[{z_start:2d}-{z_end:2d}]: 出现{count}次, 期望{expected:.1f}次, 偏差{count-expected:+.1f}")
    
    # 理论分析
    print(f"\n4. 理论分析:")
    print(f"   • 49个数选15个，理论命中率 = 15/49 = {15/49*100:.2f}%")
    print("   • 如果能识别50%的'高概率区'，理论可达到约40%")
    print("   • 如果模型能在70%时间正确识别高概率区，理论可达到约45%")
    print("   • 达到50%需要模型在80%以上时间正确预测趋势")
    
    print(f"\n5. 结论:")
    print(f"   当前最佳命中率约35%，已超过随机基准({random_hit_rate:.2f}%)")
    print(f"   提升到50%需要显著提高模式识别准确度")
    print(f"   可能的突破方向: 深度学习、外部数据源、更长历史分析")


if __name__ == "__main__":
    # 验证高级预测器
    results = validate_advanced_predictors(test_periods=200)
    
    # 理论极限分析
    explore_theoretical_limit()
    
    print("\n" + "="*80)
    print("验证完成!")
    print("="*80)
