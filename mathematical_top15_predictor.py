"""
数学模型优化TOP15预测器
尝试多种数学/统计模型提升命中率至50%
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy import stats
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')


class MathematicalTop15Predictor:
    """
    数学模型增强版TOP15预测器
    
    包含模型：
    1. 马尔可夫链模型 - 状态转移概率
    2. 贝叶斯概率模型 - 先验后验更新
    3. 熵权法 - 信息熵权重
    4. 灰色预测GM(1,1) - 小样本预测
    5. 加权移动平均 - 时间衰减
    6. 回归到均值模型 - 统计套利
    7. 热力学模型 - 能量分布
    """
    
    def __init__(self):
        self.all_numbers = list(range(1, 50))
        
    def method_markov_chain(self, numbers, k=20):
        """
        马尔可夫链模型
        基于历史状态转移概率预测下一个号码
        """
        if len(numbers) < 20:
            return self._simple_frequency(numbers, k)
        
        # 构建转移矩阵（简化为区间转移）
        # 将49个号码分成7个区间
        def get_zone(n):
            return (n - 1) // 7  # 0-6
        
        # 统计转移概率
        transitions = defaultdict(lambda: defaultdict(int))
        for i in range(len(numbers) - 1):
            from_zone = get_zone(numbers[i])
            to_zone = get_zone(numbers[i + 1])
            transitions[from_zone][to_zone] += 1
        
        # 获取当前状态
        current_zone = get_zone(numbers[-1])
        
        # 计算下一个区间的概率
        zone_probs = np.zeros(7)
        total = sum(transitions[current_zone].values())
        if total > 0:
            for zone, count in transitions[current_zone].items():
                zone_probs[zone] = count / total
        else:
            zone_probs = np.ones(7) / 7
        
        # 为每个号码分配概率
        scores = {}
        for n in self.all_numbers:
            zone = get_zone(n)
            base_prob = zone_probs[zone]
            
            # 区间内细分（基于具体数字的历史频率）
            freq = Counter(numbers[-30:])
            freq_score = freq.get(n, 0) / 30
            
            # 组合得分
            scores[n] = base_prob * 0.6 + freq_score * 0.4
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in sorted_nums[:k]]
    
    def method_bayesian(self, numbers, k=20):
        """
        贝叶斯概率模型
        使用Beta-Binomial共轭先验更新概率
        """
        if len(numbers) < 10:
            return self._simple_frequency(numbers, k)
        
        # 先验参数（均匀先验）
        alpha_prior = 1
        beta_prior = 48  # 49个数中选1个
        
        # 计算后验概率
        window_sizes = [10, 20, 30, 50]
        
        scores = {}
        for n in self.all_numbers:
            posterior_sum = 0
            weight_sum = 0
            
            for w_idx, window in enumerate(window_sizes):
                if len(numbers) >= window:
                    recent = numbers[-window:]
                    successes = recent.count(n)
                    failures = window - successes
                    
                    # Beta后验
                    alpha_post = alpha_prior + successes
                    beta_post = beta_prior + failures
                    
                    # 后验期望
                    posterior = alpha_post / (alpha_post + beta_post)
                    
                    # 权重：近期窗口权重更高
                    weight = (4 - w_idx)
                    posterior_sum += posterior * weight
                    weight_sum += weight
            
            scores[n] = posterior_sum / weight_sum if weight_sum > 0 else 1/49
        
        # 应用冷热调整
        recent_5 = numbers[-5:] if len(numbers) >= 5 else numbers
        for n in recent_5:
            scores[n] *= 0.3  # 降低最近出现的号码
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in sorted_nums[:k]]
    
    def method_entropy_weight(self, numbers, k=20):
        """
        熵权法
        基于信息熵计算各指标权重，综合多维特征
        """
        if len(numbers) < 30:
            return self._simple_frequency(numbers, k)
        
        # 构建特征矩阵（每个数字多维特征）
        features = {}
        
        recent_30 = numbers[-30:]
        recent_10 = numbers[-10:]
        recent_5 = numbers[-5:]
        
        freq_30 = Counter(recent_30)
        freq_10 = Counter(recent_10)
        
        for n in self.all_numbers:
            # 特征1: 30期频率
            f1 = freq_30.get(n, 0) / 30
            
            # 特征2: 10期频率
            f2 = freq_10.get(n, 0) / 10
            
            # 特征3: 间隔特征（距上次出现的期数）
            if n in recent_30:
                last_idx = max([i for i, x in enumerate(recent_30) if x == n])
                gap = len(recent_30) - 1 - last_idx
                f3 = 1 - gap / 30  # 间隔越短得分越低
            else:
                f3 = 0.5  # 未出现给中等分
            
            # 特征4: 奇偶平衡（根据最近趋势）
            odd_count = sum(1 for x in recent_10 if x % 2 == 1)
            if n % 2 == 1:
                f4 = 1 - odd_count / 10  # 奇数多则降低奇数权重
            else:
                f4 = odd_count / 10
            
            # 特征5: 大小平衡
            big_count = sum(1 for x in recent_10 if x > 25)
            if n > 25:
                f5 = 1 - big_count / 10
            else:
                f5 = big_count / 10
            
            features[n] = [f1, f2, f3, f4, f5]
        
        # 构建矩阵
        matrix = np.array([features[n] for n in self.all_numbers])
        
        # 标准化（避免0值）
        matrix = (matrix - matrix.min(axis=0)) / (matrix.max(axis=0) - matrix.min(axis=0) + 1e-10)
        matrix = matrix + 1e-10
        
        # 计算熵
        p = matrix / matrix.sum(axis=0)
        entropy = -np.sum(p * np.log(p + 1e-10), axis=0) / np.log(len(self.all_numbers))
        
        # 计算熵权
        weights = (1 - entropy) / (1 - entropy).sum()
        
        # 计算综合得分
        scores = {}
        for i, n in enumerate(self.all_numbers):
            scores[n] = np.dot(matrix[i], weights)
        
        # 避开最近5期
        for n in recent_5:
            scores[n] *= 0.2
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in sorted_nums[:k]]
    
    def method_grey_prediction(self, numbers, k=20):
        """
        灰色预测模型 GM(1,1)
        适合小样本、贫信息预测
        """
        if len(numbers) < 10:
            return self._simple_frequency(numbers, k)
        
        # 对每个数字的出现间隔进行灰色预测
        scores = {}
        
        for n in self.all_numbers:
            # 找出该数字的出现位置
            positions = [i for i, x in enumerate(numbers) if x == n]
            
            if len(positions) >= 4:
                # 计算间隔序列
                gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                
                # GM(1,1)预测下一个间隔
                try:
                    predicted_gap = self._gm11_predict(gaps)
                    
                    # 计算距上次出现的期数
                    current_gap = len(numbers) - positions[-1]
                    
                    # 得分：接近预测间隔的得分高
                    if predicted_gap > 0:
                        score = np.exp(-abs(current_gap - predicted_gap) / predicted_gap)
                    else:
                        score = 0.5
                except:
                    score = 0.5
            else:
                # 出现次数少，基于频率
                freq = numbers.count(n)
                expected_freq = len(numbers) / 49
                score = min(1.0, freq / expected_freq) if expected_freq > 0 else 0.5
            
            scores[n] = score
        
        # 避开最近5期
        recent_5 = numbers[-5:] if len(numbers) >= 5 else numbers
        for n in recent_5:
            scores[n] *= 0.3
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in sorted_nums[:k]]
    
    def _gm11_predict(self, sequence):
        """GM(1,1)模型预测"""
        n = len(sequence)
        if n < 3:
            return np.mean(sequence)
        
        x0 = np.array(sequence, dtype=float)
        x1 = np.cumsum(x0)  # 累加生成
        
        # 构建矩阵
        B = np.zeros((n-1, 2))
        Y = np.zeros(n-1)
        
        for i in range(n-1):
            B[i, 0] = -0.5 * (x1[i] + x1[i+1])
            B[i, 1] = 1
            Y[i] = x0[i+1]
        
        # 最小二乘求解
        try:
            params = np.linalg.lstsq(B, Y, rcond=None)[0]
            a, b = params
            
            # 预测下一个值
            predicted = (x0[0] - b/a) * np.exp(-a * n) * (1 - np.exp(a))
            return max(1, predicted)
        except:
            return np.mean(sequence)
    
    def method_regression_to_mean(self, numbers, k=20):
        """
        回归均值模型
        基于统计学的均值回归原理
        """
        if len(numbers) < 50:
            return self._simple_frequency(numbers, k)
        
        # 计算每个数字的期望频率和实际频率
        expected_freq = len(numbers) / 49
        
        scores = {}
        recent_100 = numbers[-100:] if len(numbers) >= 100 else numbers
        recent_50 = numbers[-50:] if len(numbers) >= 50 else numbers
        recent_20 = numbers[-20:] if len(numbers) >= 20 else numbers
        
        freq_100 = Counter(recent_100)
        freq_50 = Counter(recent_50)
        freq_20 = Counter(recent_20)
        
        for n in self.all_numbers:
            # 计算不同时间窗口的偏离度
            actual_100 = freq_100.get(n, 0)
            actual_50 = freq_50.get(n, 0)
            actual_20 = freq_20.get(n, 0)
            
            expected_100 = 100 / 49
            expected_50 = 50 / 49
            expected_20 = 20 / 49
            
            # 偏离度（负数表示低于预期）
            deviation_100 = (expected_100 - actual_100) / expected_100
            deviation_50 = (expected_50 - actual_50) / expected_50
            deviation_20 = (expected_20 - actual_20) / expected_20
            
            # 综合偏离度（近期权重更高）
            total_deviation = deviation_100 * 0.2 + deviation_50 * 0.3 + deviation_20 * 0.5
            
            # 转换为得分（偏离越大越可能回归）
            score = 0.5 + total_deviation * 0.3
            scores[n] = max(0, min(1, score))
        
        # 避开最近5期
        recent_5 = numbers[-5:] if len(numbers) >= 5 else numbers
        for n in recent_5:
            scores[n] *= 0.2
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in sorted_nums[:k]]
    
    def method_thermodynamic(self, numbers, k=20):
        """
        热力学模型
        将号码视为粒子，基于能量分布（玻尔兹曼分布）
        """
        if len(numbers) < 30:
            return self._simple_frequency(numbers, k)
        
        recent_50 = numbers[-50:] if len(numbers) >= 50 else numbers
        
        # 计算每个数字的"能量"（基于出现频率和间隔）
        energies = {}
        freq = Counter(recent_50)
        
        for n in self.all_numbers:
            # 频率能量（出现越多，能量越高）
            freq_energy = freq.get(n, 0)
            
            # 间隔能量（间隔越长，势能越高）
            if n in recent_50:
                positions = [i for i, x in enumerate(recent_50) if x == n]
                gap = len(recent_50) - 1 - positions[-1]
                gap_energy = gap / 10  # 归一化
            else:
                gap_energy = 3  # 高势能
            
            # 总能量（动能+势能的组合）
            energies[n] = -freq_energy * 0.3 + gap_energy * 0.7
        
        # 应用玻尔兹曼分布
        # P(n) ∝ exp(-E(n) / kT)
        temperature = 1.0  # 温度参数
        
        energy_values = np.array([energies[n] for n in self.all_numbers])
        probs = softmax(-energy_values / temperature)
        
        scores = {n: probs[i] for i, n in enumerate(self.all_numbers)}
        
        # 避开最近5期
        recent_5 = numbers[-5:] if len(numbers) >= 5 else numbers
        for n in recent_5:
            scores[n] *= 0.2
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in sorted_nums[:k]]
    
    def method_poisson_process(self, numbers, k=20):
        """
        泊松过程模型
        将号码出现视为泊松过程，计算下期出现概率
        """
        if len(numbers) < 30:
            return self._simple_frequency(numbers, k)
        
        recent_100 = numbers[-100:] if len(numbers) >= 100 else numbers
        
        scores = {}
        
        for n in self.all_numbers:
            # 计算到达率λ（平均每期出现概率）
            count = recent_100.count(n)
            lambda_rate = count / len(recent_100)
            
            if lambda_rate > 0:
                # 计算距上次出现的期数
                if n in recent_100:
                    positions = [i for i, x in enumerate(recent_100) if x == n]
                    gap = len(recent_100) - 1 - positions[-1]
                else:
                    gap = len(recent_100)
                
                # 泊松过程：等待时间服从指数分布
                # P(下期出现) = 1 - exp(-λ * t)
                expected_gap = 1 / lambda_rate if lambda_rate > 0 else 100
                
                # 如果超过期望间隔，概率更高
                prob = 1 - np.exp(-lambda_rate * (gap + 1))
                
                # 调整：超过期望间隔时加权
                if gap > expected_gap:
                    prob *= 1.2
                
                scores[n] = min(1.0, prob)
            else:
                scores[n] = 0.3  # 从未出现给基础概率
        
        # 避开最近5期
        recent_5 = numbers[-5:] if len(numbers) >= 5 else numbers
        for n in recent_5:
            scores[n] *= 0.2
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in sorted_nums[:k]]
    
    def _simple_frequency(self, numbers, k=20):
        """简单频率法（降级方案）"""
        freq = Counter(numbers)
        sorted_nums = sorted(self.all_numbers, key=lambda x: freq.get(x, 0), reverse=True)
        return sorted_nums[:k]
    
    def predict_ensemble(self, numbers, method_weights=None):
        """
        集成预测 - 融合所有数学模型
        """
        if method_weights is None:
            method_weights = {
                'markov': 0.15,
                'bayesian': 0.20,
                'entropy': 0.15,
                'grey': 0.10,
                'regression': 0.15,
                'thermo': 0.10,
                'poisson': 0.15
            }
        
        # 获取各方法的预测结果
        methods = {
            'markov': self.method_markov_chain(numbers, 25),
            'bayesian': self.method_bayesian(numbers, 25),
            'entropy': self.method_entropy_weight(numbers, 25),
            'grey': self.method_grey_prediction(numbers, 25),
            'regression': self.method_regression_to_mean(numbers, 25),
            'thermo': self.method_thermodynamic(numbers, 25),
            'poisson': self.method_poisson_process(numbers, 25)
        }
        
        # 综合评分
        scores = {}
        for method_name, predictions in methods.items():
            weight = method_weights.get(method_name, 0.1)
            for rank, num in enumerate(predictions):
                score = weight * (1.0 - rank / len(predictions))
                scores[num] = scores.get(num, 0) + score
        
        # 返回TOP15
        final = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in final[:15]]
    
    def predict_best_single(self, numbers):
        """
        返回单个最佳模型的预测
        """
        return self.method_bayesian(numbers, 15)


def validate_mathematical_models(test_periods=200):
    """验证各数学模型的表现"""
    print("="*80)
    print("数学模型TOP15预测验证")
    print("="*80)
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv')
    numbers_all = df['number'].tolist()
    
    total_periods = len(numbers_all)
    start_idx = total_periods - test_periods
    
    predictor = MathematicalTop15Predictor()
    
    # 测试各个模型
    models = {
        '马尔可夫链': predictor.method_markov_chain,
        '贝叶斯概率': predictor.method_bayesian,
        '熵权法': predictor.method_entropy_weight,
        '灰色预测': predictor.method_grey_prediction,
        '回归均值': predictor.method_regression_to_mean,
        '热力学': predictor.method_thermodynamic,
        '泊松过程': predictor.method_poisson_process,
    }
    
    results = {}
    
    for model_name, method in models.items():
        print(f"\n测试模型: {model_name}")
        hits = 0
        max_consecutive_miss = 0
        consecutive_miss = 0
        
        for i in range(start_idx, total_periods):
            history = numbers_all[:i]
            actual = numbers_all[i]
            
            predictions = method(history, k=15)
            hit = actual in predictions
            
            if hit:
                hits += 1
                consecutive_miss = 0
            else:
                consecutive_miss += 1
                max_consecutive_miss = max(max_consecutive_miss, consecutive_miss)
        
        hit_rate = hits / test_periods * 100
        results[model_name] = {
            'hits': hits,
            'hit_rate': hit_rate,
            'max_consecutive_miss': max_consecutive_miss
        }
        print(f"  命中: {hits}/{test_periods} ({hit_rate:.2f}%)")
        print(f"  最大连不中: {max_consecutive_miss}期")
    
    # 测试集成模型
    print(f"\n测试模型: 数学模型集成")
    hits = 0
    max_consecutive_miss = 0
    consecutive_miss = 0
    
    for i in range(start_idx, total_periods):
        history = numbers_all[:i]
        actual = numbers_all[i]
        
        predictions = predictor.predict_ensemble(history)
        hit = actual in predictions
        
        if hit:
            hits += 1
            consecutive_miss = 0
        else:
            consecutive_miss += 1
            max_consecutive_miss = max(max_consecutive_miss, consecutive_miss)
    
    hit_rate = hits / test_periods * 100
    results['数学模型集成'] = {
        'hits': hits,
        'hit_rate': hit_rate,
        'max_consecutive_miss': max_consecutive_miss
    }
    print(f"  命中: {hits}/{test_periods} ({hit_rate:.2f}%)")
    print(f"  最大连不中: {max_consecutive_miss}期")
    
    # 汇总输出
    print("\n" + "="*80)
    print("各模型命中率排名")
    print("="*80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['hit_rate'], reverse=True)
    
    print(f"\n{'排名':<5} {'模型名称':<15} {'命中率':<12} {'命中次数':<12} {'最大连不中':<10}")
    print("-" * 60)
    
    for rank, (name, data) in enumerate(sorted_results, 1):
        print(f"{rank:<5} {name:<15} {data['hit_rate']:<11.2f}% {data['hits']:<12}次 {data['max_consecutive_miss']:<10}期")
    
    # 对比原始TOP15
    print("\n" + "="*80)
    print("与原精准TOP15对比")
    print("="*80)
    
    from precise_top15_predictor import PreciseTop15Predictor
    original_predictor = PreciseTop15Predictor()
    
    hits = 0
    max_consecutive_miss = 0
    consecutive_miss = 0
    
    for i in range(start_idx, total_periods):
        history = numbers_all[:i]
        actual = numbers_all[i]
        
        predictions = original_predictor.predict(history)
        hit = actual in predictions
        
        if hit:
            hits += 1
            consecutive_miss = 0
        else:
            consecutive_miss += 1
            max_consecutive_miss = max(max_consecutive_miss, consecutive_miss)
    
    original_hit_rate = hits / test_periods * 100
    print(f"\n原精准TOP15: {hits}/{test_periods} ({original_hit_rate:.2f}%)")
    
    best_model = sorted_results[0]
    improvement = best_model[1]['hit_rate'] - original_hit_rate
    
    print(f"最佳数学模型 ({best_model[0]}): {best_model[1]['hit_rate']:.2f}%")
    print(f"提升幅度: {improvement:+.2f}个百分点")
    
    if best_model[1]['hit_rate'] >= 50:
        print(f"\n🎉 成功! {best_model[0]}模型达到50%命中率目标!")
    else:
        gap = 50 - best_model[1]['hit_rate']
        print(f"\n距离50%目标还差: {gap:.2f}个百分点")
    
    return results


def test_weighted_ensemble():
    """测试不同权重组合的集成效果"""
    print("\n" + "="*80)
    print("优化集成权重组合")
    print("="*80)
    
    df = pd.read_csv('data/lucky_numbers.csv')
    numbers_all = df['number'].tolist()
    
    test_periods = 100  # 快速验证
    total_periods = len(numbers_all)
    start_idx = total_periods - test_periods
    
    predictor = MathematicalTop15Predictor()
    
    # 各种权重组合
    weight_configs = [
        {'name': '均匀权重', 'weights': {'markov': 1/7, 'bayesian': 1/7, 'entropy': 1/7, 
                                        'grey': 1/7, 'regression': 1/7, 'thermo': 1/7, 'poisson': 1/7}},
        {'name': '贝叶斯主导', 'weights': {'markov': 0.10, 'bayesian': 0.40, 'entropy': 0.10, 
                                        'grey': 0.10, 'regression': 0.10, 'thermo': 0.10, 'poisson': 0.10}},
        {'name': '概率主导', 'weights': {'markov': 0.20, 'bayesian': 0.25, 'entropy': 0.10, 
                                        'grey': 0.05, 'regression': 0.15, 'thermo': 0.05, 'poisson': 0.20}},
        {'name': '统计主导', 'weights': {'markov': 0.10, 'bayesian': 0.20, 'entropy': 0.20, 
                                        'grey': 0.10, 'regression': 0.20, 'thermo': 0.05, 'poisson': 0.15}},
    ]
    
    best_config = None
    best_hit_rate = 0
    
    for config in weight_configs:
        hits = 0
        for i in range(start_idx, total_periods):
            history = numbers_all[:i]
            actual = numbers_all[i]
            
            predictions = predictor.predict_ensemble(history, config['weights'])
            if actual in predictions:
                hits += 1
        
        hit_rate = hits / test_periods * 100
        print(f"{config['name']}: {hit_rate:.2f}%")
        
        if hit_rate > best_hit_rate:
            best_hit_rate = hit_rate
            best_config = config
    
    print(f"\n最佳配置: {best_config['name']} ({best_hit_rate:.2f}%)")
    return best_config


if __name__ == "__main__":
    # 验证所有数学模型
    results = validate_mathematical_models(test_periods=200)
    
    # 测试权重优化
    test_weighted_ensemble()
    
    print("\n" + "="*80)
    print("验证完成!")
    print("="*80)
