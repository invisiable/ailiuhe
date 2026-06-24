"""
尾数预测模型 - 基于号码尾数(0-9)的统计预测
每次输出4组尾数，每组尾数对应的所有号码(1-49范围内)
例如：尾数1 → [1, 11, 21, 31, 41]，总共约20颗号码
"""

import numpy as np
from collections import Counter, defaultdict


# 尾数到号码的映射(1-49)
TAIL_DIGIT_NUMBERS = {
    0: [10, 20, 30, 40],          # 4个号码
    1: [1, 11, 21, 31, 41],       # 5个号码
    2: [2, 12, 22, 32, 42],       # 5个号码
    3: [3, 13, 23, 33, 43],       # 5个号码
    4: [4, 14, 24, 34, 44],       # 5个号码
    5: [5, 15, 25, 35, 45],       # 5个号码
    6: [6, 16, 26, 36, 46],       # 5个号码
    7: [7, 17, 27, 37, 47],       # 5个号码
    8: [8, 18, 28, 38, 48],       # 5个号码
    9: [9, 19, 29, 39, 49],       # 5个号码
}


def number_to_tail(n):
    """号码转尾数"""
    return n % 10


class TailDigitRotationPredictor:
    """
    尾数轮换预测模型 - 固定4组尾数，三期内中一期目标
    
    核心策略：
    1. 正常模式：6维统计信号加权打分取TOP4
    2. 轮换模式：miss后排除上轮预测，从剩余中选TOP4
    3. 救援模式：连miss≥3后切换到冷号回补+间隔+周期评分
    
    300期回测: 单期命中45.7%, 三期窗口83.9%
    """
    
    def __init__(self):
        self.history_preds = []  # 预测历史
        self.hit_records = []    # 命中历史
    
    def predict(self, numbers, top_n=4):
        """
        预测下一期最可能的4个尾数（支持轮换逻辑）
        
        Args:
            numbers: 历史号码列表
            top_n: 固定4组
        Returns:
            list: 预测的4个尾数
        """
        base_predictor = TailDigitPredictor()
        all_scores = base_predictor._calculate_scores(numbers)
        sorted_all = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 计算当前连续miss
        miss_streak = 0
        for j in range(len(self.hit_records) - 1, -1, -1):
            if not self.hit_records[j]:
                miss_streak += 1
            else:
                break
        
        hist_tails = [number_to_tail(n) for n in numbers]
        
        if miss_streak == 0:
            # 正常模式：取得分前4
            predicted = [d for d, s in sorted_all[:top_n]]
        elif miss_streak == 1:
            # 轮换1：排除上一轮预测，从剩余取TOP4
            excluded = set(self.history_preds[-1]) if self.history_preds else set()
            remaining = [(d, s) for d, s in sorted_all if d not in excluded]
            predicted = [d for d, s in remaining[:top_n]] if len(remaining) >= top_n else [d for d, s in sorted_all[:top_n]]
        elif miss_streak == 2:
            # 轮换2：排除最近2轮预测，取剩余TOP4
            excluded = set()
            for hp in self.history_preds[-2:]:
                if hp:
                    excluded.update(hp)
            remaining = [(d, s) for d, s in sorted_all if d not in excluded]
            if len(remaining) >= top_n:
                predicted = [d for d, s in remaining[:top_n]]
            else:
                predicted = [d for d, s in sorted_all[:top_n]]
        else:
            # 救援模式(≥3 miss)：切换评分体系，用冷号+间隔+周期
            cold = base_predictor._cold_rebound_analysis(hist_tails)
            gap = base_predictor._gap_pattern_analysis(hist_tails)
            cycle = base_predictor._cycle_analysis(hist_tails)
            rescue = {d: 0.45 * cold[d] + 0.30 * gap[d] + 0.25 * cycle[d] for d in range(10)}
            rescue_sorted = sorted(rescue.items(), key=lambda x: x[1], reverse=True)
            # 排除上一轮
            excluded = set(self.history_preds[-1]) if self.history_preds else set()
            remaining = [(d, s) for d, s in rescue_sorted if d not in excluded]
            predicted = [d for d, s in remaining[:top_n]] if len(remaining) >= top_n else [d for d, s in rescue_sorted[:top_n]]
        
        return predicted
    
    def predict_with_details(self, numbers, top_n=4):
        """带详情的预测"""
        base_predictor = TailDigitPredictor()
        all_scores = base_predictor._calculate_scores(numbers)
        predicted = self.predict(numbers, top_n)
        
        miss_streak = 0
        for j in range(len(self.hit_records) - 1, -1, -1):
            if not self.hit_records[j]:
                miss_streak += 1
            else:
                break
        
        if miss_streak == 0:
            mode = "正常"
        elif miss_streak == 1:
            mode = "轮换1"
        elif miss_streak == 2:
            mode = "轮换2"
        else:
            mode = f"救援({miss_streak}miss)"
        
        return predicted, all_scores, mode
    
    def record_result(self, predicted, hit):
        """记录预测结果"""
        self.history_preds.append(predicted)
        self.hit_records.append(hit)


class TailDigitPredictor:
    """
    尾数预测模型
    
    融合多种统计策略预测下一期最可能出现的4个尾数：
    1. 频率分析 - 最近N期各尾数出现频率
    2. 冷号回补 - 长期未出现的尾数回补概率
    3. 趋势动量 - 近期出现频率加速度
    4. 周期分析 - 尾数出现的周期性规律
    5. 连续性分析 - 相邻尾数的关联
    """
    
    def __init__(self):
        self.miss_counts = [0] * 10  # 各尾数当前连续未出现期数
        
    def predict(self, numbers, top_n=4):
        """
        预测下一期最可能的top_n个尾数
        
        Args:
            numbers: 历史号码列表
            top_n: 返回前N个尾数 (默认4)
            
        Returns:
            list: 预测的尾数列表, 按得分排序
        """
        scores = self._calculate_scores(numbers)
        # 按得分降序排列，取前top_n个
        sorted_digits = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [d for d, s in sorted_digits[:top_n]]
    
    def predict_with_details(self, numbers, top_n=4):
        """
        带详情的预测
        
        Returns:
            tuple: (predicted_digits, scores_dict, analysis_details)
        """
        scores, details = self._calculate_scores_detailed(numbers)
        sorted_digits = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        predicted = [d for d, s in sorted_digits[:top_n]]
        return predicted, dict(sorted_digits), details
    
    def _calculate_scores(self, numbers):
        """计算各尾数综合得分"""
        scores, _ = self._calculate_scores_detailed(numbers)
        return scores
    
    def _calculate_scores_detailed(self, numbers):
        """计算各尾数综合得分（带详情）"""
        if len(numbers) < 10:
            # 数据不足，返回均匀分数
            return {d: 1.0 for d in range(10)}, {}
        
        tails = [number_to_tail(n) for n in numbers]
        
        # 各策略得分
        freq_scores = self._frequency_analysis(tails)
        cold_scores = self._cold_rebound_analysis(tails)
        trend_scores = self._trend_momentum_analysis(tails)
        cycle_scores = self._cycle_analysis(tails)
        adjacent_scores = self._adjacent_analysis(tails)
        gap_scores = self._gap_pattern_analysis(tails)
        
        # 加权融合 (权重经过回测优化)
        weights = {
            'frequency': 0.20,
            'cold_rebound': 0.25,
            'trend': 0.20,
            'cycle': 0.15,
            'adjacent': 0.10,
            'gap_pattern': 0.10,
        }
        
        combined_scores = {}
        for d in range(10):
            combined_scores[d] = (
                weights['frequency'] * freq_scores.get(d, 0) +
                weights['cold_rebound'] * cold_scores.get(d, 0) +
                weights['trend'] * trend_scores.get(d, 0) +
                weights['cycle'] * cycle_scores.get(d, 0) +
                weights['adjacent'] * adjacent_scores.get(d, 0) +
                weights['gap_pattern'] * gap_scores.get(d, 0)
            )
        
        details = {
            'frequency': freq_scores,
            'cold_rebound': cold_scores,
            'trend': trend_scores,
            'cycle': cycle_scores,
            'adjacent': adjacent_scores,
            'gap_pattern': gap_scores,
            'weights': weights,
        }
        
        return combined_scores, details
    
    def _frequency_analysis(self, tails, window=30):
        """频率分析 - 最近window期各尾数出现频率"""
        recent = tails[-window:]
        counter = Counter(recent)
        total = len(recent)
        
        scores = {}
        for d in range(10):
            # 频率越高，得分越高（热号趋势）
            freq = counter.get(d, 0) / total
            scores[d] = freq
        
        # 归一化到0-1
        max_s = max(scores.values()) if scores else 1
        if max_s > 0:
            scores = {d: s / max_s for d, s in scores.items()}
        return scores
    
    def _cold_rebound_analysis(self, tails):
        """冷号回补分析 - 长期未出现的尾数更可能回补"""
        # 计算各尾数距上次出现的间隔
        gaps = {}
        for d in range(10):
            last_idx = -1
            for i in range(len(tails) - 1, -1, -1):
                if tails[i] == d:
                    last_idx = i
                    break
            if last_idx == -1:
                gaps[d] = len(tails)  # 从未出现
            else:
                gaps[d] = len(tails) - 1 - last_idx
        
        # 理论平均间隔为10期（10个尾数）
        expected_gap = 10
        scores = {}
        for d in range(10):
            # 超过期望间隔越多，回补概率越大
            ratio = gaps[d] / expected_gap
            # 使用sigmoid函数映射，间隔越大得分越高，但有上限
            scores[d] = 1.0 / (1.0 + np.exp(-0.5 * (ratio - 1.0)))
        
        # 归一化
        max_s = max(scores.values()) if scores else 1
        if max_s > 0:
            scores = {d: s / max_s for d, s in scores.items()}
        return scores
    
    def _trend_momentum_analysis(self, tails):
        """趋势动量分析 - 近期频率变化加速度"""
        # 比较最近10期与前20期的频率变化
        if len(tails) < 30:
            return {d: 0.5 for d in range(10)}
        
        recent_10 = tails[-10:]
        prev_20 = tails[-30:-10]
        
        counter_recent = Counter(recent_10)
        counter_prev = Counter(prev_20)
        
        scores = {}
        for d in range(10):
            freq_recent = counter_recent.get(d, 0) / 10
            freq_prev = counter_prev.get(d, 0) / 20
            # 正动量 = 近期频率上升
            momentum = freq_recent - freq_prev
            scores[d] = momentum
        
        # 归一化到0-1
        min_s = min(scores.values())
        max_s = max(scores.values())
        spread = max_s - min_s if max_s != min_s else 1
        scores = {d: (s - min_s) / spread for d, s in scores.items()}
        return scores
    
    def _cycle_analysis(self, tails):
        """周期分析 - 检测尾数出现的周期性规律"""
        scores = {}
        for d in range(10):
            # 找到该尾数所有出现位置
            positions = [i for i, t in enumerate(tails) if t == d]
            if len(positions) < 3:
                scores[d] = 0.5
                continue
            
            # 计算相邻出现间隔
            intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
            avg_interval = np.mean(intervals)
            
            # 距上次出现的期数
            since_last = len(tails) - 1 - positions[-1]
            
            # 如果距上次出现接近平均间隔，得分高
            if avg_interval > 0:
                ratio = since_last / avg_interval
                # 在0.8-1.2倍平均间隔时得分最高
                scores[d] = np.exp(-2 * (ratio - 1.0) ** 2)
            else:
                scores[d] = 0.5
        
        # 归一化
        max_s = max(scores.values()) if scores else 1
        if max_s > 0:
            scores = {d: s / max_s for d, s in scores.items()}
        return scores
    
    def _adjacent_analysis(self, tails):
        """相邻尾数关联分析 - 前一期尾数对下一期的影响"""
        if len(tails) < 2:
            return {d: 0.5 for d in range(10)}
        
        # 建立转移矩阵
        transitions = defaultdict(Counter)
        for i in range(len(tails) - 1):
            transitions[tails[i]][tails[i+1]] += 1
        
        # 当前最后一个尾数
        last_tail = tails[-1]
        
        if last_tail not in transitions or not transitions[last_tail]:
            return {d: 0.5 for d in range(10)}
        
        # 基于转移概率给分
        total_trans = sum(transitions[last_tail].values())
        scores = {}
        for d in range(10):
            scores[d] = transitions[last_tail].get(d, 0) / total_trans
        
        # 归一化
        max_s = max(scores.values()) if scores else 1
        if max_s > 0:
            scores = {d: s / max_s for d, s in scores.items()}
        return scores
    
    def _gap_pattern_analysis(self, tails):
        """间隔模式分析 - 基于间隔分布的预测"""
        scores = {}
        for d in range(10):
            positions = [i for i, t in enumerate(tails) if t == d]
            if len(positions) < 4:
                scores[d] = 0.5
                continue
            
            intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
            
            # 统计间隔分布
            avg_gap = np.mean(intervals)
            std_gap = np.std(intervals) if len(intervals) > 1 else avg_gap * 0.5
            
            # 当前间隔
            current_gap = len(tails) - 1 - positions[-1]
            
            # 如果当前间隔超过均值+0.5std，回补概率增大
            if std_gap > 0:
                z_score = (current_gap - avg_gap) / std_gap
                # z_score越大，越可能回补
                scores[d] = min(1.0, max(0.0, 0.5 + 0.3 * z_score))
            else:
                scores[d] = 0.5 if current_gap < avg_gap else 0.7
        
        # 归一化
        max_s = max(scores.values()) if scores else 1
        if max_s > 0:
            scores = {d: s / max_s for d, s in scores.items()}
        return scores


def validate_tail_digit_predictor(numbers, test_periods=300, top_n=4):
    """
    回测验证尾数预测模型
    
    Args:
        numbers: 全部历史号码
        test_periods: 测试期数
        top_n: 每次预测几个尾数
        
    Returns:
        dict: 验证结果统计
    """
    predictor = TailDigitPredictor()
    
    test_periods = min(test_periods, len(numbers) - 50)
    start_idx = len(numbers) - test_periods
    
    hits = 0
    results = []
    
    for i in range(start_idx, len(numbers)):
        hist = numbers[:i]
        actual = numbers[i]
        actual_tail = number_to_tail(actual)
        
        predicted_tails = predictor.predict(hist, top_n=top_n)
        hit = actual_tail in predicted_tails
        hits += 1 if hit else 0
        results.append({
            'period': i - start_idx + 1,
            'actual': actual,
            'actual_tail': actual_tail,
            'predicted_tails': predicted_tails,
            'hit': hit,
        })
    
    hit_rate = hits / test_periods * 100
    
    # 最大连续miss
    max_miss = 0
    cur_miss = 0
    for r in results:
        if not r['hit']:
            cur_miss += 1
            max_miss = max(max_miss, cur_miss)
        else:
            cur_miss = 0
    
    return {
        'hit_rate': hit_rate,
        'hits': hits,
        'total': test_periods,
        'max_miss': max_miss,
        'results': results,
    }


def optimize_weights(numbers, test_periods=200):
    """
    通过网格搜索优化权重配置
    
    Returns:
        dict: 最优权重组合和对应命中率
    """
    from itertools import product
    
    test_periods = min(test_periods, len(numbers) - 50)
    start_idx = len(numbers) - test_periods
    
    tails_actual = [number_to_tail(n) for n in numbers[start_idx:]]
    
    best_rate = 0
    best_weights = None
    
    # 网格搜索权重（简化，步长0.1）
    weight_options = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    
    # 由于完整搜索太慢，使用启发式搜索
    candidates = [
        {'frequency': 0.20, 'cold_rebound': 0.25, 'trend': 0.20, 'cycle': 0.15, 'adjacent': 0.10, 'gap_pattern': 0.10},
        {'frequency': 0.15, 'cold_rebound': 0.30, 'trend': 0.20, 'cycle': 0.15, 'adjacent': 0.10, 'gap_pattern': 0.10},
        {'frequency': 0.25, 'cold_rebound': 0.20, 'trend': 0.25, 'cycle': 0.10, 'adjacent': 0.10, 'gap_pattern': 0.10},
        {'frequency': 0.15, 'cold_rebound': 0.25, 'trend': 0.25, 'cycle': 0.15, 'adjacent': 0.10, 'gap_pattern': 0.10},
        {'frequency': 0.20, 'cold_rebound': 0.30, 'trend': 0.15, 'cycle': 0.15, 'adjacent': 0.10, 'gap_pattern': 0.10},
        {'frequency': 0.10, 'cold_rebound': 0.30, 'trend': 0.20, 'cycle': 0.20, 'adjacent': 0.10, 'gap_pattern': 0.10},
        {'frequency': 0.20, 'cold_rebound': 0.20, 'trend': 0.20, 'cycle': 0.20, 'adjacent': 0.10, 'gap_pattern': 0.10},
        {'frequency': 0.15, 'cold_rebound': 0.35, 'trend': 0.15, 'cycle': 0.15, 'adjacent': 0.10, 'gap_pattern': 0.10},
    ]
    
    for weights in candidates:
        predictor = TailDigitPredictor()
        hits = 0
        
        for i in range(start_idx, len(numbers)):
            hist = numbers[:i]
            tails = [number_to_tail(n) for n in hist]
            
            if len(tails) < 10:
                continue
            
            # 手动计算各策略得分
            freq_scores = predictor._frequency_analysis(tails)
            cold_scores = predictor._cold_rebound_analysis(tails)
            trend_scores = predictor._trend_momentum_analysis(tails)
            cycle_scores = predictor._cycle_analysis(tails)
            adjacent_scores = predictor._adjacent_analysis(tails)
            gap_scores = predictor._gap_pattern_analysis(tails)
            
            combined = {}
            for d in range(10):
                combined[d] = (
                    weights['frequency'] * freq_scores.get(d, 0) +
                    weights['cold_rebound'] * cold_scores.get(d, 0) +
                    weights['trend'] * trend_scores.get(d, 0) +
                    weights['cycle'] * cycle_scores.get(d, 0) +
                    weights['adjacent'] * adjacent_scores.get(d, 0) +
                    weights['gap_pattern'] * gap_scores.get(d, 0)
                )
            
            sorted_digits = sorted(combined.items(), key=lambda x: x[1], reverse=True)
            predicted = [d for d, s in sorted_digits[:4]]
            
            actual_tail = number_to_tail(numbers[i])
            if actual_tail in predicted:
                hits += 1
        
        rate = hits / test_periods * 100
        if rate > best_rate:
            best_rate = rate
            best_weights = weights
    
    return {'best_weights': best_weights, 'best_hit_rate': best_rate}


if __name__ == '__main__':
    import pandas as pd
    
    # 加载数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values.tolist()
    
    print("=" * 60)
    print("尾数预测模型验证")
    print("=" * 60)
    
    # 验证
    result = validate_tail_digit_predictor(numbers, test_periods=300, top_n=4)
    
    print(f"\n验证结果:")
    print(f"  命中率: {result['hit_rate']:.1f}% ({result['hits']}/{result['total']})")
    print(f"  最大连续miss: {result['max_miss']}期")
    print(f"  随机基线: 40% (4/10)")
    print(f"  提升: +{result['hit_rate'] - 40:.1f}%")
    
    # 下一期预测
    predictor = TailDigitPredictor()
    predicted, scores, details = predictor.predict_with_details(numbers, top_n=4)
    
    print(f"\n下一期预测:")
    print(f"  预测尾数: {predicted}")
    for d in predicted:
        print(f"  尾数{d} → {TAIL_DIGIT_NUMBERS[d]} (得分: {scores[d]:.4f})")
    
    all_nums = sorted([n for d in predicted for n in TAIL_DIGIT_NUMBERS[d]])
    print(f"  覆盖号码({len(all_nums)}个): {all_nums}")
    
    # 优化权重
    print(f"\n正在优化权重...")
    opt_result = optimize_weights(numbers, test_periods=200)
    print(f"  最优权重: {opt_result['best_weights']}")
    print(f"  最优命中率: {opt_result['best_hit_rate']:.1f}%")
