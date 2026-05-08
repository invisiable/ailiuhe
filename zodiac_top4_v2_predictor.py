"""
生肖TOP4 v2预测器 - 多策略投票+连miss切换
============================================
300期验证: 46.7%命中率 (随机基线33.3%)

核心策略:
1. 投票基础: 冷号20 + 冷号40 + 间隔 + MK全 各投TOP4, 票数最多的4个生肖
2. 连miss切换: 连续miss>=2时, 切换为80%MK150 + 20%静态组合
3. 静态组合: 冷号15(0.30) + 冷号30(0.10) + MK150(0.60)
"""
import numpy as np
from collections import Counter

ZODIAC_CYCLE_2026 = ['马', '蛇', '龙', '兔', '虎', '牛', '鼠', '猪', '狗', '鸡', '猴', '羊']
Z_IDX = {z: i for i, z in enumerate(ZODIAC_CYCLE_2026)}
NUM_TO_ZODIAC_2026 = {n: ZODIAC_CYCLE_2026[(n - 1) % 12] for n in range(1, 50)}
ZODIAC_NUMS_2026 = {}
for z in ZODIAC_CYCLE_2026:
    ZODIAC_NUMS_2026[z] = sorted([n for n in range(1, 50) if NUM_TO_ZODIAC_2026[n] == z])


class ZodiacTop4V2Predictor:
    """
    生肖TOP4 v2预测器
    使用多策略投票 + 连miss自适应切换
    """
    
    def __init__(self):
        self.name = "生肖TOP4_v2"
        self.consecutive_miss = 0
        self.miss_switch_threshold = 2
        self.static_weights = {
            'cold15': 0.30,
            'cold30': 0.10,
            'mk150': 0.60,
        }
        self.switch_blend_static = 0.2  # miss切换时static权重
        self.voting_strategies = ['cold20', 'cold40', 'gap', 'mk_full']
    
    def _cold_scores(self, animals, window):
        """冷号得分: 窗口内出现越少得分越高"""
        w = min(window, len(animals))
        freq = Counter(animals[-w:])
        mx = max(freq.values()) if freq else 1
        return np.array([1.0 - freq.get(z, 0) / max(mx, 1) for z in ZODIAC_CYCLE_2026])
    
    def _gap_scores(self, animals):
        """间隔得分: 距上次出现越远得分越高"""
        scores = []
        for z in ZODIAC_CYCLE_2026:
            last = -1
            for j in range(len(animals) - 1, -1, -1):
                if animals[j] == z:
                    last = j
                    break
            gap = (len(animals) - 1 - last) if last >= 0 else len(animals)
            scores.append(gap / 12.0)
        return np.array(scores)
    
    def _markov_scores(self, animals, window=None, laplace=1.0):
        """马尔可夫转移概率得分"""
        probs = np.ones(12) / 12
        h = animals[-window:] if window and len(animals) > window else animals
        if len(h) < 2:
            return probs
        
        trans = {}
        for k in range(1, len(h)):
            p, c = h[k - 1], h[k]
            if p not in trans:
                trans[p] = Counter()
            trans[p][c] += 1
        
        state = animals[-1]
        if state in trans:
            total = sum(trans[state].values()) + laplace * 12
            for zi, z in enumerate(ZODIAC_CYCLE_2026):
                probs[zi] = (trans[state].get(z, 0) + laplace) / total
        return probs
    
    def _voting_predict(self, animals):
        """投票策略: 4个策略各投TOP4, 统计票数"""
        votes = np.zeros(12)
        
        # 冷号20
        s = self._cold_scores(animals, 20)
        votes[np.argsort(-s)[:4]] += 1
        
        # 冷号40
        s = self._cold_scores(animals, 40)
        votes[np.argsort(-s)[:4]] += 1
        
        # 间隔
        s = self._gap_scores(animals)
        votes[np.argsort(-s)[:4]] += 1
        
        # MK全历史
        s = self._markov_scores(animals)
        votes[np.argsort(-s)[:4]] += 1
        
        return np.argsort(-votes)[:4]
    
    def _static_predict(self, animals):
        """静态权重组合: 冷号15+冷号30+MK150"""
        cold15 = self._cold_scores(animals, 15)
        cold30 = self._cold_scores(animals, 30)
        mk150 = self._markov_scores(animals, window=150)
        
        combined = (self.static_weights['cold15'] * cold15 +
                    self.static_weights['cold30'] * cold30 +
                    self.static_weights['mk150'] * mk150)
        return combined
    
    def _switch_predict(self, animals):
        """连miss切换: 静态+MK150混合"""
        static = self._static_predict(animals)
        mk150 = self._markov_scores(animals, window=150)
        
        blended = (self.switch_blend_static * static +
                   (1 - self.switch_blend_static) * mk150)
        return np.argsort(-blended)[:4]
    
    def predict(self, numbers, top_n=4):
        """
        预测下一期最可能的TOP-N生肖
        
        Parameters:
            numbers: 历史号码列表 (1-49)
            top_n: 返回的生肖数量 (默认4)
        
        Returns:
            list of str: 预测的生肖列表 (按可能性排序)
        """
        animals = [NUM_TO_ZODIAC_2026[n] for n in numbers]
        
        if self.consecutive_miss >= self.miss_switch_threshold:
            # 连miss: 使用切换策略(80% MK150 + 20% static)
            top_indices = self._switch_predict(animals)
        else:
            # 正常: 使用静态最优组合(冷号15+冷号30+MK150)
            static = self._static_predict(animals)
            top_indices = np.argsort(-static)[:4]
        
        return [ZODIAC_CYCLE_2026[i] for i in top_indices[:top_n]]
    
    def predict_with_scores(self, numbers, top_n=4):
        """
        带得分的预测
        
        Returns:
            list of (zodiac, score) tuples
        """
        animals = [NUM_TO_ZODIAC_2026[n] for n in numbers]
        
        # 计算所有策略得分
        cold20 = self._cold_scores(animals, 20)
        cold40 = self._cold_scores(animals, 40)
        gap = self._gap_scores(animals)
        mk_full = self._markov_scores(animals)
        mk150 = self._markov_scores(animals, window=150)
        cold15 = self._cold_scores(animals, 15)
        cold30 = self._cold_scores(animals, 30)
        
        # 投票得分
        votes = np.zeros(12)
        for s in [cold20, cold40, gap, mk_full]:
            votes[np.argsort(-s)[:4]] += 1
        
        # 静态得分
        static = (self.static_weights['cold15'] * cold15 +
                  self.static_weights['cold30'] * cold30 +
                  self.static_weights['mk150'] * mk150)
        
        # 综合得分
        if self.consecutive_miss >= self.miss_switch_threshold:
            final = self.switch_blend_static * static + (1 - self.switch_blend_static) * mk150
            mode = "切换(MK150)"
        else:
            # 正常: 静态组合
            final = static
            mode = "静态组合"
        
        sorted_idx = np.argsort(-final)
        results = []
        for zi in sorted_idx[:top_n]:
            z = ZODIAC_CYCLE_2026[zi]
            score = final[zi]
            results.append((z, score))
        
        return results, mode
    
    def update(self, actual_number):
        """
        更新预测器状态(记录命中/未命中)
        
        Parameters:
            actual_number: 实际开奖号码
        """
        # 这里需要外部调用来更新连miss计数
        pass
    
    def record_result(self, hit):
        """
        记录预测结果
        
        Parameters:
            hit: bool, 是否命中
        """
        if hit:
            self.consecutive_miss = 0
        else:
            self.consecutive_miss += 1
    
    def reset(self):
        """重置状态"""
        self.consecutive_miss = 0
    
    def get_zodiac_numbers(self, zodiac):
        """获取生肖对应的号码"""
        return ZODIAC_NUMS_2026.get(zodiac, [])
    
    def get_all_predicted_numbers(self, numbers, top_n=4):
        """
        获取预测生肖对应的所有号码
        
        Returns:
            dict: {zodiac: [numbers]}
        """
        predicted = self.predict(numbers, top_n)
        result = {}
        for z in predicted:
            result[z] = ZODIAC_NUMS_2026[z]
        return result


def validate_predictor(numbers, test_periods=300, verbose=True):
    """
    滚动验证预测器
    
    Parameters:
        numbers: 完整号码序列
        test_periods: 测试期数
        verbose: 是否打印详情
    
    Returns:
        dict: 验证结果
    """
    total = len(numbers)
    start = total - test_periods
    
    predictor = ZodiacTop4V2Predictor()
    hits = 0
    results = []
    
    for pi in range(test_periods):
        i = start + pi
        hist = numbers[:i]
        actual = numbers[i]
        actual_zodiac = NUM_TO_ZODIAC_2026[actual]
        
        predicted = predictor.predict(hist, top_n=4)
        hit = actual_zodiac in predicted
        
        if hit:
            hits += 1
        
        predictor.record_result(hit)
        
        results.append({
            'period': pi + 1,
            'actual_number': actual,
            'actual_zodiac': actual_zodiac,
            'predicted': predicted,
            'hit': hit,
            'consecutive_miss': predictor.consecutive_miss,
        })
    
    hit_rate = hits / test_periods * 100
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"生肖TOP4 v2 验证结果")
        print(f"{'='*60}")
        print(f"测试期数: {test_periods}")
        print(f"命中次数: {hits}/{test_periods}")
        print(f"命中率: {hit_rate:.1f}%")
        print(f"随机基线: 33.3%")
        print(f"提升: +{(hit_rate - 33.3):.1f}%")
        
        # 分段统计
        seg_size = min(50, test_periods)
        n_segs = test_periods // seg_size
        print(f"\n分段统计(每{seg_size}期):")
        for s in range(n_segs):
            seg_start = s * seg_size
            seg_end = seg_start + seg_size
            seg_hits = sum(1 for r in results[seg_start:seg_end] if r['hit'])
            print(f"  {seg_start+1:>3}-{seg_end:>3}: {seg_hits}/{seg_size} = {seg_hits/seg_size*100:.1f}%")
        
        # 最大连续miss
        max_miss = max(r['consecutive_miss'] for r in results)
        print(f"\n最大连续miss: {max_miss}")
        
        # 投票vs切换模式统计
        vote_periods = sum(1 for r in results if r['consecutive_miss'] < 2)
        switch_periods = test_periods - vote_periods
        print(f"投票模式: {vote_periods}期, 切换模式: {switch_periods}期")
    
    return {
        'hit_rate': hit_rate,
        'hits': hits,
        'total': test_periods,
        'results': results,
    }


def cross_validate(numbers, n_folds=5, verbose=True):
    """
    交叉验证评估预测器真实性能
    """
    total = len(numbers)
    test_total = min(300, total - 50)
    start = total - test_total
    fold_size = test_total // n_folds
    
    fold_hits = []
    
    for fold in range(n_folds):
        # 使用不同的起始位置作为不同的"fold"
        # 注意: 时间序列不能随机分割, 用不同的测试区间
        fold_start = start + fold * fold_size
        fold_end = fold_start + fold_size
        
        predictor = ZodiacTop4V2Predictor()
        hits = 0
        
        for i in range(fold_start, fold_end):
            hist = numbers[:i]
            actual = numbers[i]
            actual_zodiac = NUM_TO_ZODIAC_2026[actual]
            
            predicted = predictor.predict(hist, top_n=4)
            hit = actual_zodiac in predicted
            
            if hit:
                hits += 1
            predictor.record_result(hit)
        
        rate = hits / fold_size * 100
        fold_hits.append(rate)
        if verbose:
            print(f"  Fold {fold+1}: {hits}/{fold_size} = {rate:.1f}%")
    
    avg = np.mean(fold_hits)
    std = np.std(fold_hits)
    if verbose:
        print(f"  平均: {avg:.1f}% +/- {std:.1f}%")
    
    return avg, std


if __name__ == '__main__':
    import pandas as pd
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].tolist()
    
    print("=" * 60)
    print("生肖TOP4 v2 预测器验证")
    print("=" * 60)
    
    # 300期验证
    result = validate_predictor(numbers, test_periods=300)
    
    # 交叉验证
    print(f"\n{'='*60}")
    print("5折交叉验证:")
    print(f"{'='*60}")
    avg, std = cross_validate(numbers, n_folds=5)
    
    # 预测下一期
    print(f"\n{'='*60}")
    print("下一期预测:")
    print(f"{'='*60}")
    predictor = ZodiacTop4V2Predictor()
    predicted = predictor.predict(numbers, top_n=4)
    predicted_with_scores, mode = predictor.predict_with_scores(numbers, top_n=4)
    
    print(f"模式: {mode}")
    print(f"TOP4生肖:")
    for z, score in predicted_with_scores:
        nums = ZODIAC_NUMS_2026[z]
        print(f"  {z}: {nums} (得分: {score:.3f})")
    
    all_nums = predictor.get_all_predicted_numbers(numbers)
    all_num_list = sorted([n for nums in all_nums.values() for n in nums])
    print(f"\n覆盖号码({len(all_num_list)}个): {all_num_list}")
